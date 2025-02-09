from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union, List, Dict
import time
import logging
import asyncio
import aiohttp

import base64
import numpy as np
import torch
import uvicorn
import ssl
from PIL import Image
from io import BytesIO
import psycopg2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your existing ImageMatcher class and wiki article info function
from image_matcher import ImageMatcher, get_wiki_article_info

async def async_wiki_article_info(session: aiohttp.ClientSession, article_id: str) -> Dict[str, str]:
    """Async version of get_wiki_article_info"""
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "pageids": article_id,
        "prop": "info",
        "inprop": "url|displaytitle"
    }
    
    try:
        async with session.get(api_url, params=params) as response:
            data = await response.json()
            page = data["query"]["pages"][article_id]
            return {
                "title": page["title"],
                "url": page["fullurl"]
            }
    except Exception as e:
        logger.error(f"Error fetching article {article_id}: {str(e)}")
        return {
            "title": f"Article {article_id}",
            "url": f"https://en.wikipedia.org/?curid={article_id}"
        }

class WebImageMatcher(ImageMatcher):
    """Concrete implementation of ImageMatcher for web interface"""
    
    def __init__(self):
        start_time = time.time()
        super().__init__()
        # Cache for embeddings
        self._embedding_cache = {}
        # Ensure model is in eval mode and use half precision
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.half()  # Convert to half precision
        # Create a session for async requests
        self._session = None
        logger.info(f"WebImageMatcher initialization took {time.time() - start_time:.2f}s")
    
    async def get_session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _find_similar_articles(self, query_embedding: np.ndarray, limit: int) -> List[Dict]:
        """Find similar articles based on embedding."""
        start_time = time.time()
        
        # Convert the embedding to a string format PostgreSQL can parse
        embedding_list = query_embedding[0].tolist()
        embedding_str = f'[{",".join(map(str, embedding_list))}]'
        logger.info(f"Embedding conversion took {time.time() - start_time:.2f}s")
        
        db_start_time = time.time()
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cursor:
                # Use a more optimized query with index scan
                query_start_time = time.time()
                cursor.execute("""
                    SELECT 
                        article_id,
                        (embedding <=> %s::vector(512)) as similarity
                    FROM embeddings
                    WHERE (embedding <=> %s::vector(512)) < 0.8  -- Adjust threshold as needed
                    ORDER BY similarity ASC
                    LIMIT %s
                """, (embedding_str, embedding_str, limit))
                logger.info(f"Database query took {time.time() - query_start_time:.2f}s")
                
                # Fetch results in batch
                fetch_start_time = time.time()
                rows = cursor.fetchall()
                logger.info(f"Fetching results took {time.time() - fetch_start_time:.2f}s")
        logger.info(f"Total database operation took {time.time() - db_start_time:.2f}s")
        
        # Get article info for each result using cached function
        wiki_start_time = time.time()
        
        # Make concurrent API calls
        session = await self.get_session()
        tasks = []
        for row in rows:
            article_id, similarity = row
            tasks.append(asyncio.create_task(
                async_wiki_article_info(session, article_id)
            ))
        
        # Wait for all API calls to complete
        article_infos = await asyncio.gather(*tasks)
        
        # Combine results
        results = []
        for (article_id, similarity), article_info in zip(rows, article_infos):
            results.append({
                'article_id': article_id,
                'title': article_info['title'],
                'url': article_info['url'],
                'similarity': similarity
            })
        
        logger.info(f"Wiki info fetching took {time.time() - wiki_start_time:.2f}s")
        logger.info(f"Total _find_similar_articles took {time.time() - start_time:.2f}s")
        return results

    async def get_similar_articles(self, image_input: Union[str, BytesIO, Image.Image], limit: int = 15) -> List[Dict]:
        """Find articles similar to the input image."""
        start_time = time.time()
        # Get the embedding for the input image
        query_embedding = self.get_embedding(image_input).cpu().numpy()
        logger.info(f"Getting query embedding took {time.time() - start_time:.2f}s")
        
        # Find similar articles using the optimized method
        similar_start_time = time.time()
        results = await self._find_similar_articles(query_embedding, limit)
        logger.info(f"Finding similar articles took {time.time() - similar_start_time:.2f}s")
        
        logger.info(f"Total get_similar_articles took {time.time() - start_time:.2f}s")
        return results

    def get_embedding(self, image_input: Union[str, BytesIO, Image.Image]) -> torch.Tensor:
        """Generate embedding for input image."""
        try:
            image = self._load_image(image_input)
            with torch.no_grad():
                inputs = self.processor(
                    images=image,
                    return_tensors="pt"
                )
                image_features = self.model.get_image_features(
                    **{k: v.to(self.device) for k, v in inputs.items()}
                )
                return image_features / image_features.norm(dim=-1, keepdim=True)
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create an instance of WebImageMatcher instead of ImageMatcher
matcher = WebImageMatcher()

# Serve static files (if you have any CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# For serving the HTML template
templates = Jinja2Templates(directory="templates")

class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Serve your HTML file
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/find-similar")
async def find_similar(image_request: ImageRequest):
    start_time = time.time()
    try:
        # Remove the data URL prefix if present
        decode_start = time.time()
        if "base64," in image_request.image:
            image_data = image_request.image.split("base64,")[1]
        else:
            image_data = image_request.image
            
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        logger.info(f"Image decoding took {time.time() - decode_start:.2f}s")
        
        # Convert image to buffer
        buffer_start = time.time()
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        logger.info(f"Buffer conversion took {time.time() - buffer_start:.2f}s")
        
        # Get similar articles using your existing ImageMatcher class
        match_start = time.time()
        similar_articles = await matcher.get_similar_articles(buffer, limit=15)
        logger.info(f"Getting similar articles took {time.time() - match_start:.2f}s")
        
        # Format the results
        format_start = time.time()
        results = [
            {
                "title": article["title"],
                "similarity": article["similarity"],
                "url": article.get("url", "")  # Include URL if available
            }
            for article in similar_articles
        ]
        logger.info(f"Results formatting took {time.time() - format_start:.2f}s")
        
        logger.info(f"Total API request took {time.time() - start_time:.2f}s")
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Error in API endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Create aiohttp session on startup"""
    await matcher.get_session()

@app.on_event("shutdown")
async def shutdown_event():
    """Close aiohttp session on shutdown"""
    await matcher.close()

if __name__ == "__main__":
    # Create SSL context
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    try:
        ssl_context.load_cert_chain('cert.pem', 'key.pem')
        # Run with SSL
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000, 
            ssl_keyfile="key.pem",
            ssl_certfile="cert.pem"
        )
    except Exception as e:
        print(f"Error loading SSL certificates: {e}")
        print("Running without SSL - camera functionality will only work on localhost")
        uvicorn.run(app, host="0.0.0.0", port=8000)