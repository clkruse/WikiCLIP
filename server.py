from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union, List, Dict

import base64
import numpy as np
import torch
import uvicorn
import ssl
from PIL import Image
from io import BytesIO

# Import your existing ImageMatcher class
from image_matcher import ImageMatcher

# Import database components
from database import get_db, Embedding

class WebImageMatcher(ImageMatcher):
    """Concrete implementation of ImageMatcher for web interface"""
    
    def get_similar_articles(self, image_input: Union[str, BytesIO, Image.Image], limit: int = 15) -> List[Dict]:
        """Find articles similar to the input image."""
        # Get the embedding for the input image
        query_embedding = self.get_embedding(image_input).cpu().numpy()
        # Find similar articles using the base class method
        return self._find_similar_articles(query_embedding, limit)

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

# Create an instance of WebImageMatcher
matcher = WebImageMatcher()

# Serve static files (if you have any CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# For serving the HTML template
templates = Jinja2Templates(directory="templates")

class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

@app.get("/health")
async def health_check():
    """Health check endpoint to verify server and database status."""
    try:
        # Check database connection by making a simple query
        db = next(get_db())
        db.query(Embedding).limit(1).all()
        db.close()
        
        # Check if CLIP model is loaded
        device = matcher.device
        model_status = "loaded" if matcher.model is not None else "not loaded"
        
        return {
            "status": "healthy",
            "database": "connected",
            "model": model_status,
            "device": str(device)
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Serve your HTML file
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/find-similar")
async def find_similar(image_request: ImageRequest):
    try:
        # Remove the data URL prefix if present
        if "base64," in image_request.image:
            image_data = image_request.image.split("base64,")[1]
        else:
            image_data = image_request.image
            
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert image to buffer
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        
        # Get similar articles using your existing ImageMatcher class
        similar_articles = matcher.get_similar_articles(buffer, limit=15)
        
        # Format the results
        results = [
            {
                "title": article["title"],
                "similarity": article["similarity"],
                "url": article.get("url", "")  # Include URL if available
            }
            for article in similar_articles
        ]
        
        return {"results": results}
    
    except Exception as e:
        return {"error": str(e)}, 500

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