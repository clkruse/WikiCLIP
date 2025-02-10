from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union, List, Dict
import time
import logging
import aiohttp
import os
from dotenv import load_dotenv
import base64
from PIL import Image
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Lambda URL from environment variable
LAMBDA_URL = os.getenv('LAMBDA_URL')
if not LAMBDA_URL:
    raise ValueError("LAMBDA_URL environment variable is not set")

def resize_base64_image(base64_str: str, max_size: int = 512) -> str:
    """Resize image to reduce payload size while maintaining aspect ratio."""
    try:
        # Remove data URL prefix if present
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[1]
        
        # Decode base64 to image
        image_data = base64.b64decode(base64_str)
        img = Image.open(BytesIO(image_data))
        
        # Calculate new dimensions while maintaining aspect ratio
        ratio = max_size / max(img.size)
        if ratio < 1:  # Only resize if the image is larger than max_size
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert back to base64
        buffer = BytesIO()
        img = img.convert('RGB')  # Convert to RGB to ensure JPEG compatibility
        img.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        resized_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        logger.info(f"Image resized from {len(base64_str)} to {len(resized_base64)} chars")
        return resized_base64
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image data")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# For serving static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# For serving the HTML template
templates = Jinja2Templates(directory="templates")

class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Serve your HTML file
    return templates.TemplateResponse("index.html", {"request": request})

async def get_session():
    return aiohttp.ClientSession()

@app.post("/api/find-similar")
async def find_similar(image_request: ImageRequest):
    start_time = time.time()
    try:
        # Resize image before sending to Lambda
        resized_image = resize_base64_image(image_request.image)
        
        # Prepare the request payload for Lambda
        payload = {
            "image": resized_image,
            "limit": 15,
            "threshold": 0.5
        }
        
        # Make async request to Lambda
        async with aiohttp.ClientSession() as session:
            async with session.post(LAMBDA_URL, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Lambda request failed: {error_text}")
                    raise HTTPException(status_code=response.status, detail="Lambda processing failed")
                
                result = await response.json()
                
        logger.info(f"Total API request took {time.time() - start_time:.2f}s")
        return result
    
    except Exception as e:
        logger.error(f"Error in API endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)