from fastapi import FastAPI, Request, File, UploadFile
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
import os
from dotenv import load_dotenv

# Import your existing ImageMatcher class and Database
from image_matcher import ImageMatcher
from database import Database

# Load environment variables
load_dotenv()

# Initialize database and image matcher
db = Database()
image_matcher = ImageMatcher(db=db)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        stats = image_matcher.get_database_stats()
        
        # Check if CLIP model is loaded
        device = image_matcher.device
        model_status = "loaded" if image_matcher.model is not None else "not loaded"
        
        return {
            "status": "healthy",
            "database": "connected",
            "model": model_status,
            "device": str(device),
            "stats": stats
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
        similar_articles = image_matcher.get_similar_articles(buffer, limit=15)
        
        return {"results": similar_articles}
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/search")
async def search(image: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await image.read()
        image_file = BytesIO(contents)
        
        # Get similar articles using the image matcher
        similar_articles = image_matcher.get_similar_articles(image_file)
        
        return {"matches": similar_articles}
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

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