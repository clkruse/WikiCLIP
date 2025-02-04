import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from typing import Dict, List, Optional, Union
from transformers import CLIPProcessor, CLIPModel
from abc import ABC, abstractmethod
from sqlalchemy.orm import Session
from database import SessionLocal, Embedding, FailedArticle

class BaseEmbeddingsReader(ABC):
    """Abstract base class for embedding readers."""
    
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize the base reader with common functionality."""
        self.device = self._get_device()
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.db = SessionLocal()

    def __del__(self):
        """Cleanup database connection."""
        if hasattr(self, 'db'):
            self.db.close()

    @staticmethod
    def _get_device() -> str:
        """Determine the appropriate device for computation."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def get_article_by_title(self, title: str) -> Optional[Dict]:
        """Get a specific article by its title."""
        article = self.db.query(Embedding).filter(Embedding.title == title).first()
        if article:
            return {
                'article_id': article.article_id,
                'title': article.title,
                'url': article.url,
                'embedding': np.frombuffer(article.embedding, dtype=np.float32),
                'processed_date': article.processed_date
            }
        return None

    def get_database_stats(self) -> Dict:
        """Get statistics about the database."""
        stats = {}
        
        stats['total_articles'] = self.db.query(Embedding).count()
        
        most_recent = self.db.query(Embedding).order_by(Embedding.processed_date.desc()).first()
        if most_recent:
            stats['most_recent_article'] = {
                'title': most_recent.title,
                'date': most_recent.processed_date
            }
        
        stats['failed_articles'] = self.db.query(FailedArticle).count()
        
        return stats

    def _find_similar_articles(self, query_embedding: np.ndarray, limit: int) -> List[Dict]:
        """Find similar articles based on embedding."""
        articles = self.db.query(Embedding).all()
        
        results = [
            {
                'article_id': article.article_id,
                'title': article.title,
                'url': article.url,
                'similarity': float(np.dot(query_embedding[0], np.frombuffer(article.embedding, dtype=np.float32)))
            }
            for article in articles
        ]
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]

    @abstractmethod
    def get_embedding(self, input_data: Union[str, str]) -> torch.Tensor:
        """Generate embedding for the input data."""
        pass

    @abstractmethod
    def get_similar_articles(self, input_data: Union[str, str], limit: int = 15) -> List[Dict]:
        """Find similar articles based on input data."""
        pass

class ImageMatcher(BaseEmbeddingsReader):
    """Reader for image-based embeddings."""

    def _load_image(self, image_input: Union[str, BytesIO, Image.Image]) -> Image.Image:
        """Load an image from either a local path, URL, bytes buffer, or PIL Image."""
        try:
            if isinstance(image_input, str):
                if image_input.startswith(('http://', 'https://')):
                    response = requests.get(image_input, timeout=10)
                    response.raise_for_status()
                    return Image.open(BytesIO(response.content))
                return Image.open(image_input)
            elif isinstance(image_input, BytesIO):
                return Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                return image_input
            else:
                raise ValueError("Unsupported image input type")
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")

    def get_embedding(self, image_path: str) -> torch.Tensor:
        """Generate embedding for input image."""
        try:
            image = self._load_image(image_path)
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