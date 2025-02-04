import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from typing import Dict, List, Optional, Union
from transformers import CLIPProcessor, CLIPModel
from abc import ABC, abstractmethod
from database import Database

class BaseEmbeddingsReader(ABC):
    """Abstract base class for embedding readers."""
    
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32",
                 db: Optional[Database] = None):
        """Initialize the base reader with common functionality."""
        self.device = self._get_device()
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.db = db or Database()

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
        response = self.db.supabase.table('embeddings') \
            .select('*') \
            .eq('title', title) \
            .execute()
            
        if response.data:
            article = response.data[0]
            return {
                'article_id': article['article_id'],
                'title': article['title'],
                'url': article['url'],
                'embedding': np.array(article['embedding'], dtype=np.float32),
                'processed_date': article['processed_date']
            }
        return None

    def get_database_stats(self) -> Dict:
        """Get statistics about the database."""
        stats = {}
        
        # Get total articles count
        response = self.db.supabase.table('embeddings').select('count', count='exact').execute()
        stats['total_articles'] = response.count
        
        # Get most recent article
        response = self.db.supabase.table('embeddings') \
            .select('title,processed_date') \
            .order('processed_date', desc=True) \
            .limit(1) \
            .execute()
            
        if response.data:
            stats['most_recent_article'] = {
                'title': response.data[0]['title'],
                'date': response.data[0]['processed_date']
            }
        
        # Get failed articles count
        response = self.db.supabase.table('failed_articles').select('count', count='exact').execute()
        stats['failed_articles'] = response.count
        
        return stats

    def _find_similar_articles(self, query_embedding: np.ndarray, limit: int) -> List[Dict]:
        """Find similar articles based on embedding."""
        return self.db.get_similar_articles(query_embedding, limit)

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

    def get_similar_articles(self, image_input: Union[str, BytesIO, Image.Image], limit: int = 15) -> List[Dict]:
        """Find articles similar to the input image."""
        query_embedding = self.get_embedding(image_input).cpu().numpy()
        return self._find_similar_articles(query_embedding, limit)