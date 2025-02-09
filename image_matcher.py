import torch
import psycopg2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from typing import Dict, List, Optional, Union
from transformers import CLIPProcessor, CLIPModel
from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv

def get_wiki_article_info(article_id: str) -> Dict[str, str]:
    """Fetch article title and URL from Wikipedia API using article ID."""
    api_url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "pageids": article_id,
        "prop": "info",
        "inprop": "url|displaytitle"
    }
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        page = data["query"]["pages"][article_id]
        return {
            "title": page["title"],
            "url": page["fullurl"]
        }
    except Exception as e:
        return {
            "title": f"Article {article_id}",
            "url": f"https://en.wikipedia.org/?curid={article_id}"
        }

class BaseEmbeddingsReader(ABC):
    """Abstract base class for embedding readers."""
    
    def __init__(self, 
                 db_url: str = None,
                 model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize the base reader with common functionality."""
        load_dotenv()
        self.db_url = db_url or os.getenv('DATABASE_URL')
        if not self.db_url:
            raise ValueError("Database URL must be provided either directly or via DATABASE_URL environment variable")
            
        self.device = self._get_device()
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @staticmethod
    def _get_device() -> str:
        """Determine the appropriate device for computation."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def get_article_by_id(self, article_id: str) -> Optional[Dict]:
        """Get a specific article by its ID."""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT article_id, hash, embedding
                    FROM embeddings 
                    WHERE article_id = %s
                """, (article_id,))
                
                row = cursor.fetchone()
                if row:
                    article_info = get_wiki_article_info(row[0])
                    return {
                        'article_id': row[0],
                        'hash': row[1],
                        'embedding': np.array(row[2]),
                        'title': article_info['title'],
                        'url': article_info['url']
                    }
                return None

    def get_database_stats(self) -> Dict:
        """Get statistics about the database."""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cursor:
                stats = {}
                
                cursor.execute("SELECT COUNT(*) FROM embeddings")
                stats['total_articles'] = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT article_id 
                    FROM embeddings 
                    ORDER BY article_id DESC 
                    LIMIT 1
                """)
                row = cursor.fetchone()
                if row:
                    article_info = get_wiki_article_info(row[0])
                    stats['most_recent_article'] = {
                        'article_id': row[0],
                        'title': article_info['title'],
                        'url': article_info['url']
                    }
                
                return stats

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

    def get_similar_articles(self, image_path: str, limit: int = 15, threshold: float = 0.5) -> List[Dict]:
        """Find similar articles based on input image using the SQL function."""
        try:
            # Get the embedding for the input image
            embedding = self.get_embedding(image_path)
            embedding_list = embedding.cpu().numpy()[0].tolist()
            
            # Convert the embedding to a string format PostgreSQL can parse
            embedding_str = f'[{",".join(map(str, embedding_list))}]'
            
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    # Call the match_embeddings function
                    cursor.execute("""
                        SELECT * FROM match_embeddings(
                            %s::vector(512),
                            match_threshold := %s,
                            match_count := %s
                        )
                    """, (embedding_str, threshold, limit))
                    
                    # Fetch results
                    rows = cursor.fetchall()
                    
                    # Get article info for each result
                    results = []
                    for row in rows:
                        article_id, similarity = row
                        article_info = get_wiki_article_info(article_id)
                        results.append({
                            'article_id': article_id,
                            'title': article_info['title'],
                            'url': article_info['url'],
                            'similarity': similarity
                        })
                    
                    return results
                    
        except Exception as e:
            raise Exception(f"Error finding similar articles: {str(e)}")