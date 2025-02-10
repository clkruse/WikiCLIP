import json
import os
import torch
import psycopg2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from typing import Dict, List
from transformers import CLIPProcessor, CLIPModel
import base64
import time

# Initialize global instances for reuse across invocations
_matcher = None
_db_conn = None

def get_matcher():
    global _matcher
    if _matcher is None:
        start_time = time.time()
        _matcher = LambdaImageMatcher()
        print(f"[TIMING] Model initialization took: {time.time() - start_time:.2f}s")
    return _matcher

def get_db_connection():
    global _db_conn
    if _db_conn is None or _db_conn.closed:
        start_time = time.time()
        _db_conn = psycopg2.connect(os.environ['DATABASE_URL'])
        print(f"[TIMING] DB connection initialization took: {time.time() - start_time:.2f}s")
    return _db_conn

class LambdaImageMatcher:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize the matcher with CLIP model."""
        start_time = time.time()
        self.device = "cpu"  # Lambda only supports CPU
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        print(f"[TIMING] Model loading took: {time.time() - start_time:.2f}s")
        
        start_time = time.time()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print(f"[TIMING] Processor loading took: {time.time() - start_time:.2f}s")
        
        self.model.eval()  # Ensure model is in eval mode
        torch.set_grad_enabled(False)  # Disable gradient computation globally
        
    def _load_image(self, image_data: str) -> Image.Image:
        """Load an image from base64 string."""
        try:
            start_time = time.time()
            # Remove potential base64 prefix
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_data)
            img = Image.open(BytesIO(image_bytes))
            print(f"[TIMING] Image loading took: {time.time() - start_time:.2f}s")
            return img
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")

    def get_embedding(self, image_data: str) -> torch.Tensor:
        """Generate embedding for input image."""
        try:
            image = self._load_image(image_data)
            
            start_time = time.time()
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            )
            print(f"[TIMING] Image preprocessing took: {time.time() - start_time:.2f}s")
            
            start_time = time.time()
            image_features = self.model.get_image_features(
                **{k: v.to(self.device) for k, v in inputs.items()}
            )
            print(f"[TIMING] Model inference took: {time.time() - start_time:.2f}s")
            
            return image_features / image_features.norm(dim=-1, keepdim=True)
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")

    def get_similar_articles(self, image_data: str, limit: int = 15, threshold: float = 0.5) -> List[Dict]:
        """Find similar articles based on input image."""
        try:
            start_time = time.time()
            # Get the embedding for the input image
            embedding = self.get_embedding(image_data)
            embedding_list = embedding.cpu().numpy()[0].tolist()
            print(f"[TIMING] Embedding generation total took: {time.time() - start_time:.2f}s")
            
            # Convert the embedding to a string format PostgreSQL can parse
            embedding_str = f'[{",".join(map(str, embedding_list))}]'
            
            start_time = time.time()
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM match_embeddings(
                        %s::vector(512),
                        match_threshold := %s,
                        match_count := %s
                    )
                """, (embedding_str, threshold, limit))
                
                rows = cursor.fetchall()
            print(f"[TIMING] Database query took: {time.time() - start_time:.2f}s")
            
            if not rows:
                return []
            
            # Batch fetch article info
            start_time = time.time()
            article_ids = [str(row[0]) for row in rows]
            articles_info = self._get_wiki_articles_info(article_ids)
            print(f"[TIMING] Wikipedia API fetch took: {time.time() - start_time:.2f}s")
            
            return [
                {
                    'article_id': row[0],
                    'title': articles_info.get(str(row[0]), {}).get('title', f'Article {row[0]}'),
                    'url': articles_info.get(str(row[0]), {}).get('url', f'https://en.wikipedia.org/?curid={row[0]}'),
                    'similarity': row[1]
                }
                for row in rows
            ]
                    
        except Exception as e:
            raise Exception(f"Error finding similar articles: {str(e)}")

    def _get_wiki_articles_info(self, article_ids: List[str]) -> Dict[str, Dict[str, str]]:
        """Batch fetch article info from Wikipedia API."""
        try:
            api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "pageids": "|".join(article_ids),
                "prop": "info",
                "inprop": "url|displaytitle"
            }
            
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                pageid: {
                    "title": page["title"],
                    "url": page["fullurl"]
                }
                for pageid, page in data["query"]["pages"].items()
            }
        except Exception:
            return {}

def lambda_handler(event, context):
    # Get the origin from the request
    origin = event.get('headers', {}).get('origin', '')
    allowed_domains = [
        'calebkruse.com',
        'clkruse.github.io'
    ]
    
    # Check if origin matches any allowed domain
    is_allowed = any(domain in origin for domain in allowed_domains)
    cors_origin = origin if is_allowed else f'https://{allowed_domains[0]}'
    
    headers = {
        'Access-Control-Allow-Origin': cors_origin,
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600'
    }
    
    # Handle OPTIONS request (preflight)
    if event['requestContext']['http']['method'] == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }

    try:
        overall_start_time = time.time()
        
        # Get parameters from event
        body = json.loads(event['body']) if isinstance(event.get('body'), str) else event.get('body', {})
        image_data = body.get('image')
        limit = int(body.get('limit', 15))
        threshold = float(body.get('threshold', 0.5))
        
        if not image_data:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No image data provided'})
            }
        
        # Get similar articles using singleton matcher
        results = get_matcher().get_similar_articles(image_data, limit, threshold)
        
        total_time = time.time() - overall_start_time
        print(f"[TIMING] Total execution time: {total_time:.2f}s")
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({
                'results': results,
                'execution_time': total_time
            })
        }
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': str(e)})
        } 