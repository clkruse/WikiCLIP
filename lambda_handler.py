import json
import os
import torch
# boto3 and numpy are not used directly in this module
from PIL import Image
import requests
from io import BytesIO
from typing import Dict, List
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import base64
import time

from s3_vectors_client import S3VectorsClient, format_similarity

# Initialize global instances for reuse across invocations
_matcher = None

def get_matcher():
    global _matcher
    if _matcher is None:
        start_time = time.time()
        _matcher = LambdaImageMatcher()
        print(f"[TIMING] Model initialization took: {time.time() - start_time:.2f}s")
    return _matcher

class LambdaImageMatcher:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize the matcher with CLIP model and S3 Vectors client."""
        # Initialize CLIP vision-only model (smaller, faster to load) and image processor
        start_time = time.time()
        self.device = "cpu"  # Lambda only supports CPU
        cache_dir = os.environ.get("TRANSFORMERS_CACHE", "/opt/ml/model")
        # Prefer a local directory with pre-downloaded assets if present
        model_path = os.environ.get("CLIP_MODEL_PATH", "/opt/ml/clip")
        # Limit intraop threads to reduce init overhead in Lambda
        try:
            torch.set_num_threads(max(1, int(os.environ.get("PYTORCH_NUM_THREADS", "1"))))
        except Exception:
            pass

        self.model = CLIPVisionModelWithProjection.from_pretrained(
            model_path if os.path.isdir(model_path) else model_name,
            local_files_only=True,
            cache_dir=cache_dir,
        ).to(self.device)
        print(f"[TIMING] Model loading took: {time.time() - start_time:.2f}s")
        
        start_time = time.time()
        self.processor = CLIPImageProcessor.from_pretrained(
            model_path if os.path.isdir(model_path) else model_name,
            local_files_only=True,
            cache_dir=cache_dir,
        )
        print(f"[TIMING] Processor loading took: {time.time() - start_time:.2f}s")
        
        self.model.eval()  # Ensure model is in eval mode
        torch.set_grad_enabled(False)  # Disable gradient computation globally
        
        # Initialize S3 Vectors client (uses env vars for bucket/index/region)
        self.s3vectors = S3VectorsClient()
        
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
            inputs = self.processor(images=image, return_tensors="pt")
            print(f"[TIMING] Image preprocessing took: {time.time() - start_time:.2f}s")
            
            start_time = time.time()
            outputs = self.model(pixel_values=inputs["pixel_values"].to(self.device))
            image_features = outputs.image_embeds
            print(f"[TIMING] Model inference took: {time.time() - start_time:.2f}s")
            
            return image_features / image_features.norm(dim=-1, keepdim=True)
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")

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


    def get_similar_articles(self, image_data: str, limit: int = 15, threshold: float = 0.5) -> List[Dict]:
        """Find similar articles based on input image using S3 Vectors."""
        try:
            start_time = time.time()
            # Get the embedding for the input image
            embedding = self.get_embedding(image_data)
            embedding_list = embedding.cpu().numpy()[0].tolist()
            print(f"[TIMING] Embedding generation total took: {time.time() - start_time:.2f}s")
            start_time = time.time()
            # Query S3 Vectors
            vectors = self.s3vectors.query(
                query_vector=embedding_list,
                top_k=limit,
                return_distance=True,
                return_metadata=True,
            )
            print(f"[TIMING] S3 Vectors query took: {time.time() - start_time:.2f}s")

            results: List[Dict] = []
            for item in vectors:
                sim = format_similarity(item.get("distance"), os.environ.get("S3_VECTORS_DISTANCE", "Cosine"))
                if sim is None or sim < threshold:
                    continue
                meta = item.get("metadata", {}) or {}
                article_id = str(meta.get("article_id", item.get("key", "")))
                # Backfill titles/urls if not stored in metadata
                title = meta.get("title")
                url = meta.get("url")
                if not title or not url:
                    article_info = self._get_wiki_articles_info([article_id]).get(article_id, {})
                    title = title or article_info.get("title", f"Article {article_id}")
                    url = url or article_info.get("url", f"https://en.wikipedia.org/?curid={article_id}")

                results.append(
                    {
                        "article_id": int(article_id) if article_id.isdigit() else article_id,
                        "title": title,
                        "url": url,
                        "similarity": sim,
                    }
                )

            return results
                    
        except Exception as e:
            raise Exception(f"Error finding similar articles: {str(e)}")

def lambda_handler(event, context):
    # Get the origin from the request
    origin = event.get('headers', {}).get('origin', '')
    allowed_domains = [
        'calebkruse.com',
        'clkruse.github.io',
        'localhost',
        '127.0.0.1'
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
        threshold = float(body.get('threshold', 0.05))
        
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