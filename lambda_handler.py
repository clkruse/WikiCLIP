import json
import os
import torch
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
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

def get_matcher():
    global _matcher
    if _matcher is None:
        start_time = time.time()
        _matcher = LambdaImageMatcher()
        print(f"[TIMING] Model initialization took: {time.time() - start_time:.2f}s")
    return _matcher

class LambdaImageMatcher:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize the matcher with CLIP model and OpenSearch client."""
        # Initialize CLIP model
        start_time = time.time()
        self.device = "cpu"  # Lambda only supports CPU
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        print(f"[TIMING] Model loading took: {time.time() - start_time:.2f}s")
        
        start_time = time.time()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print(f"[TIMING] Processor loading took: {time.time() - start_time:.2f}s")
        
        self.model.eval()  # Ensure model is in eval mode
        torch.set_grad_enabled(False)  # Disable gradient computation globally
        
        # Initialize OpenSearch client
        region = os.environ.get('AWS_REGION', 'us-east-1')
        service = 'es'
        credentials = boto3.Session().get_credentials()
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            service,
            session_token=credentials.token
        )
        
        self.opensearch = OpenSearch(
            hosts=[{'host': os.environ['OPENSEARCH_ENDPOINT'], 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30
        )
        
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
        """Find similar articles based on input image using OpenSearch k-NN."""
        try:
            start_time = time.time()
            # Get the embedding for the input image
            embedding = self.get_embedding(image_data)
            embedding_list = embedding.cpu().numpy()[0].tolist()
            print(f"[TIMING] Embedding generation total took: {time.time() - start_time:.2f}s")
            start_time = time.time()
            # Query OpenSearch using k-NN
            query = {
                "size": limit,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": embedding_list,
                            "k": limit
                        }
                    }
                },
                "_source": ["article_id", "title", "url"]
            }
            
            
            response = self.opensearch.search(
                body=query,
                index="wiki-embeddings"
            )
            print(f"[TIMING] OpenSearch query took: {time.time() - start_time:.2f}s")
                
            hits = response['hits']['hits']
            if not hits:
                return []
            
            # Format results
            article_ids = [str(hit['_source']['article_id']) for hit in hits]
            articles_info = self._get_wiki_articles_info(article_ids)
            
            results = []
            for hit in hits:
                score = hit['_score']
                if score < threshold:
                    continue
                    
                source = hit['_source']
                article_id = str(source['article_id'])
                results.append({
                    'article_id': source['article_id'],
                    'title': articles_info.get(article_id, {}).get('title', f'Article {article_id}'),
                    'url': articles_info.get(article_id, {}).get('url', f'https://en.wikipedia.org/?curid={article_id}'),
                    'similarity': score
                })
            
            return results
                    
        except Exception as e:
            raise Exception(f"Error finding similar articles: {str(e)}")

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