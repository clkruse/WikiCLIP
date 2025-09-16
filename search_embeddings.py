import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict
import os
from dotenv import load_dotenv

from s3_vectors_client import S3VectorsClient, format_similarity

# Load environment variables
load_dotenv()

S3_VECTORS_DISTANCE = os.getenv("S3_VECTORS_DISTANCE", "Cosine")

class TextMatcher:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize the text matcher with CLIP model."""
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

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using CLIP.
        
        Args:
            text (str): Text to get embedding for
            
        Returns:
            List[float]: Embedding vector
        """
        with torch.no_grad():
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True
            )
            text_features = self.model.get_text_features(
                **{k: v.to(self.device) for k, v in inputs.items()}
            )
            # Normalize the features
            normalized_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return normalized_features.cpu().numpy()[0].tolist()

def search_similar_articles(query_text: str, limit: int = 5, threshold: float = 0.0) -> List[Dict]:
    """
    Search for articles similar to the query text.
    
    Args:
        query_text (str): Text to search for
        limit (int): Number of results to return
        
    Returns:
        List[Dict]: List of similar articles with their metadata
    """
    # Initialize S3 Vectors client and text matcher
    client = S3VectorsClient()
    matcher = TextMatcher()
    
    # Get embedding for query text
    query_embedding = matcher.get_embedding(query_text)
    
    # Perform vector similarity search via S3 Vectors
    results = client.query(query_vector=query_embedding, top_k=limit, return_distance=True, return_metadata=True)

    output: List[Dict] = []
    for item in results:
        distance = item.get("distance")
        similarity = format_similarity(distance, S3_VECTORS_DISTANCE)
        if similarity is None or similarity < threshold:
            continue
        meta = item.get("metadata", {}) or {}
        output.append(
            {
                "article_id": int(meta.get("article_id", item.get("key", 0))),
                "title": meta.get("title", f"Article {item.get('key')}")
                if meta.get("title")
                else f"Article {item.get('key')}",
                "url": meta.get("url", ""),
                "similarity": similarity,
            }
        )

    return output

def main():
    # Example usage
    query = input("Enter your search query: ")
    results = search_similar_articles(query)
    
    print("\nTop similar articles:")
    print("-" * 50)
    for i, item in enumerate(results, 1):
        print(f"{i}. Article ID: {item['article_id']}")
        print(f"   Title: {item.get('title', 'N/A')}")
        print(f"   URL: {item.get('url', 'N/A')}")
        print(f"   Similarity: {item['similarity']:.4f}")
        print()

if __name__ == "__main__":
    main() 
