import torch
from transformers import CLIPProcessor, CLIPModel
from supabase import create_client
from typing import List, Dict
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

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

def search_similar_articles(query_text: str, limit: int = 5) -> List[Dict]:
    """
    Search for articles similar to the query text.
    
    Args:
        query_text (str): Text to search for
        limit (int): Number of results to return
        
    Returns:
        List[Dict]: List of similar articles with their metadata
    """
    # Initialize Supabase client and text matcher
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    matcher = TextMatcher()
    
    # Get embedding for query text
    query_embedding = matcher.get_embedding(query_text)
    
    # Perform vector similarity search
    result = (
        supabase.rpc(
            'match_articles',
            {
                'query_embedding': query_embedding,
                'match_threshold': 0.5,
                'match_count': limit
            }
        )
        .execute()
    )
    
    return result.data

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
