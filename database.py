import os
from supabase import create_client, Client
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

class Database:
    def __init__(self):
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
    
    def add_embedding(self, article_id: str, title: str, url: str, embedding: np.ndarray, hash_val: str) -> None:
        """Add an embedding to the database."""
        data = {
            "article_id": article_id,
            "title": title,
            "url": url,
            "embedding": embedding.tolist(),  # Convert numpy array to list for JSON serialization
            "processed_date": datetime.now().isoformat(),
            "hash": hash_val
        }
        
        self.supabase.table('embeddings').upsert(data).execute()
    
    def add_failed_article(self, article_id: str, title: str, error_message: str) -> None:
        """Add a failed article to the database."""
        data = {
            "article_id": article_id,
            "title": title,
            "error_message": error_message,
            "attempt_date": datetime.now().isoformat()
        }
        
        self.supabase.table('failed_articles').upsert(data).execute()
    
    def get_embedding(self, article_id: str) -> Optional[Tuple[np.ndarray, str]]:
        """Get an embedding from the database by article ID."""
        response = self.supabase.table('embeddings') \
            .select('embedding, hash') \
            .eq('article_id', article_id) \
            .execute()
        
        if not response.data:
            return None
        
        embedding_list = response.data[0]['embedding']
        hash_val = response.data[0]['hash']
        return np.array(embedding_list, dtype=np.float32), hash_val
    
    def get_similar_articles(self, query_embedding: np.ndarray, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar articles using vector similarity search.
        Uses pgvector's L2 distance by default in Supabase.
        """
        # Convert numpy array to list for JSON serialization
        embedding_list = query_embedding.tolist()
        
        # Using Postgres vector operators through Supabase
        response = self.supabase.rpc(
            'match_articles',  # We'll create this function
            {
                'query_embedding': embedding_list,
                'match_threshold': 0.75,
                'match_count': limit
            }
        ).execute()
        
        return response.data
    
    def article_exists(self, article_id: str) -> bool:
        """Check if an article exists in the database."""
        response = self.supabase.table('embeddings') \
            .select('article_id') \
            .eq('article_id', article_id) \
            .execute()
        
        return len(response.data) > 0
    
    def get_failed_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Get a failed article from the database."""
        response = self.supabase.table('failed_articles') \
            .select('*') \
            .eq('article_id', article_id) \
            .execute()
        
        return response.data[0] if response.data else None 