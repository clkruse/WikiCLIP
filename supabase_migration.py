import sqlite3
import os
from supabase import create_client
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
import json

def chunk_list(lst: list, chunk_size: int):
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def migrate_to_supabase(sqlite_path: str, supabase_url: str, supabase_key: str):
    """
    Migrate data from SQLite to Supabase.
    
    Args:
        sqlite_path: Path to the SQLite database file
        supabase_url: Supabase project URL
        supabase_key: Supabase service role key (or anon key with RLS policies)
    """
    # Connect to SQLite database
    print("Connecting to SQLite database...")
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_cur = sqlite_conn.cursor()
    
    # Initialize Supabase client
    print("Connecting to Supabase...")
    supabase = create_client(supabase_url, supabase_key)
    
    try:
        # Migrate embeddings data
        print("Migrating embeddings table...")
        sqlite_cur.execute("SELECT COUNT(*) FROM embeddings")
        total_embeddings = sqlite_cur.fetchone()[0]
        
        sqlite_cur.execute("SELECT article_id, title, url, embedding, processed_date, hash FROM embeddings")
        batch_size = 50  # Smaller batch size for Supabase
        
        with tqdm(total=total_embeddings) as pbar:
            while True:
                rows = sqlite_cur.fetchmany(batch_size)
                if not rows:
                    break
                
                # Convert rows to list of dictionaries
                batch_data = []
                for row in rows:
                    article_id, title, url, embedding, processed_date, hash_val = row
                    # Convert embedding numpy array to list for JSON serialization
                    embedding_array = np.frombuffer(embedding, dtype=np.float32)
                    
                    batch_data.append({
                        "article_id": article_id,
                        "title": title,
                        "url": url,
                        "embedding": embedding_array.tolist(),
                        "processed_date": processed_date,
                        "hash": hash_val
                    })
                
                # Insert batch into Supabase
                data, count = supabase.table('embeddings').upsert(batch_data).execute()
                pbar.update(len(batch_data))
        
        # Migrate failed_articles data
        print("\nMigrating failed_articles table...")
        sqlite_cur.execute("SELECT COUNT(*) FROM failed_articles")
        total_failed = sqlite_cur.fetchone()[0]
        
        sqlite_cur.execute("SELECT article_id, title, error_message, attempt_date FROM failed_articles")
        
        with tqdm(total=total_failed) as pbar:
            while True:
                rows = sqlite_cur.fetchmany(batch_size)
                if not rows:
                    break
                
                # Convert rows to list of dictionaries
                batch_data = []
                for row in rows:
                    article_id, title, error_message, attempt_date = row
                    batch_data.append({
                        "article_id": article_id,
                        "title": title,
                        "error_message": error_message,
                        "attempt_date": attempt_date
                    })
                
                # Insert batch into Supabase
                data, count = supabase.table('failed_articles').upsert(batch_data).execute()
                pbar.update(len(batch_data))
        
        print("\nMigration completed successfully!")
        
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        raise
    
    finally:
        # Close SQLite connection
        sqlite_cur.close()
        sqlite_conn.close()

if __name__ == "__main__":
    # Get Supabase credentials from environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
    
    sqlite_path = "wiki_embeddings.db"  # Path to your local SQLite database
    
    migrate_to_supabase(sqlite_path, supabase_url, supabase_key) 