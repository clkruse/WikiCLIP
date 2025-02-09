import sqlite3
import json
import numpy as np
from pathlib import Path

def export_embeddings_to_json(db_path: str = "wiki_embeddings.db", output_path: str = "embeddings.json"):
    """
    Export embeddings from SQLite database to JSON format.
    
    Args:
        db_path (str): Path to the SQLite database
        output_path (str): Path where the JSON file will be saved
    """
    print(f"Reading embeddings from {db_path}...")
    
    # Connect to the database
    with sqlite3.connect(db_path) as conn:
        # Get all records
        cursor = conn.execute(
            "SELECT article_id, title, url, embedding, hash FROM embeddings"
        )
        
        # Convert records to dictionary format
        embeddings_data = []
        for row in cursor:
            article_id, title, url, embedding_bytes, hash_val = row
            
            # Convert embedding bytes to numpy array then to list
            embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            embeddings_data.append({
                "article_id": article_id,
                "title": title,
                "url": url,
                "embedding": embedding_array.tolist(),
                "hash": hash_val
            })
        
        print(f"Found {len(embeddings_data)} embeddings")
        
        # Save to JSON file
        print(f"Saving embeddings to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump({
                "embeddings": embeddings_data,
                "metadata": {
                    "count": len(embeddings_data),
                    "embedding_dim": len(embeddings_data[0]["embedding"]) if embeddings_data else None
                }
            }, f)
        
        print("Export completed successfully!")

if __name__ == "__main__":
    export_embeddings_to_json() 