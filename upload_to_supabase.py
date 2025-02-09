import json
import os
from supabase import create_client
from tqdm import tqdm
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

BATCH_SIZE = 500  # Number of records to insert at once
RATE_LIMIT_DELAY = 0.1  # Delay between batches in seconds

def format_vector(embedding_list):
    """Convert embedding list to PostgreSQL vector format."""
    return f"[{','.join(str(x) for x in embedding_list)}]"

def upload_embeddings_to_supabase(json_path: str = "embeddings.json"):
    """
    Upload embeddings from JSON file to Supabase.
    
    Args:
        json_path (str): Path to the JSON file containing embeddings
    """
    print(f"Reading embeddings from {json_path}...")
    
    # Initialize Supabase client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    embeddings = data['embeddings']
    total_embeddings = len(embeddings)
    print(f"Found {total_embeddings} embeddings to upload")
    
    # Process in batches
    for i in tqdm(range(0, total_embeddings, BATCH_SIZE)):
        batch = embeddings[i:i + BATCH_SIZE]
        
        # Prepare batch data with proper vector casting
        batch_data = [
            {
                "article_id": item["article_id"],
                "hash": item["hash"],
                "embedding": format_vector(item["embedding"])
            }
            for item in batch
        ]
        
        try:
            # Upload batch to Supabase
            result = supabase.table('embeddings').insert(batch_data).execute()
            
            # Add delay to avoid rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            print(f"Error uploading batch starting at index {i}: {str(e)}")
            continue
    
    print("Upload completed successfully!")

if __name__ == "__main__":
    upload_embeddings_to_supabase() 