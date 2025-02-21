import os
import psycopg2
import numpy as np
from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from requests_aws4auth import AWS4Auth
import boto3
from dotenv import load_dotenv
from tqdm import tqdm
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Initialize AWS credentials
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

# Initialize OpenSearch client with increased timeout
opensearch = OpenSearch(
    hosts=[{'host': os.environ['OPENSEARCH_ENDPOINT'], 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=30,  # Increased timeout to 30 seconds
    retry_on_timeout=True,
    max_retries=3
)

def parse_embedding(embedding_str):
    """Parse PostgreSQL vector string to list."""
    return [float(x) for x in embedding_str.strip('[]').split(',')]

def get_wiki_info(article_id):
    """Get Wikipedia article info."""
    return {
        'title': f'Article {article_id}',
        'url': f'https://en.wikipedia.org/?curid={article_id}'
    }

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def bulk_index_with_retry(client, batch):
    """Execute bulk indexing with retry logic."""
    success, failed = helpers.bulk(client, batch, stats_only=True, request_timeout=30)
    return success, failed

def migrate_data():
    """Migrate data from PostgreSQL to OpenSearch."""
    batch_size = 50  # Reduced batch size
    total_migrated = 0
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    
    try:
        # First get total count with a regular cursor
        with conn.cursor() as count_cursor:
            count_cursor.execute("SELECT COUNT(*) FROM embeddings")
            total_records = count_cursor.fetchone()[0]
            print(f"Total records to migrate: {total_records}")
        
        # Then use a named cursor for batch fetching
        with conn.cursor('fetch_cursor') as fetch_cursor:
            fetch_cursor.execute("""
                SELECT article_id, embedding::text
                FROM embeddings
                ORDER BY article_id
            """)
            
            batch = []
            with tqdm(total=total_records, desc="Migrating") as pbar:
                while True:
                    try:
                        records = fetch_cursor.fetchmany(batch_size)
                        if not records:
                            break
                        
                        for record in records:
                            article_id, embedding_str = record
                            wiki_info = get_wiki_info(article_id)
                            
                            # Prepare document
                            doc = {
                                '_index': 'wiki-embeddings',
                                '_id': article_id,
                                '_source': {
                                    'article_id': article_id,
                                    'embedding': parse_embedding(embedding_str),
                                    'title': wiki_info['title'],
                                    'url': wiki_info['url']
                                }
                            }
                            batch.append(doc)
                        
                        # Bulk index the batch
                        if batch:
                            try:
                                success, failed = bulk_index_with_retry(opensearch, batch)
                                total_migrated += success
                                pbar.update(len(batch))
                            except Exception as e:
                                print(f"\nError during bulk indexing: {str(e)}")
                                print("Retrying after a longer delay...")
                                time.sleep(5)  # Longer delay on error
                                continue
                            finally:
                                batch = []
                            
                            # Small delay between batches
                            time.sleep(0.1)  # Increased delay between batches
                            
                    except Exception as e:
                        print(f"\nUnexpected error: {str(e)}")
                        print("Continuing with next batch...")
                        batch = []
                        continue
            
            print(f"\nMigration complete! Migrated {total_migrated} records")
            
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_data() 