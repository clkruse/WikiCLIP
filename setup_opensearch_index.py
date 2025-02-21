import os
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
from dotenv import load_dotenv

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

# Initialize OpenSearch client
client = OpenSearch(
    hosts=[{'host': os.environ['OPENSEARCH_ENDPOINT'], 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

# Define the index mapping
index_name = "wiki-embeddings"
index_mapping = {
    "mappings": {
        "properties": {
            "embedding": {
                "type": "knn_vector",
                "dimension": 512,  # CLIP embedding dimension
                "method": {
                    "name": "hnsw",  # Hierarchical Navigable Small World graph
                    "space_type": "cosinesimil",  # Cosine similarity metric
                    "engine": "nmslib",  # Library for approximate nearest neighbor search
                    "parameters": {
                        "ef_construction": 128,  # Higher = more accurate but slower indexing
                        "m": 16  # Number of bi-directional links created for every new element
                    }
                }
            },
            "article_id": {
                "type": "keyword"  # Exact match field
            },
            "title": {
                "type": "text",  # Full-text searchable
                "analyzer": "standard"
            },
            "url": {
                "type": "keyword"  # Exact match field
            }
        }
    },
    "settings": {
        "index": {
            "number_of_shards": 1,  # Single shard for free tier
            "number_of_replicas": 0,  # No replicas for free tier
            "knn": True,  # Enable kNN search
            "knn.algo_param.ef_search": 50  # Controls accuracy vs. speed for search
        }
    }
}

# Delete existing index if it exists
if client.indices.exists(index_name):
    print(f"Deleting existing index: {index_name}")
    client.indices.delete(index_name)

# Create the index with the mapping
print(f"Creating index: {index_name}")
client.indices.create(index_name, body=index_mapping)

# Verify the mapping
print("\nVerifying index mapping:")
mapping = client.indices.get_mapping(index_name)
print(mapping)

print("\nIndex setup complete!") 