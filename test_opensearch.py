import os
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
from dotenv import load_dotenv
import json

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

def test_migration():
    """Test the OpenSearch setup and migration."""
    # First check the index mapping
    print("\nChecking index mapping:")
    mapping = client.indices.get_mapping(index='wiki-embeddings')
    print(json.dumps(mapping, indent=2))
    
    # Check document count
    count = client.count(index='wiki-embeddings')
    print(f"\nTotal documents in OpenSearch: {count['count']}")
    
    # Check a sample document's embedding
    print("\nChecking sample document embedding:")
    sample_query = {
        "size": 1,
        "query": {
            "match_all": {}
        },
        "_source": ["article_id", "title", "url", "embedding"]
    }
    sample_response = client.search(
        body=sample_query,
        index="wiki-embeddings"
    )

    if len(sample_response["hits"]["hits"]) > 0:
        doc = sample_response["hits"]["hits"][0]
        print(f"Article ID: {doc['_source']['article_id']}")
        print(f"Title: {doc['_source']['title']}")
        
        if "embedding" in doc["_source"]:
            embedding = doc["_source"]["embedding"]
            print("Embedding present: True")
            print(f"Embedding type: {type(embedding)}")
            print(f"Embedding length: {len(embedding)}")
            print(f"First few values: {embedding[:5]}")
            
            # Calculate the norm of the embedding
            norm = sum(x*x for x in embedding) ** 0.5
            print(f"Embedding norm: {norm}")
            
            # Testing vector search with real embedding:
            vector_query = {
                "size": 5,
                "_source": ["article_id", "title", "url"],
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": embedding,
                            "k": 5
                        }
                    }
                }
            }

            print("\nVector query:", json.dumps(vector_query, indent=2))
            vector_response = client.search(
                body=vector_query,
                index="wiki-embeddings"
            )
            print("\nVector search response:", json.dumps(vector_response, indent=2))

            # Also try a regular term query for the same document
            print("\nTrying term query for the same document:")
            term_query = {
                "query": {
                    "term": {
                        "article_id": doc["_source"]["article_id"]
                    }
                }
            }
            term_response = client.search(
                body=term_query,
                index="wiki-embeddings"
            )
            print("\nTerm search response:", json.dumps(term_response, indent=2))
        else:
            print("No embedding field found in document")
    else:
        print("No documents found")
    
    # Get some sample documents
    print("\nSample documents:")
    sample_query = {
        "size": 5,
        "query": {
            "match_all": {}
        }
    }
    
    sample_response = client.search(
        body=sample_query,
        index='wiki-embeddings'
    )
    
    for hit in sample_response['hits']['hits']:
        print(f"\nArticle ID: {hit['_source']['article_id']}")
        print(f"Title: {hit['_source']['title']}")
        print(f"URL: {hit['_source']['url']}")

if __name__ == "__main__":
    test_migration() 