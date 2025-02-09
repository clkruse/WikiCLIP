import os
from dotenv import load_dotenv
from image_matcher import ImageMatcher, get_wiki_article_info
import psycopg2
from pprint import pprint
import numpy as np
import torch
import re

def parse_embedding_string(embedding_str: str) -> np.ndarray:
    """Parse the embedding string from the database into a numpy array."""
    # Remove brackets and split by commas
    values = re.findall(r'-?\d*\.?\d+(?:e[-+]?\d+)?', embedding_str)
    return np.array([float(x) for x in values])

def check_embeddings_normalized():
    """Check if embeddings in the database are normalized"""
    print("\n=== Checking Embedding Normalization ===")
    load_dotenv()
    db_url = os.getenv('DATABASE_URL')
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        article_id, 
                        embedding,
                        sqrt(embedding <-> embedding) as l2_norm,
                        sqrt(1 - (embedding <=> embedding)) as cosine_dist,
                        (embedding <=> embedding) as cosine_sim
                    FROM embeddings
                    LIMIT 5
                """)
                results = cursor.fetchall()
                print("\nChecking norms and similarities of first 5 embeddings:")
                for article_id, embedding_str, l2_norm, cosine_dist, cosine_sim in results:
                    print(f"\nArticle {article_id}:")
                    print(f"  L2 norm: {l2_norm:.4f}")
                    print(f"  Cosine distance: {cosine_dist:.4f}")
                    print(f"  Cosine similarity: {cosine_sim:.4f}")
                    
                    # Parse and check the actual embedding values
                    embedding = parse_embedding_string(embedding_str)
                    numpy_norm = np.linalg.norm(embedding)
                    print(f"  NumPy computed norm: {numpy_norm:.4f}")
                    print(f"  First 5 values: {embedding[:5]}")
                return True
    except Exception as e:
        print(f"✗ Normalization check failed: {str(e)}")
        return False

def test_db_connection():
    """Test 1: Basic database connectivity"""
    print("\n=== Test 1: Testing Database Connection ===")
    load_dotenv()
    db_url = os.getenv('DATABASE_URL')
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM embeddings")
                count = cursor.fetchone()[0]
                print(f"✓ Successfully connected to database")
                print(f"✓ Found {count} articles in the database")
                
                # Get a sample article_id and embedding
                cursor.execute("""
                    SELECT article_id, embedding 
                    FROM embeddings 
                    LIMIT 1
                """)
                sample = cursor.fetchone()
                if sample:
                    embedding_array = parse_embedding_string(sample[1])
                    print(f"✓ Sample article_id: {sample[0]}")
                    print(f"✓ Sample embedding length: {len(embedding_array)}")
                    print(f"✓ Sample embedding norm: {np.linalg.norm(embedding_array):.4f}")
                    print(f"✓ Sample embedding first few values: {embedding_array[:5]}")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {str(e)}")
        return False


def test_wiki_api():
    """Test 2: Wikipedia API functionality"""
    print("\n=== Test 2: Testing Wikipedia API ===")
    try:
        # Get a sample article_id from the database
        matcher = ImageMatcher()
        with psycopg2.connect(matcher.db_url) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT article_id FROM embeddings LIMIT 1")
                article_id = cursor.fetchone()[0]
        
        # Test article info retrieval
        article_info = get_wiki_article_info(article_id)
        print("✓ Successfully retrieved article info from Wikipedia:")
        print(f"  Article ID: {article_id}")
        print(f"  Title: {article_info['title']}")
        print(f"  URL: {article_info['url']}")
        return True
    except Exception as e:
        print(f"✗ Wikipedia API test failed: {str(e)}")
        return False

def debug_database_embeddings():
    """Debug database embeddings storage and format"""
    print("\n=== Debugging Database Embeddings ===")
    load_dotenv()
    db_url = os.getenv('DATABASE_URL')
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                WITH sample AS (
                    SELECT 
                        article_id,
                        embedding::text as raw_embedding,
                        1 - (embedding <-> embedding)::float as self_similarity,
                        (embedding <=> embedding)::float as cosine_sim
                    FROM embeddings
                    LIMIT 5
                )
                SELECT * FROM sample;
            """)
            results = cursor.fetchall()
            for r in results:
                # Parse the embedding string into a numpy array
                embedding_str = r[1].strip('[]').split(',')
                embedding = np.array([float(x) for x in embedding_str])
                
                print(f"Article {r[0]}:")
                print(f"  Embedding length: {len(embedding)}")
                print(f"  First 5 values: {embedding[:5]}")
                print(f"  NumPy norm: {np.linalg.norm(embedding):.4f}")
                print(f"  DB self similarity: {r[2]:.4f}")
                print(f"  DB cosine similarity: {r[3]:.4f}\n")

def verify_similarity_computation(embedding1, embedding2):
    """Verify similarity computations match between numpy and database"""
    print("\n=== Verifying Similarity Computation ===")
    
    # Convert embeddings to numpy if they're torch tensors
    if torch.is_tensor(embedding1):
        embedding1 = embedding1.cpu().numpy()
    if torch.is_tensor(embedding2):
        embedding2 = embedding2.cpu().numpy()
    
    # Ensure embeddings are 1D
    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()
    
    # Compute numpy similarity
    numpy_sim = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    # Format for database
    emb1_str = f'[{",".join(map(str, embedding1))}]'
    emb2_str = f'[{",".join(map(str, embedding2))}]'
    
    load_dotenv()
    db_url = os.getenv('DATABASE_URL')
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    -(%s::vector(512) <#> %s::vector(512))::float as inner_product_sim
            """, (emb1_str, emb2_str))
            db_sim = cursor.fetchone()[0]
            
    print(f"NumPy cosine similarity: {numpy_sim:.4f}")
    print(f"DB inner product similarity: {db_sim:.4f}")
    print(f"Difference: {abs(numpy_sim - db_sim):.4f}")
    
    # Alert if the difference is significant
    if abs(numpy_sim - db_sim) > 0.01:
        print("⚠️  Warning: Large difference between NumPy and database similarity")

def test_image_matching(image_url: str):
    """Test 3: Image matching functionality"""
    print("\n=== Test 3: Testing Image Matching ===")
    matcher = ImageMatcher()
    try:
        print(f"Testing with image: {image_url}")
        
        # Debug database embeddings first
        debug_database_embeddings()
        
        # Get the embedding for the test image
        embedding = matcher.get_embedding(image_url)
        print(f"✓ Generated embedding for test image")
        print(f"  Shape: {embedding.shape}")
        print(f"  Norm: {torch.norm(embedding).item():.4f}")
        print(f"  First few values: {embedding[0, :5].tolist()}")
        
        # Get a sample embedding from the database for similarity verification
        with psycopg2.connect(matcher.db_url) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT embedding::text FROM embeddings LIMIT 1")
                embedding_str = cursor.fetchone()[0]
                # Parse the embedding string into a numpy array
                db_embedding = np.array([float(x) for x in embedding_str.strip('[]').split(',')])
        
        # Verify similarity computation
        verify_similarity_computation(
            embedding[0].cpu().numpy(),
            db_embedding
        )
        
        # Test with an image URL
        results = matcher.get_similar_articles(
            image_url,
            limit=5,
            threshold=0.2
        )
        
        if not results:
            print("\n⚠️  No matches found. Debugging information:")
            with psycopg2.connect(matcher.db_url) as conn:
                with conn.cursor() as cursor:
                    embedding_list = embedding.cpu().numpy()[0].tolist()
                    embedding_str = f'[{",".join(map(str, embedding_list))}]'
                    cursor.execute("""
                        SELECT 
                            article_id,
                            (embedding <=> %s::vector(512)) as cosine_sim,
                            sqrt(embedding <-> embedding) as db_norm,
                            sqrt(%s::vector(512) <-> %s::vector(512)) as query_norm
                        FROM embeddings
                        ORDER BY cosine_sim DESC
                        LIMIT 5
                    """, (embedding_str, embedding_str, embedding_str))
                    debug_results = cursor.fetchall()
                    print("\nRaw similarity scores:")
                    for article_id, sim, db_norm, query_norm in debug_results:
                        print(f"  Article {article_id}:")
                        print(f"    Cosine similarity: {sim:.4f}")
                        print(f"    DB vector norm: {db_norm:.4f}")
                        print(f"    Query vector norm: {query_norm:.4f}")
                        
                        article_info = get_wiki_article_info(article_id)
                        print(f"    Title: {article_info['title']}")
        else:
            print(f"\n✓ Found {len(results)} similar articles:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']}")
                print(f"   Article ID: {result['article_id']}")
                print(f"   Similarity: {result['similarity']:.4f}")
                print(f"   URL: {result['url']}")
        return True
    except Exception as e:
        print(f"✗ Image matching failed: {str(e)}")
        return False

def check_raw_embedding_format():
    """Check the raw format of embeddings in the database"""
    print("\n=== Checking Raw Embedding Format ===")
    load_dotenv()
    db_url = os.getenv('DATABASE_URL')
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        article_id,
                        embedding::text
                    FROM embeddings
                    LIMIT 3
                """)
                results = cursor.fetchall()
                print("\nRaw embedding format for first 3 records:")
                for article_id, embedding_str in results:
                    print(f"\nArticle {article_id}:")
                    print(f"Embedding format: {embedding_str[:200]}...")  # Show first 200 chars
                return True
    except Exception as e:
        print(f"✗ Format check failed: {str(e)}")
        return False

def main():
    # Test image URL - replace with a real image URL for testing
    test_image_url = "https://clkruse.github.io/photos/img/photo-2025-01-02.jpg"
    
    # Run tests sequentially
    check_raw_embedding_format()  # Add the new check
    if test_db_connection():
        check_embeddings_normalized()
        if test_wiki_api():
            test_image_matching(test_image_url)

if __name__ == "__main__":
    main() 
