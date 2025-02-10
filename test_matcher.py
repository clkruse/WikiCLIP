import os
from dotenv import load_dotenv
from image_matcher import ImageMatcher, get_wiki_article_info
import psycopg2
from pprint import pprint
import numpy as np
import torch
import re
import time
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import base64
from io import BytesIO
from typing import List, Dict

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

def verify_match_sensibility(matcher, image_path: str, results: List[Dict]):
    """Verify that the matches make semantic sense."""
    print("\n=== Verifying Match Sensibility ===")
    
    # Get the embedding for the test image
    embedding = matcher.get_embedding(image_path)
    embedding_list = embedding.cpu().numpy()[0].tolist()
    embedding_str = f'[{",".join(map(str, embedding_list))}]'
    
    # Get raw similarity scores for all articles
    with psycopg2.connect(matcher.db_url) as conn:
        with conn.cursor() as cursor:
            # Get top 100 matches to verify our results are truly the best
            cursor.execute("""
                SELECT 
                    article_id::text,
                    1 - (embedding <=> %s::vector(512))::float as similarity
                FROM embeddings
                ORDER BY embedding <=> %s::vector(512)
                LIMIT 100
            """, (embedding_str, embedding_str))
            all_matches = cursor.fetchall()
    
    # Verify our results are among the top matches
    result_ids = {r['article_id'] for r in results}
    top_ids = {row[0] for row in all_matches[:len(results)]}
    
    print("\nVerification Results:")
    print(f"Number of results: {len(results)}")
    print(f"Results in top matches: {len(result_ids.intersection(top_ids))}/{len(results)}")
    
    # Check similarity score distribution
    similarities = [row[1] for row in all_matches]
    avg_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    print(f"\nSimilarity Statistics:")
    print(f"Average similarity: {avg_sim:.4f}")
    print(f"Standard deviation: {std_sim:.4f}")
    print(f"Max similarity: {max(similarities):.4f}")
    print(f"Min similarity: {min(similarities):.4f}")
    
    # Print detailed results
    print("\nTop 10 matches from verification:")
    for i, (article_id, sim) in enumerate(all_matches[:10], 1):
        article_info = get_wiki_article_info(article_id)
        in_results = "✓" if article_id in result_ids else " "
        print(f"{in_results} {i}. {article_info['title']} (ID: {article_id}, Similarity: {sim:.4f})")
    
    return True

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
            
            # Verify the sensibility of matches
            verify_match_sensibility(matcher, image_url, results)
            
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

def test_db_query_performance(embedding_str: str, threshold: float = 0.5, limit: int = 15):
    """Test database query performance with different approaches."""
    load_dotenv()
    db_url = os.getenv('DATABASE_URL')
    
    print("\n=== Testing Database Query Performance ===")
    
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cursor:
                # Test 1: Original function approach
                print("\nTest 1: Using match_embeddings function")
                start_time = time.time()
                cursor.execute("""
                    SELECT * FROM match_embeddings(
                        %s::vector(512),
                        match_threshold := %s,
                        match_count := %s
                    )
                """, (embedding_str, threshold, limit))
                rows = cursor.fetchall()
                print(f"Query time: {time.time() - start_time:.3f}s")
                print(f"Results found: {len(rows)}")
                
                # Test 2: Direct query with CTE
                print("\nTest 2: Using CTE approach")
                start_time = time.time()
                cursor.execute("""
                    WITH similarity_scores AS (
                        SELECT 
                            article_id::text,
                            1 - (embedding <=> %s::vector(512))::float as similarity
                        FROM embeddings
                        WHERE embedding <=> %s::vector(512) < %s
                        ORDER BY embedding <=> %s::vector(512)
                        LIMIT %s
                    )
                    SELECT * FROM similarity_scores
                    WHERE similarity > %s
                    ORDER BY similarity DESC
                """, (embedding_str, embedding_str, 1 - threshold, embedding_str, limit * 2, threshold))
                rows = cursor.fetchall()
                print(f"Query time: {time.time() - start_time:.3f}s")
                print(f"Results found: {len(rows)}")
                
                # Test 3: Simple direct query
                print("\nTest 3: Simple direct query")
                start_time = time.time()
                cursor.execute("""
                    SELECT 
                        article_id::text,
                        1 - (embedding <=> %s::vector(512))::float as similarity
                    FROM embeddings
                    WHERE embedding <=> %s::vector(512) < %s
                    ORDER BY embedding <=> %s::vector(512)
                    LIMIT %s
                """, (embedding_str, embedding_str, 1 - threshold, embedding_str, limit))
                rows = cursor.fetchall()
                print(f"Query time: {time.time() - start_time:.3f}s")
                print(f"Results found: {len(rows)}")
                
                # Get query plan for the fastest approach
                print("\nAnalyzing query plan:")
                cursor.execute("""
                    EXPLAIN ANALYZE
                    SELECT 
                        article_id::text,
                        1 - (embedding <=> %s::vector(512))::float as similarity
                    FROM embeddings
                    WHERE embedding <=> %s::vector(512) < %s
                    ORDER BY embedding <=> %s::vector(512)
                    LIMIT %s
                """, (embedding_str, embedding_str, 1 - threshold, embedding_str, limit))
                plan = cursor.fetchall()
                print("\n".join([row[0] for row in plan]))
                
        return True
    except Exception as e:
        print(f"✗ Performance test failed: {str(e)}")
        return False

def generate_test_embedding():
    """Generate a test embedding using CLIP model."""
    print("\nGenerating test embedding...")
    try:
        # Initialize CLIP model
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Create a simple test image (black square)
        img = Image.new('RGB', (224, 224), color='black')
        
        # Process image
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt")
            image_features = model.get_image_features(**inputs)
            embedding = image_features / image_features.norm(dim=-1, keepdim=True)
            
        # Convert to string format
        embedding_list = embedding.cpu().numpy()[0].tolist()
        embedding_str = f'[{",".join(map(str, embedding_list))}]'
        
        return embedding_str
    except Exception as e:
        print(f"Error generating test embedding: {str(e)}")
        return None

def test_similarity_from_article(article_id: str = None):
    """Test similarity search using an existing article's embedding.
    
    Args:
        article_id (str, optional): Specific article ID to test with. If None, picks a random article.
    """
    print("\n=== Testing Similarity Using Existing Article ===")
    load_dotenv()
    db_url = os.getenv('DATABASE_URL')
    
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cursor:
                # Get source article
                if article_id is None:
                    # Get a random article
                    cursor.execute("""
                        SELECT article_id, embedding
                        FROM embeddings
                        OFFSET floor(random() * (SELECT COUNT(*) FROM embeddings))
                        LIMIT 1
                    """)
                else:
                    cursor.execute("""
                        SELECT article_id, embedding
                        FROM embeddings
                        WHERE article_id = %s
                    """, (article_id,))
                
                source = cursor.fetchone()
                if not source:
                    print("❌ No source article found")
                    return False
                
                source_id, source_embedding = source
                source_info = get_wiki_article_info(source_id)
                print(f"\nSource Article:")
                print(f"  ID: {source_id}")
                print(f"  Title: {source_info['title']}")
                print(f"  URL: {source_info['url']}")
                
                # Find similar articles
                print("\nFinding similar articles...")
                cursor.execute("""
                    SELECT 
                        article_id::text,
                        1 - (embedding <=> %s::vector(512))::float as similarity
                    FROM embeddings
                    WHERE article_id != %s  -- Exclude the source article
                    ORDER BY embedding <=> %s::vector(512)
                    LIMIT 10
                """, (source_embedding, source_id, source_embedding))
                
                similar = cursor.fetchall()
                
                print("\nMost Similar Articles:")
                for i, (similar_id, similarity) in enumerate(similar, 1):
                    article_info = get_wiki_article_info(similar_id)
                    print(f"\n{i}. {article_info['title']}")
                    print(f"   Article ID: {similar_id}")
                    print(f"   Similarity: {similarity:.4f}")
                    print(f"   URL: {article_info['url']}")
                
                # Calculate similarity statistics
                similarities = [sim for _, sim in similar]
                print(f"\nSimilarity Statistics:")
                print(f"  Average: {np.mean(similarities):.4f}")
                print(f"  Std Dev: {np.std(similarities):.4f}")
                print(f"  Max: {max(similarities):.4f}")
                print(f"  Min: {min(similarities):.4f}")
                
                return True
                
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    # Add test_similarity_from_article to the test suite
    print("\nTesting similarity with existing article...")
    test_similarity_from_article()
    
    # Original tests
    test_images = [
        "https://calebkruse.com/photos/img/photo-2025-02-01.jpg",
        "https://calebkruse.com/photos/img/photo-2025-01-03.jpg",  # Cactus garden
        "https://calebkruse.com/photos/img/photo-2025-01-02.jpg"  # Cat
    ]
    
    for image_url in test_images:
        print(f"\nTesting with image: {image_url}")
        test_image_matching(image_url)

if __name__ == "__main__":
    main() 
