import os
import numpy as np
import psycopg2
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Generate a random normalized vector
vector = np.random.randn(512)
vector = vector / np.linalg.norm(vector)
vector_str = f'[{",".join(map(str, vector))}]'

# Connect to database
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
with conn.cursor() as cursor:
    # First check if the HNSW index exists
    cursor.execute("""
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = 'embeddings';
    """)
    indexes = cursor.fetchall()
    print("\nExisting Indexes:")
    print("-" * 50)
    for idx_name, idx_def in indexes:
        print(f"Index: {idx_name}")
        print(f"Definition: {idx_def}\n")

    # Try different thresholds
    thresholds = [0.05]
    
    for threshold in thresholds:
        print(f"\nTesting with threshold: {threshold}")
        print("-" * 50)
        
        # Analyze query performance
        explain_query = f"""
        EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
        SELECT 
            article_id::text,
            1 - (embedding <=> '{vector_str}'::vector(512))::float as similarity
        FROM embeddings
        WHERE 1 - (embedding <=> '{vector_str}'::vector(512)) > {threshold}
        ORDER BY embedding <=> '{vector_str}'::vector(512)
        LIMIT 15;
        """
        
        cursor.execute(explain_query)
        results = cursor.fetchall()
        
        # Print the full execution plan
        plan = results[0][0]
        
        # Print formatted plan for debugging
        print("\nFull Execution Plan:")
        print(json.dumps(plan, indent=2))
        
        # Extract and print relevant metrics
        print("\nQuery Metrics:")
        node = plan[0]['Plan']
        print(f"Operation: {node['Node Type']}")
        print(f"Total Time: {node['Actual Total Time']:.2f} ms")
        print(f"Rows Retrieved: {node['Actual Rows']}")
        
        if 'Index Name' in node:
            print(f"Index Used: {node['Index Name']}")
        
        print(f"Planning Time: {plan[0]['Planning Time']:.2f} ms")
        print(f"Execution Time: {plan[0]['Execution Time']:.2f} ms")
        
        # Run actual query
        actual_query = f"""
        SELECT 
            article_id::text,
            1 - (embedding <=> '{vector_str}'::vector(512))::float as similarity
        FROM embeddings
        WHERE 1 - (embedding <=> '{vector_str}'::vector(512)) > {threshold}
        ORDER BY embedding <=> '{vector_str}'::vector(512)
        LIMIT 15;
        """
        
        cursor.execute(actual_query)
        results = cursor.fetchall()
        
        print("\nResults:")
        print(f"Number of results: {len(results)}")
        if results:
            print("\nTop 3 matches:")
            for article_id, similarity in results[:3]:
                print(f"Article ID: {article_id}, Similarity: {similarity:.4f}")
        print("\n" + "="*60 + "\n") 