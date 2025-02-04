import sqlite3
import psycopg2
from psycopg2.extras import execute_batch
import os
from urllib.parse import urlparse
import numpy as np
from tqdm import tqdm

def migrate_database(sqlite_path: str, postgres_url: str):
    """
    Migrate data from SQLite to PostgreSQL database.
    
    Args:
        sqlite_path: Path to the SQLite database file
        postgres_url: PostgreSQL connection URL from Render
    """
    # Convert postgres:// to postgresql:// if necessary
    if postgres_url.startswith("postgres://"):
        postgres_url = postgres_url.replace("postgres://", "postgresql://", 1)
    
    # Connect to SQLite database
    print("Connecting to SQLite database...")
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_cur = sqlite_conn.cursor()
    
    # Connect directly to PostgreSQL using the full URL
    print("Connecting to PostgreSQL database...")
    postgres_conn = psycopg2.connect(postgres_url, sslmode='require')
    postgres_cur = postgres_conn.cursor()
    
    try:
        # Create tables in PostgreSQL
        print("Creating tables...")
        postgres_cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                article_id TEXT PRIMARY KEY,
                title TEXT,
                url TEXT,
                embedding BYTEA,
                processed_date TEXT,
                hash TEXT
            )
        """)
        
        postgres_cur.execute("""
            CREATE TABLE IF NOT EXISTS failed_articles (
                article_id TEXT PRIMARY KEY,
                title TEXT,
                error_message TEXT,
                attempt_date TEXT
            )
        """)
        
        # Create index on title
        postgres_cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_title ON embeddings (title)
        """)
        
        # Migrate embeddings data
        print("Migrating embeddings table...")
        sqlite_cur.execute("SELECT COUNT(*) FROM embeddings")
        total_embeddings = sqlite_cur.fetchone()[0]
        
        sqlite_cur.execute("SELECT article_id, title, url, embedding, processed_date, hash FROM embeddings")
        batch_size = 100  # Reduced batch size for better stability
        batch = []
        
        with tqdm(total=total_embeddings) as pbar:
            for row in sqlite_cur:
                batch.append(row)
                if len(batch) >= batch_size:
                    execute_batch(postgres_cur, """
                        INSERT INTO embeddings (article_id, title, url, embedding, processed_date, hash)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (article_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        url = EXCLUDED.url,
                        embedding = EXCLUDED.embedding,
                        processed_date = EXCLUDED.processed_date,
                        hash = EXCLUDED.hash
                    """, batch)
                    postgres_conn.commit()
                    pbar.update(len(batch))
                    batch = []
            
            # Insert remaining records
            if batch:
                execute_batch(postgres_cur, """
                    INSERT INTO embeddings (article_id, title, url, embedding, processed_date, hash)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (article_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    url = EXCLUDED.url,
                    embedding = EXCLUDED.embedding,
                    processed_date = EXCLUDED.processed_date,
                    hash = EXCLUDED.hash
                """, batch)
                postgres_conn.commit()
                pbar.update(len(batch))
        
        # Migrate failed_articles data
        print("\nMigrating failed_articles table...")
        sqlite_cur.execute("SELECT COUNT(*) FROM failed_articles")
        total_failed = sqlite_cur.fetchone()[0]
        
        sqlite_cur.execute("SELECT article_id, title, error_message, attempt_date FROM failed_articles")
        batch = []
        
        with tqdm(total=total_failed) as pbar:
            for row in sqlite_cur:
                batch.append(row)
                if len(batch) >= batch_size:
                    execute_batch(postgres_cur, """
                        INSERT INTO failed_articles (article_id, title, error_message, attempt_date)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (article_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        error_message = EXCLUDED.error_message,
                        attempt_date = EXCLUDED.attempt_date
                    """, batch)
                    postgres_conn.commit()
                    pbar.update(len(batch))
                    batch = []
            
            # Insert remaining records
            if batch:
                execute_batch(postgres_cur, """
                    INSERT INTO failed_articles (article_id, title, error_message, attempt_date)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (article_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    error_message = EXCLUDED.error_message,
                    attempt_date = EXCLUDED.attempt_date
                """, batch)
                postgres_conn.commit()
                pbar.update(len(batch))
        
        print("\nMigration completed successfully!")
        
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        postgres_conn.rollback()
        raise
    
    finally:
        # Close connections
        sqlite_cur.close()
        sqlite_conn.close()
        postgres_cur.close()
        postgres_conn.close()

if __name__ == "__main__":
    # Get PostgreSQL URL from environment variable or use provided URL
    postgres_url = os.getenv("DATABASE_URL")
    if not postgres_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    sqlite_path = "wiki_embeddings.db"  # Path to your local SQLite database
    
    migrate_database(sqlite_path, postgres_url)