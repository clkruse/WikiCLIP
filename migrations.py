import sqlite3
import os
from database import engine, SessionLocal, Base, Embedding, FailedArticle
from sqlalchemy.orm import Session

def get_table_columns(cursor, table_name):
    """Get column names for a given table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]

def migrate_data():
    """Migrate data from SQLite to PostgreSQL."""
    # Drop and recreate all tables in PostgreSQL
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    # Only proceed with migration if SQLite database exists
    if os.path.exists('wiki_embeddings.db'):
        # Connect to SQLite database
        sqlite_conn = sqlite3.connect('wiki_embeddings.db')
        sqlite_cursor = sqlite_conn.cursor()
        
        # Create PostgreSQL session
        db = SessionLocal()
        
        try:
            # Get column names for embeddings table
            embedding_columns = get_table_columns(sqlite_cursor, "embeddings")
            if embedding_columns:
                # Construct dynamic query based on available columns
                columns_str = ", ".join(embedding_columns)
                sqlite_cursor.execute(f"SELECT {columns_str} FROM embeddings")
                
                # Batch process to avoid memory issues
                batch_size = 1000
                count = 0
                batch = []
                
                for row in sqlite_cursor:
                    # Create dict of column names and values
                    row_dict = dict(zip(embedding_columns, row))
                    batch.append(Embedding(**row_dict))
                    count += 1
                    
                    if len(batch) >= batch_size:
                        db.bulk_save_objects(batch)
                        db.commit()
                        print(f"Migrated {count} embeddings")
                        batch = []
                
                # Save any remaining records
                if batch:
                    db.bulk_save_objects(batch)
                    db.commit()
                    print(f"Migrated {count} embeddings")
            
            # Get column names for failed_articles table
            failed_columns = get_table_columns(sqlite_cursor, "failed_articles")
            if failed_columns:
                # Construct dynamic query based on available columns
                columns_str = ", ".join(failed_columns)
                sqlite_cursor.execute(f"SELECT {columns_str} FROM failed_articles")
                
                count = 0
                batch = []
                
                for row in sqlite_cursor:
                    # Create dict of column names and values
                    row_dict = dict(zip(failed_columns, row))
                    batch.append(FailedArticle(**row_dict))
                    count += 1
                    
                    if len(batch) >= batch_size:
                        db.bulk_save_objects(batch)
                        db.commit()
                        print(f"Migrated {count} failed articles")
                        batch = []
                
                # Save any remaining records
                if batch:
                    db.bulk_save_objects(batch)
                    db.commit()
                    print(f"Migrated {count} failed articles")
            
            print("Migration completed successfully!")
            
        except Exception as e:
            print(f"Error during migration: {str(e)}")
            db.rollback()
        
        finally:
            db.close()
            sqlite_conn.close()
    else:
        print("No SQLite database found. Creating fresh PostgreSQL database.")
        Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    migrate_data() 