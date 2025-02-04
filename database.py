from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime

# Debug environment variables
print("All environment variables:", dict(os.environ))
print("Current working directory:", os.getcwd())

# Get database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")
print("Raw DATABASE_URL:", DATABASE_URL)

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set. Please ensure it is configured in Render.")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    print("Modified DATABASE_URL to use postgresql:// prefix")

# Create SQLAlchemy engine
print("Final database URL:", DATABASE_URL.split("@")[0] + "@" + "XXXXX")  # Hide credentials
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Embedding(Base):
    __tablename__ = "embeddings"

    article_id = Column(String, primary_key=True)
    title = Column(String, index=True)
    url = Column(String)
    embedding = Column(LargeBinary)
    processed_date = Column(String)
    hash = Column(String)

class FailedArticle(Base):
    __tablename__ = "failed_articles"

    article_id = Column(String, primary_key=True)
    title = Column(String)
    error_message = Column(String)
    attempt_date = Column(String)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create tables
def init_db():
    Base.metadata.create_all(bind=engine) 