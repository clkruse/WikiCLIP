from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime

# Get database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL or "sqlite:///wiki_embeddings.db")
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