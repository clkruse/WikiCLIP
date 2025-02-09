-- First drop the existing index
DROP INDEX IF EXISTS embeddings_embedding_idx;

-- Then create the new index with cosine similarity
CREATE INDEX embeddings_embedding_idx 
ON embeddings 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 8, ef_construction = 32);