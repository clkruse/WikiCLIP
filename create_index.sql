-- Set a longer statement timeout (15 minutes)
SET statement_timeout = '900000';

-- First terminate any running queries that might block the index drop
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE pid <> pg_backend_pid()
AND query LIKE '%embeddings_embedding_idx%';

-- First drop the existing index
DROP INDEX IF EXISTS embeddings_embedding_idx;

-- Then create the new index with cosine similarity
CREATE INDEX embeddings_embedding_idx 
ON embeddings 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);