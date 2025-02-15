-- Set a longer statement timeout (15 minutes)
SET statement_timeout = '1800000';

-- Show current PostgreSQL settings
SHOW work_mem;
SHOW maintenance_work_mem;


-- First terminate any running queries that might block the index drop
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE pid <> pg_backend_pid()
AND query LIKE '%embeddings_embedding_idx%';

-- First drop the existing index if it exists
DROP INDEX IF EXISTS embeddings_embedding_idx;

-- Then create the new index with cosine similarity
CREATE INDEX embeddings_embedding_idx 
ON embeddings 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 32, ef_construction = 128);

-- Verify index was created
SELECT schemaname, tablename, indexname, indexdef
FROM pg_indexes
WHERE tablename = 'embeddings';

-- Reset maintenance work memory
RESET maintenance_work_mem;