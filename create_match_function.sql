-- Drop the old function
DROP FUNCTION IF EXISTS match_embeddings(vector,double precision,integer);

-- Function to find similar articles given an embedding vector
CREATE OR REPLACE FUNCTION match_embeddings(
    query_embedding vector(512),
    match_threshold float DEFAULT 0.2,
    match_count int DEFAULT 15,
    ef_search int DEFAULT 100
)
RETURNS TABLE (
    article_id text,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    -- Use the HNSW index directly for efficient similarity search
    RETURN QUERY
    SELECT
        e.article_id::text,
        1 - (embedding <=> query_embedding)::float as similarity
    FROM embeddings e
    WHERE 1 - (embedding <=> query_embedding) > match_threshold
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
END;
$$;