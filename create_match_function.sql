-- Drop the old function
DROP FUNCTION IF EXISTS match_embeddings(vector,double precision,integer);

-- Function to find similar articles given an embedding vector
CREATE OR REPLACE FUNCTION match_embeddings(
    query_embedding vector(512),
    match_threshold float DEFAULT 0.2,
    match_count int DEFAULT 15,
    ef_search int DEFAULT 40
)
RETURNS TABLE (
    article_id text,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    -- Set search quality parameter
    SET LOCAL hnsw.ef_search = ef_search;
    
    -- Use the HNSW index for efficient similarity search
    RETURN QUERY
    SELECT
        e.article_id::text,
        1 - (embedding <=> query_embedding)::float as similarity
    FROM embeddings e
    WHERE embedding <=> query_embedding < (1 - match_threshold)  -- Inverted threshold since <=> returns distance
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
END;
$$;