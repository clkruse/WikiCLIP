-- Drop the old function
DROP FUNCTION match_embeddings(vector,double precision,integer);

-- Function to find similar articles given an embedding vector
CREATE OR REPLACE FUNCTION match_embeddings(
    query_embedding vector(512),
    match_threshold float DEFAULT 0.2,
    match_count int DEFAULT 15
)
RETURNS TABLE (
    article_id text,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    -- Negate the <#> operator to get positive similarity
    RETURN QUERY
    SELECT
        e.article_id::text,
        -(embedding <#> query_embedding)::float as similarity
    FROM embeddings e
    WHERE -(embedding <#> query_embedding)::float > match_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;