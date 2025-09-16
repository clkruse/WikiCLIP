import os
import json
from typing import Any, Dict, List, Optional

import boto3


class S3VectorsClient:
    """Lightweight wrapper around the Amazon S3 Vectors preview client.

    Environment variables:
    - S3_VECTORS_BUCKET: Vector bucket name
    - S3_VECTORS_INDEX: Vector index name within the bucket
    - S3_VECTORS_REGION: AWS region (e.g., us-east-1)
    - S3_VECTORS_DISTANCE: Distance metric (Cosine or Euclidean). Defaults to Cosine
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        index_name: Optional[str] = None,
        region_name: Optional[str] = None,
    ) -> None:
        self.bucket_name = bucket_name or os.environ.get("S3_VECTORS_BUCKET")
        self.index_name = index_name or os.environ.get("S3_VECTORS_INDEX")
        self.region_name = region_name or os.environ.get("S3_VECTORS_REGION", "us-east-1")

        if not self.bucket_name or not self.index_name:
            raise ValueError("S3_VECTORS_BUCKET and S3_VECTORS_INDEX must be set in env or passed to constructor")

        # Boto3 adds the preview client in recent versions per AWS announcement.
        # If your environment lacks the s3vectors client, upgrade boto3 or use the console for setup.
        self.client = boto3.client("s3vectors", region_name=self.region_name)

    def put_vectors(self, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Insert vectors into the configured S3 Vectors index.

        Each vector dict should have the shape:
        {
            "key": "unique-id",
            "vector": List[float],
            "metadata": { ... }  # optional
        }
        """
        payload: List[Dict[str, Any]] = []
        for item in vectors:
            payload.append(
                {
                    "key": str(item["key"]),
                    "data": {"float32": item["vector"]},
                    "metadata": item.get("metadata", {}),
                }
            )

        response = self.client.put_vectors(
            vectorBucketName=self.bucket_name,
            indexName=self.index_name,
            vectors=payload,
        )
        return response

    def query(
        self,
        query_vector: List[float],
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        return_distance: bool = True,
        return_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """Query the configured index using a vector.

        Returns a list of results with keys: key, distance (if requested), metadata (if requested)
        """
        request: Dict[str, Any] = {
            "vectorBucketName": self.bucket_name,
            "indexName": self.index_name,
            "queryVector": {"float32": query_vector},
            "topK": top_k,
            "returnDistance": return_distance,
            "returnMetadata": return_metadata,
        }
        if metadata_filter:
            request["filter"] = metadata_filter

        result = self.client.query_vectors(**request)
        return result.get("vectors", [])

    # Optional helpers for setup; API names may change during preview. These will no-op on errors.
    def ensure_bucket_and_index(self, vector_dimension: int, distance_metric: str = "Cosine") -> None:
        try:
            # Attempt to create bucket; ignore if already exists
            self.client.create_vector_bucket(vectorBucketName=self.bucket_name)
        except Exception:
            pass
        try:
            self.client.create_vector_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                vectorDimension=vector_dimension,
                distanceMetric=distance_metric,
            )
        except Exception:
            pass


def format_similarity(distance: Optional[float], distance_metric: str = "Cosine") -> Optional[float]:
    """Convert a distance into a similarity for display.

    For cosine distance d in [0, 2], we approximate similarity as 1 - d.
    """
    if distance is None:
        return None
    if distance_metric.lower() == "cosine":
        try:
            return 1.0 - float(distance)
        except Exception:
            return None
    # For Euclidean, users typically treat lower as better; keep as negative distance similarity
    try:
        return 1.0 / (1.0 + float(distance))
    except Exception:
        return None


