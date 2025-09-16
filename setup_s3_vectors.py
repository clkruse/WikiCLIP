import os
from dotenv import load_dotenv

from s3_vectors_client import S3VectorsClient


load_dotenv()


def main() -> None:
    bucket = os.environ.get("S3_VECTORS_BUCKET")
    index = os.environ.get("S3_VECTORS_INDEX")
    region = os.environ.get("S3_VECTORS_REGION", "us-east-1")
    dimension = int(os.environ.get("S3_VECTORS_DIMENSION", "512"))
    metric = os.environ.get("S3_VECTORS_DISTANCE", "Cosine")

    if not bucket or not index:
        raise SystemExit("S3_VECTORS_BUCKET and S3_VECTORS_INDEX must be set")

    client = S3VectorsClient(bucket, index, region)
    client.ensure_bucket_and_index(vector_dimension=dimension, distance_metric=metric)
    print(f"Ensured vector bucket '{bucket}', index '{index}', dim={dimension}, metric={metric}")


if __name__ == "__main__":
    main()


