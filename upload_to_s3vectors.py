import os
import json
import time
from typing import Dict, Any, List

from dotenv import load_dotenv
from tqdm import tqdm

from s3_vectors_client import S3VectorsClient


load_dotenv()


BATCH_SIZE = int(os.environ.get("S3_VECTORS_BATCH_SIZE", "500"))
RATE_LIMIT_DELAY = float(os.environ.get("S3_VECTORS_RATE_LIMIT_DELAY", "0.1"))


def upload_embeddings_to_s3_vectors(json_path: str = "embeddings.json") -> None:
    print(f"Reading embeddings from {json_path}...")

    client = S3VectorsClient()

    with open(json_path, "r") as f:
        data: Dict[str, Any] = json.load(f)

    embeddings: List[Dict[str, Any]] = data["embeddings"]
    total_embeddings = len(embeddings)
    print(f"Found {total_embeddings} embeddings to upload")

    for i in tqdm(range(0, total_embeddings, BATCH_SIZE)):
        batch = embeddings[i : i + BATCH_SIZE]

        vectors_payload = []
        for item in batch:
            vectors_payload.append(
                {
                    "key": str(item["article_id"]),
                    "vector": item["embedding"],
                    "metadata": {
                        "article_id": str(item["article_id"]),
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "hash": item.get("hash", ""),
                    },
                }
            )

        try:
            client.put_vectors(vectors_payload)
            time.sleep(RATE_LIMIT_DELAY)
        except Exception as e:
            print(f"Error uploading batch starting at index {i}: {str(e)}")
            continue

    print("Upload completed successfully!")


if __name__ == "__main__":
    upload_embeddings_to_s3_vectors('/Users/clkruse/Documents/wikiclip-original/embeddings.json')


