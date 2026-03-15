"""
Index data/expert_advice.json into Qdrant rei_expert_advice collection.

Usage:
    uv run python scripts/index_expert_advice.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

COLLECTION_NAME = "rei_expert_advice"
DENSE_MODEL = "BAAI/bge-large-en-v1.5"
DENSE_DIM = 1024
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def _stable_id(url: str, section: str) -> int:
    raw = f"{url}|{section}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:15], 16)


def create_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' exists — dropping and recreating.")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense_text": models.VectorParams(
                size=DENSE_DIM,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
            ),
        },
    )

    for field, schema_type in [
        ("category", models.PayloadSchemaType.KEYWORD),
        ("url",      models.PayloadSchemaType.KEYWORD),
        ("source",   models.PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(COLLECTION_NAME, field, schema_type)

    print(f"Collection '{COLLECTION_NAME}' created.")


def main() -> None:
    data_path = Path(__file__).parent.parent / "data" / "expert_advice.json"
    chunks = json.loads(data_path.read_text())
    print(f"Loaded {len(chunks)} chunks.")

    texts = [c["chunk_text"] for c in chunks]

    print(f"Embedding with {DENSE_MODEL} @ {DENSE_DIM} dims...")
    model = TextEmbedding(model_name=DENSE_MODEL)
    vectors = [list(v) for v in model.embed(texts)]
    print("Embeddings done.")

    qdrant = QdrantClient(url=QDRANT_URL)
    create_collection(qdrant)

    points = [
        PointStruct(
            id=_stable_id(chunks[i]["url"], chunks[i]["section"]),
            vector={"dense_text": vectors[i]},
            payload={
                "url":        chunks[i]["url"],
                "title":      chunks[i]["title"],
                "section":    chunks[i]["section"],
                "category":   chunks[i]["category"],
                "chunk_text": chunks[i]["chunk_text"],
                "source":     chunks[i].get("source", "synthetic"),
            },
        )
        for i in range(len(chunks))
    ]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"\nDone. {len(points)} chunks indexed into '{COLLECTION_NAME}'.")
    print(f"Qdrant dashboard: http://localhost:6333/dashboard")


if __name__ == "__main__":
    main()
