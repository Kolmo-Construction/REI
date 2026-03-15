"""
One-time indexing script: embed data/sample_products.json and load into Qdrant.

Usage:
    # 1. Start Qdrant
    docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

    # 2. Run (no API key needed — uses local FastEmbed models)
    uv run python scripts/index_catalog.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, SparseVector

COLLECTION_NAME = "rei_products"
DENSE_MODEL = "BAAI/bge-large-en-v1.5"
DENSE_DIM = 1024
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def build_product_document(product: dict) -> str:
    """Convert a product dict into a natural-language string for embedding."""
    parts = [product["name"]]
    if product.get("category"):
        parts.append(f"Category: {product['category'].replace('_', ' ')}")
    if product.get("fill_type"):
        parts.append(f"Fill: {product['fill_type']}")
    if product.get("temp_rating_f") is not None:
        parts.append(f"Temperature rating: {product['temp_rating_f']}°F")
    if product.get("weight_oz") is not None:
        parts.append(f"Weight: {product['weight_oz']} oz")
    if product.get("r_value") is not None:
        parts.append(f"R-value: {product['r_value']}")
    if product.get("water_resistance"):
        parts.append(f"Water resistance: {product['water_resistance']}")
    if product.get("tags"):
        parts.append(f"Use cases: {', '.join(product['tags'])}")
    return ". ".join(parts)


def create_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists — dropping and recreating.")
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
        sparse_vectors_config={
            "sparse_text": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),
            ),
        },
    )

    # Payload indexes for pre-filtering (critical for filter-during-traversal)
    for field, schema_type in [
        ("temp_rating_f", models.PayloadSchemaType.FLOAT),
        ("r_value",       models.PayloadSchemaType.FLOAT),
        ("weight_oz",     models.PayloadSchemaType.FLOAT),
        ("category",      models.PayloadSchemaType.KEYWORD),
        ("fill_type",     models.PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(COLLECTION_NAME, field, schema_type)

    print(f"Collection '{COLLECTION_NAME}' created with payload indexes.")


def main() -> None:
    products_path = Path(__file__).parent.parent / "data" / "sample_products.json"
    products = json.loads(products_path.read_text())
    print(f"Loaded {len(products)} products.")

    docs = [build_product_document(p) for p in products]

    # Dense embeddings via FastEmbed (local, no API key)
    print(f"Generating dense embeddings with {DENSE_MODEL} @ {DENSE_DIM} dims...")
    dense_model = TextEmbedding(model_name=DENSE_MODEL)
    dense_vectors = [list(v) for v in dense_model.embed(docs)]
    print("Dense embeddings done.")

    # Sparse embeddings via FastEmbed (SPLADE — local, no API key)
    print("Generating sparse embeddings via SPLADE (FastEmbed)...")
    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    sparse_results = list(sparse_model.embed(docs))
    print("Sparse embeddings done.")

    # Build and upsert points
    qdrant = QdrantClient(url=QDRANT_URL)
    create_collection(qdrant)

    points = [
        PointStruct(
            id=i,
            vector={
                "dense_text": dense_vectors[i],
                "sparse_text": SparseVector(
                    indices=sparse_results[i].indices.tolist(),
                    values=sparse_results[i].values.tolist(),
                ),
            },
            payload={**products[i]},
        )
        for i in range(len(products))
    ]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"\nDone. {len(points)} products indexed into '{COLLECTION_NAME}'.")
    print(f"Qdrant dashboard: http://localhost:6333/dashboard")


if __name__ == "__main__":
    main()
