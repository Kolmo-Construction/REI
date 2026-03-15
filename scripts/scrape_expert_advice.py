"""
Scrape REI Expert Advice articles and index them into Qdrant as rei_expert_advice.

What it does:
  1. Fetches article URLs from REI's expert advice category pages
  2. Extracts each article's text, chunked by <h2> section
  3. Embeds chunks with BAAI/bge-large-en-v1.5 (same model as rei_products)
  4. Upserts into Qdrant collection "rei_expert_advice"

Usage:
    docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
    uv run python scripts/scrape_expert_advice.py

Rate limiting: 1 request/second, respects robots.txt delay.
For development only — check REI's ToS before production use.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Iterator
from urllib.parse import urljoin

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from bs4 import BeautifulSoup
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

COLLECTION_NAME = "rei_expert_advice"
DENSE_MODEL = "BAAI/bge-large-en-v1.5"
DENSE_DIM = 1024
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CHUNK_MIN_WORDS = 40       # discard sections that are too short to be useful
CRAWL_DELAY_S = 1.0        # be polite

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; GreenvestDev/0.1; "
        "+https://github.com/your-org/greenvest)"
    ),
}

# Category slugs on rei.com/learn/expert-advice that match our catalog
CATEGORY_SLUGS = [
    "sleeping-bags",
    "sleeping-pads",
    "backpacking-sleeping-bags",
    "camping-sleeping-bags",
    "backpacks",
    "rain-jackets",
    "insulated-jackets",
    "hiking-boots-and-shoes",
    "layering",
    "camping",
    "backpacking",
]

BASE_LEARN_URL = "https://www.rei.com/learn/expert-advice"


# ---------------------------------------------------------------------------
# Crawling helpers
# ---------------------------------------------------------------------------

def _get(client: httpx.Client, url: str) -> BeautifulSoup | None:
    try:
        resp = client.get(url, headers=HEADERS, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as exc:
        print(f"  [skip] {url} — {exc}")
        return None


def _article_urls_from_category(client: httpx.Client, slug: str) -> list[str]:
    url = f"{BASE_LEARN_URL}/{slug}.html"
    soup = _get(client, url)
    if not soup:
        return []

    urls = set()
    for a in soup.select("a[href]"):
        href = a["href"]
        if "/learn/expert-advice/" in href and href.endswith(".html"):
            urls.add(urljoin("https://www.rei.com", href))

    print(f"  {slug}: {len(urls)} article links found")
    return list(urls)


def _chunks_from_article(
    client: httpx.Client, url: str
) -> Iterator[dict]:
    """Yield one dict per <h2> section in the article."""
    soup = _get(client, url)
    if not soup:
        return

    # Article title
    title_el = soup.select_one("h1")
    title = title_el.get_text(strip=True) if title_el else url.split("/")[-1]

    # Infer category from URL path
    parts = url.rstrip("/").split("/")
    category = parts[-2] if len(parts) >= 2 else "general"

    # Walk h2 boundaries — each h2 + following paragraphs = one chunk
    article_body = soup.select_one("article") or soup.select_one("main") or soup
    h2s = article_body.find_all("h2")

    if not h2s:
        # No sections — treat whole article as single chunk
        text = article_body.get_text(separator=" ", strip=True)
        words = text.split()
        if len(words) >= CHUNK_MIN_WORDS:
            yield {
                "doc_id": _stable_id(url, ""),
                "url": url,
                "title": title,
                "section": title,
                "category": category,
                "chunk_text": " ".join(words),
            }
        return

    for h2 in h2s:
        section_title = h2.get_text(strip=True)
        parts_text: list[str] = [section_title]

        for sibling in h2.find_next_siblings():
            if sibling.name == "h2":
                break
            if sibling.name in ("p", "ul", "ol", "li", "blockquote"):
                parts_text.append(sibling.get_text(separator=" ", strip=True))

        chunk_text = " ".join(parts_text)
        words = chunk_text.split()
        if len(words) < CHUNK_MIN_WORDS:
            continue

        yield {
            "doc_id": _stable_id(url, section_title),
            "url": url,
            "title": title,
            "section": section_title,
            "category": category,
            "chunk_text": chunk_text,
        }


def _stable_id(url: str, section: str) -> str:
    """Deterministic integer ID from url+section so re-runs are idempotent (upsert)."""
    raw = f"{url}|{section}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:15], 16)


# ---------------------------------------------------------------------------
# Qdrant setup
# ---------------------------------------------------------------------------

def create_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' exists — will upsert (idempotent).")
        return

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
    ]:
        client.create_payload_index(COLLECTION_NAME, field, schema_type)

    print(f"Collection '{COLLECTION_NAME}' created.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    qdrant = QdrantClient(url=QDRANT_URL)
    create_collection(qdrant)

    dense_model = TextEmbedding(model_name=DENSE_MODEL)

    all_chunks: list[dict] = []
    seen_urls: set[str] = set()

    with httpx.Client() as http:
        for slug in CATEGORY_SLUGS:
            print(f"\n[category] {slug}")
            article_urls = _article_urls_from_category(http, slug)
            time.sleep(CRAWL_DELAY_S)

            for url in article_urls:
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                print(f"  [article] {url}")
                chunks = list(_chunks_from_article(http, url))
                print(f"    → {len(chunks)} chunks")
                all_chunks.extend(chunks)
                time.sleep(CRAWL_DELAY_S)

    if not all_chunks:
        print("\nNo chunks collected — check network access and URL patterns.")
        return

    print(f"\nTotal chunks to index: {len(all_chunks)}")
    print(f"Embedding with {DENSE_MODEL}...")

    texts = [c["chunk_text"] for c in all_chunks]
    vectors = [list(v) for v in dense_model.embed(texts)]

    points = [
        PointStruct(
            id=all_chunks[i]["doc_id"],
            vector={"dense_text": vectors[i]},
            payload={
                "url":        all_chunks[i]["url"],
                "title":      all_chunks[i]["title"],
                "section":    all_chunks[i]["section"],
                "category":   all_chunks[i]["category"],
                "chunk_text": all_chunks[i]["chunk_text"],
                "source":     "rei_scrape",
            },
        )
        for i in range(len(all_chunks))
    ]

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points[i:i + batch_size])
        print(f"  upserted {min(i + batch_size, len(points))}/{len(points)}")

    print(f"\nDone. {len(points)} chunks in '{COLLECTION_NAME}'.")
    print(f"Qdrant dashboard: http://localhost:6333/dashboard")


if __name__ == "__main__":
    main()
