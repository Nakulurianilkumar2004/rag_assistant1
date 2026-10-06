"""Indexing pipeline for zoikorooms.com.

RAG steps handled here (run once, and again whenever the website changes):
  1. LOAD   - crawl every page listed in the site's sitemap
  2. SPLIT  - clean the HTML and split the text into overlapping chunks
  3. EMBED  - turn each chunk into a vector with a local embedding model
  4. STORE  - save chunks + vectors to data/ (a simple on-disk vector store)
"""

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
from bs4 import BeautifulSoup
from fastembed import TextEmbedding

SITE = "https://zoikorooms.com"
SITEMAP_URL = f"{SITE}/sitemap.xml"
DATA_DIR = Path(__file__).parent / "data"
CHUNKS_FILE = DATA_DIR / "chunks.json"
VECTORS_FILE = DATA_DIR / "vectors.npy"
HEADERS = {"User-Agent": "ZoikoRoomsRAG/1.0 (+local RAG demo)"}

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 150


# ---------- Step 1: LOAD ----------
def get_urls() -> list[str]:
    resp = requests.get(SITEMAP_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    root = ET.fromstring(resp.content)
    return [loc.text.strip() for loc in root.findall(".//sm:loc", ns) if loc.text]


def load_pages() -> list[dict]:
    urls = get_urls()
    print(f"[1/4 LOAD] {len(urls)} URLs in sitemap")
    pages = []
    for i, url in enumerate(urls, 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [{i}/{len(urls)}] SKIP {url}: {e}")
            continue
        pages.append({"url": url, "html": resp.text})
        print(f"  [{i}/{len(urls)}] {url}")
        time.sleep(0.3)  # be polite
    return pages


# ---------- Step 2: SPLIT ----------
def extract_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    # Drop boilerplate that repeats on every page and would pollute retrieval.
    for tag in soup(["script", "style", "noscript", "svg", "nav", "header", "footer", "form"]):
        tag.decompose()
    main = soup.find("main") or soup.body or soup
    lines = (line.strip() for line in main.get_text("\n").splitlines())
    return title, "\n".join(line for line in lines if line)


def chunk_text(text: str) -> list[str]:
    paragraphs = [p for p in re.split(r"\n+", text) if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if current and len(current) + len(para) + 1 > CHUNK_SIZE:
            chunks.append(current)
            current = current[-CHUNK_OVERLAP:]  # carry a little context over
        current = f"{current}\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def split_pages(pages: list[dict]) -> list[dict]:
    chunks = []
    for page in pages:
        title, text = extract_text(page["html"])
        for j, chunk in enumerate(chunk_text(text)):
            chunks.append({"id": f"{page['url']}#{j}", "url": page["url"], "title": title, "text": chunk})
    print(f"[2/4 SPLIT] {len(chunks)} chunks from {len(pages)} pages")
    return chunks


# ---------- Step 3: EMBED ----------
def embed_chunks(chunks: list[dict]) -> np.ndarray:
    model = TextEmbedding(EMBED_MODEL)
    vectors = np.array(list(model.embed([f"{c['title']}\n{c['text']}" for c in chunks])), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # normalise so dot product = cosine
    print(f"[3/4 EMBED] {vectors.shape[0]} vectors x {vectors.shape[1]} dims ({EMBED_MODEL})")
    return vectors


# ---------- Step 4: STORE ----------
def store(chunks: list[dict], vectors: np.ndarray) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    CHUNKS_FILE.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    np.save(VECTORS_FILE, vectors)
    print(f"[4/4 STORE] saved {CHUNKS_FILE.name} and {VECTORS_FILE.name} in {DATA_DIR}")


def main() -> None:
    chunks = split_pages(load_pages())
    store(chunks, embed_chunks(chunks))


if __name__ == "__main__":
    main()
