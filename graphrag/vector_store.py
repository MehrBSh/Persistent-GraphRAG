from functools import lru_cache

import chromadb

from .config import CHROMA_DIR, COLLECTION_NAME


@lru_cache(maxsize=1)
def get_collection():
    """Open the persistent Chroma collection (created on first use)."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
