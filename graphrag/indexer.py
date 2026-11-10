from pathlib import Path

import networkx as nx

from .config import (
    CHROMA_DIR, DEFAULT_MAX_PAGES, EMBED_MODEL, GRAPH_PATH, LLM_MODEL,
)
from .knowledge_graph import add_graph_data, extract_graph, load_graph, save_graph
from .llm import get_embedding
from .pdf_loader import read_pdf, split_into_chunks
from .utils import file_sha256, load_metadata, save_metadata
from .vector_store import get_collection


def _rebuild_graph_from_index(collection):
    """Rebuild the graph from all chunks currently stored in the index."""
    remaining = collection.get(include=["documents", "metadatas"])
    graph = nx.Graph()
    for chunk_id, chunk_text in zip(remaining["ids"], remaining["documents"]):
        entities, relationships = extract_graph(chunk_text)
        add_graph_data(graph, entities, relationships, chunk_id)
    return graph


def index_pdf(pdf_path, max_pages=DEFAULT_MAX_PAGES, force=False):
    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    collection = get_collection()
    digest = file_sha256(path)
    metadata = load_metadata()
    document_key = str(path)
    existing = metadata["documents"].get(document_key)

    if existing and existing.get("sha256") == digest and not force:
        print("This unchanged PDF is already indexed. Skipping.")
        print("Use --force to re-index it.")
        return

    if existing and existing.get("sha256") != digest and not force:
        print("The PDF has changed since the previous indexing.")
        print("Re-indexing this document and replacing its previous chunks.")

    # Remove prior chunks for this exact file.
    old_ids = collection.get(
        where={"source_path": document_key}, include=[]
    )["ids"]
    if old_ids:
        collection.delete(ids=old_ids)

    # When replacing a document, rebuild the graph from the remaining chunks.
    graph = _rebuild_graph_from_index(collection) if existing else load_graph()

    pages = read_pdf(path, max_pages=max_pages)
    indexed_ids = []

    for page_data in pages:
        chunks = split_into_chunks(page_data["text"])

        for chunk_number, chunk in enumerate(chunks, start=1):
            chunk_id = f"{digest[:12]}_p{page_data['page']}_c{chunk_number}"
            print(f"Indexing page {page_data['page']}, chunk {chunk_number}...")

            embedding = get_embedding(chunk).tolist()
            entities, relationships = extract_graph(chunk)

            collection.upsert(
                ids=[chunk_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "source": path.name,
                    "source_path": document_key,
                    "page": int(page_data["page"]),
                    "chunk_number": int(chunk_number),
                    "sha256": digest,
                }],
            )

            add_graph_data(graph, entities, relationships, chunk_id)
            indexed_ids.append(chunk_id)

    save_graph(graph)

    metadata["documents"][document_key] = {
        "filename": path.name,
        "sha256": digest,
        "chunk_ids": indexed_ids,
        "chunk_count": len(indexed_ids),
        "embedding_model": EMBED_MODEL,
        "llm_model": LLM_MODEL,
        "pages_indexed": len(pages),
    }
    save_metadata(metadata)

    print("\nIndexing complete.")
    print(f"Chunks indexed: {len(indexed_ids)}")
    print(f"Entities: {graph.number_of_nodes()}")
    print(f"Relationships: {graph.number_of_edges()}")
    print(f"Saved database: {CHROMA_DIR}")
    print(f"Saved graph: {GRAPH_PATH}")
