from pathlib import Path

# Models (served locally by Ollama)
LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"

# Storage
BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
GRAPH_PATH = STORAGE_DIR / "knowledge_graph.graphml"
METADATA_PATH = STORAGE_DIR / "document_metadata.json"
COLLECTION_NAME = "pdf_chunks"

# Chunking
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 250
DEFAULT_MAX_PAGES = 0  # 0 means all pages

# Retrieval
TOP_K = 5
GRAPH_HOPS = 1
MAX_CONTEXT_CHUNKS = 10

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
