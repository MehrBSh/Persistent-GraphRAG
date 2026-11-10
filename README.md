# PDF GraphRAG

A local, persistent GraphRAG tool for PDFs. It combines vector search with a
knowledge graph built from each chunk, so answers can draw on passages that are
related through shared entities, not only on the closest embeddings.
Everything runs locally via [Ollama](https://ollama.com).

## How it works

1. **Index**: extract text page by page with PyMuPDF, split it into overlapping
   chunks, embed each chunk (`nomic-embed-text`) into ChromaDB, and extract
   entities and relationships (`llama3.2`) into a NetworkX graph.
2. **Query**: embed the question, retrieve the top-k chunks, expand with chunks
   linked through the graph (1 hop), and generate an answer that cites file and page.
3. **Persistence**: vectors, graph (GraphML) and document hashes are stored in
   `storage/`, so unchanged PDFs are skipped on re-indexing.

## Project structure

```
main.py                     entry point
graphrag/
  config.py                 models, paths, chunking and retrieval settings
  llm.py                    Ollama chat and embeddings
  pdf_loader.py             PDF text extraction and chunking
  vector_store.py           persistent ChromaDB collection
  knowledge_graph.py        graph extraction, building and persistence
  indexer.py                indexing pipeline
  retriever.py              vector search, graph expansion, answer generation
  visualize.py              graph plot (matplotlib)
  cli.py                    command-line interface
  utils.py                  hashing, JSON parsing, metadata file
```

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Usage

```bash
python main.py index path/to/document.pdf          # add --force to re-index
python main.py query                               # interactive; --retrieval-only, --top-k N
python main.py graph                               # visualize the knowledge graph

