import argparse

from .config import DEFAULT_MAX_PAGES, TOP_K
from .indexer import index_pdf
from .retriever import query_index
from .vector_store import get_collection
from .visualize import visualize_graph


def build_parser():
    parser = argparse.ArgumentParser(description="Persistent local PDF GraphRAG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser(
        "index", help="Index a PDF and persist its data"
    )
    index_parser.add_argument("pdf_path", help="Path to the PDF")
    index_parser.add_argument(
        "--max-pages", type=int, default=DEFAULT_MAX_PAGES,
        help="Maximum pages to index; 0 means all pages",
    )
    index_parser.add_argument(
        "--force", action="store_true",
        help="Force re-indexing even if the PDF hash is unchanged",
    )

    query_parser = subparsers.add_parser(
        "query", help="Query the existing persistent index"
    )
    query_parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help="Number of initial semantic search results",
    )
    query_parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Print retrieved passages without LLM answer generation",
    )

    subparsers.add_parser("graph", help="Visualize the saved knowledge graph")
    return parser


def run_query_loop(top_k, retrieval_only):
    collection = get_collection()
    if collection.count() == 0:
        print("No indexed chunks. Run the index command first.")
        return

    print(f"Loaded {collection.count()} persisted chunks.")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            query_index(
                question, top_k=top_k, generate_answer=not retrieval_only
            )
        except Exception as error:
            print(f"Query failed: {error}")


def main():
    args = build_parser().parse_args()

    try:
        if args.command == "index":
            index_pdf(args.pdf_path, max_pages=args.max_pages, force=args.force)
        elif args.command == "query":
            run_query_loop(args.top_k, args.retrieval_only)
        elif args.command == "graph":
            visualize_graph()
    except Exception as error:
        print(f"Error: {error}")
