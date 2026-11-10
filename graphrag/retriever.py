import json

import networkx as nx

from .config import GRAPH_HOPS, MAX_CONTEXT_CHUNKS, TOP_K
from .knowledge_graph import load_graph, parse_chunk_ids
from .llm import get_embedding, local_chat
from .vector_store import get_collection


def retrieve(question, top_k=TOP_K):
    """Search persisted vectors; no document re-embedding occurs here."""
    collection = get_collection()
    if collection.count() == 0:
        raise ValueError("The vector database is empty. Index a PDF first.")

    query_embedding = get_embedding(question).tolist()
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    return [
        {"id": chunk_id, "text": text, "metadata": metadata,
         "distance": float(distance)}
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0], result["documents"][0],
            result["metadatas"][0], result["distances"][0],
        )
    ]


def expand_with_graph(retrieved, hops=GRAPH_HOPS):
    """Add chunks linked to the retrieved ones through the knowledge graph."""
    graph = load_graph()
    selected_ids = {hit["id"] for hit in retrieved}
    starting_nodes = set()

    for node, attributes in graph.nodes(data=True):
        if selected_ids.intersection(parse_chunk_ids(attributes.get("chunks"))):
            starting_nodes.add(node)

    related_nodes = set(starting_nodes)
    for node in starting_nodes:
        related_nodes.update(
            nx.single_source_shortest_path_length(graph, node, cutoff=hops).keys()
        )

    for node in related_nodes:
        selected_ids.update(parse_chunk_ids(graph.nodes[node].get("chunks")))

    if not selected_ids:
        return retrieved, graph, related_nodes

    result = get_collection().get(
        ids=list(selected_ids), include=["documents", "metadatas"]
    )

    by_id = {hit["id"]: hit for hit in retrieved}
    for chunk_id, text, metadata in zip(
        result["ids"], result["documents"], result["metadatas"]
    ):
        if chunk_id not in by_id:
            by_id[chunk_id] = {
                "id": chunk_id, "text": text,
                "metadata": metadata, "distance": None,
            }

    expanded = list(by_id.values())
    # Semantically retrieved chunks first, then graph-added chunks.
    expanded.sort(key=lambda hit: (
        hit["distance"] is None,
        hit["distance"] if hit["distance"] is not None else 0,
    ))
    return expanded[:MAX_CONTEXT_CHUNKS], graph, related_nodes


def format_graph_context(graph, node_ids):
    lines = []
    for source in node_ids:
        if source not in graph:
            continue

        for target in graph.neighbors(source):
            if target not in node_ids:
                continue

            source_name = graph.nodes[source].get("name", source)
            target_name = graph.nodes[target].get("name", target)
            try:
                relations = json.loads(
                    graph.edges[source, target].get("relations", "[]")
                )
            except (TypeError, json.JSONDecodeError):
                relations = ["RELATED_TO"]

            for relation in relations:
                lines.append(f"{source_name} --[{relation}]--> {target_name}")

    return "\n".join(sorted(set(lines)))


def query_index(question, top_k=TOP_K, generate_answer=True):
    retrieved = retrieve(question, top_k=top_k)
    expanded, graph, related_nodes = expand_with_graph(retrieved)

    print("\n--- Retrieved passages ---")
    for index, hit in enumerate(expanded, start=1):
        metadata = hit["metadata"]
        score = (
            f"{hit['distance']:.4f}"
            if hit["distance"] is not None else "graph expansion"
        )
        print(
            f"\n[{index}] {metadata.get('source')} | "
            f"page {metadata.get('page')} | distance: {score}"
        )
        print(hit["text"][:1000])

    if not generate_answer:
        return

    text_context = "\n\n".join(
        f"Source: {hit['metadata'].get('source')}, "
        f"page {hit['metadata'].get('page')}\n{hit['text']}"
        for hit in expanded
    )
    graph_context = format_graph_context(graph, related_nodes)

    prompt = f"""
Answer the question using only the evidence below.
Cite the source filename and page number where possible.
If the evidence is insufficient, state that clearly.
Do not invent facts.

DOCUMENT EXCERPTS:
{text_context}

KNOWLEDGE GRAPH:
{graph_context or "No relevant graph relationships found."}

QUESTION:
{question}
"""
    print("\n--- Answer ---")
    print(local_chat(prompt))
