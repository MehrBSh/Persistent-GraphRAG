import json

import networkx as nx

from .knowledge_graph import load_graph


def visualize_graph(max_nodes=60):
    graph = load_graph()
    if graph.number_of_nodes() == 0:
        print("Knowledge graph is empty.")
        return

    if graph.number_of_nodes() > max_nodes:
        important = sorted(
            graph.degree, key=lambda item: item[1], reverse=True
        )[:max_nodes]
        graph = graph.subgraph([node for node, _ in important]).copy()

    labels = {n: graph.nodes[n].get("name", n) for n in graph.nodes}

    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 10))
    positions = nx.spring_layout(graph, seed=42)
    nx.draw_networkx(
        graph, positions, labels=labels, node_size=1600,
        font_size=8, with_labels=True,
    )

    edge_labels = {}
    for source, target, attrs in graph.edges(data=True):
        try:
            relations = json.loads(attrs.get("relations", "[]"))
        except (TypeError, json.JSONDecodeError):
            relations = []
        edge_labels[(source, target)] = ", ".join(relations)

    nx.draw_networkx_edge_labels(
        graph, positions, edge_labels=edge_labels, font_size=7
    )
    plt.title("Persistent PDF Knowledge Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
