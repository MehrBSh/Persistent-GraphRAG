import json
import re

import networkx as nx

from .config import GRAPH_PATH
from .llm import local_chat
from .utils import parse_json_response


# ---------- persistence ----------

def load_graph():
    if GRAPH_PATH.exists():
        return nx.read_graphml(GRAPH_PATH)
    return nx.Graph()


def save_graph(graph):
    # GraphML supports scalar attributes, not Python sets.
    graph_to_save = graph.copy()
    for _, attributes in graph_to_save.nodes(data=True):
        if isinstance(attributes.get("chunks"), set):
            attributes["chunks"] = json.dumps(sorted(attributes["chunks"]))
    for _, _, attributes in graph_to_save.edges(data=True):
        if isinstance(attributes.get("relations"), set):
            attributes["relations"] = json.dumps(
                sorted(attributes["relations"])
            )
    nx.write_graphml(graph_to_save, GRAPH_PATH)


def parse_chunk_ids(value):
    try:
        return set(json.loads(value or "[]"))
    except (TypeError, json.JSONDecodeError):
        return set()


# ---------- extraction ----------

def normalize_entity(name):
    return re.sub(r"\s+", " ", str(name)).strip().casefold()


def extract_graph(text):
    prompt = f"""
Extract important entities and explicit relationships from this text.
Return valid JSON with exactly these keys:
{{
  "entities": [
    {{"name": "System Design", "type": "CONCEPT"}}
  ],
  "relationships": [
    {{
      "source": "System Design",
      "target": "Verification",
      "relation": "RELATED_TO"
    }}
  ]
}}

Rules:
- Include important concepts, methods, systems, organizations and people.
- Entity names should be concise and consistent.
- Every relationship must be supported by the text.
- Do not invent entities or relationships.
- Use empty lists if no useful information is present.

TEXT:
{text}
"""
    try:
        data = parse_json_response(local_chat(prompt, json_mode=True))
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        if not isinstance(entities, list):
            entities = []
        if not isinstance(relationships, list):
            relationships = []
        return entities, relationships
    except Exception as error:
        print(f"  Entity extraction warning: {error}")
        return [], []


def add_graph_data(graph, entities, relationships, chunk_id):
    valid_entities = {}

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        name = str(entity.get("name", "")).strip()
        entity_type = str(entity.get("type", "CONCEPT")).strip()
        if not name:
            continue

        key = normalize_entity(name)
        valid_entities[key] = name

        if graph.has_node(key):
            graph.nodes[key]["mentions"] = int(
                graph.nodes[key].get("mentions", 0)
            ) + 1
        else:
            graph.add_node(
                key, name=name, type=entity_type, mentions=1, chunks="[]"
            )

        chunk_ids = parse_chunk_ids(graph.nodes[key].get("chunks", "[]"))
        chunk_ids.add(str(chunk_id))
        graph.nodes[key]["chunks"] = json.dumps(sorted(chunk_ids))

    for relation in relationships:
        if not isinstance(relation, dict):
            continue

        source = normalize_entity(relation.get("source", ""))
        target = normalize_entity(relation.get("target", ""))
        relation_type = str(relation.get("relation", "RELATED_TO")).strip()

        if (
            not source or not target or source == target
            or source not in valid_entities or target not in valid_entities
        ):
            continue

        if graph.has_edge(source, target):
            edge = graph.edges[source, target]
            edge["weight"] = int(edge.get("weight", 1)) + 1
            relations_set = parse_chunk_ids(edge.get("relations", "[]"))
            relations_set.add(relation_type)
            edge["relations"] = json.dumps(sorted(relations_set))
        else:
            graph.add_edge(
                source, target, weight=1, relations=json.dumps([relation_type])
            )
