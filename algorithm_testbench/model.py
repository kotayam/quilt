"""Validated JSON scenarios. Unknown JSON fields survive round trips."""

from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import networkx as nx


def number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number")
    return value


def mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


@dataclass
class Scenario:
    graph: nx.DiGraph
    metadata: dict

    @classmethod
    def from_dict(cls, document):
        document = deepcopy(mapping(document, "graph document"))
        if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
            raise ValueError("schema_version must be 1")
        invocations = document.get("workflow_invocations")
        number(invocations, "workflow_invocations", positive=True)
        if not isinstance(invocations, int):
            raise ValueError("workflow_invocations must be an integer")
        graph = nx.DiGraph()
        if not isinstance(document.get("nodes"), list) or not isinstance(document.get("edges"), list):
            raise ValueError("nodes and edges must be arrays")
        for node in document.pop("nodes"):
            node = mapping(node, "node")
            node_id = node.pop("id", None)
            if not isinstance(node_id, str) or not node_id or node_id in graph:
                raise ValueError("node IDs must be unique nonempty strings")
            for key, value in mapping(node.get("resources"), f"{node_id}.resources").items():
                number(value, f"{node_id}.resources.{key}")
            mapping(node.setdefault("attributes", {}), f"{node_id}.attributes")
            graph.add_node(node_id, **node)
        for edge in document.pop("edges"):
            edge = mapping(edge, "edge")
            source, target = edge.pop("source", None), edge.pop("target", None)
            if not isinstance(source, str) or not isinstance(target, str) or source not in graph or target not in graph:
                raise ValueError("edge endpoints must reference existing nodes")
            if graph.has_edge(source, target):
                raise ValueError(f"duplicate edge {source} -> {target}")
            calls = edge.get("invocations")
            number(calls, "edge.invocations")
            if not isinstance(calls, int):
                raise ValueError("edge.invocations must be an integer")
            if edge.get("type") not in ("sync", "async"):
                raise ValueError("edge.type must be sync or async")
            mapping(edge.setdefault("attributes", {}), "edge.attributes")
            graph.add_edge(source, target, **edge)
        entry = document.get("entry")
        if not isinstance(entry, str) or entry not in graph:
            raise ValueError("entry must reference an existing node")
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("graph must be a DAG")
        if [n for n in graph if graph.in_degree(n) == 0] != [entry]:
            raise ValueError("graph must have exactly one root, matching entry")
        if len(nx.descendants(graph, entry)) + 1 != len(graph):
            raise ValueError("all nodes must be reachable from entry")
        return cls(graph, document)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self):
        document = deepcopy(self.metadata)
        document["nodes"] = [dict(id=n, **deepcopy(data)) for n, data in self.graph.nodes(data=True)]
        document["edges"] = [dict(source=u, target=v, **deepcopy(data))
                             for u, v, data in self.graph.edges(data=True)]
        return document

    def save(self, path):
        write_json(path, self.to_dict())

    @property
    def entry(self):
        return self.metadata["entry"]

    def alpha(self, source, target):
        calls = self.graph.edges[source, target]["invocations"]
        count = self.metadata["workflow_invocations"]
        return (calls + count - 1) // count


@dataclass
class AlgorithmResult:
    groups: dict[str, list[str]] = field(default_factory=dict)
    status: str = "feasible"
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self):
        return {"groups": self.groups, "status": self.status, "diagnostics": self.diagnostics}
