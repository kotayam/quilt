"""Profiling inputs for custom memory models, independent of Gurobi."""

import math

import networkx as nx


def prepare_memory_inputs(graph, workflow_invocations):
    """Annotate a rooted DAG in place before root search or direct ILP calls.

    The original entry is the unique node with no incoming edges. Its invocation
    count is ``workflow_invocations``; other counts sum incoming ``weight`` values.
    ``calls_per_caller`` averages over all observed invocations of the caller,
    including shared callers. It implies neither peak concurrency nor calling
    context. A zero-invocation caller has an undefined ratio (None).

    Existing attributes, including workflow-normalized ``alpha``, are preserved.
    Invalid counts or positive calls from an uninvoked caller raise ValueError
    before any attributes are written. Returns the annotated graph.
    """
    def validate_count(value, name):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite nonnegative number")

    validate_count(workflow_invocations, "workflow_invocations")
    if not graph.is_directed() or graph.is_multigraph() or not nx.is_directed_acyclic_graph(graph):
        raise ValueError("graph must be a directed acyclic graph without parallel edges")
    roots = [node for node in graph if graph.in_degree(node) == 0]
    if len(roots) != 1:
        raise ValueError("graph must have exactly one original entry root")
    for u, v, data in graph.edges(data=True):
        validate_count(data.get("weight"), f"edge {u!r} -> {v!r} weight")
    entry = roots[0]
    counts = {
        node: workflow_invocations if node == entry else sum(
            data["weight"] for _, _, data in graph.in_edges(node, data=True))
        for node in graph
    }
    ratios = {}
    for u, v, data in graph.edges(data=True):
        if counts[u] == 0 and data["weight"] > 0:
            raise ValueError(f"positive outgoing calls from uninvoked caller {u!r}")
        ratios[u, v] = data["weight"] / counts[u] if counts[u] else None
    nx.set_node_attributes(graph, counts, "profile_invocations")
    nx.set_edge_attributes(graph, ratios, "calls_per_caller")
    return graph
