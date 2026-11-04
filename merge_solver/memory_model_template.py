"""Numeric experimental memory model and its symbolic solver formulation."""

import math

import networkx as nx


def estimate_memory_by_node(graph, root, members):
    """Return bottom-up memory estimates for a selected, rooted group.

    First run prepare_memory_inputs(graph, workflow_invocations) on the original
    graph. Read ratios from graph.edges[u, v]["calls_per_caller"]. Only internal
    group edges contribute. Zero and undefined ratios contribute no child memory.

    Each estimate is own memory + sum(ratio * async child estimate) + maximum
    positive-call sync child estimate. The parent retains its memory, sync calls
    are sequential, and async calls overlap each other and the largest sync call.
    Ratios are averages, so these estimates are not guaranteed peak bounds.
    Shared callees contribute separately along each incoming execution path.
    Inputs and existing alpha values are never modified. No Gurobi is required.
    """
    members = set(members)
    if root not in members or not members <= set(graph):
        raise ValueError("members must contain root and reference existing nodes")
    if (not graph.is_directed() or graph.is_multigraph()
            or not nx.is_directed_acyclic_graph(graph)):
        raise ValueError("graph must be a DAG without parallel edges")
    group = graph.subgraph(members)
    if nx.descendants(group, root) | {root} != members:
        raise ValueError("all members must be reachable from root within the group")

    def valid_number(value):
        return (not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value) and value >= 0)

    for node, data in group.nodes(data=True):
        if not valid_number(data.get("m")):
            raise ValueError(f"node {node!r} memory must be finite and nonnegative")
    for u, v, data in group.edges(data=True):
        if data.get("type") not in ("sync", "async"):
            raise ValueError(f"unknown call type on {u!r} -> {v!r}")
        if "calls_per_caller" not in data:
            raise ValueError("missing calls_per_caller; run prepare_memory_inputs first")
        ratio = data["calls_per_caller"]
        if ratio is not None and not valid_number(ratio):
            raise ValueError(f"invalid calls_per_caller on {u!r} -> {v!r}")

    memory = {}
    for node in reversed(list(nx.topological_sort(group))):
        async_total, sync_peak = 0, 0
        for child in group.successors(node):
            edge = group.edges[node, child]
            ratio = edge["calls_per_caller"]
            if ratio is None or ratio == 0:
                continue
            if edge["type"] == "async":
                async_total += ratio * memory[child]
            else:
                sync_peak = max(sync_peak, memory[child])
        memory[node] = group.nodes[node]["m"] + async_total + sync_peak
    return memory


def estimate_memory(graph, root, members):
    """Return the root estimate; see estimate_memory_by_node for assumptions.

    Prepare the original graph with prepare_memory_inputs before calling.
    """
    return estimate_memory_by_node(graph, root, members)[root]


def build_memory_expression(model, graph, root, roots, y, z):
    """Return a Gurobi linear expression for one candidate root's memory usage.

    Called during ILP construction, before group membership has been selected:
      * ``model`` is the active Gurobi model;
      * ``roots`` contains all candidate roots in this ILP;
      * ``y[node, root]`` is the binary node-membership variable;
      * ``z[source, target, root]`` is binary internal-edge membership.

    First call ``prepare_memory_inputs(graph, workflow_invocations)`` from
    ``merge_solver.memory_inputs`` once before root search or a direct solve.
    Read caller-relative ratios through
    ``graph.edges[u, v]["calls_per_caller"]`` and counts through node
    ``profile_invocations``. Ratios are unrounded averages across all observed
    caller invocations (including shared callers), or None for uninvoked callers;
    they do not establish peak concurrency or context-specific behavior.

    The caller adds ``expression <= M``. You may add auxiliary variables and
    constraints here, using names unique to this root. Variables have no solution
    values yet: do not read .X or branch on symbolic membership. In particular,
    estimate_memory cannot be called on an assumed final group to encode the ILP.

    Formulation stages:
      1. Precompute any membership-independent traversal data for this root.
      2. Express membership-dependent concurrency and bottom-up aggregation using
         y/z and auxiliary linear constraints as needed. Avoid variable products
         or Python max() on symbolic expressions without a valid formulation.
      3. Return the resulting linear memory expression. Check it against
         estimate_memory for fixed memberships in tests.

    Leave the original graph alpha unchanged so CPU accounting is unaffected.
    """
    import gurobipy as gp

    # Validate prepared data using the numeric implementation.
    estimate_memory_by_node(graph, root, {root} | nx.descendants(graph, root))
    nodes = list(graph)
    prefix = f"recursive_{list(roots).index(root)}"
    memory = model.addVars(nodes, lb=0, name=f"{prefix}_memory")
    for index, node in enumerate(nodes):
        asynchronous, synchronous = [], []
        for edge_index, child in enumerate(graph.successors(node)):
            edge = graph.edges[node, child]
            ratio = edge["calls_per_caller"]
            if ratio is None or ratio == 0:
                continue
            contribution = model.addVar(lb=0, name=f"{prefix}_edge_{index}_{edge_index}")
            model.addGenConstrIndicator(z[node, child, root], False, contribution == 0)
            model.addGenConstrIndicator(z[node, child, root], True,
                                        contribution == memory[child])
            if edge["type"] == "async":
                asynchronous.append(ratio * contribution)
            else:
                synchronous.append(contribution)
        peak = model.addVar(lb=0, name=f"{prefix}_sync_{index}")
        if synchronous:
            model.addGenConstrMax(peak, synchronous, constant=0)
        else:
            model.addConstr(peak == 0)
        model.addConstr(memory[node] == graph.nodes[node]["m"] * y[node, root]
                        + gp.quicksum(asynchronous) + peak)
    return memory[root]
