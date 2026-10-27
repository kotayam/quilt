"""Independent verification of assignments; never trust an algorithm's cost."""

import networkx as nx

from .resources import get_model


def evaluate(scenario, groups, limits, resource_model="quilt"):
    model = get_model(resource_model)
    model.validate(scenario, limits)
    graph = scenario.graph
    errors = []
    members_by_root = {}
    if not isinstance(groups, dict) or not groups:
        return {"valid": False, "errors": ["groups must be a nonempty object"]}
    for root, members in groups.items():
        if (not isinstance(members, list) or any(not isinstance(n, str) for n in members)
                or len(set(members)) != len(members)):
            errors.append(f"{root}: members must be a list of unique node IDs")
            continue
        selected = set(members)
        if root not in graph or root not in selected or selected - set(graph):
            errors.append(f"{root}: unknown nodes or missing root membership")
            continue
        subgraph = graph.subgraph(selected)
        if nx.descendants(subgraph, root) | {root} != selected:
            errors.append(f"{root}: members are not directed-reachable from group root")
        members_by_root[root] = selected
    covered = set().union(*members_by_root.values()) if members_by_root else set()
    if set(graph) - covered:
        errors.append(f"uncovered nodes: {sorted(set(graph) - covered)}")
    roots = set(members_by_root)
    cut_edges = []
    for source, target, data in graph.edges(data=True):
        # Match Quilt x_ij >= y_ir - y_jr for EVERY root r, not just a
        # test of whether any group happens to contain both endpoints.
        cut = any(source in selected and target not in selected
                  for selected in members_by_root.values())
        if cut:
            cut_edges.append({"source": source, "target": target, "cost": data["invocations"]})
            if target not in roots:
                errors.append(f"illegal boundary {source} -> {target}: target is not a group root")
    resources = {}
    for root, members in members_by_root.items():
        usage = model.usage(scenario, root, members)
        slack = {key: limits[key] - value for key, value in usage.items()}
        resources[root] = {"usage": usage, "slack": slack}
        for key, value in usage.items():
            if value > limits[key] + 1e-9:
                errors.append(f"{root}: {key} usage {value} exceeds {limits[key]}")
    baseline = sum(data["invocations"] for _, _, data in graph.edges(data=True))
    cost = sum(edge["cost"] for edge in cut_edges)
    metrics = {
        "valid": not errors, "errors": errors, "resource_model": resource_model,
        "group_count": len(groups),
        "duplicated_placements": sum(map(len, members_by_root.values())) - len(covered),
        "resources": resources,
    }
    if not errors:
        metrics.update(remote_call_cost=cost, baseline_cost=baseline, cut_edges=cut_edges,
                       reduction_from_baseline=1 - cost / baseline if baseline else None,
                       absolute_savings=baseline - cost)
    return metrics
