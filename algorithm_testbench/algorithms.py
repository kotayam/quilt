"""Algorithm entry points. Plugins use module:function(scenario, limits, config)."""

from copy import deepcopy
import importlib
from pathlib import Path
import sys

from .evaluate import evaluate
from .model import AlgorithmResult


def no_merge(scenario, limits, config):
    return AlgorithmResult({n: [n] for n in scenario.graph})


def greedy(scenario, limits, config):
    """Simple disjoint agglomerative baseline; accepts only valid improvements."""
    groups = no_merge(scenario, limits, config).groups
    current = evaluate(scenario, groups, limits, config["resource_model"])
    if not current["valid"]:
        return AlgorithmResult(groups, "infeasible", {"reason": "a singleton exceeds capacity"})
    while True:
        best = None
        best_cost = current["remote_call_cost"]
        for source in sorted(groups):
            for target in sorted(groups):
                if source == target:
                    continue
                candidate = deepcopy(groups)
                candidate[source] = sorted(set(candidate[source]) | set(candidate.pop(target)))
                report = evaluate(scenario, candidate, limits, config["resource_model"])
                if report["valid"] and report["remote_call_cost"] < best_cost:
                    best, best_cost = candidate, report["remote_call_cost"]
        if best is None:
            return AlgorithmResult(groups)
        groups = best
        current = evaluate(scenario, groups, limits, config["resource_model"])


def quilt(scenario, limits, config, strategy):
    """Adapt the original solver in the isolated worker; never edit source files."""
    if config["resource_model"] != "quilt":
        raise ValueError("Quilt adapters support only the quilt resource model")
    original_dir = Path(__file__).resolve().parents[1] / "merge_solver"
    # The adapter runs in its own worker. Preserve even the original directory's
    # generated state by keeping imports from creating __pycache__ there.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(original_dir))
    try:
        import gurobipy as gp
        import root_selector
        from rdag import preprocess_graph
        from memory_inputs import prepare_memory_inputs
        from downstream_impact import select_downstream_candidate_roots
        from weighted_degree import select_weighted_degree_candidates
    except ImportError as error:
        raise ImportError("Quilt baselines require optional gurobipy and a working Gurobi license") from error

    graph = deepcopy(scenario.graph)
    for _, data in graph.nodes(data=True):
        data["c"], data["m"] = data["resources"]["cpu"], data["resources"]["memory"]
    for u, v, data in graph.edges(data=True):
        data["weight"] = data["invocations"]
        data["alpha"] = scenario.alpha(u, v)

    prepare_memory_inputs(graph, scenario.metadata["workflow_invocations"])

    statuses = []
    original_solve = root_selector.solve_subgraph_construction

    def observed_solve(*args, **kwargs):
        try:
            result = original_solve(*args, **kwargs)
        except (AttributeError, gp.GurobiError) as error:
            # The original solver reads .X/.objVal at TIME_LIMIT even when no
            # incumbent exists. Never interpret that case as proven infeasible.
            if isinstance(error, gp.GurobiError) and error.errno != 10005:
                raise
            if isinstance(error, AttributeError) and "retrieve attribute" not in str(error):
                raise
            statuses.append(None)
            return gp.GRB.TIME_LIMIT, None, None
        statuses.append(result[0])
        return result

    root, nodes, predecessors, reach = preprocess_graph(graph)
    parameters = config.get("parameters", {})
    max_k = parameters.get("max_k", len(nodes))
    if type(max_k) is not int or not 1 <= max_k <= len(nodes):
        raise ValueError("max_k must be between 1 and the graph's node count")
    if strategy == "exact" and len(nodes) > parameters.get("max_nodes", 10):
        raise ValueError("quilt_exact defaults to at most 10 nodes; explicitly raise max_nodes if needed")
    selector = None
    selector_args = {}
    if strategy != "exact":
        selector = select_downstream_candidate_roots if strategy == "downstream" else select_weighted_degree_candidates
        selector_args = {"num_candidates": len(nodes) - 1, "rcl_size": parameters.get("rcl_size", 1)}
        if strategy == "downstream":
            selector_args.update(M=limits["memory"], C=limits["cpu"],
                                 beta=parameters.get("beta", 0.3), gamma=parameters.get("gamma", 0.35),
                                 delta=parameters.get("delta", 0.35))
    # Instrument only the imported function reference, and restore it afterward.
    root_selector.solve_subgraph_construction = observed_solve
    try:
        if strategy == "exact":
            cost, roots, assignment = _exact_roots(
                graph, root, nodes, predecessors, reach, max_k, limits, parameters, observed_solve)
        else:
            cost, roots, assignment, *rest = root_selector.run_root_selection_strategy(
                strategy, graph, limits["memory"], limits["cpu"], root, nodes, predecessors, reach,
                max_k=max_k, candidate_selector_fn=selector, selector_args=selector_args,
                strategy_mode="greedy", num_threads=1,
                ilp_time_limit=parameters.get("solve_time_limit_seconds", 5), ilp_mip_gap=0.0)
    finally:
        root_selector.solve_subgraph_construction = original_solve
    exhaustive = strategy == "exact" and max_k == len(nodes)
    proven = exhaustive and bool(statuses) and all(s in (gp.GRB.OPTIMAL, gp.GRB.INFEASIBLE) for s in statuses)
    status = ("optimal" if proven else "feasible") if assignment else ("infeasible" if proven else "unknown")
    return AlgorithmResult({r: sorted(members) for r, members in (assignment or {}).items()}, status,
                           {"solver_cost": cost, "subproblem_statuses": statuses,
                            "exhaustive_root_search": exhaustive,
                            "optimality_proven": bool(proven and assignment)})


def _exact_roots(graph, root, nodes, predecessors, reach, max_k, limits, parameters, solve):
    """Enumerate serially so every solver status participates in certification."""
    from itertools import combinations

    best_cost, best_roots, best_assignment = None, None, None
    candidates = sorted(set(nodes) - {root})
    for count in range(max_k):
        for chosen in combinations(candidates, count):
            roots = set(chosen) | {root}
            status, cost, assignment = solve(
                graph, roots, limits["memory"], limits["cpu"], nodes, predecessors, reach,
                time_limit=parameters.get("solve_time_limit_seconds", 5), num_threads=1)
            if assignment is not None and (best_cost is None or cost < best_cost):
                best_cost, best_roots, best_assignment = cost, roots, assignment
    return best_cost, best_roots, best_assignment


def execute(name, scenario, limits, config):
    if name == "no_merge":
        return no_merge(scenario, limits, config)
    if name == "greedy":
        return greedy(scenario, limits, config)
    if name in ("quilt_exact", "quilt_downstream", "quilt_weighted_degree"):
        strategy = {"quilt_exact": "exact", "quilt_downstream": "downstream",
                    "quilt_weighted_degree": "weighted"}[name]
        return quilt(scenario, limits, config, strategy)
    if ":" not in name:
        raise ValueError(f"unknown algorithm: {name}; custom algorithms use module:function")
    module, function = name.split(":", 1)
    result = getattr(importlib.import_module(module), function)(scenario, limits, config)
    if not isinstance(result, AlgorithmResult):
        raise TypeError("custom algorithm must return AlgorithmResult")
    # Custom algorithms cannot certify themselves as optimal to the benchmark.
    if result.status == "optimal":
        result.status = "feasible"
        result.diagnostics["uncertified_optimality_claim"] = True
    return result
