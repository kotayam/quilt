"""Direct solver-hook checks, separate from the original Quilt tests."""

import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from merge_solver.memory_inputs import prepare_memory_inputs
from algorithm_testbench.model import Scenario
from merge_solver.memory_model_template import (
    build_memory_expression,
    estimate_memory,
)

try:
    import gurobipy as gp
    from merge_solver.ilp import solve_subgraph_construction
except ImportError:
    gp = None


class MemoryTemplateTests(unittest.TestCase):
    def test_unfinished_functions_fail_explicitly(self):
        with self.assertRaises(NotImplementedError):
            estimate_memory(nx.DiGraph(), "a", {"a"})
        with self.assertRaises(NotImplementedError):
            build_memory_expression(None, nx.DiGraph(), "a", {"a"}, {}, {})


class MemoryInputsTests(unittest.TestCase):
    def test_counterexample_and_alpha_preservation(self):
        scenario = Scenario.load(Path(__file__).parents[1] / "examples/graphs/counterexample.json")
        graph = scenario.graph.copy()
        for u, v, data in graph.edges(data=True):
            data.update(weight=data["invocations"], alpha=scenario.alpha(u, v))
        alpha = nx.get_edge_attributes(graph, "alpha")
        self.assertIs(prepare_memory_inputs(graph, 10), graph)
        self.assertEqual(list(nx.get_edge_attributes(graph, "calls_per_caller").values()), [10, 2, 3])
        self.assertEqual(nx.get_node_attributes(graph, "profile_invocations"),
                         {"a": 10, "b": 100, "c": 200, "d": 300})
        self.assertEqual(nx.get_edge_attributes(graph, "alpha"), alpha)
        self.assertNotIn("profile_invocations", scenario.graph.nodes["a"])

    def test_fractional_shared_caller_and_zero_counts(self):
        graph = nx.DiGraph()
        graph.add_weighted_edges_from([
            ("a", "b", 2), ("a", "c", 4), ("b", "d", 1),
            ("c", "d", 2), ("d", "e", 1), ("a", "z", 0), ("z", "e", 0)])
        prepare_memory_inputs(graph, 3)
        self.assertEqual(graph.edges["a", "b"]["calls_per_caller"], 2 / 3)
        self.assertEqual(graph.nodes["d"]["profile_invocations"], 3)
        self.assertEqual(graph.edges["d", "e"]["calls_per_caller"], 1 / 3)
        self.assertEqual(graph.edges["a", "z"]["calls_per_caller"], 0)
        self.assertIsNone(graph.edges["z", "e"]["calls_per_caller"])

    def test_inconsistent_counts_are_rejected_before_mutation(self):
        for count, weights in [(0, [1, 0]), (1, [0, 1])]:
            graph = nx.DiGraph()
            graph.add_weighted_edges_from([("a", "b", weights[0]), ("b", "c", weights[1])])
            with self.assertRaisesRegex(ValueError, "uninvoked caller"):
                prepare_memory_inputs(graph, count)
            self.assertEqual(nx.get_node_attributes(graph, "profile_invocations"), {})
        graph = nx.DiGraph()
        graph.add_edge("a", "b", weight=0)
        prepare_memory_inputs(graph, 0)
        self.assertIsNone(graph.edges["a", "b"]["calls_per_caller"])

    def test_invalid_inputs(self):
        for value in (-1, float("nan"), float("inf"), None, True):
            graph = nx.DiGraph()
            graph.add_edge("a", "b", weight=1)
            with self.assertRaises(ValueError):
                prepare_memory_inputs(graph, value)
            graph.edges["a", "b"]["weight"] = value
            with self.assertRaises(ValueError):
                prepare_memory_inputs(graph, 1)
        for graph in (nx.DiGraph(), nx.DiGraph([(1, 2), (2, 1)]),
                      nx.DiGraph([(1, 3), (2, 3)])):
            with self.assertRaises(ValueError):
                prepare_memory_inputs(graph, 1)


@unittest.skipIf(gp is None, "optional gurobipy is not installed")
class MemoryHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            with gp.Env(empty=True) as env:
                env.setParam("LogToConsole", 0)
                env.start()
        except gp.GurobiError as error:
            raise unittest.SkipTest(f"Gurobi license unavailable: {error}")

    def setUp(self):
        self.graph = nx.DiGraph()
        self.graph.add_node("a", m=10, c=1)
        self.graph.add_node("b", m=20, c=1)
        self.graph.add_edge("a", "b", weight=100, alpha=10, type="async")

    def solve(self, memory, **kwargs):
        return solve_subgraph_construction(
            graph=self.graph, R_set=kwargs.pop("roots", {"a"}),
            M=memory, C=kwargs.pop("cpu", 100), all_nodes=list(self.graph),
            predecessors={n: list(self.graph.predecessors(n)) for n in self.graph},
            full_reachable_from={n: nx.descendants(self.graph, n) for n in self.graph},
            **kwargs,
        )

    def test_default_formula_is_preserved(self):
        status, cost, groups = self.solve(210)
        self.assertEqual(status, gp.GRB.OPTIMAL)
        self.assertEqual(cost, 0)
        self.assertEqual(set(groups["a"]), {"a", "b"})
        self.assertEqual(self.solve(40)[0], gp.GRB.INFEASIBLE)

    def test_builder_receives_variables_for_each_root_and_can_add_constraints(self):
        seen = []
        prepare_memory_inputs(self.graph, 10)

        def builder(*, model, graph, root, roots, y, z):
            self.assertIs(graph, self.graph)
            self.assertEqual(graph.nodes["a"]["profile_invocations"], 10)
            self.assertEqual(graph.nodes["b"]["profile_invocations"], 100)
            self.assertEqual(graph.edges["a", "b"]["calls_per_caller"], 10)
            self.assertEqual(roots, {"a", "b"})
            self.assertIsInstance(y["a", root], gp.Var)
            self.assertIsInstance(z["a", "b", root], gp.Var)
            seen.append(root)
            memory = model.addVar(lb=0, name=f"test_memory_{root}")
            model.addConstr(memory == gp.quicksum(
                graph.nodes[n]["m"] * y[n, root] for n in graph))
            return memory

        status, cost, _ = self.solve(40, roots={"a", "b"},
                                    memory_expression_builder=builder)
        self.assertEqual(status, gp.GRB.OPTIMAL)
        self.assertEqual(cost, 0)
        self.assertCountEqual(seen, ["a", "b"])
        self.assertEqual(self.graph.edges["a", "b"]["alpha"], 10)

    def test_adapter_prepares_private_graph_before_root_search(self):
        from algorithm_testbench.algorithms import quilt

        scenario = Scenario.load(Path(__file__).parents[1] / "examples/graphs/counterexample.json")
        original = scenario.to_dict()

        def search(graph, *args):
            self.assertIsNot(graph, scenario.graph)
            self.assertEqual(graph.edges["b", "c"]["calls_per_caller"], 2)
            self.assertEqual(graph.edges["b", "d"]["calls_per_caller"], 3)
            self.assertEqual(graph.edges["b", "c"]["alpha"], 20)
            return None, None, None

        with patch("algorithm_testbench.algorithms._exact_roots", side_effect=search):
            quilt(scenario, {"cpu": 100, "memory": 1000},
                  {"resource_model": "quilt"}, "exact")
        self.assertEqual(scenario.to_dict(), original)

    def test_custom_expression_controls_memory_but_not_cpu(self):
        def builder(*, model, graph, root, roots, y, z):
            return gp.quicksum(graph.nodes[n]["m"] * y[n, root] for n in graph)

        self.assertEqual(self.solve(40, memory_expression_builder=builder)[0], gp.GRB.OPTIMAL)
        self.assertEqual(self.solve(25, memory_expression_builder=builder)[0], gp.GRB.INFEASIBLE)
        self.assertEqual(self.solve(40, cpu=5, memory_expression_builder=builder)[0], gp.GRB.INFEASIBLE)

    def test_selecting_unfinished_builder_fails(self):
        with self.assertRaisesRegex(NotImplementedError, "symbolic"):
            self.solve(40, memory_expression_builder=build_memory_expression)
