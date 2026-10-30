"""Direct solver-hook checks, separate from the original Quilt tests."""

import unittest

import networkx as nx

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

        def builder(*, model, graph, root, roots, y, z):
            self.assertIs(graph, self.graph)
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

    def test_custom_expression_controls_memory_but_not_cpu(self):
        def builder(*, model, graph, root, roots, y, z):
            return gp.quicksum(graph.nodes[n]["m"] * y[n, root] for n in graph)

        self.assertEqual(self.solve(40, memory_expression_builder=builder)[0], gp.GRB.OPTIMAL)
        self.assertEqual(self.solve(25, memory_expression_builder=builder)[0], gp.GRB.INFEASIBLE)
        self.assertEqual(self.solve(40, cpu=5, memory_expression_builder=builder)[0], gp.GRB.INFEASIBLE)

    def test_selecting_unfinished_builder_fails(self):
        with self.assertRaisesRegex(NotImplementedError, "symbolic"):
            self.solve(40, memory_expression_builder=build_memory_expression)
