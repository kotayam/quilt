from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from algorithm_testbench.algorithms import execute
from algorithm_testbench.evaluate import evaluate
from algorithm_testbench.generate import generate
from algorithm_testbench.model import Scenario, write_json
from algorithm_testbench.run import load_config, run_experiment, run_isolated


def fixture(nodes=None, edges=None, invocations=100):
    nodes = nodes or {"A": (1, 10), "B": (1, 10), "C": (1, 10)}
    edges = edges if edges is not None else [("A", "B", 100, "sync"), ("B", "C", 100, "sync")]
    return Scenario.from_dict({
        "schema_version": 1, "name": "fixture", "entry": "A", "workflow_invocations": invocations,
        "nodes": [{"id": name, "resources": {"cpu": cpu, "memory": mem}}
                  for name, (cpu, mem) in nodes.items()],
        "edges": [{"source": source, "target": target, "invocations": calls, "type": kind}
                  for source, target, calls, kind in edges],
    })


class GraphTests(unittest.TestCase):
    def test_round_trip_preserves_future_fields(self):
        document = fixture().to_dict()
        document["future_metadata"] = {"version": 7}
        document["nodes"][0]["resources"]["gpu_memory"] = 1024
        document["nodes"][0]["attributes"] = {"device": "gpu-family"}
        document["edges"][0]["attributes"] = {"request_bytes": 4096}
        document["edges"][0]["future"] = [1, 2]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "graph.json"
            Scenario.from_dict(document).save(path)
            self.assertEqual(Scenario.load(path).to_dict(), document)

    def test_seeded_generation_and_shapes(self):
        for shape in ("chain", "tree", "diamond", "fanout", "random"):
            with self.subTest(shape=shape):
                left, right = generate(shape, 8, 42), generate(shape, 8, 42)
                self.assertEqual(left.to_dict(), right.to_dict())
                self.assertEqual(len(left.graph), 4 if shape == "diamond" else 8)

    def test_reject_invalid_graphs(self):
        base = fixture().to_dict()
        mutations = [
            lambda d: d.update(schema_version=2),
            lambda d: d.update(workflow_invocations=0),
            lambda d: d.update(entry="missing"),
            lambda d: d["nodes"].append(deepcopy(d["nodes"][0])),
            lambda d: d["nodes"][0]["resources"].update(cpu=float("nan")),
            lambda d: d["nodes"][0]["resources"].update(cpu=True),
            lambda d: d["edges"][0].update(target="missing"),
            lambda d: d["edges"][0].update(type="unknown"),
            lambda d: d["edges"][0].update(invocations=-1),
            lambda d: d["edges"].append(deepcopy(d["edges"][0])),
            lambda d: d["edges"].append({"source": "C", "target": "A", "invocations": 1, "type": "sync"}),
            lambda d: d["edges"].clear(),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                document = deepcopy(base)
                mutation(document)
                with self.assertRaises(ValueError):
                    Scenario.from_dict(document)

    def test_alpha_uses_counts_not_future_cost(self):
        scenario = fixture(edges=[("A", "B", 201, "async"), ("B", "C", 20, "sync")])
        scenario.graph.edges["A", "B"]["attributes"]["cost"] = 99999
        self.assertEqual(scenario.alpha("A", "B"), 3)
        self.assertEqual(scenario.alpha("B", "C"), 1)


class EvaluationTests(unittest.TestCase):
    def test_hand_calculated_chain_and_capacity(self):
        scenario = fixture(edges=[("A", "B", 100, "sync"), ("B", "C", 200, "sync")])
        groups = {"A": ["A"], "B": ["B", "C"]}
        report = evaluate(scenario, groups, {"cpu": 4, "memory": 25})
        self.assertTrue(report["valid"])
        self.assertEqual(report["remote_call_cost"], 100)
        self.assertAlmostEqual(report["reduction_from_baseline"], 2 / 3)
        self.assertEqual(report["resources"]["B"]["usage"], {"cpu": 3, "memory": 20})
        invalid = evaluate(scenario, groups, {"cpu": 2, "memory": 25})
        self.assertFalse(invalid["valid"])
        self.assertNotIn("remote_call_cost", invalid)

    def test_overlapping_groups_and_per_root_cut_semantics(self):
        scenario = fixture(nodes={n: (1, 1) for n in "ABCD"}, edges=[
            ("A", "B", 10, "async"), ("A", "C", 20, "async"),
            ("B", "D", 30, "sync"), ("C", "D", 40, "sync")])
        report = evaluate(scenario, {"A": ["A", "B", "D"], "C": ["C", "D"]}, {"cpu": 10, "memory": 10})
        self.assertTrue(report["valid"])
        self.assertEqual(report["remote_call_cost"], 20)
        self.assertEqual(report["duplicated_placements"], 1)
        # B-C is cut in A's group despite both endpoints being together in B's.
        chain = fixture()
        report = evaluate(chain, {"A": ["A", "B"], "B": ["B", "C"]}, {"cpu": 10, "memory": 100})
        self.assertFalse(report["valid"])  # B-C leaves A's group, but C is not a root.
        report = evaluate(chain, {"A": ["A", "B"], "B": ["B", "C"], "C": ["C"]},
                          {"cpu": 10, "memory": 100})
        self.assertTrue(report["valid"])
        self.assertEqual(report["remote_call_cost"], 100)

    def test_invalid_assignments(self):
        for groups in ({}, {"A": ["A"]}, {"A": ["A", "C"], "B": ["B"]},
                       {"A": ["A", "B"], "C": ["C", "unknown"]},
                       {"A": ["A", "A", "B", "C"]}, {"A": ["B", "C"]}):
            with self.subTest(groups=groups):
                self.assertFalse(evaluate(fixture(), groups, {"cpu": 10, "memory": 100})["valid"])

    def test_quilt_inherited_concurrency_limitation_is_explicit(self):
        scenario = fixture(edges=[("A", "B", 400, "async"), ("B", "C", 400, "sync")])
        report = evaluate(scenario, {"A": ["A", "B", "C"]}, {"cpu": 10, "memory": 100})
        self.assertEqual(report["resources"]["A"]["usage"], {"cpu": 9, "memory": 60})
        # Four concurrent B->C instances could instead reach 10 + 4*10 + 4*10 = 90.
        self.assertEqual(report["resource_model"], "quilt")

    def test_unsupported_constraints_fail_but_unused_resources_survive(self):
        scenario = fixture()
        scenario.graph.nodes["A"]["resources"]["gpu_memory"] = 1024
        self.assertTrue(evaluate(scenario, {"A": ["A", "B", "C"]}, {"cpu": 10, "memory": 100})["valid"])
        with self.assertRaises(ValueError):
            evaluate(scenario, {"A": ["A", "B", "C"]}, {"cpu": 10, "memory": 100, "gpu_memory": 100})

    def test_zero_cost_graph(self):
        scenario = fixture(nodes={"A": (1, 1)}, edges=[])
        report = evaluate(scenario, {"A": ["A"]}, {"cpu": 1, "memory": 1})
        self.assertEqual(report["remote_call_cost"], 0)
        self.assertIsNone(report["reduction_from_baseline"])
        self.assertEqual(report["absolute_savings"], 0)


class RunnerTests(unittest.TestCase):
    def test_greedy_improves_cost_and_respects_limits(self):
        scenario = fixture()
        result = execute("greedy", scenario, {"cpu": 2, "memory": 20}, {"resource_model": "quilt"})
        report = evaluate(scenario, result.groups, {"cpu": 2, "memory": 20})
        self.assertTrue(report["valid"])
        self.assertEqual(report["remote_call_cost"], 100)

    def test_timeout_and_input_isolation(self):
        scenario = fixture()
        config = {"resource_model": "quilt", "parameters": {}}
        limits = {"cpu": 2, "memory": 20}
        before = scenario.to_dict()
        result = run_isolated("algorithm_testbench.tests.plugins:mutate", scenario, limits, config, timeout_seconds=10)
        self.assertEqual(result["status"], "feasible")
        self.assertFalse(evaluate(scenario, result["groups"], limits)["valid"])
        self.assertEqual(scenario.to_dict(), before)
        self.assertEqual(limits, {"cpu": 2, "memory": 20})
        slow = run_isolated("algorithm_testbench.tests.plugins:slow", scenario, limits, config, timeout_seconds=0.2)
        self.assertEqual(slow["status"], "timeout")

    def test_plugins_cannot_claim_certified_optimality(self):
        result = execute("algorithm_testbench.tests.plugins:false_optimum", fixture(), {}, {})
        self.assertEqual(result.status, "feasible")
        self.assertTrue(result.diagnostics["uncertified_optimality_claim"])

    def test_complete_file_based_run(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fixture().save(directory / "graphs" / "fixture.json")
            config = {"schema_version": 1, "graph": "graphs/fixture.json", "resource_model": "quilt",
                      "limits": {"cpu": 2, "memory": 20}, "units": {"cpu": "units", "memory": "MiB"},
                      "algorithms": [{"name": "no_merge"}, {"name": "greedy"}], "seeds": [0, 7]}
            write_json(directory / "experiment.json", config)
            result = run_experiment(directory / "experiment.json", directory / "output")
            self.assertEqual(len(result["runs"]), 4)
            self.assertTrue(all(run["evaluation"]["valid"] for run in result["runs"]))
            self.assertEqual(json.loads((directory / "output/results.json").read_text()), result)
            self.assertEqual(len((directory / "output/summary.csv").read_text().splitlines()), 5)
            config["limits"]["gpu"] = 1
            write_json(directory / "experiment.json", config)
            with self.assertRaises(ValueError):
                load_config(directory / "experiment.json")


class VisualizationTests(unittest.TestCase):
    def test_exports_and_deterministic_layout(self):
        from algorithm_testbench.visualize import layered_positions, visualize

        scenario = fixture()
        scenario.graph.nodes["A"]["resources"]["gpu_memory"] = 1024
        scenario.graph.nodes["B"]["attributes"]["label"] = "a long descriptive label " * 3
        self.assertEqual(layered_positions(scenario.graph), layered_positions(scenario.graph.copy()))
        with tempfile.TemporaryDirectory() as directory:
            paths = visualize(scenario, Path(directory) / "graph", resources=["memory", "gpu_memory"],
                              node_attributes=["label"])
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(p.stat().st_size > 100 for p in paths))
            svg = paths[0].read_text()
            self.assertIn("gpu_memory", svg)
            group_paths = visualize(scenario, Path(directory) / "groups.svg", orientation="horizontal",
                                    groups={"A": ["A", "B"], "B": ["B", "C"]})
            self.assertIn("Group rooted at A", group_paths[0].read_text())
            self.assertIn("Group rooted at B", group_paths[0].read_text())


@unittest.skipUnless(importlib.util.find_spec("gurobipy"), "optional gurobipy is not installed")
class QuiltAdapterTests(unittest.TestCase):
    def test_original_adapters(self):
        for algorithm in ("quilt_exact", "quilt_downstream", "quilt_weighted_degree"):
            with self.subTest(algorithm=algorithm):
                result = run_isolated(algorithm, fixture(), {"cpu": 2, "memory": 20},
                                      {"resource_model": "quilt", "parameters": {}}, timeout_seconds=20)
                self.assertIn(result["status"], ("optimal", "feasible"), result)
                report = evaluate(fixture(), result["groups"], {"cpu": 2, "memory": 20})
                self.assertTrue(report["valid"], report)
                self.assertEqual(report["remote_call_cost"], 100)
                self.assertEqual(result["diagnostics"]["optimality_proven"], algorithm == "quilt_exact")

    def test_partial_root_search_and_no_incumbent_are_not_certified(self):
        partial = run_isolated("quilt_exact", fixture(), {"cpu": 10, "memory": 100},
                               {"resource_model": "quilt", "parameters": {"max_k": 1}}, timeout_seconds=20)
        self.assertEqual(partial["status"], "feasible", partial)
        self.assertFalse(partial["diagnostics"]["optimality_proven"])
        timed = run_isolated("quilt_exact", fixture(), {"cpu": 2, "memory": 20},
                             {"resource_model": "quilt", "parameters": {"solve_time_limit_seconds": 0}},
                             timeout_seconds=20)
        self.assertEqual(timed["status"], "unknown", timed)
        self.assertFalse(timed["diagnostics"]["optimality_proven"])

    def test_zero_optimum_reports_absolute_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fixture(nodes={"A": (1, 1)}, edges=[]).save(directory / "graph.json")
            write_json(directory / "experiment.json", {
                "schema_version": 1, "graph": "graph.json", "resource_model": "quilt",
                "limits": {"cpu": 1, "memory": 1}, "units": {"cpu": "units", "memory": "MiB"},
                "algorithms": [{"name": "no_merge"}, {"name": "quilt_exact"}],
            })
            results = run_experiment(directory / "experiment.json", directory / "out")
            for run in results["runs"]:
                self.assertEqual(run["evaluation"]["absolute_gap"], 0)
                self.assertIsNone(run["evaluation"]["optimality_gap"])


if __name__ == "__main__":
    unittest.main()
