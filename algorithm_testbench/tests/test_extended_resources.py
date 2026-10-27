"""Optional resource data must not change Quilt's optimization problem."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from algorithm_testbench.evaluate import evaluate
from algorithm_testbench.generate import generate
from algorithm_testbench.model import Scenario
from algorithm_testbench.run import run_isolated


class ExtendedResourceTests(unittest.TestCase):
    def test_default_generation_matches_pre_extension_snapshot(self):
        document = generate("random", 10, 42).to_dict()
        fingerprint = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()
        self.assertEqual(fingerprint, "746dbd6f05d3a282ba70159dc538b01f2922b1c0813cccd11d24efc9e5f14493")
        self.assertEqual(document, generate("random", 10, 42, resource_profile="basic").to_dict())

    def test_extended_generation_is_repeatable_and_preserves_original_measurements(self):
        for shape in ("chain", "fanout", "tree", "diamond", "random"):
            with self.subTest(shape=shape):
                basic = generate(shape, 10, 42).to_dict()
                extended = generate(shape, 10, 42, resource_profile="extended").to_dict()
                self.assertEqual(extended, generate(shape, 10, 42, resource_profile="extended").to_dict())
                projected = deepcopy(extended)
                projected["generator"].pop("resource_profile")
                for original, enriched in zip(basic["nodes"], projected["nodes"]):
                    self.assertEqual(original["id"], enriched["id"])
                    self.assertEqual(original["resources"]["cpu"], enriched["resources"]["cpu"])
                    self.assertEqual(original["resources"]["memory"], enriched["resources"]["memory"])
                    enriched["resources"] = deepcopy(original["resources"])
                    enriched["attributes"] = deepcopy(original["attributes"])
                for original, enriched in zip(basic["edges"], projected["edges"]):
                    enriched["attributes"] = deepcopy(original["attributes"])
                self.assertEqual(basic, projected)

    def test_extended_values_and_round_trip(self):
        scenario = generate("random", 30, 42, resource_profile="extended")
        for _, node in scenario.graph.nodes(data=True):
            resources = node["resources"]
            for direction in ("read", "write"):
                operations = resources[f"storage_{direction}_ops_per_invocation"]
                self.assertTrue(0 <= operations <= 128)
                self.assertEqual(resources[f"storage_{direction}_bytes_per_invocation"], operations * 4096)
            for direction in ("rx", "tx"):
                self.assertTrue(0 <= resources[f"network_external_{direction}_bytes_per_invocation"] <= 1024 ** 2)
            self.assertIsInstance(node["attributes"]["requires_gpu"], bool)
            if node["attributes"]["requires_gpu"]:
                self.assertIn(resources["gpu_peak_memory_bytes"], [256 * 1024 ** 2, 512 * 1024 ** 2, 1024 ** 3])
            else:
                self.assertNotIn("gpu_peak_memory_bytes", resources)
        for _, _, edge in scenario.graph.edges(data=True):
            self.assertTrue(0 <= edge["attributes"]["request_bytes"] <= 1024 ** 2)
            self.assertTrue(0 <= edge["attributes"]["response_bytes"] <= 1024 ** 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            scenario.save(path)
            self.assertEqual(Scenario.load(path).to_dict(), scenario.to_dict())

    def test_requires_gpu_is_optional_but_must_be_boolean(self):
        base = generate("chain", 2).to_dict()
        for value in (True, False):
            document = deepcopy(base)
            document["nodes"][0]["attributes"]["requires_gpu"] = value
            self.assertIs(Scenario.from_dict(document).graph.nodes["f0"]["attributes"]["requires_gpu"], value)
        for value in (0, 1, "true", None, [], {}):
            with self.subTest(value=value):
                document = deepcopy(base)
                document["nodes"][0]["attributes"]["requires_gpu"] = value
                with self.assertRaisesRegex(ValueError, "requires_gpu must be a boolean"):
                    Scenario.from_dict(document)
        document = deepcopy(base)
        document["nodes"][0]["resources"]["requires_gpu"] = True
        with self.assertRaises(ValueError):
            Scenario.from_dict(document)

    def test_payload_fields_and_extended_resources_reject_invalid_values(self):
        base = generate("chain", 2, resource_profile="extended").to_dict()
        for value in (-1, float("inf"), True, "4096"):
            with self.subTest(value=value):
                for key in ("request_bytes", "response_bytes"):
                    document = deepcopy(base)
                    document["edges"][0]["attributes"][key] = value
                    with self.assertRaises(ValueError):
                        Scenario.from_dict(document)
                document = deepcopy(base)
                document["nodes"][0]["resources"]["storage_read_ops_per_invocation"] = value
                with self.assertRaises(ValueError):
                    Scenario.from_dict(document)

    def test_missing_is_distinct_from_zero(self):
        document = generate("chain", 2).to_dict()
        document["nodes"][0]["resources"]["storage_read_ops_per_invocation"] = 0
        document["nodes"][0]["attributes"]["requires_gpu"] = True
        scenario = Scenario.from_dict(document)
        self.assertEqual(scenario.graph.nodes["f0"]["resources"]["storage_read_ops_per_invocation"], 0)
        self.assertNotIn("storage_read_ops_per_invocation", scenario.graph.nodes["f1"]["resources"])
        self.assertNotIn("gpu_peak_memory_bytes", scenario.graph.nodes["f0"]["resources"])

    def test_quilt_evaluation_ignores_new_fields_but_rejects_new_limits(self):
        basic = generate("diamond", seed=42)
        extended = generate("diamond", seed=42, resource_profile="extended")
        groups = {node: [node] for node in basic.graph}
        limits = {"cpu": 10, "memory": 128}
        self.assertEqual(evaluate(basic, groups, limits), evaluate(extended, groups, limits))
        for key in ("storage_read_ops_per_invocation", "network_external_rx_bytes_per_invocation", "gpu_peak_memory_bytes"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    evaluate(extended, groups, {**limits, key: 100})

    def test_cli_and_checked_in_example(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extended.json"
            subprocess.run([sys.executable, "-m", "algorithm_testbench.generate", "--shape", "diamond",
                            "--seed", "42", "--resource-profile", "extended", "--output", str(path)],
                           check=True, capture_output=True, text=True, timeout=10)
            self.assertEqual(Scenario.load(path).to_dict(),
                             generate("diamond", seed=42, resource_profile="extended").to_dict())
        example = Path(__file__).parents[1] / "examples/graphs/diamond_extended.json"
        self.assertEqual(Scenario.load(example).to_dict(), generate("diamond", seed=42, resource_profile="extended").to_dict())
        with self.assertRaises(ValueError):
            generate(resource_profile="unknown")

    def test_visualization_displays_gpu_flag_and_optional_fields(self):
        from algorithm_testbench.visualize import visualize

        scenario = generate("diamond", seed=42, resource_profile="extended")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extended.svg"
            visualize(scenario, path, resources=["memory", "gpu_peak_memory_bytes"],
                      node_attributes=["requires_gpu"], edge_attribute="request_bytes")
            svg = path.read_text()
            self.assertIn("GPU memory: 256 MiB", svg)
            self.assertIn("GPU required: yes", svg)
            self.assertIn("GPU required: no", svg)


@unittest.skipUnless(importlib.util.find_spec("gurobipy"), "optional gurobipy is not installed")
class ExtendedQuiltCompatibilityTests(unittest.TestCase):
    def test_original_quilt_results_are_identical_with_extended_data(self):
        basic = generate("diamond", seed=42)
        extended = generate("diamond", seed=42, resource_profile="extended")
        config = {"resource_model": "quilt", "parameters": {}}
        limits = {"cpu": 10, "memory": 128}
        for algorithm in ("quilt_exact", "quilt_downstream", "quilt_weighted_degree"):
            with self.subTest(algorithm=algorithm):
                original = run_isolated(algorithm, basic, limits, config, seed=42, timeout_seconds=20)
                enriched = run_isolated(algorithm, extended, limits, config, seed=42, timeout_seconds=20)
                self.assertIn(original["status"], ("optimal", "feasible"), original)
                self.assertEqual(original["status"], enriched["status"])
                self.assertEqual(original["groups"], enriched["groups"])
                original_report = evaluate(basic, original["groups"], limits)
                self.assertTrue(original_report["valid"])
                self.assertEqual(original_report, evaluate(extended, enriched["groups"], limits))
                self.assertEqual(original["diagnostics"]["solver_cost"], enriched["diagnostics"]["solver_cost"])


if __name__ == "__main__":
    unittest.main()
