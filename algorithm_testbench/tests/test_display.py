"""Display semantics and node layout must not alter measured graph data."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from algorithm_testbench.display import DEFAULT_RESOURCES, edge_text, format_bytes, memory_unit_for
from algorithm_testbench.generate import generate
from algorithm_testbench.model import write_json
from algorithm_testbench.visualize import _label, _measure_labels, layered_positions, visualize


class DisplayTests(unittest.TestCase):
    def test_byte_units_and_boundaries(self):
        cases = [(0, "0 B"), (0.5, "0.5 B"), (100, "100 B"), (1023, "1023 B"),
                 (1024, "1 KiB"), (1536, "1.5 KiB"), (1024 ** 2, "1 MiB"),
                 (256 * 1024 ** 2, "256 MiB"), (1024 ** 3, "1 GiB"),
                 (1024 ** 4, "1 TiB"), (1024 ** 4 * 1.23456, "1.23 TiB"),
                 (1024 ** 2 - 1, "1 MiB")]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(format_bytes(value), expected)
        for value in (-1, float("inf"), True):
            with self.assertRaises(ValueError):
                format_bytes(value)

    def test_default_labels_group_storage_and_keep_cpu_unitless(self):
        scenario = generate("diamond", seed=42, resource_profile="extended")
        label = _label("f1", scenario.graph.nodes["f1"], DEFAULT_RESOURCES, ["requires_gpu"], False)
        self.assertIn("CPU: 3\n", label)
        self.assertNotIn("profile units", label)
        self.assertIn("Memory: 16 MiB", label)
        self.assertIn("Storage read: 22 ops/inv, 88 KiB/inv", label)
        self.assertIn("Storage write: 39 ops/inv, 156 KiB/inv", label)
        self.assertIn("Network RX: 473.75 KiB/inv", label)
        self.assertIn("Network TX: 244.03 KiB/inv", label)
        self.assertIn("GPU memory: 256 MiB", label)
        self.assertIn("GPU required: yes", label)
        self.assertNotIn("e+", label)

    def test_field_selection_missing_zero_and_hidden_labels(self):
        scenario = generate("chain", 2)
        data = deepcopy(scenario.graph.nodes["f0"])
        data["resources"]["storage_read_ops_per_invocation"] = 0
        data["resources"]["network_external_rx_bytes_per_invocation"] = 0
        label = _label("f0", data, DEFAULT_RESOURCES, ["requires_gpu"], False)
        self.assertIn("Storage read: 0 ops/inv", label)
        self.assertIn("Network RX: 0 B/inv", label)
        self.assertNotIn("Storage write", label)
        self.assertNotIn("GPU", label)
        self.assertEqual(_label("f0", data, ["memory"], [], False), f"f0\nMemory: {data['resources']['memory']} MiB")
        self.assertEqual(_label("f0", data, DEFAULT_RESOURCES, ["requires_gpu"], True), "f0")
        data["resources"]["storage_read_bytes_per_invocation"] = 1024
        label = _label("f0", data, ["storage_read_bytes_per_invocation"], [], False)
        self.assertEqual(label, "f0\nStorage read: 1 KiB/inv")

    def test_source_memory_units_and_precedence(self):
        scenario = generate("chain", 1)
        self.assertEqual(memory_unit_for(scenario), "MiB")
        scenario.metadata["units"] = {"memory": "KiB", "cpu": "ignored"}
        self.assertEqual(memory_unit_for(scenario), "KiB")
        self.assertEqual(memory_unit_for(scenario, result_units={"memory": "B"}), "B")
        self.assertEqual(memory_unit_for(scenario, "GiB", {"memory": "B"}), "GiB")
        data = {"resources": {"cpu": 2, "memory": 1024, "gpu_peak_memory_bytes": 1024 ** 3}}
        label = _label("node", data, DEFAULT_RESOURCES, [], False, memory_unit="KiB")
        self.assertIn("Memory: 1 MiB", label)
        self.assertIn("GPU memory: 1 GiB", label)
        label = _label("node", data, ["memory"], [], False, memory_unit="pages")
        self.assertIn("Memory: 1024 pages", label)
        with self.assertRaises(ValueError):
            memory_unit_for(scenario, "")

    def test_edge_labels_have_correct_denominators(self):
        self.assertEqual(edge_text("invocations", 200), "200 calls/window")
        self.assertEqual(edge_text("request_bytes", 1024), "1 KiB/call")
        self.assertEqual(edge_text("response_bytes", 1024 ** 3), "1 GiB/call")

    def test_measured_spacing_grows_with_labels(self):
        scenario = generate("diamond", seed=42, resource_profile="extended")
        compact = _measure_labels(["f0"])
        full = _measure_labels([_label(n, data, DEFAULT_RESOURCES, ["requires_gpu"], False)
                                for n, data in scenario.graph.nodes(data=True)])
        self.assertGreater(full[0], compact[0])
        self.assertGreater(full[1], compact[1])
        for orientation in ("vertical", "horizontal"):
            positions = layered_positions(scenario.graph, orientation, full)
            for first, a in positions.items():
                for second, b in positions.items():
                    if first < second:
                        self.assertTrue(abs(a[0] - b[0]) >= full[0] or abs(a[1] - b[1]) >= full[1])

    def test_default_svg_and_group_panels_preserve_graph(self):
        scenario = generate("diamond", seed=42, resource_profile="extended")
        original = scenario.to_dict()
        with tempfile.TemporaryDirectory() as directory:
            for orientation in ("vertical", "horizontal"):
                path = Path(directory) / f"{orientation}.svg"
                visualize(scenario, path, orientation=orientation)
                svg = path.read_text()
                for expected in ("Storage read:", "Storage write:", "Network RX:", "Network TX:",
                                 "GPU memory: 1 GiB", "CPU: 3", "100 calls/window", "workflow invocations"):
                    self.assertIn(expected, svg)
                self.assertNotIn("profile units", svg)
            path = Path(directory) / "groups.svg"
            visualize(scenario, path, groups={"f0": ["f0", "f1"], "f2": ["f2", "f3"]})
            self.assertIn("Group rooted at f0", path.read_text())
            self.assertIn("Storage read:", path.read_text())
        self.assertEqual(original, scenario.to_dict())

    def test_cli_units_result_units_and_field_controls(self):
        scenario = generate("chain", 1, seed=42)
        node = scenario.entry
        scenario.graph.nodes[node]["resources"]["memory"] = 1024
        scenario.metadata["units"] = {"memory": "B"}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            graph = directory / "graph.json"
            scenario.save(graph)
            result = directory / "results.json"
            write_json(result, {"graph": scenario.to_dict(), "experiment": {"units": {"memory": "KiB"}},
                                "runs": [{"algorithm": "no_merge", "seed": 0, "groups": {node: [node]},
                                          "evaluation": {"valid": True}}]})
            svg = directory / "image.svg"
            command = [sys.executable, "-m", "algorithm_testbench.visualize", str(graph), "--output", str(svg)]
            subprocess.run(command, check=True, capture_output=True, timeout=15)
            self.assertIn("Memory: 1 KiB", svg.read_text())
            subprocess.run(command + ["--result", str(result)], check=True, capture_output=True, timeout=15)
            self.assertIn("Memory: 1 MiB", svg.read_text())
            subprocess.run(command + ["--result", str(result), "--memory-unit", "MiB"],
                           check=True, capture_output=True, timeout=15)
            self.assertIn("Memory: 1 GiB", svg.read_text())
            subprocess.run(command + ["--resources", "cpu", "--node-attributes", ""],
                           check=True, capture_output=True, timeout=15)
            self.assertNotIn("Memory:", svg.read_text())
            subprocess.run(command + ["--hide-labels"], check=True, capture_output=True, timeout=15)
            self.assertNotIn("CPU:", svg.read_text())


if __name__ == "__main__":
    unittest.main()
