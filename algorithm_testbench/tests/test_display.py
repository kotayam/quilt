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
        label = _label("f1", scenario.graph.nodes["f1"], DEFAULT_RESOURCES, [], False)
        self.assertIn("CPU: 3\n", label)
        self.assertNotIn("profile units", label)
        self.assertIn("Memory [MiB]: 16", label)
        self.assertIn("Storage R/W [KiB]: 88 / 156", label)
        self.assertIn("Network RX/TX [KiB]: 473.75 / 244.03", label)
        self.assertIn("GPU memory [MiB]: 256", label)
        self.assertNotIn("GPU required", label)
        self.assertNotIn("ops", label)
        self.assertNotIn("/inv", label)
        self.assertNotIn("e+", label)

    def test_field_selection_missing_zero_and_hidden_labels(self):
        scenario = generate("chain", 2)
        data = deepcopy(scenario.graph.nodes["f0"])
        data["resources"]["storage_read_ops_per_invocation"] = 0
        data["resources"]["network_external_rx_bytes_per_invocation"] = 0
        label = _label("f0", data, DEFAULT_RESOURCES, ["requires_gpu"], False)
        self.assertNotIn("Storage", label)  # Operation counts alone are hidden by default.
        self.assertNotIn("Network", label)
        self.assertNotIn("GPU", label)
        self.assertEqual(_label("f0", data, ["memory"], [], False), f"f0\nMemory [MiB]: {data['resources']['memory']}")
        self.assertEqual(_label("f0", data, DEFAULT_RESOURCES, ["requires_gpu"], True), "f0")
        data["resources"]["storage_read_bytes_per_invocation"] = 1024
        label = _label("f0", data, ["storage_read_bytes_per_invocation"], [], False)
        self.assertEqual(label, "f0\nStorage R/W [KiB]: 1 / —")

    def test_paired_units_missing_values_and_manual_operations(self):
        data = {"resources": {"storage_read_bytes_per_invocation": 1024,
                              "storage_write_bytes_per_invocation": 1024 ** 2,
                              "network_external_rx_bytes_per_invocation": 0,
                              "network_external_tx_bytes_per_invocation": 0,
                              "storage_read_ops_per_invocation": 22,
                              "storage_write_ops_per_invocation": 39}}
        label = _label("n", data, DEFAULT_RESOURCES, [], False)
        self.assertIn("Storage R/W [MiB]: <0.01 / 1", label)
        self.assertNotIn("Network", label)
        label = _label("n", data, ["storage_write_bytes_per_invocation"], [], False)
        self.assertEqual(label, "n\nStorage R/W [MiB]: — / 1")
        label = _label("n", data, ["storage_read_ops_per_invocation", "storage_write_ops_per_invocation"], [], False)
        self.assertEqual(label, "n\nStorage ops R/W [ops]: 22 / 39")
        self.assertEqual(_label("n", {"resources": {}}, DEFAULT_RESOURCES, [], False), "n")

    def test_zero_resources_are_omitted_without_changing_data(self):
        data = {"resources": {key: 0 for key in DEFAULT_RESOURCES}}
        original = deepcopy(data)
        self.assertEqual(_label("n", data, DEFAULT_RESOURCES, [], False), "n")
        self.assertEqual(data, original)
        data["resources"]["network_external_tx_bytes_per_invocation"] = 1024 ** 2
        data["resources"]["storage_read_bytes_per_invocation"] = 1024
        label = _label("n", data, DEFAULT_RESOURCES, [], False)
        self.assertEqual(label, "n\nStorage R [KiB]: 1\nNetwork TX [MiB]: 1")
        # Omitting the other member is distinct from knowing it is zero.
        del data["resources"]["network_external_rx_bytes_per_invocation"]
        self.assertIn("Network RX/TX [MiB]: — / 1", _label("n", data, DEFAULT_RESOURCES, [], False))
        zero_ops = {"resources": {"storage_read_ops_per_invocation": 0,
                                  "storage_write_ops_per_invocation": 5}}
        selected = ["storage_read_ops_per_invocation", "storage_write_ops_per_invocation"]
        self.assertEqual(_label("n", zero_ops, selected, [], False), "n\nStorage ops W [ops]: 5")
        self.assertEqual(edge_text("invocations", 0), "0 calls/window")

    def test_source_memory_units_and_precedence(self):
        scenario = generate("chain", 1)
        self.assertEqual(memory_unit_for(scenario), "MiB")
        scenario.metadata["units"] = {"memory": "KiB", "cpu": "ignored"}
        self.assertEqual(memory_unit_for(scenario), "KiB")
        self.assertEqual(memory_unit_for(scenario, result_units={"memory": "B"}), "B")
        self.assertEqual(memory_unit_for(scenario, "GiB", {"memory": "B"}), "GiB")
        data = {"resources": {"cpu": 2, "memory": 1024, "gpu_peak_memory_bytes": 1024 ** 3}}
        label = _label("node", data, DEFAULT_RESOURCES, [], False, memory_unit="KiB")
        self.assertIn("Memory [MiB]: 1", label)
        self.assertIn("GPU memory [GiB]: 1", label)
        label = _label("node", data, ["memory"], [], False, memory_unit="pages")
        self.assertIn("Memory [pages]: 1024", label)
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
                for expected in ("Storage R/W [KiB]", "Network RX/TX [KiB]",
                                 "GPU memory [GiB]: 1", "CPU: 3", "100 calls/window", "workflow invocations"):
                    self.assertIn(expected, svg)
                self.assertNotIn("profile units", svg)
                self.assertNotIn("GPU required", svg)
                self.assertNotIn("ops/inv", svg)
                self.assertNotIn("/inv", svg)
            path = Path(directory) / "groups.svg"
            visualize(scenario, path, groups={"f0": ["f0", "f1"], "f2": ["f2", "f3"]})
            self.assertIn("Group rooted at f0", path.read_text())
            self.assertIn("Storage R/W [KiB]", path.read_text())
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
            self.assertIn("Memory [KiB]: 1", svg.read_text())
            subprocess.run(command + ["--result", str(result)], check=True, capture_output=True, timeout=15)
            self.assertIn("Memory [MiB]: 1", svg.read_text())
            subprocess.run(command + ["--result", str(result), "--memory-unit", "MiB"],
                           check=True, capture_output=True, timeout=15)
            self.assertIn("Memory [GiB]: 1", svg.read_text())
            subprocess.run(command + ["--resources", "cpu", "--node-attributes", ""],
                           check=True, capture_output=True, timeout=15)
            self.assertNotIn("Memory [", svg.read_text())
            subprocess.run(command + ["--hide-labels"], check=True, capture_output=True, timeout=15)
            self.assertNotIn("CPU:", svg.read_text())


if __name__ == "__main__":
    unittest.main()
