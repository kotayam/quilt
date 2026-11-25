"""Check the curated examples' provenance, workload assumptions, and payload budgets."""

from pathlib import Path
import re
import tempfile
import unittest

from algorithm_testbench.catalog import WORKFLOWS, load_workflows, render_catalog
from algorithm_testbench.evaluate import evaluate
from algorithm_testbench.model import Scenario


class CatalogTests(unittest.TestCase):
    def test_examples_validate_round_trip_and_identify_estimates(self):
        examples = load_workflows()
        self.assertEqual(set(examples), set(WORKFLOWS))
        repo = Path(__file__).resolve().parents[2]
        for name, scenario in examples.items():
            with self.subTest(workflow=name):
                document = scenario.to_dict()
                self.assertEqual(Scenario.from_dict(document).to_dict(), document)
                self.assertEqual(scenario.metadata["workflow_invocations"], 100)
                self.assertEqual(scenario.metadata["units"]["memory"], "MiB")
                self.assertIn("Illustrative resource estimates, not measured production data", scenario.metadata["name"])
                provenance = scenario.metadata["provenance"]
                self.assertEqual(provenance["resource_status"], "illustrative_estimates")
                self.assertEqual(provenance["catalog_id"], name)
                self.assertTrue(provenance["sources"])
                for source in provenance["sources"]:
                    self.assertTrue(source.get("url") or source.get("path"))
                    if "path" in source:
                        self.assertTrue((repo / source["path"]).is_file(), source)
                for u, v, data in scenario.graph.edges(data=True):
                    self.assertEqual(data["invocations"], 100)
                    self.assertEqual(scenario.alpha(u, v), 1)
                    self.assertGreaterEqual(data["attributes"]["request_bytes"], 0)
                    self.assertGreaterEqual(data["attributes"]["response_bytes"], 0)
                for _, data in scenario.graph.nodes(data=True):
                    self.assertTrue(data["attributes"]["role"])
                    self.assertIsInstance(data["attributes"]["external_services"], list)

    def test_training_dataset_accounting_and_cpu_only_parallelism(self):
        scenario = load_workflows()["ml_training"]
        graph = scenario.graph
        workload = scenario.metadata["workload"]
        size = (workload["samples"] * workload["features"] * workload["feature_element_bytes"]
                + workload["samples"] * workload["label_element_bytes"])
        self.assertEqual(size, workload["dataset_bytes"])
        self.assertEqual(len(graph), 3)
        self.assertEqual(set(graph.successors("generate_dataset")), {"train_svm", "train_random_forest"})
        generator = graph.nodes["generate_dataset"]["resources"]
        self.assertEqual(generator["network_external_tx_bytes_per_invocation"], size)
        for trainer, artifact in (("train_svm", "svm_artifact_bytes"),
                                  ("train_random_forest", "random_forest_artifact_bytes")):
            resources = graph.nodes[trainer]["resources"]
            self.assertEqual(resources["network_external_rx_bytes_per_invocation"], size)
            self.assertEqual(resources["network_external_tx_bytes_per_invocation"], workload[artifact])
            edge = graph.edges["generate_dataset", trainer]
            self.assertEqual(edge["type"], "async")
            self.assertEqual(edge["attributes"]["payload_kind"], "dataset_reference_and_parameters")
            self.assertLess(edge["attributes"]["request_bytes"], size)
        for _, data in graph.nodes(data=True):
            self.assertFalse(data["attributes"]["requires_gpu"])
            self.assertNotIn("gpu_peak_memory_bytes", data["resources"])
            self.assertEqual(data["resources"]["storage_read_bytes_per_invocation"], 0)
            self.assertEqual(data["resources"]["storage_write_bytes_per_invocation"], 0)

    def test_gpu_inference_payloads_local_scratch_and_warm_model(self):
        scenario = load_workflows()["gpu_image_inference"]
        graph = scenario.graph
        workload = scenario.metadata["workload"]
        tensor_size = workload["tensor_element_bytes"]
        for dimension in workload["tensor_shape"]:
            tensor_size *= dimension
        self.assertEqual(workload["tensor_bytes"], tensor_size)
        self.assertEqual(workload["scores_bytes"], workload["class_count"] * workload["score_element_bytes"])
        expected = [("fetch_image", "preprocess", "encoded_image_bytes"),
                    ("preprocess", "infer_gpu", "tensor_bytes"),
                    ("infer_gpu", "postprocess", "scores_bytes"),
                    ("postprocess", "store_result", "result_record_bytes")]
        self.assertEqual(set(graph.edges()), {(source, target) for source, target, _ in expected})
        for source, target, key in expected:
            edge = graph.edges[source, target]
            self.assertEqual(edge["attributes"]["request_bytes"], workload[key])
            self.assertEqual(edge["attributes"]["response_bytes"], workload["acknowledgment_bytes"])
            self.assertEqual(edge["type"], "sync")
        gpu_nodes = [node for node, data in graph.nodes(data=True) if data["attributes"]["requires_gpu"]]
        self.assertEqual(gpu_nodes, ["infer_gpu"])
        inference = graph.nodes["infer_gpu"]
        self.assertEqual(inference["attributes"]["model_state"], "warm")
        self.assertEqual(inference["resources"]["gpu_peak_memory_bytes"], 2 * 1024 ** 3)
        self.assertGreater(inference["resources"]["gpu_peak_memory_bytes"], workload["resident_gpu_weights_bytes"])
        for node, data in graph.nodes(data=True):
            resources = data["resources"]
            scratch = workload["encoded_image_bytes"] if node == "preprocess" else 0
            self.assertEqual(resources["storage_read_bytes_per_invocation"], scratch)
            self.assertEqual(resources["storage_write_bytes_per_invocation"], scratch)
            if node not in ("fetch_image", "store_result"):
                self.assertEqual(resources["network_external_rx_bytes_per_invocation"], 0)
                self.assertEqual(resources["network_external_tx_bytes_per_invocation"], 0)
        self.assertEqual(graph.nodes["fetch_image"]["resources"]["network_external_rx_bytes_per_invocation"],
                         workload["encoded_image_bytes"])
        self.assertEqual(graph.nodes["store_result"]["resources"]["network_external_tx_bytes_per_invocation"],
                         workload["result_record_bytes"])

    def test_compose_post_matches_source_calls_and_backend_roles(self):
        scenario = load_workflows()["compose_post"]
        graph = scenario.graph
        repo = Path(__file__).resolve().parents[2]
        path = repo / "benchmark/DeathStarBench/social_network/merge/funcTrees/funcTree.compose_post"
        pairs = {tuple(line.split()) for line in path.read_text().splitlines() if line.strip()}
        self.assertEqual(set(graph.edges()), pairs)
        self.assertEqual(len(graph), 11)
        self.assertTrue(all(data["type"] == "sync" for _, _, data in graph.edges(data=True)))
        for node in ("media-service", "compose-creator-with-userid", "unique-id-service"):
            data = graph.nodes[node]
            self.assertEqual(data["attributes"]["external_services"], [])
            self.assertEqual(data["resources"]["network_external_rx_bytes_per_invocation"], 0)
            self.assertEqual(data["resources"]["network_external_tx_bytes_per_invocation"], 0)
        for node in ("user-mention-service", "social-graph-get-followers"):
            self.assertEqual(graph.nodes[node]["attributes"]["external_services"], ["memcached", "redis"])
        workload = scenario.metadata["workload"]
        updates = workload["follower_count"] + workload["mention_count"]
        self.assertEqual(updates, workload["home_timeline_updates"])
        resources = graph.nodes["write-home-timeline"]["resources"]
        self.assertEqual(resources["network_external_tx_bytes_per_invocation"], updates * workload["redis_update_request_bytes"])
        self.assertEqual(resources["network_external_rx_bytes_per_invocation"], updates * workload["redis_update_response_bytes"])
        for _, data in graph.nodes(data=True):
            self.assertFalse(data["attributes"]["requires_gpu"])
            self.assertEqual(data["resources"]["storage_read_bytes_per_invocation"], 0)
            self.assertEqual(data["resources"]["storage_write_bytes_per_invocation"], 0)

    def test_examples_work_with_existing_quilt_resource_evaluator(self):
        for name, scenario in load_workflows().items():
            with self.subTest(workflow=name):
                report = evaluate(scenario, {node: [node] for node in scenario.graph}, {"cpu": 4, "memory": 1024})
                self.assertTrue(report["valid"], report)
                self.assertEqual(report["remote_call_cost"], len(scenario.graph.edges) * 100)
                self.assertEqual(report["resource_model"], "quilt")

    def test_catalog_exports_snapshots_images_and_context(self):
        originals = {name: scenario.to_dict() for name, scenario in load_workflows().items()}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            index = render_catalog(output)
            text = index.read_text()
            self.assertIn("not measured production data", text)
            self.assertIn("External stores remain external after merging", text)
            profiled_count = sum(any("execution_profiles" in data["attributes"]
                                     for data in document["nodes"]) for document in originals.values())
            self.assertEqual(len(list(output.iterdir())), 1 + 5 * len(originals) + 5 * profiled_count)
            for name, document in originals.items():
                self.assertEqual(Scenario.load(output / f"{name}.json").to_dict(), document)
                for suffix in (".svg", ".png", "_payloads.svg", "_payloads.png"):
                    artifact = output / f"{name}{suffix}"
                    self.assertGreater(artifact.stat().st_size, 100)
                    self.assertIn(artifact.name, text)
                svg = (output / f"{name}.svg").read_text()
                labels = " ".join(re.findall(r"<!--(.*?)-->", svg, re.DOTALL))
                self.assertIn("not measured production data", " ".join(labels.split()))
        self.assertEqual(originals, {name: scenario.to_dict() for name, scenario in load_workflows().items()})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            render_catalog(output, workflow="ml_training", orientation="horizontal")
            self.assertEqual(len(list(output.iterdir())), 6)
            self.assertFalse((output / "compose_post.json").exists())
        with self.assertRaises(ValueError):
            render_catalog("unused", workflow="unknown")


if __name__ == "__main__":
    unittest.main()
