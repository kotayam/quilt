"""Video-feed structure, context-aware traffic budgets and profile alternatives."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from algorithm_testbench.catalog import EXAMPLES, load_workflows, render_catalog
from algorithm_testbench.evaluate import evaluate
from algorithm_testbench.model import Scenario
from algorithm_testbench.profiles import execution_profile_view
from algorithm_testbench.video_feed import CASES, build_video_feed


class VideoFeedTests(unittest.TestCase):
    def test_saved_cases_match_builder_and_share_graph(self):
        shapes = []
        for case in CASES:
            scenario = Scenario.load(EXAMPLES / f"video_feed_{case}.json")
            self.assertEqual(scenario.to_dict(), build_video_feed(case).to_dict())
            self.assertEqual(len(scenario.graph), 21)
            self.assertEqual(scenario.graph.number_of_edges(), 22)
            self.assertEqual(scenario.entry, "feed_handler")
            self.assertTrue(all(edge["invocations"] == 100 for _, _, edge in scenario.graph.edges(data=True)))
            shapes.append({(source, target, data["type"]) for source, target, data in scenario.graph.edges(data=True)})
            for node in ("feature_lookup", "video_metadata"):
                self.assertEqual(scenario.graph.in_degree(node), 2)
                total_calls = sum(scenario.graph.edges[parent, node]["invocations"]
                                  for parent in scenario.graph.predecessors(node))
                self.assertEqual(total_calls, 200)
                self.assertTrue(all(scenario.alpha(parent, node) == 1 for parent in scenario.graph.predecessors(node)))
        self.assertEqual(shapes[0], shapes[1])
        self.assertEqual(shapes[0], shapes[2])
        with self.assertRaises(ValueError):
            build_video_feed("unknown")

    def test_candidate_counts_and_payload_contracts(self):
        for case, count, eligible in (("small", 100, 64), ("large", 5000, 3200), ("cache_miss", 100, 64)):
            scenario = build_video_feed(case)
            workload, graph = scenario.metadata["workload"], scenario.graph
            self.assertEqual(workload["raw_candidates"], count)
            self.assertEqual(sum(workload["branch_candidates"].values()), count)
            self.assertEqual(workload["unique_candidates"], count * 0.8)
            self.assertEqual(workload["ranking_candidates"], eligible)
            self.assertEqual(workload["returned_videos"], 10)
            for branch, items in workload["branch_candidates"].items():
                self.assertEqual(graph.edges["candidate_pool", branch]["attributes"]["response_bytes"], items * 12)
            self.assertEqual(graph.edges["recommend", "candidate_pool"]["attributes"]["response_bytes"], eligible * 8)
            self.assertEqual(graph.edges["candidate_pool", "candidate_filter"]["attributes"]["request_bytes"],
                             workload["unique_candidates"] * 8)
            tensor_bytes = eligible * workload["feature_width"] * workload["feature_element_bytes"]
            self.assertEqual(workload["ranking_feature_bytes"], tensor_bytes)
            self.assertEqual(graph.edges["rank_candidates", "feature_lookup"]["attributes"]["response_bytes"], tensor_bytes)
            self.assertEqual(graph.edges["recommend", "rank_candidates"]["attributes"]["response_bytes"], eligible * 12)
            self.assertEqual(graph.edges["recommend", "rerank"]["attributes"]["request_bytes"], eligible * 12)
            self.assertEqual(graph.edges["feed_handler", "recommend"]["attributes"]["response_bytes"], 10 * 8)
            cards = 10 * (workload["card_metadata_bytes"] + workload["playback_url_bytes_per_item"])
            self.assertEqual(graph.edges["feed_handler", "hydrate_feed"]["attributes"]["response_bytes"], cards)
            self.assertEqual(workload["rendered_response_bytes"], cards + workload["render_envelope_bytes"])
            self.assertEqual(graph.edges["feed_handler", "render_response"]["attributes"]["response_bytes"],
                             workload["rendered_response_bytes"])

    def test_call_order_and_join_metadata_cover_exact_edges(self):
        graph = build_video_feed().graph
        actual_calls = {}
        for caller, data in graph.nodes(data=True):
            for step in data["attributes"]["execution_plan"]:
                if step["kind"] == "parallel":
                    self.assertEqual(step["join"], "all")
                    for callee in step["callees"]:
                        actual_calls[caller, callee] = "async"
                else:
                    actual_calls[caller, step["callee"]] = step["kind"]
        self.assertEqual(actual_calls, {(u, v): data["type"] for u, v, data in graph.edges(data=True)})
        feed_plan = graph.nodes["feed_handler"]["attributes"]["execution_plan"]
        self.assertEqual([step["callee"] for step in feed_plan],
                         ["authenticate", "load_context", "recommend", "hydrate_feed", "render_response", "record_event"])
        self.assertEqual(feed_plan[-1]["join"], "none")
        candidate_plan = graph.nodes["candidate_pool"]["attributes"]["execution_plan"]
        self.assertEqual(candidate_plan[0]["kind"], "parallel")
        self.assertEqual(candidate_plan[0]["join"], "all")
        self.assertEqual(candidate_plan[1], {"kind": "sync", "callee": "candidate_filter"})

    def test_external_cache_traffic_is_averaged_not_double_counted(self):
        for case in CASES:
            scenario = build_video_feed(case)
            hit = scenario.metadata["workload"]["cache_hit_rate"]
            for helper in ("feature_lookup", "video_metadata"):
                node = scenario.graph.nodes[helper]
                contexts = node["attributes"]["call_contexts"]
                self.assertEqual(len(contexts), 2)
                for context in contexts:
                    self.assertEqual(scenario.graph.edges[context["caller"], helper]["attributes"]["response_bytes"],
                                     context["response_bytes"])
                total_items = sum(context["items"] for context in contexts)
                total_values = sum(context["response_bytes"] for context in contexts)
                resources = node["resources"]
                self.assertAlmostEqual(resources["network_external_rx_bytes_per_invocation"] * 2,
                                       total_values + (1 - hit) * total_items * 8)
                self.assertAlmostEqual(resources["network_external_tx_bytes_per_invocation"] * 2,
                                       total_items * 8 * (2 - hit))
        small, missed = build_video_feed("small"), build_video_feed("cache_miss")
        for helper in ("feature_lookup", "video_metadata"):
            warm = small.graph.nodes[helper]
            cold = missed.graph.nodes[helper]
            self.assertGreater(cold["attributes"]["estimated_service_time_ms"], warm["attributes"]["estimated_service_time_ms"])
            self.assertGreater(cold["resources"]["network_external_tx_bytes_per_invocation"],
                               warm["resources"]["network_external_tx_bytes_per_invocation"])
        self.assertEqual(small.graph.nodes["rank_candidates"], missed.graph.nodes["rank_candidates"])

    def test_cpu_gpu_tradeoffs_and_baseline_compatibility(self):
        for case in CASES:
            scenario = build_video_feed(case)
            profiled = {node for node, data in scenario.graph.nodes(data=True) if "execution_profiles" in data["attributes"]}
            self.assertEqual(profiled, {"encode_query", "rank_candidates"})
            for _, data in scenario.graph.nodes(data=True):
                self.assertFalse(data["attributes"]["requires_gpu"])
                self.assertEqual(data["resources"]["storage_read_bytes_per_invocation"], 0)
                self.assertEqual(data["resources"]["storage_write_bytes_per_invocation"], 0)
                alternatives = data["attributes"].get("execution_profiles")
                if alternatives:
                    self.assertEqual(alternatives["cpu"]["resources"], data["resources"])
                    self.assertFalse(alternatives["cpu"]["requires_gpu"])
                    self.assertTrue(alternatives["gpu"]["requires_gpu"])
                    self.assertGreater(alternatives["gpu"]["resources"]["gpu_peak_memory_bytes"], 0)
            encoder = scenario.graph.nodes["encode_query"]["attributes"]["execution_profiles"]
            self.assertLess(encoder["cpu"]["estimated_service_time_ms"], encoder["gpu"]["estimated_service_time_ms"])
            ranking = scenario.graph.nodes["rank_candidates"]["attributes"]["execution_profiles"]
            if case == "large":
                self.assertGreater(ranking["cpu"]["estimated_service_time_ms"], ranking["gpu"]["estimated_service_time_ms"])
            else:
                self.assertLess(ranking["cpu"]["estimated_service_time_ms"], ranking["gpu"]["estimated_service_time_ms"])
            self.assertTrue(evaluate(scenario, {n: [n] for n in scenario.graph}, {"cpu": 4, "memory": 1024})["valid"])

    def test_profile_views_are_copies_and_not_algorithm_outputs(self):
        scenario = build_video_feed("large")
        original = scenario.to_dict()
        gpu = execution_profile_view(scenario)
        self.assertFalse(gpu.metadata["execution_profile_view"]["optimizer_output"])
        self.assertNotIn("CPU baseline", gpu.metadata["name"])
        self.assertIn("not optimizer output", gpu.metadata["name"])
        self.assertEqual([n for n, data in gpu.graph.nodes(data=True) if data["attributes"]["requires_gpu"]],
                         ["encode_query", "rank_candidates"])
        self.assertEqual(dict(scenario.graph.edges), dict(gpu.graph.edges))
        self.assertEqual(scenario.to_dict(), original)
        cpu_view = execution_profile_view(scenario, "cpu")
        for node in scenario.graph:
            self.assertEqual(scenario.graph.nodes[node]["resources"], cpu_view.graph.nodes[node]["resources"])
        for bad in ("unknown", None):
            with self.assertRaises(ValueError):
                execution_profile_view(scenario, bad)
        with self.assertRaises(ValueError):
            execution_profile_view(load_workflows()["ml_training"])
        malformed = Scenario.from_dict(original)
        malformed.graph.nodes["encode_query"]["attributes"]["execution_profiles"]["gpu"]["estimated_service_time_ms"] = -1
        with self.assertRaises(ValueError):
            execution_profile_view(malformed)

    def test_catalog_exposes_gpu_alternatives_and_response_traffic(self):
        scenario = load_workflows()["video_feed_small"]
        original = scenario.to_dict()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            index = render_catalog(output, workflow="video_feed_small")
            text = index.read_text()
            for expected in ("CPU/GPU implementation alternatives", "Exclusive service time [ms]",
                             "Caller-local execution plan", "join all", "detached", "Shared helper inputs",
                             "Response-payload SVG", "not a placement or merging decision"):
                self.assertIn(expected, text)
            self.assertEqual(len(list(output.iterdir())), 11)
            self.assertEqual(Scenario.load(output / "video_feed_small.json").to_dict(), original)
            view = Scenario.load(output / "video_feed_small_gpu.json")
            self.assertFalse(view.metadata["execution_profile_view"]["optimizer_output"])
            self.assertIn("GPU memory", (output / "video_feed_small_gpu.svg").read_text())
            self.assertIn("response payload per call", (output / "video_feed_small_responses.svg").read_text())
        self.assertEqual(scenario.to_dict(), original)


if __name__ == "__main__":
    unittest.main()
