"""Build reproducible, illustrative video-feed fixtures (not measured workloads)."""

import argparse
from copy import deepcopy
from pathlib import Path

from .model import Scenario


CASES = {"small": (100, 0.9), "large": (5000, 0.9), "cache_miss": (100, 0.0)}
EVIDENCE = "Illustrative resource estimates, not measured production data"
CPU_DEVICE = "Illustrative CPU worker with up to 4 cores; no measured performance"
GPU_DEVICE = "Illustrative compatible CUDA GPU with 16 GiB memory; no measured performance"


def _resources(cpu, memory, rx=0, tx=0, gpu=0):
    values = {"cpu": cpu, "memory": memory,
              "storage_read_bytes_per_invocation": 0, "storage_write_bytes_per_invocation": 0,
              "network_external_rx_bytes_per_invocation": rx,
              "network_external_tx_bytes_per_invocation": tx}
    if gpu:
        values["gpu_peak_memory_bytes"] = gpu
    return values


def _node(name, role, cpu, memory, time_ms, *, services=(), rx=0, tx=0):
    return {"id": name, "resources": _resources(cpu, memory, rx, tx),
            "attributes": {"role": role, "requires_gpu": False, "external_services": list(services),
                           "estimated_service_time_ms": time_ms, "execution_plan": []}}


def _profile(resources, time_ms, gpu=False):
    return {"resources": deepcopy(resources), "requires_gpu": gpu,
            "estimated_service_time_ms": time_ms,
            "reference_device": GPU_DEVICE if gpu else CPU_DEVICE,
            "assumptions": ["Exclusive service time excludes nested function calls and queueing.",
                            "Warm model; GPU time includes illustrative launch and host/device-transfer overhead.",
                            "Synthetic tradeoff hypothesis, not a measured speedup or end-to-end prediction."]}


def _backend_traffic(contexts, hit_rate):
    """Mean per helper call: all cache requests plus miss-only backend requests.

    An item is returned once (by cache OR backend). A cache miss also returns an
    8-byte placeholder. Do not count the full value twice on misses.
    """
    items = sum(context["items"] for context in contexts) / len(contexts)
    values = sum(context["response_bytes"] for context in contexts) / len(contexts)
    return {"mean_items": items, "rx": round(values + (1 - hit_rate) * items * 8, 6),
            "tx": round(items * 8 * (2 - hit_rate), 6)}


def build_video_feed(case="small"):
    if case not in CASES:
        raise ValueError(f"unknown video-feed case: {case}")
    raw, hit_rate = CASES[case]
    unique, eligible = raw * 4 // 5, raw * 16 // 25
    branch_counts = {"similar_candidates": raw * 4 // 10,
                     "trending_candidates": raw * 3 // 10, "followed_candidates": raw * 3 // 10}
    top_k, context_bytes, feature_width, element_bytes, embedding_width = 10, 512, 128, 4, 256
    id_bytes, candidate_bytes, metadata_bytes, feature_bytes = 8, 12, 512, feature_width * element_bytes
    query_bytes, ranking_bytes, scores_bytes = embedding_width * element_bytes, eligible * feature_bytes, eligible * 4
    feature_contexts = [
        {"caller": "similar_candidates", "items": 1, "response_bytes": context_bytes},
        {"caller": "rank_candidates", "items": eligible, "response_bytes": ranking_bytes},
    ]
    metadata_contexts = [
        {"caller": "candidate_filter", "items": unique, "response_bytes": unique * 64},
        {"caller": "hydrate_feed", "items": top_k, "response_bytes": top_k * metadata_bytes},
    ]
    features = _backend_traffic(feature_contexts, hit_rate)
    metadata = _backend_traffic(metadata_contexts, hit_rate)
    miss = 1 - hit_rate
    nodes = [
        _node("feed_handler", "Coordinate the feed request, retaining completed stage results", 0.08, 96, 0.5),
        _node("authenticate", "Validate a warm local signing key; no remote authentication call", 0.05, 32, 0.3),
        _node("load_context", "Launch profile/history lookups and await both results", 0.03, 64, 0.2),
        _node("profile_lookup", "Read a compact user profile from the external profile store", 0.03, 32, 1.5,
              services=("profile_store",), rx=256, tx=64),
        _node("history_lookup", "Read recent viewing IDs from the external session cache", 0.03, 48, 1,
              services=("session_cache",), rx=2048, tx=64),
        _node("recommend", "Run candidate retrieval, ranking and re-ranking in that order", 0.05, 128, 0.4),
        _node("candidate_pool", "Join three retrieval branches, deduplicate, then call eligibility filtering",
              0.15 if raw == 100 else 0.8, 64 if raw == 100 else 128, raw * 0.003),
        _node("similar_candidates", "Read query features, encode query, and access an external vector index",
              0.08, 96, 2, services=("vector_index",),
              tx=query_bytes + 64, rx=branch_counts["similar_candidates"] * candidate_bytes),
        _node("trending_candidates", "Retrieve popular IDs/scores from the external trends cache", 0.03, 48, 1,
              services=("trends_cache",), tx=64, rx=branch_counts["trending_candidates"] * candidate_bytes),
        _node("followed_candidates", "Read social IDs, then retrieve their recent videos from an external index",
              0.05, 64, 1.5, services=("creator_video_index",),
              tx=256, rx=branch_counts["followed_candidates"] * candidate_bytes),
        _node("candidate_filter", "Fetch eligibility metadata and remove 20% of unique candidates",
              0.1, 64, unique * 0.002),
        _node("feature_lookup", "Shared batched feature helper; resources are averaged over two call contexts",
              0.05, 64 if raw == 100 else 128, 1 + miss * 10 + features["mean_items"] * 0.002,
              services=("feature_cache", "feature_store"), rx=features["rx"], tx=features["tx"]),
        _node("encode_query", "Warm neural query encoder with selectable CPU/GPU implementation", 1, 256, 3),
        _node("social_lookup", "Read followed creator IDs from the external social store", 0.03, 48, 1.5,
              services=("social_store",), tx=64, rx=256),
        _node("rank_candidates", "Fetch feature tensors and score the eligible candidate batch",
              1 if raw == 100 else 4, 512 if raw == 100 else 768, 8 if raw == 100 else 400),
        _node("rerank", "Apply CPU-based diversity rules and select ten items", 0.1, 64, eligible * 0.003),
        _node("video_metadata", "Shared metadata helper with different filter/hydration projections",
              0.04, 64 if raw == 100 else 128, 1 + miss * 8 + metadata["mean_items"] * 0.001,
              services=("metadata_cache", "video_catalog"), rx=metadata["rx"], tx=metadata["tx"]),
        _node("hydrate_feed", "Fetch ten video cards then request their playback URLs", 0.05, 64, 0.3),
        _node("playback_urls", "Generate locally signed CDN URLs for ten items", 0.05, 32, 0.5),
        _node("render_response", "Render feed JSON or a page shell containing ten cards and playback URLs", 0.1, 64, 1),
        _node("record_event", "Publish a detached recommendation event to an external event sink", 0.02, 32, 1,
              services=("event_sink",), tx=1024, rx=64),
    ]
    by_id = {node["id"]: node for node in nodes}
    by_id["feature_lookup"]["attributes"]["call_contexts"] = feature_contexts
    by_id["video_metadata"]["attributes"]["call_contexts"] = metadata_contexts
    for name, gpu_resources, gpu_time in (
        ("encode_query", _resources(0.15, 256, gpu=768 * 1024 ** 2), 7),
        ("rank_candidates", _resources(0.25, 512, gpu=(2 if raw == 100 else 4) * 1024 ** 3),
         10 if raw == 100 else 30),
    ):
        node = by_id[name]
        node["attributes"]["execution_profiles"] = {
            "cpu": _profile(node["resources"], node["attributes"]["estimated_service_time_ms"]),
            "gpu": _profile(gpu_resources, gpu_time, gpu=True),
        }

    edges = []

    def call(source, target, request, response, *, kind="sync", inputs=()):
        edges.append({"source": source, "target": target, "invocations": 100, "type": kind,
                      "attributes": {"request_bytes": request, "response_bytes": response,
                                     "arguments_from": list(inputs)}})

    call("feed_handler", "authenticate", 512, 64, inputs=("request.credentials",))
    call("feed_handler", "load_context", 64, context_bytes, inputs=("authenticate.user_id",))
    call("feed_handler", "recommend", context_bytes, top_k * id_bytes, inputs=("load_context.result",))
    call("feed_handler", "hydrate_feed", top_k * id_bytes, top_k * (metadata_bytes + 256), inputs=("recommend.result",))
    call("feed_handler", "render_response", top_k * (metadata_bytes + 256), 8192, inputs=("hydrate_feed.result",))
    call("feed_handler", "record_event", 1024, 0, kind="async", inputs=("recommend.result", "request.context"))
    call("load_context", "profile_lookup", 64, 256, kind="async", inputs=("user_id",))
    call("load_context", "history_lookup", 64, 2048, kind="async", inputs=("user_id",))
    call("recommend", "candidate_pool", context_bytes, eligible * id_bytes, inputs=("context",))
    call("recommend", "rank_candidates", eligible * id_bytes + context_bytes, eligible * (id_bytes + 4),
         inputs=("candidate_pool.result", "context"))
    call("recommend", "rerank", eligible * (id_bytes + 4), top_k * id_bytes, inputs=("rank_candidates.result",))
    for branch, count in branch_counts.items():
        call("candidate_pool", branch, context_bytes, count * candidate_bytes, kind="async", inputs=("context",))
    call("candidate_pool", "candidate_filter", unique * id_bytes, eligible * id_bytes,
         inputs=("deduplicated_results_of_all_three_branches",))
    call("similar_candidates", "feature_lookup", 64 + id_bytes, context_bytes, inputs=("user_id",))
    call("similar_candidates", "encode_query", context_bytes, query_bytes, inputs=("feature_lookup.result",))
    call("followed_candidates", "social_lookup", 64, 256, inputs=("user_id",))
    call("rank_candidates", "feature_lookup", 64 + eligible * id_bytes, ranking_bytes, inputs=("candidate_ids",))
    call("candidate_filter", "video_metadata", 64 + unique * id_bytes, unique * 64, inputs=("candidate_ids", "eligibility_projection"))
    call("hydrate_feed", "video_metadata", 64 + top_k * id_bytes, top_k * metadata_bytes, inputs=("selected_ids", "card_projection"))
    call("hydrate_feed", "playback_urls", top_k * id_bytes, top_k * 256, inputs=("selected_ids",))

    def sync(*callees):
        return [{"kind": "sync", "callee": callee} for callee in callees]

    plans = {
        "feed_handler": sync("authenticate", "load_context", "recommend", "hydrate_feed", "render_response")
                        + [{"kind": "async", "callee": "record_event", "join": "none"}],
        "load_context": [{"kind": "parallel", "callees": ["profile_lookup", "history_lookup"], "join": "all"}],
        "recommend": sync("candidate_pool", "rank_candidates", "rerank"),
        "candidate_pool": [{"kind": "parallel", "callees": list(branch_counts), "join": "all"}]
                          + sync("candidate_filter"),
        "similar_candidates": sync("feature_lookup", "encode_query"),
        "followed_candidates": sync("social_lookup"), "rank_candidates": sync("feature_lookup"),
        "candidate_filter": sync("video_metadata"), "hydrate_feed": sync("video_metadata", "playback_urls"),
    }
    for name, plan in plans.items():
        by_id[name]["attributes"]["execution_plan"] = plan
    return Scenario.from_dict({
        "schema_version": 1, "name": f"Video feed ({case}, CPU baseline) — {EVIDENCE}", "entry": "feed_handler",
        "workflow_invocations": 100, "units": {"cpu": "illustrative active core equivalents", "memory": "MiB"},
        "description": "A 21-function direct-call adaptation combining web serving, shared helpers, parallel retrieval and optional GPU inference; not TikTok's actual architecture.",
        "provenance": {"catalog_id": f"video_feed_{case}", "resource_status": "illustrative_estimates",
                       "structure_status": "illustrative_direct_call_adaptation",
                       "sources": [
                           {"label": "Google recommendation-system stages", "url": "https://developers.google.com/machine-learning/recommendation/overview/types"},
                           {"label": "Deep Neural Networks for YouTube Recommendations", "url": "https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/"},
                           {"label": "CloudFront video delivery", "url": "https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/on-demand-streaming-video.html"},
                       ]},
        "workload": {"case": case, "raw_candidates": raw, "branch_candidates": branch_counts,
                     "duplicate_fraction": 0.2, "unique_candidates": unique,
                     "eligibility_rejection_fraction": 0.2, "ranking_candidates": eligible,
                     "returned_videos": top_k, "cache_hit_rate": hit_rate,
                     "id_bytes": id_bytes, "candidate_record_bytes": candidate_bytes,
                     "context_bytes": context_bytes, "feature_width": feature_width,
                     "feature_element_bytes": element_bytes, "ranking_feature_bytes": ranking_bytes,
                     "query_embedding_width": embedding_width, "query_embedding_bytes": query_bytes,
                     "ranking_scores_bytes": scores_bytes, "card_metadata_bytes": metadata_bytes,
                     "playback_url_bytes_per_item": 256, "render_envelope_bytes": 512, "rendered_response_bytes": 8192,
                     "model_state": "warm", "default_execution_profile": "cpu",
                     "reference_cpu": CPU_DEVICE, "reference_gpu": GPU_DEVICE},
        "assumptions": [
            "This is an illustrative direct-call architecture inspired by published recommendation stages, not a reproduction of TikTok or YouTube. All resource, timing, and payload budgets are estimates.",
            "Each edge is called once per successful workflow execution in N=100 executions. Shared feature_lookup and video_metadata each run twice per workflow; their external volumes and service times are means per function invocation over two call contexts, and host memory is a chosen peak envelope across contexts.",
            "execution_plan records caller-local order and join-all points. Each async retrieval/lookup branch is awaited; record_event is detached. Sibling nodes do not imply parallel execution.",
            "Context aggregates a subset/summary of profile and history. Similarity retrieval uses a 256-float query embedding; each ranking item has 128 float32 features. IDs use 8 bytes; retrieved candidate records add a 4-byte score.",
            "20% of raw candidates are removed as duplicates; eligibility rejects 20% of unique candidates. Re-ranking selects ten IDs. Metadata uses a 64-byte eligibility projection or a 512-byte card projection.",
            "The external feature and metadata cache-hit rate is an expected item-level rate, not a new function branch. A cache lookup sends 8 ID bytes per item, hits return the value, misses return an 8-byte placeholder and trigger an 8-byte backend ID request with the value returned once. Helper resource profiles average both call contexts equally.",
            "Cache misses mainly increase external service time; data is still returned exactly once. Exclusive service-time estimates include own computation/external I/O and exclude nested function calls, queueing, and end-to-end scheduling. Profile time is not an evaluator objective.",
            "CPU/GPU alternatives exist only for encode_query and rank_candidates. Baseline resources equal the CPU profile. GPU alternatives assume warm models and include illustrative GPU launch and host/device-transfer overhead. No placement or GPU capacity is enforced.",
            "Node network volumes describe external services only. RPC request/response payloads are on edges and are not double-counted in node fields. Storage, caches, indexes, event sinks and the CDN remain external after merging.",
            "The feed returns an 8192-byte JSON/page-shell budget with metadata and URLs. The client downloads video segments from the CDN separately; video bodies do not pass through these functions.",
            "Local disk I/O is zero for this warm in-memory serving example. Model downloads, retries, failures, client-to-entry traffic and most protocol framing are excluded. No bandwidth rate or end-to-end latency is inferred.",
        ],
        "merging_questions": [
            "Should batched feature hydration merge with ranking when its large tensor is a response payload?",
            "Can the large ranking batch justify GPU overhead while the small request remains on CPU?",
            "Should cheap web wrappers share a GPU-requiring group, or retain independent scaling?",
            "Should feature_lookup or video_metadata be duplicated across groups, given different caller contexts and retained external I/O?",
            "How do cache misses change waiting time without changing call counts or automatically increasing GPU compute demand?",
        ],
        "nodes": nodes, "edges": edges,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path(__file__).parent / "examples" / "workflows"))
    args = parser.parse_args()
    for case in CASES:
        path = Path(args.output) / f"video_feed_{case}.json"
        build_video_feed(case).save(path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
