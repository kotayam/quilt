# Personalized video-feed family

**Illustrative resource estimates, not measured production data.** This is a
direct-call application design, not TikTok's architecture, a deployed service, or
an optimizer result. It combines published patterns from
[Google's recommendation stages](https://developers.google.com/machine-learning/recommendation/overview/types),
[YouTube candidate generation/ranking](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/),
and [separate CDN video delivery](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/on-demand-streaming-video.html).
Those sources motivate stages, not our function boundaries or resource numbers.

## Generate and inspect

Three fixed JSON cases are checked in beside this guide. To reproduce just these
fixtures, without changing any of the original graph generators:

```bash
python -m algorithm_testbench.video_feed
```

Use `--output DIRECTORY` to write copies elsewhere. Render the whole catalog with
`python -m algorithm_testbench.catalog`, or one case:

```bash
python -m algorithm_testbench.catalog --workflow video_feed_large --output algorithm_testbench/outputs/video_feed_large
```

Each case includes CPU-baseline call/request-payload images, a response-payload
image, and a GPU-alternative image, all in SVG and PNG. The generated `index.md`
includes per-stage resources/times, CPU/GPU alternatives, actual caller-local
execution order, shared-helper contexts, and all payload budgets. Baseline JSON
is copied unchanged; `*_gpu.json` is a separate illustrative view.

## Request cases

| Case | Raw candidates | Similar / trending / followed | Unique | Eligible for ranking | Cache hit rate | Returned items |
| --- | --- | --- | --- | --- | --- | --- |
| `video_feed_small` | 100 | 40 / 30 / 30 | 80 | 64 | 90% | 10 |
| `video_feed_large` | 5,000 | 2,000 / 1,500 / 1,500 | 4,000 | 3,200 | 90% | 10 |
| `video_feed_cache_miss` | 100 | 40 / 30 / 30 | 80 | 64 | 0% | 10 |

All cases have **21 functions and 22 call edges**, with one call along each edge
per workflow execution and N=100 executions in the synthetic profile window.
`feature_lookup` and `video_metadata` each have two callers, so each helper runs
200 times in that window. Alpha is still one on each incoming edge. These are not
200 concurrent instances: the metadata describes when each call actually occurs.

## Actual call structure and ordering

| Caller | Ordered actions |
| --- | --- |
| `feed_handler` | Authenticate; load context; recommend; hydrate cards; render; detach event publication |
| `load_context` | Launch profile/history lookups concurrently and join both |
| `recommend` | Retrieve/filter candidate pool; rank; re-rank |
| `candidate_pool` | Launch similar/trending/followed retrieval concurrently; join all, deduplicate, then filter |
| `similar_candidates` | Read query features through the shared helper; encode query; access external vector index |
| `followed_candidates` | Read followed IDs through `social_lookup`; query external creator-video index |
| `rank_candidates` | Read batched features through the shared helper; compute ranking scores |
| `candidate_filter` | Read eligibility metadata through the shared metadata helper; reject ineligible IDs |
| `hydrate_feed` | Read selected card metadata through the same helper; generate playback URLs |

Other stages are leaves with their own work/external accesses. `rerank` performs
CPU-based diversity selection. `playback_urls` signs URLs using local warm keys.
`render_response` builds JSON or a page shell. Authentication uses warm local
signing keys. `record_event` publishes asynchronously and is not awaited by the
feed response. Every node's `attributes.execution_plan` records the ordered
actions and joins; every call edge records its argument dependencies.

The graph is a **call graph**, not a diagram that connects successive dataflow
stages with fictitious RPCs. For example, `feed_handler` retains `load_context`'s
result and passes it to `recommend`; the context loader does not call the
recommender. Siblings are not necessarily parallel.

## Payload and shared-helper semantics

Candidate records contain an 8-byte ID and a 4-byte retrieval score. Deduplication
removes 20%, and filtering removes 20% of the remaining items. Each ranking item
has 128 float32 features (512 bytes), so the tensor returned from `feature_lookup`
is **32 KiB** for the small case or **1.5625 MiB** for the large one. Inspect the
**response-payload** view: these bytes travel back from a helper, not in the RPC
request. The initial query input is 512 bytes; its embedding has 256 float32 values
(1 KiB). Ranking returns IDs/scores, and re-ranking selects ten IDs.

Shared helpers have different input contexts:

- `feature_lookup` serves one query-feature record during similarity retrieval,
  then 64 or 3,200 candidate-feature records during ranking.
- `video_metadata` serves an eligibility projection (64 bytes/item) for 80 or
  4,000 candidates, then a card projection (512 bytes/item) for ten selected items.

Node external-network volumes and exclusive service times are equal-weight means
over those two helper invocations, **not sums charged on each call**. Host memory
is a chosen peak envelope. Context-specific logical output sizes remain on edges
and in `attributes.call_contexts`; a future algorithm could use them rather than
the mean profile.

Cache modeling uses 8 ID bytes per item per query. Cache hits return the value;
cache misses return an 8-byte placeholder, then trigger an 8-byte backend request
and return the value from the backend. Thus the value is transferred once, not
once from each tier. Expected hit rates can produce fractional per-invocation
average volumes. Misses mainly increase waiting/service time, not tensor size or
the amount of ML computation. Only feature/metadata cache rates change between
small and cache-miss cases; retrieval counts and ranker profiles stay the same.

The final ten cards contain 512 metadata bytes plus a 256-byte signed URL each.
Rendering adds a 512-byte shell, producing an 8 KiB response budget. The client
fetches video segments separately from the CDN. No video bodies flow through
these function nodes.

## CPU versus GPU choices

`encode_query` and `rank_candidates` have optional
`attributes.execution_profiles.cpu` and `.gpu` dictionaries. Each contains a full
resource dictionary, `requires_gpu`, `estimated_service_time_ms`, reference-device
assumptions, and profile notes. Ordinary node resources equal the CPU alternative.

| Stage/case | CPU exclusive time [ms] | GPU exclusive time [ms] | GPU peak memory |
| --- | --- | --- | --- |
| Query encoder, all cases | 3 | 7 | 768 MiB |
| Ranker, small/cache-miss | 8 | 10 | 2 GiB |
| Ranker, large | 400 | 30 | 4 GiB |

These deliberately illustrate that GPU overhead can lose on small work but win on
a larger batch. They are **invented comparison profiles, not measurements**. The
reference envelope is a CPU worker with up to four cores and a compatible 16 GiB
CUDA GPU. Device names do not establish actual performance or allocation policy.
Models are warm; GPU times include illustrative launch and host/device-transfer
overhead. Cold initialization/model downloads, queueing, device contention, and
request batching across workflows are excluded. Peak memory is not deduplicated
across groups or invocations.

Every stage has an illustrative exclusive service time for documentation. It
includes its own CPU work/external I/O, **excluding child function calls**. Times
cannot simply be summed to predict workflow latency: joins, overlap, detached
work, and queueing need an execution model. They do not change Quilt's objective.

The catalog creates a deep-copied GPU view selecting both GPU alternatives. That
is intentionally not an optimal choice: in our profiles the query encoder is
still faster on CPU. The view is labeled **not optimizer output**. It is neither
a scheduling result nor a merge assignment, and it does not modify source JSON.

## Decisions to investigate

- Does merging feature hydration with ranking justify placing external-I/O work
  in a GPU group, especially when the tensor is large?
- Should CPU query encoding remain separate even if GPU ranking is attractive?
- Are CPU-light response/event wrappers worth moving into GPU-requiring groups?
- Should shared helpers be duplicated across merged groups, or remain independent?
- Can retrieval branches retain independent scaling and parallelism after merging?
- Does a cache miss change the best grouping even though ML compute is unchanged?

External databases, caches, vector indexes, event sinks, and the CDN are not
mergeable nodes. Their traffic survives function merging. Local disk usage is
zero in this warm, in-memory serving design. CPU/resource numbers, payload
budgets, cache policies, and function boundaries are illustrative choices.

No placement algorithm, timing simulator, new objective, or GPU/I/O evaluator has
been added. Existing Quilt interfaces and CPU/memory rules remain unchanged.
GPU profiles and timing fields are optional data for future algorithms; validity
under the current evaluator does not establish GPU feasibility or a speedup.
