# Standalone merging algorithm testbench

This directory is independent of the existing tests and experiments. It runs on
one machine without deploying functions, containers, or a serverless platform.
Original `merge_solver/` source is only imported by optional Quilt adapters.

For source-based workflow examples with illustrative extended resource profiles,
see the [curated workflow guide](examples/workflows/README.md): parallel CPU ML
training, warm GPU image inference, and database-backed compose-post. Render all
graphs and a self-contained catalog with `python -m algorithm_testbench.catalog`.

## Install and try it

Run all commands from the repository root:

```bash
python3 -m venv algorithm_testbench/.venv
algorithm_testbench/.venv/bin/python -m pip install -r algorithm_testbench/requirements.txt
source algorithm_testbench/.venv/bin/activate

python -m algorithm_testbench.generate --shape diamond --seed 42 --output algorithm_testbench/outputs/generated.json
python -m algorithm_testbench.visualize algorithm_testbench/outputs/generated.json --output algorithm_testbench/outputs/generated

python -m algorithm_testbench.run --config algorithm_testbench/examples/experiments/diamond.json --output algorithm_testbench/outputs/diamond
python -m algorithm_testbench.visualize algorithm_testbench/examples/graphs/diamond.json --output algorithm_testbench/outputs/diamond/workflow
python -m algorithm_testbench.visualize algorithm_testbench/examples/graphs/diamond.json --result algorithm_testbench/outputs/diamond/results.json --algorithm greedy --seed 0 --output algorithm_testbench/outputs/diamond/groups

python -m unittest discover -s algorithm_testbench/tests -v
```

An extensionless visualization output writes both SVG and PNG. Specify `.svg` or
`.png` to write only that format. No Graphviz or display server is required.
Generated outputs, virtual environments, and plotting caches are ignored within
this directory. A checked-in example is handwritten; generated graphs include
the generator seed and parameters.

## Saved graph format

`examples/graphs/diamond.json` is a complete editable example:

```json
{
  "schema_version": 1,
  "name": "Two-function chain",
  "entry": "A",
  "workflow_invocations": 100,
  "nodes": [
    {"id": "A", "resources": {"cpu": 1, "memory": 16}, "attributes": {}},
    {"id": "B", "resources": {"cpu": 2, "memory": 32, "gpu_peak_memory_bytes": 1073741824}, "attributes": {"requires_gpu": true}}
  ],
  "edges": [
    {"source": "A", "target": "B", "invocations": 200, "type": "async", "attributes": {"request_bytes": 4096}}
  ]
}
```

IDs are unique strings. Graphs are directed acyclic graphs with exactly one root,
matching `entry`. Resource values are finite, nonnegative numbers. Invocation
counts are nonnegative integers; `workflow_invocations` must be positive.
Unknown JSON fields, resource keys, and attributes survive save/load round trips.
Zero-count edges can document paths not observed during profiling; the current
Quilt equations then give alpha zero, exactly as in the original code. Avoid
using such counts to claim conservative resource bounds.

The graph loader does not require CPU/memory: those fields are required by the
active `quilt` resource model at experiment time. Resources and limits must use
the same units; experiment `units` records those units, without conversion.

Invocation count is the aggregate caller-to-callee count in the profiling window.
`alpha = ceil(edge.invocations / workflow_invocations)` is derived, never taken
from an unrelated cost attribute. The initial objective is remote-call count.
Additional edge attributes can later support other objectives without redefining
invocation multiplicity.

Generation supports `chain`, `fanout`, `tree`, `diamond` (always four nodes), and
`random`, with `--nodes`, `--seed`, `--workflow-invocations`,
`--async-probability`, and `--extra-edge-probability`. Random generation uses
abstract synthetic counts/resources, not an execution simulator or measured trace.
The saved file is the experiment input; reuse it rather than regenerating graphs
between algorithms. To test a generated file, copy an example experiment and set
`graph` relative to that experiment's location. Vary resource ranges by editing
the saved node dictionaries or constructing a `Scenario` in Python.

## Optional storage, network, and GPU profiles

Enable synthetic extended data without changing the original graph:

```bash
python -m algorithm_testbench.generate --shape diamond --seed 42 --resource-profile extended --output algorithm_testbench/outputs/extended.json
python -m algorithm_testbench.visualize algorithm_testbench/outputs/extended.json --output algorithm_testbench/outputs/extended
```

`examples/graphs/diamond_extended.json` is a saved output of this command. The
default `--resource-profile basic` preserves the previous generator output.
Extended generation uses a separate seeded random stream, leaving topology,
invocation counts, edge types, CPU, and memory identical for the same parameters.

Optional node fields in `resources`:

| Field | Meaning |
| --- | --- |
| `storage_read_ops_per_invocation` | Average local disk read operations per invocation |
| `storage_write_ops_per_invocation` | Average local disk write operations per invocation |
| `storage_read_bytes_per_invocation` | Average bytes read from local disk per invocation |
| `storage_write_bytes_per_invocation` | Average bytes written to local disk per invocation |
| `network_external_rx_bytes_per_invocation` | Average external received bytes per invocation |
| `network_external_tx_bytes_per_invocation` | Average external sent bytes per invocation |
| `gpu_peak_memory_bytes` | Peak device-memory footprint of one invocation on its reference GPU |

These measurements are finite, nonnegative numbers. Fractional per-invocation
averages are allowed. New byte fields always use **bytes**, independent of the
existing CPU/memory units. Missing fields mean unknown; zero means explicitly no
demand. The loader does not fill missing measurements with zeros.

Store `requires_gpu` as a **boolean in node `attributes`**, not `resources`, whose
values must be numeric. Its absence means unspecified. A GPU-required function may
omit its memory footprint when that measurement is unknown. A requirement flag
describes hardware eligibility; optional peak memory adds useful capacity data.
Neither implies a device count, exclusive allocation, or a compute utilization
share. The generator omits GPU memory for nodes with `requires_gpu: false`.
If device-specific profiles are needed, reference-device information can be
recorded in arbitrary node attributes; it is not inferred or enforced.

Node network data **excludes traffic represented by graph edges**. Optional edge
`attributes.request_bytes` and `attributes.response_bytes` describe average
payload bytes per call and must be finite, nonnegative numbers when present.
Keeping these separate avoids double counting and lets a future model remove
network traffic on merged call edges while retaining external traffic. Payload
sizes do not automatically include protocol overhead. Remote database/object-store
operations are not local disk operations; they need a separate service profile.

The generator uses 0–128 reads/writes per invocation at 4 KiB per operation,
0–1 MiB external rx/tx and per-call request/response sizes, a 25% GPU requirement
probability, and GPU peaks chosen from 256, 512, or 1024 MiB. These are illustrative
synthetic ranges, not hardware measurements. Edit the JSON for other workloads.

**IOPS and bandwidth require a workload rate.** Per-invocation operations and bytes
record work, rather than implicitly assuming a request rate. For example, 1 MiB
per invocation at 100 function invocations/second implies 100 MiB/second average
throughput. Peak demand additionally needs timing/concurrency information. Both
operation counts and byte volumes matter for storage; neither alone captures all
IOPS and throughput bottlenecks. GPU utilization is deferred because activity
percentages do not directly describe an additive capacity requirement.

These fields are **data only**. Quilt's algorithms and `quilt` evaluator continue
using CPU, memory, invocation counts, and sync/async types; assignments, costs, and
resource accounting do not change. New limits remain unsupported and are rejected
explicitly. Validity under Quilt does not imply validity under GPU, disk, or
network limits. GPU allocations shared across merged functions are not automatically
deduplicated; a future model must define that behavior.

## Algorithms and custom implementations

Built-ins:

| Name | Behavior |
| --- | --- |
| `no_merge` | Each function remains its own group |
| `greedy` | Simple disjoint pairwise merging, accepting valid cost improvements |
| `quilt_exact` | Enumerates root sets and calls the original ILP; defaults to at most 10 nodes |
| `quilt_downstream` | Original downstream-impact greedy root selection and ILP |
| `quilt_weighted_degree` | Original weighted-degree greedy root selection and ILP |

The simple `greedy` baseline is included so merging experiments work without a
Gurobi installation. It is not Quilt's heuristic and cannot duplicate functions.

For original Quilt baselines:

```bash
python -m pip install -r algorithm_testbench/requirements-quilt.txt
python -m algorithm_testbench.run --config algorithm_testbench/examples/experiments/diamond_quilt.json --output algorithm_testbench/outputs/quilt
```

Gurobi needs a working license appropriate to the problem size. A missing binding
is recorded as `unavailable`; solver/license errors are recorded as `error`.
Optional adapter integration tests run when `gurobipy` is installed and require a
working license. The ordinary suite skips them when the binding is absent.

Add a Python module accessible from the repository root, and reference its
function as `"name": "my_algorithm:merge"` in an experiment:

```python
from algorithm_testbench.model import AlgorithmResult

def merge(scenario, limits, config):
    graph = scenario.graph  # NetworkX DiGraph; metadata on scenario.metadata
    parameters = config["parameters"]
    return AlgorithmResult(
        groups={node: [node] for node in graph},
        status="feasible",
        diagnostics={"description": "replace this baseline with your algorithm"},
    )
```

Each group maps its entry/root ID to unique member IDs including that root.
Groups may overlap. Use `feasible`, `infeasible`, or `unknown` as completion
statuses. Custom optimality claims are downgraded to feasible: only the exhaustive
Quilt adapter currently supplies certification for comparison gaps.

Every run uses a separate spawned process with its own graph, limits, and Python
random seed. Algorithms using NumPy or other random generators must seed those
explicitly using `config["seed"]`. `timeout_seconds` bounds each entire worker
run including startup; timed-out runs have no retained incumbent. The optional
`solve_time_limit_seconds` parameter limits each Quilt ILP, not the root search.
Existing heuristics retain their original behavior; `max_k` restricts only exact
root enumeration. Use identical configurations when comparing algorithms.

## Resource models and evaluation

`resources.py` provides `RESOURCE_MODELS` as the extension point. Each model
implements `validate(scenario, limits)` and `usage(scenario, root, members)`.
Register future models in that module so they are available in spawned workers.
Optimization algorithms must also honor the selected model; the original Quilt
adapters explicitly support only `quilt`. The independent evaluator checks them.

For a group with root r, the initial Quilt model computes:

```text
CPU = cpu[r] + sum(alpha[u,v] * cpu[v] for internal edges)
memory = memory[r] + sum(memory[v] for internal edges)
         + sum((alpha[u,v] - 1) * memory[v] for internal async edges)
```

Its known upstream-async/downstream-sync limitation is preserved for baseline
comparability, with `examples/graphs/inherited_concurrency.json` illustrating it.
A valid result means valid under this model, not guaranteed real peak usage.

Storing `gpu_peak_memory_bytes`, bandwidth, IOPS, or other fields does not enforce them.
Activating unsupported limits fails explicitly. Future models must specify units,
aggregation, concurrency, local versus shared bottlenecks, and (for GPUs) device
compatibility. Bandwidth demand also depends on traffic remaining after merging.

Evaluation independently checks coverage, root inclusion, directed connectivity,
legal boundaries, and resource limits. For overlapping groups, an edge is cut if
ANY group contains its caller but not its callee. Every cut must target a group
root. This matches Quilt's ILP; merely finding both endpoints together in some
other group does not remove a cut.

`results.json` embeds the complete input graph/configuration, assignments, seeds,
statuses, diagnostics, resource usage/slack, and evaluation metrics. `summary.csv`
contains one row per run, including:

- `remote_call_cost`: sum of invocation counts on cut edges, counted once per edge.
- `reduction_from_baseline`: `1 - cost / sum(all edge counts)`; null for zero baseline.
- `absolute_savings`: baseline cost minus remaining cost.
- Group count, duplicated placements, algorithm time, and wall time including startup.
- `optimality_gap`: `(cost - optimum) / optimum`, only with a proven optimum in the same seed/configuration; null for zero optimum, with `absolute_gap` retained.

An exact result is certified only when every relevant root set is considered and
all subproblems finish optimally or prove infeasible. A restricted `max_k`, time
limit, or missing incumbent prevents certification. No fabricated lower bounds
are reported. Invalid or incomplete runs are excluded from quality comparisons.
The no-merging communication cost remains a reference even when singleton groups
cannot satisfy resource limits.

## Visualization options

The default node label shows CPU, memory, paired storage/network volumes, and GPU
memory when present:

```text
f1
CPU: 3
Memory [MiB]: 16
Storage R/W [KiB]: 88 / 156
Network RX/TX [KiB]: 473.75 / 244.03
GPU memory [MiB]: 256
```

Storage operation counts and the GPU-required flag are hidden by default; their
JSON fields and generation remain unchanged. CPU has no units. Other units appear
in brackets beside labels. Byte quantities automatically scale to B, KiB, MiB,
GiB, TiB (and larger binary units as needed), with at most two decimal places and
no scientific notation. Both values in a pair use the same unit, chosen from the
larger selected value. Zero-valued node measurements are hidden. A pair with both
values zero is omitted; a known-zero side is removed and the remaining direction
is named explicitly, e.g. `Network TX [MiB]: 3.91` or `Storage R [KiB]: 88`.
A missing or unselected partner is still shown as a dash when the other value is
nonzero; a pair with no nonzero measurements is omitted. Positive
values too small for two decimal places in a shared unit appear as `<0.01`.
This affects labels only: stored zeros and algorithm inputs are preserved, and
zero-valued edge labels retain their existing behavior.

Storage and external network volumes remain **per function invocation**, with the
repeated `/inv` suffix omitted from node labels for readability. They are not
bandwidth rates. Edge labels are unchanged: invocation counts use `calls/window`,
and request/response payloads use bytes per call, such as `12 KiB/call`.

The original `memory` field defaults to source units of MiB for generated graphs.
A graph's optional top-level `units.memory` declaration supplies another source
unit. When visualizing a selected benchmark result, its experiment's memory unit
takes precedence over the graph declaration. `--memory-unit` overrides both.
Recognized byte units (including decimal MB/GB) are converted to binary display
units; custom units such as `pages` are displayed as declared. New `_bytes` fields
always contain bytes and are unaffected by this override. Graph JSON, generator
output, and resource accounting remain unchanged.

Select fewer measurements or change edge labels when needed:

```bash
python -m algorithm_testbench.visualize algorithm_testbench/examples/graphs/diamond_extended.json --edge-attribute request_bytes --orientation horizontal --output algorithm_testbench/outputs/traffic.svg
```

Layouts use deterministic topological layers; synchronous and asynchronous edges
have distinct colors/styles. Node spacing and figure dimensions adapt to measured
label sizes. The root is highlighted. `--resources` and `--node-attributes` replace
their respective default field selections; pass an empty string to omit a category.
Arbitrary custom keys can still be selected explicitly. Explicit
`--node-attributes requires_gpu` restores the GPU-required flag if
needed. Operation counts can be requested with
`--resources storage_read_ops_per_invocation,storage_write_ops_per_invocation`,
which renders a separate `Storage ops R/W [ops]` line. `--hide-labels` keeps IDs
and hides resources/edge labels. Large dense DAGs can be cluttered: select fewer
fields or export SVG for zooming. `--result`, `--algorithm`, and `--seed` select
a validated assignment for one panel per merge group, repeating shared functions
across panels. Results from a different input graph are rejected.

This harness compares graph-level decisions and computational effort. It does not
estimate actual application latency, throughput, or execution-time concurrency.
