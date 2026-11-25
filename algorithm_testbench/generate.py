"""Generate and save scenarios without running an optimizer."""

import argparse
import random

from .model import Scenario


def generate(shape="random", nodes=10, seed=0, workflow_invocations=100,
             extra_edge_probability=0.15, async_probability=0.2, resource_profile="basic"):
    if nodes < 1:
        raise ValueError("nodes must be positive")
    if not 0 <= extra_edge_probability <= 1 or not 0 <= async_probability <= 1:
        raise ValueError("probabilities must be between 0 and 1")
    if resource_profile not in ("basic", "extended"):
        raise ValueError("resource_profile must be basic or extended")
    rng = random.Random(seed)
    if shape == "diamond":
        nodes = 4
        pairs = [(0, 1), (0, 2), (1, 3), (2, 3)]
    elif shape == "chain":
        pairs = [(i - 1, i) for i in range(1, nodes)]
    elif shape == "fanout":
        pairs = [(0, i) for i in range(1, nodes)]
    elif shape == "tree":
        pairs = [(rng.randrange(i), i) for i in range(1, nodes)]
    elif shape == "random":
        pairs = [(rng.randrange(i), i) for i in range(1, nodes)]
        existing = set(pairs)
        for i in range(nodes):
            for j in range(i + 1, nodes):
                if (i, j) not in existing and rng.random() < extra_edge_probability:
                    pairs.append((i, j))
    else:
        raise ValueError(f"unknown shape: {shape}")
    document = {
        "schema_version": 1,
        "name": f"{shape}-{nodes}-seed-{seed}",
        "entry": "f0",
        "workflow_invocations": workflow_invocations,
        "generator": {"shape": shape, "seed": seed, "nodes": nodes,
                      "extra_edge_probability": extra_edge_probability,
                      "async_probability": async_probability},
        "nodes": [{"id": f"f{i}", "resources": {"cpu": rng.randint(1, 5),
                   "memory": rng.choice([16, 32, 64])}, "attributes": {}}
                  for i in range(nodes)],
        "edges": [{"source": f"f{i}", "target": f"f{j}",
                   "invocations": workflow_invocations * rng.randint(1, 4),
                   "type": "async" if rng.random() < async_probability else "sync",
                   "attributes": {}} for i, j in sorted(pairs)],
    }
    if resource_profile == "extended":
        _extend_resources(document, seed)
    return Scenario.from_dict(document)


def _extend_resources(document, seed):
    """Decorate the completed graph without consuming the topology RNG stream."""
    rng = random.Random(f"extended-resources:{seed}")
    document["generator"]["resource_profile"] = "extended"
    for node in document["nodes"]:
        read_ops = rng.randint(0, 128)
        write_ops = rng.randint(0, 128)
        node["resources"].update({
            "storage_read_ops_per_invocation": read_ops,
            "storage_write_ops_per_invocation": write_ops,
            "storage_read_bytes_per_invocation": read_ops * 4096,
            "storage_write_bytes_per_invocation": write_ops * 4096,
            "network_external_rx_bytes_per_invocation": rng.randint(0, 1024 ** 2),
            "network_external_tx_bytes_per_invocation": rng.randint(0, 1024 ** 2),
        })
        requires_gpu = rng.random() < 0.25
        node["attributes"]["requires_gpu"] = requires_gpu
        if requires_gpu:
            node["resources"]["gpu_peak_memory_bytes"] = rng.choice([256, 512, 1024]) * 1024 ** 2
    for edge in document["edges"]:
        edge["attributes"].update({"request_bytes": rng.randint(0, 1024 ** 2),
                                   "response_bytes": rng.randint(0, 1024 ** 2)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=["chain", "fanout", "tree", "diamond", "random"], default="random")
    parser.add_argument("--nodes", type=int, default=10, help="diamond always has four nodes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workflow-invocations", type=int, default=100)
    parser.add_argument("--extra-edge-probability", type=float, default=0.15)
    parser.add_argument("--async-probability", type=float, default=0.2)
    parser.add_argument("--resource-profile", choices=["basic", "extended"], default="basic",
                        help="extended adds optional storage/network/GPU data without changing the graph")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        scenario = generate(args.shape, args.nodes, args.seed, args.workflow_invocations,
                            args.extra_edge_probability, args.async_probability, args.resource_profile)
        scenario.save(args.output)
    except ValueError as error:
        parser.error(str(error))
    print(f"Saved {len(scenario.graph)} nodes and {len(scenario.graph.edges)} edges to {args.output}")


if __name__ == "__main__":
    main()
