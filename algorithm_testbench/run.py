"""Run isolated algorithms on one saved graph and write JSON/CSV comparisons."""

import argparse
from copy import deepcopy
import csv
import io
import json
import multiprocessing as mp
from pathlib import Path
import random
import time
from contextlib import redirect_stdout, redirect_stderr

from .algorithms import execute
from .evaluate import evaluate
from .model import AlgorithmResult, Scenario, mapping, number, write_json
from .resources import get_model


def _worker(connection, name, document, limits, config, seed):
    log = io.StringIO()
    started = time.perf_counter()
    try:
        random.seed(seed)
        with redirect_stdout(log), redirect_stderr(log):
            result = execute(name, Scenario.from_dict(document), limits, config)
        payload = result.to_dict()
    except Exception as error:
        payload = AlgorithmResult(status="unavailable" if isinstance(error, ImportError) else "error",
                                  diagnostics={"error": f"{type(error).__name__}: {error}"}).to_dict()
    payload["algorithm_seconds"] = time.perf_counter() - started
    if log.getvalue():
        payload["diagnostics"]["log_tail"] = log.getvalue()[-8000:]
    try:
        connection.send(payload)
    finally:
        connection.close()


def run_isolated(name, scenario, limits, config, seed=0, timeout_seconds=60):
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_worker,
                              args=(send, name, scenario.to_dict(), deepcopy(limits), deepcopy(config), seed))
    started = time.perf_counter()
    process.start()
    send.close()
    try:
        if receive.poll(timeout_seconds):
            try:
                result = receive.recv()
            except EOFError:
                result = AlgorithmResult(status="error", diagnostics={"error": "worker exited without a result"}).to_dict()
        else:
            result = AlgorithmResult(status="timeout", diagnostics={"timeout_seconds": timeout_seconds}).to_dict()
        process.join(timeout=0.5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join()
    finally:
        receive.close()
    result["wall_seconds"] = time.perf_counter() - started
    return result


def load_config(path):
    path = Path(path)
    config = mapping(json.loads(path.read_text()), "experiment")
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("experiment schema_version must be 1")
    graph_path = path.parent / config["graph"]
    scenario = Scenario.load(graph_path)
    config.setdefault("resource_model", "quilt")
    get_model(config["resource_model"]).validate(scenario, config.get("limits"))
    if not isinstance(config.get("algorithms"), list) or not config["algorithms"]:
        raise ValueError("algorithms must be a nonempty array")
    config.setdefault("seeds", [0])
    if not isinstance(config["seeds"], list) or not config["seeds"] or any(type(s) is not int for s in config["seeds"]):
        raise ValueError("seeds must be a nonempty array of integers")
    number(config.setdefault("timeout_seconds", 60), "timeout_seconds", positive=True)
    units = mapping(config.get("units"), "units")
    for key in config["limits"]:
        if not isinstance(units.get(key), str) or not units[key]:
            raise ValueError(f"units.{key} must be specified")
    names = []
    for specification in config["algorithms"]:
        mapping(specification, "algorithm")
        name = specification.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("algorithm name must be a nonempty string")
        mapping(specification.setdefault("parameters", {}), "algorithm.parameters")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("algorithm names must be unique within an experiment")
    return config, scenario


def run_experiment(config_path, output):
    config, scenario = load_config(config_path)
    output = Path(output)
    runs = []
    for seed in config["seeds"]:
        for specification in config["algorithms"]:
            name = specification["name"]
            algorithm_config = {"resource_model": config["resource_model"],
                                "parameters": specification["parameters"], "seed": seed}
            result = run_isolated(name, scenario, config["limits"], algorithm_config,
                                  seed, config["timeout_seconds"])
            # An optimizer receives its own graph; evaluation always uses the pristine input.
            report = evaluate(scenario, result["groups"], config["limits"], config["resource_model"])
            if result["status"] not in ("feasible", "optimal") and report["valid"]:
                report["valid"] = False
                report["errors"].append("algorithm did not report a completed feasible solution")
                for key in ("remote_call_cost", "baseline_cost", "cut_edges", "reduction_from_baseline", "absolute_savings"):
                    report.pop(key, None)
            if not report["valid"] and result["status"] in ("feasible", "optimal"):
                result["status"] = "invalid"
            runs.append({"algorithm": name, "seed": seed, **result, "evaluation": report})
            print(f"{name} seed={seed}: {result['status']}; valid={report['valid']}; cost={report.get('remote_call_cost', 'n/a')}")

    optima = {run["seed"]: run["evaluation"]["remote_call_cost"] for run in runs
              if run["status"] == "optimal" and run["evaluation"]["valid"]
              and run["diagnostics"].get("optimality_proven")}
    for run in runs:
        report = run["evaluation"]
        if report["valid"] and run["seed"] in optima:
            optimum = optima[run["seed"]]
            difference = report["remote_call_cost"] - optimum
            report["absolute_gap"] = difference
            report["optimality_gap"] = difference / optimum if optimum else None
    document = {"schema_version": 1, "experiment": config, "graph": scenario.to_dict(), "runs": runs}
    write_json(output / "results.json", document)
    with (output / "summary.csv").open("w", newline="") as handle:
        fields = ["algorithm", "seed", "status", "valid", "remote_call_cost", "reduction_from_baseline",
                  "absolute_savings", "optimality_gap", "absolute_gap", "group_count", "duplicated_placements",
                  "algorithm_seconds", "wall_seconds", "errors"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            flat = {**run, **run["evaluation"]}
            writer.writerow({key: flat.get(key) for key in fields})
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="algorithm_testbench/outputs/latest")
    args = parser.parse_args()
    try:
        run_experiment(args.config, args.output)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
