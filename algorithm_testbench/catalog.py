"""Render the curated workflow examples and a self-contained Markdown catalog."""

import argparse
from pathlib import Path

from .display import DEFAULT_RESOURCES, resource_lines
from .model import Scenario
from .profiles import execution_profile_view


WORKFLOWS = ("ml_training", "gpu_image_inference", "compose_post",
             "video_feed_small", "video_feed_large", "video_feed_cache_miss")
EXAMPLES = Path(__file__).parent / "examples" / "workflows"


def load_workflows():
    """Load saved examples without fetching sources or constructing new graphs."""
    return {name: Scenario.load(EXAMPLES / f"{name}.json") for name in WORKFLOWS}


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _profile_comparison(scenario):
    lines = ["", "CPU/GPU implementation alternatives (illustrative, **not optimizer outputs**):", "",
             "| Function | Implementation | Requires GPU | Exclusive service time [ms] | Estimated resources | Reference device |",
             "| --- | --- | --- | --- | --- | --- |"]
    for node, data in scenario.graph.nodes(data=True):
        for implementation, profile in data["attributes"].get("execution_profiles", {}).items():
            resources = "<br>".join(resource_lines(profile["resources"], DEFAULT_RESOURCES, "MiB"))
            lines.append(f"| `{node}` | {implementation} | {profile['requires_gpu']} | "
                         f"{profile['estimated_service_time_ms']} | {_cell(resources)} | {_cell(profile['reference_device'])} |")
    lines.extend(["", "These times exclude nested function calls and queueing; GPU times include",
                  "illustrative launch/transfer overhead. They are not end-to-end latency estimates.", ""])
    return lines


def _execution_notes(scenario):
    lines = ["", "Caller-local execution plan (edge layout alone does not imply this order):", "",
             "| Caller | Ordered actions |", "| --- | --- |"]
    for node, data in scenario.graph.nodes(data=True):
        actions = []
        for step in data["attributes"].get("execution_plan", []):
            if step["kind"] == "parallel":
                actions.append(f"parallel [{', '.join(step['callees'])}], join {step['join']}")
            else:
                actions.append(f"{step['kind']} {step['callee']}" + (" (detached)" if step.get("join") == "none" else ""))
        if actions:
            lines.append(f"| `{node}` | {_cell('; then '.join(actions))} |")
    lines.extend(["", "Shared helper inputs (external volumes are averaged per helper invocation):", "",
                  "| Helper | Caller | Items/call | Response bytes/call |", "| --- | --- | --- | --- |"])
    for node, data in scenario.graph.nodes(data=True):
        for context in data["attributes"].get("call_contexts", []):
            lines.append(f"| `{node}` | `{context['caller']}` | {context['items']} | {context['response_bytes']} |")
    return lines


def render_catalog(output, *, workflow=None, orientation="vertical"):
    from .visualize import visualize

    scenarios = load_workflows()
    if workflow is not None:
        if workflow not in scenarios:
            raise ValueError(f"unknown workflow: {workflow}")
        scenarios = {workflow: scenarios[workflow]}
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be vertical or horizontal")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lines = ["# Curated workflow catalog", "",
             "**Illustrative resource estimates, not measured production data.**", "",
             "These are graph-level examples for exploring merging ideas. Sources establish",
             "patterns or call relationships; all CPU, memory, storage, GPU, and payload budgets",
             "are explicitly estimated. No functions or external services are deployed.", "",
             "Node storage/network volumes are per invocation; CPU is an illustrative active-core",
             "proxy and memory is peak host memory. Neither is an allocated capacity or a measured",
             "duration. Edges show direct calls in these adaptations, not generic data dependencies.", "",
             "The payload view labels request bodies; response budgets remain in JSON and the table.", "",
             "External stores remain external after merging. Quilt's evaluator currently enforces",
             "only CPU and memory, so its validity result says nothing about GPU or I/O feasibility.", ""]
    for name, scenario in scenarios.items():
        provenance = scenario.metadata["provenance"]
        if provenance.get("resource_status") != "illustrative_estimates":
            raise ValueError(f"{name}: expected explicit illustrative-estimate provenance")
        scenario.save(output / f"{name}.json")
        visualize(scenario, output / name, orientation=orientation)
        visualize(scenario, output / f"{name}_payloads", orientation=orientation, edge_attribute="request_bytes")
        has_profiles = any("execution_profiles" in data["attributes"] for _, data in scenario.graph.nodes(data=True))
        lines.extend([f"## {name.replace('_', ' ').title()}", "", scenario.metadata["description"], "",
                      f"Structure: `{provenance['structure_status']}`. N = {scenario.metadata['workflow_invocations']}.", "",
                      f"[JSON]({name}.json) · [Call graph SVG]({name}.svg) · [PNG]({name}.png) · "
                      f"[Request-payload SVG]({name}_payloads.svg) · [PNG]({name}_payloads.png)", "",
                      f"![{name} — illustrative resource estimates]({name}.png)", ""])
        if has_profiles:
            gpu_view = execution_profile_view(scenario, "gpu")
            gpu_view.save(output / f"{name}_gpu.json")
            visualize(gpu_view, output / f"{name}_gpu", orientation=orientation)
            visualize(scenario, output / f"{name}_responses", orientation=orientation, edge_attribute="response_bytes")
            lines.extend(["The ordinary graph is the CPU baseline. The GPU view selects both available",
                          "GPU alternatives for illustration; it is **not a placement or merging decision**.", "",
                          f"[GPU-view JSON]({name}_gpu.json) · [GPU-view SVG]({name}_gpu.svg) · [PNG]({name}_gpu.png) · "
                          f"[Response-payload SVG]({name}_responses.svg) · [PNG]({name}_responses.png)", "",
                          "Inspect response payloads too: feature tensors are returned by shared helpers,",
                          "so request-only views understate the communication opportunity.", ""])
        lines.extend(["Sources:", ""])
        for source in provenance["sources"]:
            if "url" in source:
                lines.append(f"- [{source['label']}]({source['url']})")
            else:
                lines.append(f"- {source['label']}: repository path `{source['path']}`")
        lines.extend(["", "Stage profiles" + (" (CPU baseline)" if has_profiles else "") + ":", "",
                      "| Function | Role | External services | Estimated resources |" + (" Exclusive service time [ms] |" if has_profiles else ""),
                      "| --- | --- | --- | --- |" + (" --- |" if has_profiles else "")])
        for node, data in scenario.graph.nodes(data=True):
            attributes = data["attributes"]
            resources = "<br>".join(resource_lines(data["resources"], DEFAULT_RESOURCES, "MiB"))
            services = ", ".join(attributes.get("external_services", [])) or "None"
            lines.append(f"| `{node}` | {_cell(attributes['role'])} | {_cell(services)} | {_cell(resources)} |"
                         + (f" {attributes['estimated_service_time_ms']:.3f} |" if has_profiles else ""))
        if has_profiles:
            lines.extend(_profile_comparison(scenario))
            lines.extend(_execution_notes(scenario))
        lines.extend(["", "Call payload budgets (bytes/call):", "",
                      "| Caller | Callee | Type | Requests/window | Request bytes | Response bytes |",
                      "| --- | --- | --- | --- | --- | --- |"])
        for source, target, data in scenario.graph.edges(data=True):
            attributes = data["attributes"]
            lines.append(f"| `{source}` | `{target}` | {data['type']} | {data['invocations']} | "
                         f"{attributes['request_bytes']} | {attributes['response_bytes']} |")
        lines.extend(["", "Assumptions:", ""])
        lines.extend(f"- {assumption}" for assumption in scenario.metadata["assumptions"])
        lines.extend(["", "Merging questions:", ""])
        lines.extend(f"- {question}" for question in scenario.metadata["merging_questions"])
        lines.append("")
    index = output / "index.md"
    index.write_text("\n".join(lines) + "\n")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="algorithm_testbench/outputs/workflow_catalog")
    parser.add_argument("--workflow", choices=WORKFLOWS, help="render only this example")
    parser.add_argument("--orientation", choices=("vertical", "horizontal"), default="vertical")
    args = parser.parse_args()
    try:
        index = render_catalog(args.output, workflow=args.workflow, orientation=args.orientation)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Saved catalog: {index}")


if __name__ == "__main__":
    main()
