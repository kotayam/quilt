"""Render saved graphs and optional merge groups to SVG/PNG, without Graphviz."""

import argparse
import json
import math
import os
from pathlib import Path
import textwrap

# Keep Matplotlib's cache in a writable location and work without a display.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx

from .model import Scenario


def layered_positions(graph, orientation="vertical"):
    """Stable topological layers, with predecessor barycenters reducing crossings."""
    positions = {}
    order = {}
    for depth, generation in enumerate(nx.topological_generations(graph)):
        def sort_key(node):
            predecessors = list(graph.predecessors(node))
            center = sum(order[p] for p in predecessors) / len(predecessors) if predecessors else 0
            return center, node
        layer = sorted(generation, key=sort_key)
        for index, node in enumerate(layer):
            across = (index - (len(layer) - 1) / 2) * 4.0
            order[node] = across
            positions[node] = (across, -depth * 3.2) if orientation == "vertical" else (depth * 5.5, -across)
    return positions


def _value(value):
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def _label(node, data, resources, attributes, hide_labels):
    lines = [textwrap.fill(node, width=25, break_long_words=True)]
    if not hide_labels:
        for key in resources:
            if key in data["resources"]:
                lines.append(f"{key}: {_value(data['resources'][key])}")
        for key in attributes:
            if key in data.get("attributes", {}):
                lines.append(textwrap.fill(f"{key}: {_value(data['attributes'][key])}", 30))
    return "\n".join(lines)


def _curve_point(ax, start, end, rad, fraction):
    """Evaluate the same display-space quadratic used by arc3 connections."""
    x0, y0 = ax.transData.transform(start)
    x1, y1 = ax.transData.transform(end)
    control = ((x0 + x1) / 2 + rad * (y1 - y0),
               (y0 + y1) / 2 - rad * (x1 - x0))
    t = fraction
    return ((1 - t) ** 2 * x0 + 2 * t * (1 - t) * control[0] + t ** 2 * x1,
            (1 - t) ** 2 * y0 + 2 * t * (1 - t) * control[1] + t ** 2 * y1)


def _draw(ax, graph, root, title, resources, node_attributes, edge_attribute,
          hide_labels, orientation, color):
    positions = layered_positions(graph, orientation)
    boxes = {}
    for node, (x, y) in positions.items():
        is_root = node == root
        label = _label(node, graph.nodes[node], resources, node_attributes, hide_labels)
        boxes[node] = ax.text(x, y, label, ha="center", va="center", fontsize=10,
                              fontweight="bold" if is_root else "normal", zorder=3,
                              bbox={"boxstyle": "round,pad=0.65", "facecolor": "#dcfce7" if is_root else color,
                                    "edgecolor": "#15803d" if is_root else "#64748b",
                                    "linewidth": 2.0 if is_root else 1.1})
    xs, ys = zip(*positions.values())
    ax.set_xlim(min(xs) - 2.3, max(xs) + 2.3)
    ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
    ax.set_title(title, fontsize=13, pad=20, fontweight="bold")
    ax.axis("off")
    # Measure labels once, then use data-space patches. Clipping arrows against
    # cached text boxes would leave heads under nodes when saving at another DPI.
    ax.figure.canvas.draw()
    patches = {}
    for node, text in boxes.items():
        bounds = text.get_bbox_patch().get_window_extent().transformed(ax.transData.inverted())
        root_node = node == root
        patch = FancyBboxPatch((bounds.x0, bounds.y0), bounds.width, bounds.height,
                               boxstyle="round,pad=0.03", zorder=2,
                               facecolor="#dcfce7" if root_node else color,
                               edgecolor="#15803d" if root_node else "#64748b",
                               linewidth=2 if root_node else 1.1)
        ax.add_patch(patch)
        patches[node] = patch
        text.set_bbox(None)
    for source, target, data in graph.edges(data=True):
        start, end = positions[source], positions[target]
        obstacles = [patch.get_window_extent().padded(8) for node, patch in patches.items()
                     if node not in (source, target)]
        radii = (0, 0.18, -0.18, 0.35, -0.35, 0.55, -0.55, 0.8, -0.8, 1.1, -1.1)
        # Prefer a straight edge; bend around intervening nodes on long edges.
        def intersections(radius):
            points = [_curve_point(ax, start, end, radius, t / 30) for t in range(3, 28)]
            return sum(box.contains(*point) for box in obstacles for point in points)
        rad, best_count = 0, math.inf
        for radius in radii:
            count = intersections(radius)
            if count < best_count:
                rad, best_count = radius, count
            if count == 0:
                break
        asynchronous = data["type"] == "async"
        arrow = FancyArrowPatch(start, end, patchA=patches[source],
                                patchB=patches[target], shrinkA=5, shrinkB=5,
                                arrowstyle="-|>", mutation_scale=15,
                                connectionstyle=f"arc3,rad={rad}",
                                linestyle="--" if asynchronous else "-",
                                color="#9333ea" if asynchronous else "#475569", linewidth=1.5,
                                zorder=1, clip_on=False)
        ax.add_patch(arrow)
        if not hide_labels:
            value = data.get(edge_attribute, data.get("attributes", {}).get(edge_attribute))
            if value is not None:
                all_boxes = [patch.get_window_extent().padded(14) for patch in patches.values()]
                candidates = [_curve_point(ax, start, end, rad, t) for t in (0.5, 0.4, 0.6, 0.3, 0.7)]
                point = min(candidates, key=lambda p: sum(box.contains(*p) for box in all_boxes))
                x, y = ax.transData.inverted().transform(point)
                ax.text(x, y, textwrap.fill(_value(value), 22), ha="center", va="center", fontsize=9,
                        color="#334155", zorder=2,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2})


def visualize(scenario, output, *, resources=("cpu", "memory"), node_attributes=(),
              edge_attribute="invocations", hide_labels=False, orientation="vertical", groups=None):
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be vertical or horizontal")
    panels = [(scenario.entry, scenario.graph, "Workflow")]
    if groups is not None:
        panels = []
        for root, members in sorted(groups.items()):
            if root not in scenario.graph or root not in members or set(members) - set(scenario.graph):
                raise ValueError("result contains unknown nodes or a missing group root")
            panels.append((root, scenario.graph.subgraph(members), f"Group rooted at {root}"))
        if not panels:
            raise ValueError("result contains no groups")
    max_layers = max(len(list(nx.topological_generations(graph))) for _, graph, _ in panels)
    max_width = max(max(map(len, nx.topological_generations(graph))) for _, graph, _ in panels)
    columns = min(3, len(panels))
    rows = math.ceil(len(panels) / columns)
    width, height = max(6.5, max_width * 3.8), max(4.5, max_layers * 2.0)
    if orientation == "horizontal":
        width, height = max(7, max_layers * 3.4), max(4.5, max_width * 2.0)
    fig, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows), squeeze=False)
    fig.subplots_adjust(top=0.86, bottom=0.13, hspace=0.35, wspace=0.25)
    colors = ["#eff6ff", "#fff7ed", "#f5f3ff", "#fdf2f8", "#ecfeff"]
    try:
        for index, (root, graph, title) in enumerate(panels):
            _draw(axes.flat[index], graph, root, title, resources, node_attributes,
                  edge_attribute, hide_labels, orientation, colors[index % len(colors)])
        for ax in list(axes.flat)[len(panels):]:
            ax.axis("off")
        fig.suptitle(textwrap.fill(
            f"{scenario.metadata.get('name', 'Call graph')}  |  N = {scenario.metadata['workflow_invocations']}",
            width=max(45, int(width * columns * 9))), fontsize=16, fontweight="bold", y=0.99)
        legend = [Line2D([0], [0], color="#475569", label="Synchronous call"),
                  Line2D([0], [0], color="#9333ea", linestyle="--", label="Asynchronous call"),
                  Line2D([0], [0], marker="s", color="#15803d", linestyle="none", label="Entry / group root")]
        fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
        fig.text(0.5, 0.052, f"Edge labels: {edge_attribute}" + (" | Shared functions repeat across group panels" if groups else ""),
                 ha="center", fontsize=9, color="#475569")
        output = Path(output)
        if output.suffix.lower() not in ("", ".svg", ".png"):
            raise ValueError("output must have .svg/.png extension, or no extension to write both")
        paths = [output] if output.suffix else [output.with_suffix(".svg"), output.with_suffix(".png")]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        return paths
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph")
    parser.add_argument("--output", required=True, help=".svg, .png, or extensionless stem for both")
    parser.add_argument("--resources", default="cpu,memory", help="comma-separated resource keys")
    parser.add_argument("--node-attributes", default="", help="comma-separated attribute keys")
    parser.add_argument("--edge-attribute", default="invocations")
    parser.add_argument("--hide-labels", action="store_true", help="show IDs but hide resource and edge labels")
    parser.add_argument("--orientation", choices=["vertical", "horizontal"], default="vertical")
    parser.add_argument("--result", help="results.json from a benchmark run")
    parser.add_argument("--algorithm", help="algorithm name to select from --result")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        scenario = Scenario.load(args.graph)
        groups = None
        if args.result:
            result = json.loads(Path(args.result).read_text())
            if result["graph"] != scenario.to_dict():
                raise ValueError("result was produced for a different graph")
            matches = [run for run in result["runs"] if run["seed"] == args.seed
                       and (args.algorithm is None or run["algorithm"] == args.algorithm)]
            if len(matches) != 1:
                raise ValueError("select exactly one result using --algorithm and --seed")
            if not matches[0]["evaluation"]["valid"]:
                raise ValueError("selected result is invalid or has no feasible assignment")
            groups = matches[0]["groups"]
        paths = visualize(scenario, args.output, resources=[r for r in args.resources.split(",") if r],
                          node_attributes=[r for r in args.node_attributes.split(",") if r],
                          edge_attribute=args.edge_attribute, hide_labels=args.hide_labels,
                          orientation=args.orientation, groups=groups)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
