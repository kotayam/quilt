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
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox
import networkx as nx

from .model import Scenario
from .display import DEFAULT_RESOURCES, edge_text, memory_unit_for, resource_lines, value_text


def layered_positions(graph, orientation="vertical", node_size=(2.5, 1.0)):
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
            node_width, node_height = node_size
            across = index - (len(layer) - 1) / 2
            order[node] = across
            positions[node] = ((across * (node_width + 0.75), -depth * (node_height + 1.0))
                               if orientation == "vertical"
                               else (depth * (node_width + 1.4), -across * (node_height + 0.6)))
    return positions


def _label(node, data, resources, attributes, hide_labels, memory_unit="MiB"):
    lines = [textwrap.fill(node, width=25, break_long_words=True)]
    if not hide_labels:
        lines.extend(resource_lines(data["resources"], resources, memory_unit))
        for key in attributes:
            if key in data.get("attributes", {}):
                label = "GPU required" if key == "requires_gpu" else key
                lines.append(textwrap.fill(f"{label}: {value_text(data['attributes'][key])}", 30))
    return "\n".join(lines)


def _measure_labels(labels):
    """Measure actual font extents in inches, including room for box padding."""
    fig = plt.figure()
    try:
        artists = [fig.text(0, 0, label, fontsize=10, fontweight="bold") for label in labels]
        fig.canvas.draw()
        extents = [artist.get_window_extent() for artist in artists]
        return (max(extent.width for extent in extents) / fig.dpi + 0.3,
                max(extent.height for extent in extents) / fig.dpi + 0.3)
    finally:
        plt.close(fig)


def _panel_bounds(positions, node_size):
    xs, ys = zip(*positions.values())
    width, height = node_size
    return (min(xs) - width / 2 - 0.35, max(xs) + width / 2 + 0.35,
            min(ys) - height / 2 - 0.35, max(ys) + height / 2 + 0.35)


def _curve_point(ax, start, end, rad, fraction):
    """Evaluate the same display-space quadratic used by arc3 connections."""
    x0, y0 = ax.transData.transform(start)
    x1, y1 = ax.transData.transform(end)
    control = ((x0 + x1) / 2 + rad * (y1 - y0),
               (y0 + y1) / 2 - rad * (x1 - x0))
    t = fraction
    return ((1 - t) ** 2 * x0 + 2 * t * (1 - t) * control[0] + t ** 2 * x1,
            (1 - t) ** 2 * y0 + 2 * t * (1 - t) * control[1] + t ** 2 * y1)


def _draw(ax, graph, root, title, labels, positions, edge_attribute, hide_labels, color):
    boxes = {}
    for node, (x, y) in positions.items():
        is_root = node == root
        label = labels[node]
        boxes[node] = ax.text(x, y, label, ha="center", va="center", fontsize=10,
                              fontweight="bold" if is_root else "normal", zorder=3,
                              bbox={"boxstyle": "round,pad=0.65", "facecolor": "#dcfce7" if is_root else color,
                                    "edgecolor": "#15803d" if is_root else "#64748b",
                                    "linewidth": 2.0 if is_root else 1.1})
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
                label = textwrap.fill(edge_text(edge_attribute, value), 24)
                renderer = ax.figure.canvas.get_renderer()
                font = FontProperties(size=9)
                label_width = max(renderer.get_text_width_height_descent(line, font, False)[0]
                                  for line in label.splitlines())
                label_height = len(label.splitlines()) * 9 * ax.figure.dpi / 72 * 1.2
                all_boxes = [patch.get_window_extent().padded(4) for patch in patches.values()]
                candidates = [_curve_point(ax, start, end, rad, t) for t in (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8)]
                def label_collisions(point):
                    bounds = Bbox.from_bounds(point[0] - label_width / 2 - 4,
                                              point[1] - label_height / 2 - 4,
                                              label_width + 8, label_height + 8)
                    return sum(bounds.overlaps(box) for box in all_boxes)
                point = min(candidates, key=label_collisions)
                x, y = ax.transData.inverted().transform(point)
                ax.text(x, y, label, ha="center", va="center", fontsize=9,
                        color="#334155", zorder=2,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2})


def visualize(scenario, output, *, resources=None, node_attributes=None,
              edge_attribute="invocations", hide_labels=False, orientation="vertical", groups=None,
              memory_unit=None, result_units=None):
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be vertical or horizontal")
    resources = list(DEFAULT_RESOURCES if resources is None else resources)
    node_attributes = list(() if node_attributes is None else node_attributes)
    memory_unit = memory_unit_for(scenario, memory_unit, result_units)
    labels = {node: _label(node, data, resources, node_attributes, hide_labels, memory_unit)
              for node, data in scenario.graph.nodes(data=True)}
    panels = [(scenario.entry, scenario.graph, "Workflow")]
    if groups is not None:
        panels = []
        for root, members in sorted(groups.items()):
            if root not in scenario.graph or root not in members or set(members) - set(scenario.graph):
                raise ValueError("result contains unknown nodes or a missing group root")
            panels.append((root, scenario.graph.subgraph(members), f"Group rooted at {root}"))
        if not panels:
            raise ValueError("result contains no groups")
    node_size = _measure_labels(labels.values())
    layouts = [layered_positions(graph, orientation, node_size) for _, graph, _ in panels]
    bounds = [_panel_bounds(positions, node_size) for positions in layouts]
    panel_width = max(4.5, max(right - left for left, right, _, _ in bounds))
    panel_height = max(2.5, max(top - bottom for _, _, bottom, top in bounds))
    columns = min(3, len(panels))
    rows = math.ceil(len(panels) / columns)
    width = columns * panel_width + (columns - 1) * 0.6 + 0.8
    graph_title = textwrap.fill(
        f"{scenario.metadata.get('name', 'Call graph')}  |  N = {scenario.metadata['workflow_invocations']} workflow invocations",
        width=max(35, int(width * 8)))
    header_height = 0.5 + 0.27 * len(graph_title.splitlines())
    # Narrow chain graphs stack the three legend entries; reserve enough space
    # to keep their top entry below the edge-description text.
    footer_height = 1.25 if width < 6 else 0.9
    height = rows * (panel_height + 0.5) + (rows - 1) * 0.6 + header_height + footer_height
    fig = plt.figure(figsize=(width, height))
    colors = ["#eff6ff", "#fff7ed", "#f5f3ff", "#fdf2f8", "#ecfeff"]
    try:
        for index, (root, graph, title) in enumerate(panels):
            row, column = divmod(index, columns)
            bottom = height - header_height - (row + 1) * (panel_height + 0.5) - row * 0.6
            ax = fig.add_axes(((0.4 + column * (panel_width + 0.6)) / width,
                               bottom / height, panel_width / width, panel_height / height))
            left, right, lower, upper = bounds[index]
            center_x, center_y = (left + right) / 2, (lower + upper) / 2
            ax.set_xlim(center_x - panel_width / 2, center_x + panel_width / 2)
            ax.set_ylim(center_y - panel_height / 2, center_y + panel_height / 2)
            _draw(ax, graph, root, title, labels, layouts[index], edge_attribute,
                  hide_labels, colors[index % len(colors)])
        fig.suptitle(graph_title, fontsize=16, fontweight="bold", y=1 - 0.12 / height)
        legend = [Line2D([0], [0], color="#475569", label="Synchronous call"),
                  Line2D([0], [0], color="#9333ea", linestyle="--", label="Asynchronous call"),
                  Line2D([0], [0], marker="s", color="#15803d", linestyle="none", label="Entry / group root")]
        fig.legend(handles=legend, loc="lower center", ncol=3 if width >= 6 else 1, frameon=False,
                   fontsize=9, bbox_to_anchor=(0.5, 0.03 / height))
        edge_description = {"invocations": "calls per profiling window", "request_bytes": "request payload per call",
                            "response_bytes": "response payload per call"}.get(edge_attribute, edge_attribute)
        fig.text(0.5, (footer_height - 0.25) / height, textwrap.fill(
            f"Edge labels: {edge_description}" + (" | Shared functions repeat across group panels" if groups else ""),
            width=max(40, int(width * 12))),
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
    parser.add_argument("--resources", help="comma-separated resource keys; default: CPU, memory, storage/network volumes, GPU memory")
    parser.add_argument("--node-attributes", help="comma-separated attribute keys; default: none")
    parser.add_argument("--memory-unit", help="source unit of existing memory values; overrides declared units (default: MiB)")
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
        result_units = None
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
            result_units = result.get("experiment", {}).get("units", {})
        paths = visualize(scenario, args.output,
                          resources=None if args.resources is None else [r for r in args.resources.split(",") if r],
                          node_attributes=None if args.node_attributes is None else [r for r in args.node_attributes.split(",") if r],
                          edge_attribute=args.edge_attribute, hide_labels=args.hide_labels,
                          orientation=args.orientation, groups=groups, memory_unit=args.memory_unit,
                          result_units=result_units)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
