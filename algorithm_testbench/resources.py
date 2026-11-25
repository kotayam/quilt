"""Resource policies are independent of graph storage and optimization."""

from .model import mapping, number


class QuiltResources:
    """The original equations, including their inherited-concurrency limitation."""

    name = "quilt"
    supported = {"cpu", "memory"}

    def validate(self, scenario, limits):
        mapping(limits, "limits")
        if set(limits) != self.supported:
            raise ValueError("quilt model requires exactly cpu and memory limits; other constraints are unsupported")
        for key, value in limits.items():
            number(value, f"limits.{key}", positive=True)
        for node, data in scenario.graph.nodes(data=True):
            if not self.supported <= data["resources"].keys():
                raise ValueError(f"{node} requires cpu and memory resources for the quilt model")

    def usage(self, scenario, root, members):
        graph = scenario.graph
        usage = {key: graph.nodes[root]["resources"][key] for key in self.supported}
        for source, target, data in graph.edges(data=True):
            if source in members and target in members:
                alpha = scenario.alpha(source, target)
                resources = graph.nodes[target]["resources"]
                usage["cpu"] += alpha * resources["cpu"]
                usage["memory"] += (alpha if data["type"] == "async" else 1) * resources["memory"]
        return usage


class RecursiveResources(QuiltResources):
    """Bottom-up memory with caller-relative average multiplicities."""
    name = "recursive"

    def usage(self, scenario, root, members):
        from copy import deepcopy
        from merge_solver.memory_inputs import prepare_memory_inputs
        from merge_solver.memory_model_template import estimate_memory
        graph = deepcopy(scenario.graph)
        for _, data in graph.nodes(data=True):
            data["m"] = data["resources"]["memory"]
        for _, _, data in graph.edges(data=True):
            data["weight"] = data["invocations"]
        prepare_memory_inputs(graph, scenario.metadata["workflow_invocations"])
        usage = super().usage(scenario, root, members)
        usage["memory"] = estimate_memory(graph, root, members)
        return usage


RESOURCE_MODELS = {"quilt": QuiltResources, "recursive": RecursiveResources}


def get_model(name):
    if name not in RESOURCE_MODELS:
        raise ValueError(f"unsupported resource model: {name}")
    return RESOURCE_MODELS[name]()
