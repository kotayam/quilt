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


RESOURCE_MODELS = {"quilt": QuiltResources}


def get_model(name):
    if name not in RESOURCE_MODELS:
        raise ValueError(f"unsupported resource model: {name}")
    return RESOURCE_MODELS[name]()
