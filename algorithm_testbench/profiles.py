"""Materialize execution-profile copies for explanation, never optimizer results."""

from copy import deepcopy

from .model import Scenario, mapping, number


def execution_profile_view(scenario, profile="gpu"):
    if profile not in ("cpu", "gpu"):
        raise ValueError("execution profile must be cpu or gpu")
    document = scenario.to_dict()
    selected = []
    for node in document["nodes"]:
        profiles = node["attributes"].get("execution_profiles")
        if profiles is None:
            continue
        mapping(profiles, "execution_profiles")
        alternative = mapping(profiles.get(profile), f"execution_profiles.{profile}")
        if type(alternative.get("requires_gpu")) is not bool:
            raise ValueError("profile requires_gpu must be a boolean")
        number(alternative.get("estimated_service_time_ms"), "estimated_service_time_ms")
        resources = mapping(alternative.get("resources"), "profile resources")
        if not {"cpu", "memory"} <= resources.keys():
            raise ValueError("execution profiles require cpu and memory resources")
        node["resources"] = deepcopy(resources)
        node["attributes"]["requires_gpu"] = alternative["requires_gpu"]
        node["attributes"]["estimated_service_time_ms"] = alternative["estimated_service_time_ms"]
        node["attributes"]["selected_execution_profile"] = profile
        selected.append(node["id"])
    if not selected:
        raise ValueError("scenario has no execution-profile alternatives")
    if "CPU baseline" in document["name"]:
        document["name"] = document["name"].replace("CPU baseline", f"{profile.upper()} alternative view")
        document["name"] += " — not optimizer output"
    else:
        document["name"] += f" — {profile.upper()} alternative view (not optimizer output)"
    document["execution_profile_view"] = {"profile": profile, "selected_nodes": selected,
                                          "optimizer_output": False}
    return Scenario.from_dict(document)
