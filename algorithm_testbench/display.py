"""Human-readable labels; storage values and algorithm inputs are never changed."""

from .model import number


DEFAULT_RESOURCES = (
    "cpu", "memory", "storage_read_ops_per_invocation", "storage_read_bytes_per_invocation",
    "storage_write_ops_per_invocation", "storage_write_bytes_per_invocation",
    "network_external_rx_bytes_per_invocation", "network_external_tx_bytes_per_invocation",
    "gpu_peak_memory_bytes",
)
BYTE_UNITS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB")
MEMORY_SCALES = {unit: 1024 ** index for index, unit in enumerate(BYTE_UNITS)}
MEMORY_SCALES.update({"bytes": 1, "kB": 1000, "KB": 1000, "MB": 1000 ** 2,
                      "GB": 1000 ** 3, "TB": 1000 ** 4})


def value_text(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def format_bytes(value):
    number(value, "byte measurement")
    index = 0
    while value >= 1024 and index < len(BYTE_UNITS) - 1:
        value /= 1024
        index += 1
    # Avoid presenting a rounded boundary as e.g. 1024 MiB.
    if round(value, 2) >= 1024 and index < len(BYTE_UNITS) - 1:
        value /= 1024
        index += 1
    return f"{value_text(value)} {BYTE_UNITS[index]}"


def memory_unit_for(scenario, override=None, result_units=None):
    declared = scenario.metadata.get("units", {})
    if not isinstance(declared, dict):
        raise ValueError("graph units must be an object")
    unit = override if override is not None else (result_units or {}).get("memory", declared.get("memory", "MiB"))
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("memory unit must be a nonempty string")
    return unit


def resource_lines(data, resources, memory_unit):
    """Group storage operations and volumes, respecting explicit field selection."""
    selected = set(resources)
    handled = set()
    for key in resources:
        if key in handled or key not in data:
            continue
        value = data[key]
        if key == "cpu":
            yield f"CPU: {value_text(value)}"
        elif key == "memory":
            amount = (format_bytes(value * MEMORY_SCALES[memory_unit]) if memory_unit in MEMORY_SCALES
                      else f"{value_text(value)} {memory_unit}")
            yield f"Memory: {amount}"
        elif key.startswith("storage_") and key in DEFAULT_RESOURCES:
            direction = key.split("_")[1]
            operations = f"storage_{direction}_ops_per_invocation"
            volume = f"storage_{direction}_bytes_per_invocation"
            parts = []
            if operations in selected and operations in data:
                parts.append(f"{value_text(data[operations])} ops/inv")
            if volume in selected and volume in data:
                parts.append(f"{format_bytes(data[volume])}/inv")
            handled.update((operations, volume))
            yield f"Storage {direction}: {', '.join(parts)}"
        elif key.startswith("network_external_") and key in DEFAULT_RESOURCES:
            direction = "RX" if "_rx_" in key else "TX"
            yield f"Network {direction}: {format_bytes(value)}/inv"
        elif key == "gpu_peak_memory_bytes":
            yield f"GPU memory: {format_bytes(value)}"
        elif key.endswith("_bytes"):
            yield f"{key}: {format_bytes(value)}"
        elif key.endswith("_bytes_per_invocation"):
            yield f"{key}: {format_bytes(value)}/inv"
        else:
            yield f"{key}: {value_text(value)}"


def edge_text(key, value):
    if key == "invocations":
        return f"{value_text(value)} calls/window"
    if key in ("request_bytes", "response_bytes"):
        return f"{format_bytes(value)}/call"
    if key.endswith("_bytes"):
        return format_bytes(value)
    return value_text(value)
