"""Human-readable labels; storage values and algorithm inputs are never changed."""

from .model import number


DEFAULT_RESOURCES = (
    "cpu", "memory", "storage_read_bytes_per_invocation", "storage_write_bytes_per_invocation",
    "network_external_rx_bytes_per_invocation", "network_external_tx_bytes_per_invocation",
    "gpu_peak_memory_bytes",
)
BYTE_UNITS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB")
MEMORY_SCALES = {unit: 1024 ** index for index, unit in enumerate(BYTE_UNITS)}
MEMORY_SCALES.update({"bytes": 1, "kB": 1000, "KB": 1000, "MB": 1000 ** 2,
                      "GB": 1000 ** 3, "TB": 1000 ** 4})
PAIRED_RESOURCES = (
    ("Storage R/W", "storage_read_bytes_per_invocation", "storage_write_bytes_per_invocation"),
    ("Network RX/TX", "network_external_rx_bytes_per_invocation", "network_external_tx_bytes_per_invocation"),
    ("Storage ops R/W", "storage_read_ops_per_invocation", "storage_write_ops_per_invocation"),
)


def value_text(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _scaled_bytes(value):
    number(value, "byte measurement")
    index = 0
    while value >= 1024 and index < len(BYTE_UNITS) - 1:
        value /= 1024
        index += 1
    # Avoid presenting a rounded boundary as e.g. 1024 MiB.
    if round(value, 2) >= 1024 and index < len(BYTE_UNITS) - 1:
        value /= 1024
        index += 1
    return value, index


def format_byte_parts(value):
    value, index = _scaled_bytes(value)
    return value_text(value), BYTE_UNITS[index]


def format_bytes(value):
    return " ".join(format_byte_parts(value))


def _paired_line(label, values, *, operations=False):
    if operations:
        unit, scale = "ops", 1
    else:
        _, index = _scaled_bytes(max(value for value in values if value is not None))
        unit, scale = BYTE_UNITS[index], 1024 ** index
    parts = []
    for value in values:
        if value is None:
            parts.append("—")
        else:
            scaled = value / scale
            # A large shared unit must not turn a nonzero member into an apparent zero.
            parts.append("<0.01" if 0 < scaled < 0.005 else value_text(scaled))
    return f"{label} [{unit}]: {' / '.join(parts)}"


def memory_unit_for(scenario, override=None, result_units=None):
    declared = scenario.metadata.get("units", {})
    if not isinstance(declared, dict):
        raise ValueError("graph units must be an object")
    unit = override if override is not None else (result_units or {}).get("memory", declared.get("memory", "MiB"))
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("memory unit must be a nonempty string")
    return unit


def resource_lines(data, resources, memory_unit):
    """Compact paired volumes with shared units; operation counts are opt-in."""
    selected = set(resources)
    handled = set()
    for key in resources:
        if key in handled or key not in data:
            continue
        value = data[key]
        if key == "cpu":
            yield f"CPU: {value_text(value)}"
        elif key == "memory":
            amount, unit = (format_byte_parts(value * MEMORY_SCALES[memory_unit]) if memory_unit in MEMORY_SCALES
                            else (value_text(value), memory_unit))
            yield f"Memory [{unit}]: {amount}"
        elif any(key in (left, right) for _, left, right in PAIRED_RESOURCES):
            label, left, right = next(pair for pair in PAIRED_RESOURCES if key in pair[1:])
            values = [data.get(member) if member in selected else None for member in (left, right)]
            handled.update((left, right))
            yield _paired_line(label, values, operations=left.endswith("_ops_per_invocation"))
        elif key == "gpu_peak_memory_bytes":
            amount, unit = format_byte_parts(value)
            yield f"GPU memory [{unit}]: {amount}"
        elif key.endswith(("_bytes", "_bytes_per_invocation")):
            amount, unit = format_byte_parts(value)
            yield f"{key} [{unit}]: {amount}"
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
