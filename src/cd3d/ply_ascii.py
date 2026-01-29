from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PlyVertexLayout:
    vertex_count: int
    properties: list[tuple[str, str]]  # (name, type)

    def index_of(self, name: str) -> int | None:
        for idx, (prop_name, _) in enumerate(self.properties):
            if prop_name == name:
                return idx
        return None

    def dtype_of(self, name: str) -> str | None:
        for prop_name, prop_type in self.properties:
            if prop_name == name:
                return prop_type
        return None


_FLOAT_TYPES = {"float", "float32", "double", "float64"}
_INT_TYPES = {"char", "uchar", "short", "ushort", "int", "uint", "int32", "uint32"}


def read_ply_ascii_vertex_layout(path: Path) -> PlyVertexLayout:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()
        if first != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        fmt = None
        vertex_count = None
        in_vertex = False
        props: list[tuple[str, str]] = []

        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading PLY header: {path}")
            line = line.strip()
            if line.startswith("format "):
                fmt = line.split(" ", 1)[1]
            elif line.startswith("element "):
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "vertex":
                    vertex_count = int(parts[2])
                    in_vertex = True
                else:
                    in_vertex = False
            elif in_vertex and line.startswith("property "):
                parts = line.split()
                if len(parts) < 3:
                    continue
                prop_type = parts[1]
                prop_name = parts[-1]
                props.append((prop_name, prop_type))
            elif line == "end_header":
                break

        if fmt is None:
            raise ValueError(f"Missing PLY format in header: {path}")
        if not fmt.startswith("ascii"):
            raise ValueError(f"Only ASCII PLY is supported (got {fmt}): {path}")
        if vertex_count is None:
            raise ValueError(f"Missing vertex element in PLY header: {path}")

        return PlyVertexLayout(vertex_count=vertex_count, properties=props)


def read_ply_ascii_vertices(
    path: Path,
    fields: Iterable[str],
) -> dict[str, np.ndarray]:
    """
    Reads selected per-vertex fields from an ASCII PLY file.

    Returns a dict mapping field -> numpy array (length N).
    """
    fields = list(fields)
    layout = read_ply_ascii_vertex_layout(path)

    field_indices: dict[str, int] = {}
    field_types: dict[str, str] = {}
    for name in fields:
        idx = layout.index_of(name)
        if idx is None:
            raise KeyError(f"Missing PLY vertex property '{name}' in {path}")
        field_indices[name] = idx
        field_types[name] = layout.dtype_of(name) or "float"

    out: dict[str, np.ndarray] = {}
    for name in fields:
        typ = field_types[name]
        if typ in _FLOAT_TYPES:
            out[name] = np.empty((layout.vertex_count,), dtype=np.float64)
        elif typ in _INT_TYPES:
            out[name] = np.empty((layout.vertex_count,), dtype=np.int64)
        else:
            # Default to float if we do not recognize the type.
            out[name] = np.empty((layout.vertex_count,), dtype=np.float64)

    with path.open("r", encoding="utf-8", errors="replace") as f:
        # Skip header.
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while skipping PLY header: {path}")
            if line.strip() == "end_header":
                break

        prop_count = len(layout.properties)
        for i in range(layout.vertex_count):
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading vertices ({i}/{layout.vertex_count}): {path}")
            parts = line.strip().split()
            if len(parts) < prop_count:
                raise ValueError(
                    f"Invalid vertex row at {i}: expected >= {prop_count} values, got {len(parts)} ({path})"
                )
            for name, idx in field_indices.items():
                if out[name].dtype.kind in {"i", "u"}:
                    out[name][i] = int(float(parts[idx]))
                else:
                    out[name][i] = float(parts[idx])

    return out


def read_3rscan_instance_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads `labels.instances.annotated.v2.ply` as:
      - points: (N, 3) float64
      - object_ids: (N,) int64
    """
    data = read_ply_ascii_vertices(path, fields=("x", "y", "z", "objectId"))
    points = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64, copy=False)
    object_ids = data["objectId"].astype(np.int64, copy=False)
    return points, object_ids


def write_ply_ascii(
    path: Path,
    *,
    points: np.ndarray,
    colors_rgb: np.ndarray | None = None,
    extra_properties: dict[str, np.ndarray] | None = None,
    comments: list[str] | None = None,
) -> None:
    """
    Writes an ASCII PLY with vertex properties:
      - x, y, z (float)
      - optional red, green, blue (uchar)
      - optional extra properties (int/float inferred from dtype)
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    n = points.shape[0]

    if colors_rgb is not None:
        colors_rgb = np.asarray(colors_rgb)
        if colors_rgb.shape != (n, 3):
            raise ValueError("colors_rgb must have shape (N, 3)")
        if colors_rgb.dtype != np.uint8:
            colors_rgb = np.clip(colors_rgb, 0, 255).astype(np.uint8)

    extra_properties = extra_properties or {}
    for k, v in extra_properties.items():
        v = np.asarray(v)
        if v.shape[0] != n:
            raise ValueError(f"extra property '{k}' must have length N={n}")

    comments = comments or []

    lines: list[str] = []
    lines.append("ply")
    lines.append("format ascii 1.0")
    for c in comments:
        lines.append(f"comment {c}")
    lines.append(f"element vertex {n}")
    lines.append("property float x")
    lines.append("property float y")
    lines.append("property float z")
    if colors_rgb is not None:
        lines.append("property uchar red")
        lines.append("property uchar green")
        lines.append("property uchar blue")

    extra_keys = sorted(extra_properties.keys())
    for key in extra_keys:
        v = np.asarray(extra_properties[key])
        if v.dtype.kind in {"i", "u"}:
            lines.append(f"property int {key}")
        else:
            lines.append(f"property float {key}")

    lines.append("end_header")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        for i in range(n):
            x, y, z = points[i]
            row = [f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"]
            if colors_rgb is not None:
                r, g, b = colors_rgb[i]
                row.extend([str(int(r)), str(int(g)), str(int(b))])
            for key in extra_keys:
                v = np.asarray(extra_properties[key])[i]
                if np.asarray(extra_properties[key]).dtype.kind in {"i", "u"}:
                    row.append(str(int(v)))
                else:
                    row.append(f"{float(v):.6f}")
            f.write(" ".join(row) + "\n")

