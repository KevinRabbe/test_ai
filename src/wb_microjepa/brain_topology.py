from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import numpy as np
import torch


REGION_TO_ID = {
    "cortex": 0,
    "thalamus": 1,
    "hippocampus": 2,
    "basal_ganglia": 3,
    "cerebellum": 4,
    "brainstem": 5,
}

ID_TO_REGION = {v: k for k, v in REGION_TO_ID.items()}


@dataclass(frozen=True)
class BrainUnit:
    unit_id: int
    region: str
    x: float
    y: float
    z: float


@dataclass
class BrainGraph:
    units: List[BrainUnit]
    neighbor_indices: torch.Tensor
    neighbor_mask: torch.Tensor
    coords: torch.Tensor
    region_ids: torch.Tensor

    @property
    def num_units(self) -> int:
        return len(self.units)


def _ellipsoid_score(x: float, y: float, z: float, center: Tuple[float, float, float], radii: Tuple[float, float, float]) -> float:
    cx, cy, cz = center
    rx, ry, rz = radii
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2


def _inside_ellipsoid(x: float, y: float, z: float, center: Tuple[float, float, float], radii: Tuple[float, float, float]) -> bool:
    return _ellipsoid_score(x, y, z, center, radii) <= 1.0


def _make_grid(grid: Dict[str, int]) -> List[Tuple[int, int, int]]:
    return [(x, y, z) for x in range(grid["x"]) for y in range(grid["y"]) for z in range(grid["z"])]


def _normalized_point(point: Tuple[int, int, int], grid: Dict[str, int]) -> Tuple[float, float, float]:
    x, y, z = point
    return (
        (x / max(1, grid["x"] - 1)) * 2.0 - 1.0,
        (y / max(1, grid["y"] - 1)) * 2.0 - 1.0,
        (z / max(1, grid["z"] - 1)) * 2.0 - 1.0,
    )


def _candidate_region(point: Tuple[int, int, int], grid: Dict[str, int]) -> str | None:
    x, y, z = _normalized_point(point, grid)

    # Rough functional brain-shaped masks. These masks define topology only.
    # They are not answer logic and do not encode task solutions.
    left_cortex = _inside_ellipsoid(x, y, z, center=(-0.32, 0.05, 0.18), radii=(0.72, 0.92, 0.72))
    right_cortex = _inside_ellipsoid(x, y, z, center=(0.32, 0.05, 0.18), radii=(0.72, 0.92, 0.72))
    cortex_outer = left_cortex or right_cortex

    thalamus = _inside_ellipsoid(x, y, z, center=(0.0, 0.0, -0.05), radii=(0.28, 0.26, 0.22))
    basal = _inside_ellipsoid(x, y, z, center=(0.0, -0.02, -0.18), radii=(0.42, 0.30, 0.20))
    hippocampus_left = _inside_ellipsoid(x, y, z, center=(-0.42, -0.32, -0.12), radii=(0.24, 0.18, 0.16))
    hippocampus_right = _inside_ellipsoid(x, y, z, center=(0.42, -0.32, -0.12), radii=(0.24, 0.18, 0.16))
    cerebellum = _inside_ellipsoid(x, y, z, center=(0.0, -0.82, -0.52), radii=(0.62, 0.30, 0.26))
    brainstem = abs(x) <= 0.12 and y < -0.48 and -0.92 <= z <= -0.22

    # More central/specialized masks win before cortex.
    if thalamus:
        return "thalamus"
    if basal:
        return "basal_ganglia"
    if hippocampus_left or hippocampus_right:
        return "hippocampus"
    if cerebellum:
        return "cerebellum"
    if brainstem:
        return "brainstem"
    if cortex_outer:
        return "cortex"
    return None


def _select_region_points(
    candidates: List[Tuple[int, int, int]],
    grid: Dict[str, int],
    count: int,
    seed: int,
) -> List[Tuple[int, int, int]]:
    if count <= 0:
        return []
    if not candidates:
        raise ValueError("No topology candidates available for region.")

    rng = np.random.default_rng(seed)
    if len(candidates) <= count:
        selected = list(candidates)
    else:
        indices = rng.choice(len(candidates), size=count, replace=False)
        selected = [candidates[int(i)] for i in indices]

    # Stable ordering after seeded sampling.
    selected.sort()
    return selected


def build_brain_graph(cfg: Dict) -> BrainGraph:
    topo = cfg.get("topology", {})
    if topo.get("type", "brain_graph_3d") != "brain_graph_3d":
        raise ValueError("Only brain_graph_3d topology is supported here.")

    seed = int(topo.get("seed", cfg.get("run", {}).get("seed", 42)))
    grid = topo.get("grid", {"x": 24, "y": 16, "z": 12})
    region_counts = topo.get("regions", {"cortex": cfg["model"]["num_microbrains"]})

    grouped: Dict[str, List[Tuple[int, int, int]]] = {name: [] for name in REGION_TO_ID}
    for p in _make_grid(grid):
        region = _candidate_region(p, grid)
        if region is not None:
            grouped[region].append(p)

    units: List[BrainUnit] = []
    used: set[Tuple[int, int, int]] = set()
    for region_name, count in region_counts.items():
        available = [p for p in grouped[region_name] if p not in used]
        selected = _select_region_points(available, grid, int(count), seed + REGION_TO_ID[region_name] * 997)
        for p in selected:
            used.add(p)
            x, y, z = _normalized_point(p, grid)
            units.append(BrainUnit(unit_id=len(units), region=region_name, x=x, y=y, z=z))

    if not units:
        raise ValueError("BrainGraph has zero units. Check topology region counts/grid.")

    coords_np = np.array([[u.x, u.y, u.z] for u in units], dtype=np.float32)
    region_ids_np = np.array([REGION_TO_ID[u.region] for u in units], dtype=np.int64)

    max_neighbors = int(topo.get("max_neighbors", 12))
    local_radius = float(topo.get("local_radius", 0.34))
    coords = torch.tensor(coords_np, dtype=torch.float32)
    dist = torch.cdist(coords, coords)

    neighbor_indices = []
    neighbor_mask = []
    for i in range(len(units)):
        d = dist[i].numpy()
        candidates = [j for j in np.argsort(d).tolist() if j != i and d[j] <= local_radius]
        candidates = candidates[:max_neighbors]
        mask = [1.0] * len(candidates)
        while len(candidates) < max_neighbors:
            candidates.append(i)
            mask.append(0.0)
        neighbor_indices.append(candidates)
        neighbor_mask.append(mask)

    return BrainGraph(
        units=units,
        neighbor_indices=torch.tensor(neighbor_indices, dtype=torch.long),
        neighbor_mask=torch.tensor(neighbor_mask, dtype=torch.float32),
        coords=coords,
        region_ids=torch.tensor(region_ids_np, dtype=torch.long),
    )


def region_count_summary(graph: BrainGraph) -> Dict[str, int]:
    out = {name: 0 for name in REGION_TO_ID}
    for unit in graph.units:
        out[unit.region] += 1
    return out
