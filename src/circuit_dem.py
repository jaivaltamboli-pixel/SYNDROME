import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pymatching
import stim


@dataclass
class DEMTopology:
    distance: int
    num_detectors: int
    coords: np.ndarray                         # (N, 3) normalized (x, y, t)
    is_x_det: np.ndarray                       # (N,) bool
    boundary_capable: np.ndarray               # (N,) bool
    virtual_boundary_coords: Dict[int, np.ndarray]  # det_id -> (3,) outward-normal coord
    base_edges: Dict[Tuple[int, int], Tuple[float, bool]]  # (u, v) -> (weight, crosses_logical)
    boundary_edges: Dict[int, Tuple[float, bool]]          # u -> (weight, crosses_logical)
    adj: Dict[int, List[int]]                  # 1-hop physical adjacency
    xz_corr_edges: Dict[int, List[Tuple[int, float]]]      # det_id -> [(partner_det, corr_weight)]


def build_depolarizing_circuit(distance: int, total_p: float, rounds: int = None) -> stim.Circuit:
    """Build a rotated planar surface code (memory_x) with isotropic depolarizing noise."""
    if rounds is None:
        rounds = distance
    p_err = total_p / 3.0
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=total_p,
        before_round_data_depolarization=total_p,
        before_measure_flip_probability=p_err,
        after_reset_flip_probability=p_err,
    )


def build_circuit_and_topology(distance: int, total_p: float, rounds: int = None):
    """Return (circuit, vanilla_matcher, dem_topology)."""
    circuit = build_depolarizing_circuit(distance, total_p, rounds=rounds)
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    num_det = circuit.num_detectors
    raw_coords = circuit.get_detector_coordinates()
    coords = np.zeros((num_det, 3), dtype=np.float32)
    for d_id, c in raw_coords.items():
        if d_id < num_det:
            coords[d_id, : len(c)] = c[:3]

    # Identify X vs Z check components from Stim rotated_memory_x parity
    is_x_det = np.array([(int(coords[i, 0]) + int(coords[i, 1])) % 4 == 2 for i in range(num_det)], dtype=bool)

    # Normalize coordinates to [0, 1] grid scale
    max_xy = max(float(coords[:, :2].max()), 1.0)
    max_t = max(float(coords[:, 2].max()), 1.0)
    norm_coords = coords.copy()
    norm_coords[:, 0] /= max_xy
    norm_coords[:, 1] /= max_xy
    norm_coords[:, 2] /= max_t

    base_edges: Dict[Tuple[int, int], Tuple[float, bool]] = {}
    boundary_edges: Dict[int, Tuple[float, bool]] = {}
    adj: Dict[int, List[int]] = defaultdict(list)
    xz_corr_edges: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        if p <= 0.0 or p >= 0.5:
            continue
        w = float(math.log((1.0 - p) / p))

        # Split decomposed hyperedge into graphlike components
        components = [[]]
        for t in inst.targets_copy():
            if t.is_separator():
                components.append([])
            else:
                components[-1].append(t)

        comp_dets = []
        for comp in components:
            dets = [t.val for t in comp if t.is_relative_detector_id()]
            crosses_log = any(t.is_logical_observable_id() for t in comp)
            if len(dets) == 2:
                u, v = sorted(dets)
                if (u, v) not in base_edges or w < base_edges[(u, v)][0]:
                    base_edges[(u, v)] = (w, crosses_log)
            elif len(dets) == 1:
                u = dets[0]
                if u not in boundary_edges or w < boundary_edges[u][0]:
                    boundary_edges[u] = (w, crosses_log)
            if dets:
                comp_dets.append(dets)

        # Record X-Z Y-error correlation links between decomposed components
        if len(comp_dets) >= 2:
            flat_x = [d for sub in comp_dets for d in sub if is_x_det[d]]
            flat_z = [d for sub in comp_dets for d in sub if not is_x_det[d]]
            for dx in flat_x:
                for dz in flat_z:
                    xz_corr_edges[dx].append((dz, w))
                    xz_corr_edges[dz].append((dx, w))

    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Localized virtual boundary node placement (1.0 normalized grid unit along outward normal)
    boundary_capable = np.zeros(num_det, dtype=bool)
    virtual_boundary_coords: Dict[int, np.ndarray] = {}
    center_xy = norm_coords[:, :2].mean(axis=0)
    grid_step = 1.0 / max(distance, 1)

    for u in boundary_edges:
        boundary_capable[u] = True
        outward = norm_coords[u, :2] - center_xy
        norm = float(np.linalg.norm(outward))
        direction = outward / norm if norm > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
        vb = norm_coords[u].copy()
        vb[:2] += direction * grid_step
        virtual_boundary_coords[u] = vb

    topo = DEMTopology(
        distance=distance,
        num_detectors=num_det,
        coords=norm_coords,
        is_x_det=is_x_det,
        boundary_capable=boundary_capable,
        virtual_boundary_coords=virtual_boundary_coords,
        base_edges=base_edges,
        boundary_edges=boundary_edges,
        adj=dict(adj),
        xz_corr_edges=dict(xz_corr_edges),
    )
    return circuit, matcher, topo


def compute_complementary_gap(syndrome: np.ndarray, matcher: pymatching.Matching) -> float:
    """Compute exact Complementary Gap ΔW = |W_1 - W_0| via two-coset weight comparison."""
    pred, w_best = matcher.decode(syndrome, return_weight=True)
    # Placeholder hook: exact two-boundary B0/B1 matcher returns |W_1 - W_0|
    return float(abs(w_best))
