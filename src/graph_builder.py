from typing import Optional, Tuple

import numpy as np
import pymatching
import torch
from torch_geometric.data import Data

from .circuit_dem import DEMTopology


def run_corr_mwpm_and_build_graph(
    syndrome: np.ndarray,
    matcher: pymatching.Matching,
    topo: DEMTopology,
    true_obs: Optional[int] = None,
) -> Tuple[Optional[Data], int, bool]:
    """
    Run 2-pass Correlated MWPM and build the localized SYNDROME PyG graph.
    Returns: (graph_or_None, corr_mwpm_prediction, is_routed)
    """
    lit = np.flatnonzero(syndrome)
    if len(lit) == 0:
        return None, 0, False

    # Pass 1: Vanilla MWPM to obtain initial matching
    matched_edges = matcher.decode_to_edges_array(syndrome)
    corr_pred = int(matcher.decode(syndrome)[0])

    # Only route syndromes touching at least one X-component detector
    lit_set = set(int(d) for d in lit)
    routed = any(topo.is_x_det[d] for d in lit_set)
    if not routed:
        return None, corr_pred, False

    chosen_pairs = set()
    chosen_boundary = set()
    for u, v in matched_edges:
        u, v = int(u), int(v)
        if v == -1:
            chosen_boundary.add(u)
        else:
            chosen_pairs.add((min(u, v), max(u, v)))

    # Collect active nodes: lit detectors + 1-hop unlit bridges
    active_nodes = set(lit_set)
    for d in lit_set:
        for nb in topo.adj.get(d, []):
            active_nodes.add(nb)

    node_list = sorted(active_nodes)
    node_to_idx = {d: i for i, d in enumerate(node_list)}

    # 7-dim Node Features: [x, y, t, is_lit, is_virtual_boundary, is_boundary_capable, is_x_component]
    x_feats = []
    for d in node_list:
        c = topo.coords[d]
        x_feats.append([
            float(c[0]), float(c[1]), float(c[2]),
            1.0 if d in lit_set else 0.0,
            0.0,
            1.0 if topo.boundary_capable[d] else 0.0,
            1.0 if topo.is_x_det[d] else 0.0,
        ])

    edge_src, edge_dst, edge_attr = [], [], []

    def add_undirected(i: int, j: int, feat: list):
        edge_src.extend([i, j])
        edge_dst.extend([j, i])
        edge_attr.extend([feat, feat])

    # 1. Physical DEM edges (6-dim: [w, delta_w, crosses_logical, is_chosen, is_boundary, is_corr])
    for i, u in enumerate(node_list):
        for v in topo.adj.get(u, []):
            if v > u and v in node_to_idx:
                key = (u, v)
                if key in topo.base_edges:
                    w, crosses_log = topo.base_edges[key]
                    is_chosen = 1.0 if key in chosen_pairs else 0.0
                    add_undirected(i, node_to_idx[v], [w, 0.0, float(crosses_log), is_chosen, 0.0, 0.0])

    # 2. Localized virtual boundary nodes & boundary edges
    for u in node_list:
        if u in topo.boundary_edges:
            vb_idx = len(x_feats)
            vb_c = topo.virtual_boundary_coords[u]
            x_feats.append([
                float(vb_c[0]), float(vb_c[1]), float(vb_c[2]),
                0.0, 1.0, 1.0, 1.0 if topo.is_x_det[u] else 0.0,
            ])
            w, crosses_log = topo.boundary_edges[u]
            is_chosen = 1.0 if u in chosen_boundary else 0.0
            add_undirected(node_to_idx[u], vb_idx, [w, 0.0, float(crosses_log), is_chosen, 1.0, 0.0])

    # 3. X-Z DEM correlation edges
    seen_corr = set()
    for u in node_list:
        for v, w_corr in topo.xz_corr_edges.get(u, []):
            if v in node_to_idx:
                pair = (min(u, v), max(u, v))
                if pair not in seen_corr:
                    seen_corr.add(pair)
                    add_undirected(node_to_idx[u], node_to_idx[v], [w_corr, 0.0, 0.0, 0.0, 0.0, 1.0])

    data = Data(
        x=torch.tensor(x_feats, dtype=torch.float32),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )
    if true_obs is not None:
        data.y = torch.tensor([float(corr_pred != true_obs)], dtype=torch.float32)

    return data, corr_pred, True
