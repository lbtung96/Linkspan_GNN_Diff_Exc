"""Export a 2D projected graph by collapsing MPC offset nodes to masters.

This script builds the upgraded 3D grid model, collapses slave MPC nodes to their
representative master nodes, projects geometry to x-y (z=0), and exports an
undirected graph suitable for GNN pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from scipy import sparse

from grid_modal_Kinematic_MPC import grid_modal_demo_3d_offset_upgraded_v22


# Group IDs from solver
G_GIRDER = 1
G_CROSS = 2
G_DIAG = 3
G_RIGID = 4
G_BOTTOM = 5
G_FRONT = 6
G_BALCONY = 7
G_EDGE = 8
G_BBRACE = 9


# Include flags (group 4 rigid is always excluded in logic)
INCLUDE_GIRDER = True
INCLUDE_CROSS = True
INCLUDE_DIAG = True
INCLUDE_BOTTOM = True
INCLUDE_FRONT = True
INCLUDE_BALCONY = True
INCLUDE_EDGE = True
INCLUDE_BBRACE = True


EXPORT_DENSE_A = False
OUTPUT_NPZ = "projected_graph.npz"


SENSOR_MAP_ENABLE = False
SENSOR_XY: List[Tuple[float, float]] = []
SENSOR_TOL_MM = 5.0


PLOT_GRAPH = True
PLOT_NODE_LABELS = True
PLOT_EDGE_LABELS = False
NODE_LABEL_MODE = "rep"  # "idx" | "rep" | "both"
LABEL_EVERY = 1
EDGE_LABEL_EVERY = 1


@dataclass(frozen=True)
class ExportStats:
    n_original_nodes: int
    n_mpc_pairs: int
    n_rep_nodes: int
    n_edges: int
    sensor_enabled: bool
    n_sensors_mapped: int


def _load_model() -> dict:
    """Build and return the solver model using the public demo entrypoint."""
    try:
        res = grid_modal_demo_3d_offset_upgraded_v22(do_plot=False, run_modal=False)
    except TypeError:
        # Fallback for any variant that does not accept run_modal=False.
        res = grid_modal_demo_3d_offset_upgraded_v22(do_plot=False, run_modal=True)

    if "model" not in res:
        raise KeyError("Solver result does not contain 'model'.")
    return res["model"]


def _build_slave_to_master(mpc_pairs: np.ndarray) -> Dict[int, int]:
    """Build a slave->master map from 1-based MPC node pairs."""
    out: Dict[int, int] = {}
    for pair in np.asarray(mpc_pairs, dtype=int):
        if pair.size < 2:
            continue
        master, slave = int(pair[0]), int(pair[1])
        out[slave] = master
    return out


def make_find_rep(slave_to_master: Dict[int, int]) -> Callable[[int], int]:
    """Return a representative finder with loop protection."""

    def find_rep(nid: int) -> int:
        cur = int(nid)
        visited = set()
        while cur in slave_to_master:
            if cur in visited:
                raise RuntimeError(f"Loop detected while collapsing MPC mapping at node {cur}.")
            visited.add(cur)
            cur = int(slave_to_master[cur])
        return cur

    return find_rep


def _group_enabled(group_id: int) -> bool:
    """Check whether an element group should be kept."""
    if group_id == G_RIGID:
        return False
    mapping = {
        G_GIRDER: INCLUDE_GIRDER,
        G_CROSS: INCLUDE_CROSS,
        G_DIAG: INCLUDE_DIAG,
        G_BOTTOM: INCLUDE_BOTTOM,
        G_FRONT: INCLUDE_FRONT,
        G_BALCONY: INCLUDE_BALCONY,
        G_EDGE: INCLUDE_EDGE,
        G_BBRACE: INCLUDE_BBRACE,
    }
    return bool(mapping.get(group_id, False))


def _build_collapsed_edges(elems: np.ndarray, find_rep) -> List[Tuple[int, int]]:
    """Build deduplicated undirected edges from structural elements only."""
    edge_set = set()
    for e in np.asarray(elems):
        if e.size < 4:
            continue
        n1 = int(e[1])
        n2 = int(e[2])
        grp = int(e[3])
        if not _group_enabled(grp):
            continue
        r1 = find_rep(n1)
        r2 = find_rep(n2)
        if r1 == r2:
            continue
        a, b = (r1, r2) if r1 < r2 else (r2, r1)
        edge_set.add((a, b))
    return sorted(edge_set)


def _coords_from_rep_nodes(nodes: np.ndarray, rep_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build (N,2) xy and (N,3) xyz0 arrays from representative 1-based IDs."""
    xyz = np.asarray(nodes, dtype=float)
    xy = np.zeros((rep_ids.size, 2), dtype=float)
    xyz0 = np.zeros((rep_ids.size, 3), dtype=float)
    for i, rep in enumerate(rep_ids.astype(int)):
        row = xyz[rep - 1]
        x = float(row[1])
        y = float(row[2])
        xy[i, 0] = x
        xy[i, 1] = y
        xyz0[i, 0] = x
        xyz0[i, 1] = y
        xyz0[i, 2] = 0.0
    return xy, xyz0


def _build_adjacency(edges_ij: np.ndarray, n_nodes: int) -> sparse.csr_matrix:
    """Build undirected no-self-loop adjacency matrix in CSR format."""
    if n_nodes == 0:
        return sparse.csr_matrix((0, 0), dtype=np.int8)
    if edges_ij.size == 0:
        return sparse.csr_matrix((n_nodes, n_nodes), dtype=np.int8)

    ii = edges_ij[:, 0].astype(int)
    jj = edges_ij[:, 1].astype(int)
    data = np.ones(ii.size * 2, dtype=np.int8)
    rows = np.concatenate([ii, jj])
    cols = np.concatenate([jj, ii])
    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.int8).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def _edge_geometry_from_coords(coords_xy: np.ndarray, edge_index_undirected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute undirected edge lengths and projected angles from node coordinates."""
    xy = np.asarray(coords_xy, dtype=float)
    eij = np.asarray(edge_index_undirected, dtype=int)
    if eij.shape[0] != 2:
        raise ValueError(f"edge_index_undirected must have shape (2,E), got {eij.shape}.")
    if eij.size == 0:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)

    src = eij[0]
    dst = eij[1]
    dxy = xy[dst] - xy[src]
    length_mm = np.sqrt(np.sum(dxy * dxy, axis=1))
    theta_rad = np.arctan2(dxy[:, 1], dxy[:, 0])
    return np.asarray(length_mm, dtype=float), np.asarray(theta_rad, dtype=float)


def _empty_sensor_result() -> dict[str, np.ndarray]:
    return {
        "sensor_xy": np.zeros((0, 2), dtype=float),
        "sensor_node_idx": np.zeros((0,), dtype=int),
        "sensor_rep_node_id": np.zeros((0,), dtype=int),
        "sensor_dist_mm": np.zeros((0,), dtype=float),
    }


def _group_include_payload() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_names = np.array(
        ["G_GIRDER", "G_CROSS", "G_DIAG", "G_BOTTOM", "G_FRONT", "G_BALCONY", "G_EDGE", "G_BBRACE", "G_RIGID"],
        dtype="<U16",
    )
    group_ids = np.array([1, 2, 3, 5, 6, 7, 8, 9, 4], dtype=int)
    group_enabled = np.array(
        [
            INCLUDE_GIRDER,
            INCLUDE_CROSS,
            INCLUDE_DIAG,
            INCLUDE_BOTTOM,
            INCLUDE_FRONT,
            INCLUDE_BALCONY,
            INCLUDE_EDGE,
            INCLUDE_BBRACE,
            False,
        ],
        dtype=bool,
    )
    return group_names, group_ids, group_enabled


def _validate_graph_payload(payload: dict[str, Any]) -> None:
    graph_node_ids = np.asarray(payload["graph_node_ids"], dtype=int).reshape(-1)
    coords_xy = np.asarray(payload["coords_xy_mm"], dtype=float)
    edge_index_undirected = np.asarray(payload["edge_index_undirected"], dtype=int)
    edge_index = np.asarray(payload["edge_index"], dtype=int)
    edge_attr_geom = np.asarray(payload["edge_attr_geom"], dtype=float)

    if coords_xy.shape != (graph_node_ids.size, 2):
        raise ValueError(
            f"coords_xy_mm must have shape ({graph_node_ids.size}, 2), got {coords_xy.shape}."
        )
    if edge_index_undirected.ndim != 2 or edge_index_undirected.shape[0] != 2:
        raise ValueError(f"edge_index_undirected must have shape (2,E), got {edge_index_undirected.shape}.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2,E), got {edge_index.shape}.")
    if edge_index.shape[1] != 2 * edge_index_undirected.shape[1]:
        raise ValueError(
            "Directed edge count must be exactly 2x the undirected edge count. "
            f"Got {edge_index.shape[1]} vs {edge_index_undirected.shape[1]}."
        )
    if edge_attr_geom.shape != (edge_index.shape[1], 2):
        raise ValueError(
            f"edge_attr_geom must have shape ({edge_index.shape[1]}, 2), got {edge_attr_geom.shape}."
        )
    if edge_index.size:
        lo = int(np.min(edge_index))
        hi = int(np.max(edge_index))
        if lo < 0 or hi >= graph_node_ids.size:
            raise ValueError(
                f"edge_index contains out-of-range node indices [{lo}, {hi}] for N={graph_node_ids.size}."
            )
    if edge_index_undirected.size and not np.all(edge_index_undirected[0] < edge_index_undirected[1]):
        raise ValueError("edge_index_undirected must satisfy i < j for every column.")


def build_projected_graph_from_model(model: dict, *, include_sensor_map: bool | None = None) -> dict[str, Any]:
    """Build projected/collapsed graph arrays for an already-built solver model."""
    if include_sensor_map is None:
        include_sensor_map = bool(SENSOR_MAP_ENABLE)

    nodes = np.asarray(model.get("nodes", np.zeros((0, 4))), dtype=float)
    elems = np.asarray(model.get("elems", np.zeros((0, 7))), dtype=float)
    mpc_pairs = np.asarray(model.get("mpc_pairs", np.zeros((0, 2), dtype=int)), dtype=int)

    if elems.size == 0:
        raise ValueError("Model does not contain structural elements: 'elems' is missing or empty.")

    slave_to_master = _build_slave_to_master(mpc_pairs)
    find_rep = make_find_rep(slave_to_master)
    collapsed_edges = _build_collapsed_edges(elems, find_rep)

    if collapsed_edges:
        graph_node_ids = np.unique(np.asarray(collapsed_edges, dtype=int).reshape(-1))
    else:
        graph_node_ids = np.zeros((0,), dtype=int)

    graph_node_ids = np.sort(graph_node_ids.astype(int))
    rep_to_idx = {int(rep): i for i, rep in enumerate(graph_node_ids.tolist())}

    edge_index_undirected = np.zeros((2, len(collapsed_edges)), dtype=int)
    for k, (a, b) in enumerate(collapsed_edges):
        edge_index_undirected[0, k] = rep_to_idx[int(a)]
        edge_index_undirected[1, k] = rep_to_idx[int(b)]

    coords_xy_mm, coords_xyz0_mm = _coords_from_rep_nodes(nodes, graph_node_ids)
    A = _build_adjacency(edge_index_undirected.T, graph_node_ids.size)
    length_u, theta_u = _edge_geometry_from_coords(coords_xy_mm, edge_index_undirected)

    edge_index = np.zeros((2, 2 * edge_index_undirected.shape[1]), dtype=int)
    edge_length_mm = np.zeros((edge_index.shape[1],), dtype=float)
    edge_theta_rad = np.zeros((edge_index.shape[1],), dtype=float)
    if edge_index_undirected.shape[1]:
        n_undir = edge_index_undirected.shape[1]
        edge_index[:, :n_undir] = edge_index_undirected
        edge_index[0, n_undir:] = edge_index_undirected[1]
        edge_index[1, n_undir:] = edge_index_undirected[0]
        edge_length_mm[:n_undir] = length_u
        edge_length_mm[n_undir:] = length_u
        edge_theta_rad[:n_undir] = theta_u
        edge_theta_rad[n_undir:] = theta_u + np.pi
    edge_attr_geom = np.column_stack([edge_length_mm, edge_theta_rad]) if edge_index.shape[1] else np.zeros((0, 2), dtype=float)

    group_names, group_ids, group_enabled = _group_include_payload()
    sensor_result = _sensor_map(coords_xy_mm, graph_node_ids) if include_sensor_map else _empty_sensor_result()

    payload: dict[str, Any] = {
        "graph_node_ids": graph_node_ids.astype(np.int64),
        "coords_xy_mm": np.asarray(coords_xy_mm, dtype=float),
        "coords_xyz0_mm": np.asarray(coords_xyz0_mm, dtype=float),
        "edge_index_undirected": np.asarray(edge_index_undirected, dtype=np.int64),
        "edge_index": np.asarray(edge_index, dtype=np.int64),
        "edge_length_mm": np.asarray(edge_length_mm, dtype=float),
        "edge_theta_rad": np.asarray(edge_theta_rad, dtype=float),
        "edge_attr_geom": np.asarray(edge_attr_geom, dtype=float),
        "adjacency_csr": A,
        "group_include_names": group_names,
        "group_include_ids": group_ids.astype(np.int64),
        "group_include_enabled": group_enabled.astype(bool),
        "sensor_map_enabled": np.asarray([bool(include_sensor_map)], dtype=bool),
        "sensor_tol_mm": np.asarray([float(SENSOR_TOL_MM)], dtype=float),
    }
    payload.update(sensor_result)
    _validate_graph_payload(payload)
    return payload


def _sensor_map(coords_xy: np.ndarray, rep_node_ids: np.ndarray) -> dict:
    """Map configured sensor points to nearest projected nodes."""
    sensor_xy = np.asarray(SENSOR_XY, dtype=float).reshape(-1, 2)
    ns = sensor_xy.shape[0]
    if ns == 0:
        return {
            "sensor_xy": sensor_xy,
            "sensor_node_idx": np.zeros((0,), dtype=int),
            "sensor_rep_node_id": np.zeros((0,), dtype=int),
            "sensor_dist_mm": np.zeros((0,), dtype=float),
        }
    if coords_xy.shape[0] == 0:
        raise ValueError("Sensor mapping requested but no projected nodes are available.")

    sensor_node_idx = np.zeros((ns,), dtype=int)
    sensor_rep_node_id = np.zeros((ns,), dtype=int)
    sensor_dist_mm = np.zeros((ns,), dtype=float)

    for i in range(ns):
        diff = coords_xy - sensor_xy[i]
        d = np.sqrt(np.sum(diff * diff, axis=1))
        idx = int(np.argmin(d))
        sensor_node_idx[i] = idx
        sensor_rep_node_id[i] = int(rep_node_ids[idx])
        sensor_dist_mm[i] = float(d[idx])

    over_tol = sensor_dist_mm > float(SENSOR_TOL_MM)
    if np.any(over_tol):
        n_bad = int(np.count_nonzero(over_tol))
        print(
            f"Warning: {n_bad} sensor(s) exceed tolerance "
            f"{SENSOR_TOL_MM:.3f} mm (max={sensor_dist_mm.max():.3f} mm)."
        )

    return {
        "sensor_xy": sensor_xy,
        "sensor_node_idx": sensor_node_idx,
        "sensor_rep_node_id": sensor_rep_node_id,
        "sensor_dist_mm": sensor_dist_mm,
    }


def plot_projected_graph(coords_xy: np.ndarray, edges_ij: np.ndarray, rep_node_ids: np.ndarray) -> None:
    """Plot projected graph using exported arrays only."""
    import matplotlib.pyplot as plt

    xy = np.asarray(coords_xy, dtype=float)
    eij = np.asarray(edges_ij, dtype=int)
    rep_ids = np.asarray(rep_node_ids, dtype=int).reshape(-1)

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"coords_xy must have shape (N,2), got {xy.shape}.")
    if eij.ndim != 2 or eij.shape[1] != 2:
        raise ValueError(f"edges_ij must have shape (E,2), got {eij.shape}.")
    if rep_ids.size != xy.shape[0]:
        raise ValueError(
            f"rep_node_ids size must match coords_xy rows, got {rep_ids.size} vs {xy.shape[0]}."
        )

    if NODE_LABEL_MODE not in ("idx", "rep", "both"):
        raise ValueError(f"NODE_LABEL_MODE must be 'idx', 'rep', or 'both', got '{NODE_LABEL_MODE}'.")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Draw edges first.
    for edge_id, (i, j) in enumerate(eij):
        xi, yi = xy[int(i), 0], xy[int(i), 1]
        xj, yj = xy[int(j), 0], xy[int(j), 1]
        ax.plot([xi, xj], [yi, yj], color="0.45", linewidth=0.8, zorder=1)

        if PLOT_EDGE_LABELS and EDGE_LABEL_EVERY > 0 and (edge_id % EDGE_LABEL_EVERY) == 0:
            xm = 0.5 * (xi + xj)
            ym = 0.5 * (yi + yj)
            ax.text(xm, ym, str(edge_id), fontsize=6, color="tab:red", ha="center", va="center", zorder=3)

    # Draw nodes on top.
    ax.scatter(xy[:, 0], xy[:, 1], s=14, c="tab:blue", zorder=2)

    if PLOT_NODE_LABELS and LABEL_EVERY > 0:
        for i in range(xy.shape[0]):
            if (i % LABEL_EVERY) != 0:
                continue
            if NODE_LABEL_MODE == "idx":
                txt = str(i)
            elif NODE_LABEL_MODE == "rep":
                txt = str(int(rep_ids[i]))
            else:
                txt = f"{i}|{int(rep_ids[i])}"
            ax.text(xy[i, 0], xy[i, 1], txt, fontsize=7, color="black", ha="left", va="bottom", zorder=4)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"Projected Graph (after MPC collapse): N={xy.shape[0]}, E={eij.shape[0]}")
    plt.tight_layout()
    plt.show()


def export_projected_graph(output_npz: str | Path = OUTPUT_NPZ) -> ExportStats:
    """Export projected graph arrays to an NPZ file."""
    model = _load_model()
    graph_payload = build_projected_graph_from_model(model, include_sensor_map=bool(SENSOR_MAP_ENABLE))
    nodes = np.asarray(model.get("nodes", np.zeros((0, 4))), dtype=float)
    mpc_pairs = np.asarray(model.get("mpc_pairs", np.zeros((0, 2), dtype=int)), dtype=int)
    A = graph_payload["adjacency_csr"]

    save_payload = {
        "coords_xy": np.asarray(graph_payload["coords_xy_mm"], dtype=float),
        "coords_xyz0": np.asarray(graph_payload["coords_xyz0_mm"], dtype=float),
        "rep_node_ids": np.asarray(graph_payload["graph_node_ids"], dtype=np.int64),
        "edges_ij": np.asarray(graph_payload["edge_index_undirected"], dtype=np.int64).T,
        "edge_index": np.asarray(graph_payload["edge_index"], dtype=np.int64),
        "edge_index_undirected": np.asarray(graph_payload["edge_index_undirected"], dtype=np.int64),
        "edge_length_mm": np.asarray(graph_payload["edge_length_mm"], dtype=float),
        "edge_theta_rad": np.asarray(graph_payload["edge_theta_rad"], dtype=float),
        "edge_attr_geom": np.asarray(graph_payload["edge_attr_geom"], dtype=float),
        "A_data": A.data.astype(np.int8),
        "A_indices": A.indices.astype(int),
        "A_indptr": A.indptr.astype(int),
        "A_shape": np.asarray(A.shape, dtype=int),
        "group_include_names": np.asarray(graph_payload["group_include_names"]),
        "group_include_ids": np.asarray(graph_payload["group_include_ids"], dtype=np.int64),
        "group_include_enabled": np.asarray(graph_payload["group_include_enabled"], dtype=bool),
        "sensor_map_enabled": np.asarray(graph_payload["sensor_map_enabled"], dtype=bool),
        "sensor_tol_mm": np.asarray(graph_payload["sensor_tol_mm"], dtype=float),
    }
    sensor_result = {
        "sensor_xy": np.asarray(graph_payload["sensor_xy"], dtype=float),
        "sensor_node_idx": np.asarray(graph_payload["sensor_node_idx"], dtype=np.int64),
        "sensor_rep_node_id": np.asarray(graph_payload["sensor_rep_node_id"], dtype=np.int64),
        "sensor_dist_mm": np.asarray(graph_payload["sensor_dist_mm"], dtype=float),
    }
    save_payload.update(sensor_result)

    if EXPORT_DENSE_A:
        save_payload["A_dense"] = A.toarray()

    np.savez(str(output_npz), **save_payload)

    return ExportStats(
        n_original_nodes=int(nodes.shape[0]),
        n_mpc_pairs=int(mpc_pairs.shape[0]) if mpc_pairs.ndim == 2 else 0,
        n_rep_nodes=int(np.asarray(graph_payload["graph_node_ids"]).size),
        n_edges=int(np.asarray(graph_payload["edge_index_undirected"]).shape[1]),
        sensor_enabled=bool(SENSOR_MAP_ENABLE),
        n_sensors_mapped=int(sensor_result["sensor_xy"].shape[0]) if SENSOR_MAP_ENABLE else 0,
    )


if __name__ == "__main__":
    stats = export_projected_graph(OUTPUT_NPZ)
    print(f"Exported projected graph to: {OUTPUT_NPZ}")
    print(f"Original nodes: {stats.n_original_nodes}")
    print(f"MPC pairs: {stats.n_mpc_pairs}")
    print(f"Representative nodes: {stats.n_rep_nodes}")
    print(f"Edges (undirected): {stats.n_edges}")
    print(f"Sensor mapping enabled: {stats.sensor_enabled}")
    print(f"Sensors mapped: {stats.n_sensors_mapped}")
    print("Reconstruct A:")
    print(
        "  d=np.load('projected_graph.npz'); "
        "A=sparse.csr_matrix((d['A_data'], d['A_indices'], d['A_indptr']), shape=tuple(d['A_shape']))"
        "For correspondence between node indices and original model node IDs, use d['rep_node_ids'] array. or simply rep_node_ids[idx]"
    )

    if PLOT_GRAPH:
        d = np.load(OUTPUT_NPZ)
        plot_projected_graph(d["coords_xy"], d["edges_ij"], d["rep_node_ids"])
