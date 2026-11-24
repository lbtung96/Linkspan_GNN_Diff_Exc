"""Build Phase 3 graph packages for the 5 real linkspan structures.

For each structure:
  1. Build FE projected graph (nodes + directed edges + geom features).
  2. Load measured sensor coordinates from grid_<structure>.txt.
  3. Auto-snap each sensor to the nearest FE node.
  4. Append explicit sensor nodes and bidirectional sensor-FE edges.
  5. Save phase3_real_graphs/<structure>/graph_package.npz.

Outputs (graph_package.npz keys):
  node_coords_xy_mm    (n_nodes, 2)   FE nodes first, then sensor nodes
  edge_index           (2, E_dir)     directed edges
  edge_attr_geom       (E_dir, 2)     [length_mm, theta_rad]
  sensor_node_indices  (n_sensors,)   row indices into node array for sensor nodes
  sensor_physical_ids  (n_sensors,)   physical IDs 125-140
  nearest_fe_node_idx  (n_sensors,)   FE node each sensor is snapped to
  n_fe_nodes           ()             number of original FE nodes
  structure_name       str
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from export_projected_graph import build_projected_graph_from_model
from grid_modal_Kinematic_MPC import (
    build_grid_model_3d_upgraded_v22,
    default_params_grid_3d_offset_upgraded_v22,
)

# grid_<structure>.txt files are copied verbatim into SSI_modal_infor/ by
# build_ssi_modal_infor.py (byte-identical to the field-measurement originals),
# so the whole real-structure input set lives in one in-repo folder.
SENSOR_GRID_DIR = Path("SSI_modal_infor")
OUTPUT_ROOT = Path("phase3_real_graphs")

SUPPORT_TXT_IDS = {1, 2}
SLAVE_TXT_IDS = {19, 20}
MEASURED_TXT_RANGE = range(3, 19)  # 3..18 inclusive
PHYSICAL_ID_OFFSET = 122           # txt_node_id + 122 = physical sensor ID

STRUCTURES: dict[str, dict[str, Any]] = {
    "Eidsdal":{
        "L_mm": 22000, "Nx": 35, "Ng": 4, "dy_mm": 3000, "diag_pattern": "None",
        "bottom_beam": {"x_frac": 0.6754, "overhang_mm": 600},
        "balcony": {
            "enable": True,
            "side": "both",           # ← left, right, or both?
            "x_center_frac": 0.895,     # ← balcony x-position / L_mm
            "nBeams": 3,              # ← number of balcony beams
            "dx_mm": 790,           # ← beam spacing
            "length_mm": 2650.0,      # ← balcony length
            "edge_beam_enable": True,
            "brace_enable": True,
        },
        "nsm": {
            "enable": True, "total_mass_tonne": 30,
            "to_girders": True, "to_cross": True, "to_balcony": True,
            "to_edge": False, "to_brace": False, "to_front": False,
            "to_bottom": False, "to_diagonals": False,
        },        
    },
    "Festoya":{
        "L_mm": 15000, "Nx": 18, "Ng": 5, "dy_mm": 3000, "diag_pattern": "None",
        "bottom_beam": {"x_frac": 0.686, "overhang_mm": 600},
        "balcony": {"enable": False},
        "nsm": {
            "enable": True, "total_mass_tonne": 30,
            "to_girders": True, "to_cross": True, "to_balcony": False,
            "to_edge": False, "to_brace": False, "to_front": False,
            "to_bottom": False, "to_diagonals": False,
        },
    },

    "Hellesylt": {
        "L_mm": 14920, "Nx": 15, "Ng": 2, "dy_mm": 4000, "diag_pattern": "None",
        "bottom_beam": {"x_frac": 0.6726, "overhang_mm": 600},
        "balcony": {"enable": False},
        "nsm": {
            "enable": True, "total_mass_tonne": 20,
            "to_girders": True, "to_cross": True, "to_balcony": True,
            "to_edge": False, "to_brace": False, "to_front": False,
            "to_bottom": False, "to_diagonals": False,
        },
    },
    "Hundeidvik": {
        "L_mm": 14555, "Nx": 15, "Ng": 4, "dy_mm": 2333, "diag_pattern": "staggered_x",
        "bottom_beam": {"x_frac": 0.6166667, "overhang_mm": 600},
        "balcony": {
            "enable": True, "side": "left", "x_center_frac": 0.88,
            "nBeams": 1, "dx_mm": 950.0, "length_mm": 2250,
            "edge_beam_enable": True, "brace_enable": True,
        },
        "nsm": {
            "enable": True, "total_mass_tonne": 20,
            "to_girders": True, "to_cross": True, "to_balcony": True,
            "to_edge": False, "to_brace": False, "to_front": False,
            "to_bottom": False, "to_diagonals": False,
        },
    },
    "Leknes": {
        "L_mm": 14900, "Nx": 15, "Ng": 2, "dy_mm": 4500, "diag_pattern": "staggered_x",
        "bottom_beam": {"x_frac": 9190/15350, "overhang_mm": 600},
        "balcony": {
            "enable": True, "side": "both", "x_center_frac": 0.939,
            "nBeams": 2, "dx_mm": 650.0, "length_mm": 1000,
            "edge_beam_enable": True, "brace_enable": True,
        },
        "nsm": {
            "enable": True, "total_mass_tonne": 20,
            "to_girders": True, "to_cross": True, "to_balcony": True,
            "to_edge": False, "to_brace": False, "to_front": False,
            "to_bottom": False, "to_diagonals": False,
        },
    },
    "Liabygda":{
        "L_mm": 18090, "Nx": 29, "Ng": 4, "dy_mm": 3000, "diag_pattern": "None",
        "bottom_beam": {"x_frac": 0.67, "overhang_mm": 600},
        "balcony": {
            "enable": True,
            "side": "both",           # ← left, right, or both?
            "x_center_frac": 0.8572,     # ← balcony x-position / L_mm
            "nBeams": 3,              # ← number of balcony beams
            "dx_mm": 675,           # ← beam spacing
            "length_mm": 2400.0,      # ← balcony length
            "edge_beam_enable": True,
            "brace_enable": True,
        },
        "nsm": {
            "enable": True, "total_mass_tonne": 30,
            "to_girders": True, "to_cross": True, "to_balcony": True,
            "to_edge": False, "to_brace": False, "to_front": False,
            "to_bottom": False, "to_diagonals": False,
        },        
    },

    "Linge": {
        "L_mm": 18000, "Nx": 29, "Ng": 5, "dy_mm": 3000, "diag_pattern": "None",
        "bottom_beam": {"x_frac": 0.667, "overhang_mm": 600},
        "balcony": {"enable": False},
        "nsm": {
            "enable": True, "total_mass_tonne": 30,
            "to_girders": True, "to_cross": True, "to_balcony": True,
            "to_edge": False, "to_brace": False, "to_front": False,
            "to_bottom": False, "to_diagonals": False,
        },
    },
    "Saebo": {
        "L_mm": 18000, "Nx": 18, "Ng": 4, "dy_mm": 2333, "diag_pattern": "single_partial_x",
        "bottom_beam": {"x_frac": 0.667, "overhang_mm": 600},
        "balcony": {"enable": False},
        "nsm": {
            "enable": True, "total_mass_tonne": 30,
            "to_girders": True, "to_cross": True, "to_balcony": True,
            "to_edge": False, "to_brace": False, "to_front": False,
            "to_bottom": False, "to_diagonals": False,
        },
    },
    "Stranda":{
        "L_mm": 18090, "Nx": 29, "Ng": 4, "dy_mm": 3000, "diag_pattern": "None",
        "bottom_beam": {"x_frac": 0.667, "overhang_mm": 600},
        "balcony": {
            "enable": True,
            "side": "both",           # ← left, right, or both?
            "x_center_frac": 0.8572,     # ← balcony x-position / L_mm
            "nBeams": 3,              # ← number of balcony beams
            "dx_mm": 675,           # ← beam spacing
            "length_mm": 2500.0,      # ← balcony length
            "edge_beam_enable": True,
            "brace_enable": True,
        },
        "nsm": {
            "enable": True, "total_mass_tonne": 30,
            "to_girders": True, "to_cross": True, "to_balcony": True,
            "to_edge": False, "to_brace": False, "to_front": False,
            "to_bottom": False, "to_diagonals": False,
        },        
    },
    "Sykkylven":{
        "L_mm": 15270, "Nx": 15, "Ng": 4, "dy_mm": 2333, "diag_pattern": "staggered_x",
        "bottom_beam": {"x_frac": 0.602, "overhang_mm": 600},
        "balcony": {
            "enable": True,
            "side": "both",           # ← left, right, or both?
            "x_center_frac": 0.8572,     # ← balcony x-position / L_mm
            "nBeams": 3,              # ← number of balcony beams
            "dx_mm": 570.0,
            "length_left_mm": 800,                      # ← beam spacing
            "length_right_mm": 4300,      # ← balcony length
            "edge_beam_enable": True,
            "brace_enable": True,
        },
        "nsm": {
            "enable": True, "total_mass_tonne": 30,
            "to_girders": True, "to_cross": True, "to_balcony": True,
            "to_edge": False, "to_brace": False, "to_front": False,
            "to_bottom": False, "to_diagonals": False,
        },        
    },
    "Magerholm":{
        "L_mm": 14905, "Nx": 16, "Ng": 5, "dy_mm": 2250, "diag_pattern": "staggered_x",
        "bottom_beam": {"x_frac": 9190/15350, "overhang_mm": 570},
        "balcony": {
            "enable": True,
            "side": "both",           # ← left, right, or both?
            "x_center_frac": 13290 / 14905,     # ← balcony x-position / L_mm
            "nBeams": 3,              # ← number of balcony beams
            "dx_mm": 950,           # ← beam spacing
            "length_mm": 2500.0,      # ← balcony length
            "edge_beam_enable": True,
            "brace_enable": True,
        },
        "nsm": {
            "enable": True, "total_mass_tonne": 30,
            "to_girders": True, "to_cross": True, "to_balcony": True,
            "to_edge": False, "to_brace": False, "to_front": False,
            "to_bottom": False, "to_diagonals": False,
        },        
    },
    "Rykkjem":{
        "L_mm": 17695, "Nx": 18, "Ng": 4, "dy_mm": 2333, "diag_pattern": "staggered_x",
        "bottom_beam": {"x_frac": 11050/18140, "overhang_mm": 600},
        "balcony": {
            "enable": False,
            "side": "both",           # ← left, right, or both?
            "x_center_frac": 0.8572,     # ← balcony x-position / L_mm
            "nBeams": 3,              # ← number of balcony beams
            "dx_mm": 570.0,           # ← beam spacing
            "length_mm": 2500.0,      # ← balcony length
            "edge_beam_enable": True,
            "brace_enable": True,
        },
        "nsm": {
            "enable": True, "total_mass_tonne": 30,
            "to_girders": True, "to_cross": True, "to_balcony": True,
            "to_edge": False, "to_brace": False, "to_front": False,
            "to_bottom": False, "to_diagonals": False,
        },        
    },            
}


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _build_fe_params(overrides: dict) -> dict:
    # An empty overrides dict would silently return the solver defaults, which are
    # Magerholm's geometry (see grid_modal_Kinematic_MPC.py). That produced a
    # Magerholm-shaped FE model wearing another structure's SSI labels, with no
    # error. Refuse it instead.
    if not overrides:
        raise ValueError(
            "Empty FE parameter dict. The solver defaults are Magerholm's geometry, "
            "so an empty entry silently builds the wrong structure. Populate the "
            "entry in STRUCTURES before building this structure."
        )
    params = default_params_grid_3d_offset_upgraded_v22()
    _deep_update(params, deepcopy(overrides))
    return params


def _load_sensor_grid(txt_path: Path) -> list[dict]:
    """Parse grid_<structure>.txt, return only measured nodes (3-18)."""
    rows = []
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip().lstrip("\ufeff")
            if not line or line.startswith(("#", "%", "//")):
                continue
            tokens = line.replace(",", " ").split()
            if len(tokens) < 4:
                continue
            try:
                nid = int(float(tokens[0]))
                x_m, y_m = float(tokens[1]), float(tokens[2])
            except ValueError:
                continue
            if nid not in MEASURED_TXT_RANGE:
                continue
            rows.append({
                "txt_node_id": nid,
                "physical_sensor_id": nid + PHYSICAL_ID_OFFSET,
                "x_mm": x_m * 1000.0,
                "y_mm": y_m * 1000.0,
            })
    rows.sort(key=lambda r: r["txt_node_id"])
    return rows


def _y_snap_and_find_nearest(
    sensor_rows: list[dict],
    fe_coords_xy_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Snap each sensor y to the nearest girder y, then find nearest FE node on that girder."""
    girder_ys = np.unique(fe_coords_xy_mm[:, 1])
    raw_xy = np.array([[r["x_mm"], r["y_mm"]] for r in sensor_rows], dtype=np.float32)
    snapped = raw_xy.copy()
    for i in range(len(sensor_rows)):
        best_y = girder_ys[np.argmin(np.abs(girder_ys - raw_xy[i, 1]))]
        snapped[i, 1] = best_y
    dists = np.linalg.norm(
        fe_coords_xy_mm[np.newaxis, :, :] - snapped[:, np.newaxis, :], axis=2
    )
    nearest_fe_idx = np.argmin(dists, axis=1).astype(np.int64)
    return snapped, nearest_fe_idx


def _find_bracketing_edge(
    sensor_xy: np.ndarray,
    fe_coords: np.ndarray,
    undirected_edges: set,
    y_tol: float = 1.0,
) -> tuple[int, int] | None:
    """Return (left_fe_idx, right_fe_idx) of the FE-FE edge the sensor lies on, or None."""
    sx, gy = float(sensor_xy[0]), float(sensor_xy[1])
    on_girder = np.where(np.abs(fe_coords[:, 1] - gy) < y_tol)[0]
    if len(on_girder) < 2:
        return None
    order = np.argsort(fe_coords[on_girder, 0])
    girder_nodes = on_girder[order]
    girder_xs = fe_coords[girder_nodes, 0]
    pos = int(np.searchsorted(girder_xs, sx))
    if pos == 0 or pos >= len(girder_xs):
        return None
    left_idx = int(girder_nodes[pos - 1])
    right_idx = int(girder_nodes[pos])
    if frozenset([left_idx, right_idx]) not in undirected_edges:
        return None
    return left_idx, right_idx


def _edge_geom(src_xy: np.ndarray, dst_xy: np.ndarray) -> tuple[float, float]:
    dx = dst_xy[0] - src_xy[0]
    dy = dst_xy[1] - src_xy[1]
    length = float(np.hypot(dx, dy))
    theta = float(np.arctan2(dy, dx))
    return length, theta


def build_graph_package(structure_name: str, overrides: dict) -> dict:
    params = _build_fe_params(overrides)
    model = build_grid_model_3d_upgraded_v22(params)
    fe_graph = build_projected_graph_from_model(model, include_sensor_map=False)

    fe_coords = np.asarray(fe_graph["coords_xy_mm"], dtype=np.float32)
    fe_edge_index = np.asarray(fe_graph["edge_index"], dtype=np.int64)
    n_fe = fe_coords.shape[0]

    txt_path = SENSOR_GRID_DIR / f"grid_{structure_name}.txt"
    sensor_rows = _load_sensor_grid(txt_path)
    n_sensors = len(sensor_rows)
    if n_sensors == 0:
        raise RuntimeError(f"No measured sensor nodes found in {txt_path}")

    # Y-snap sensors to nearest girder line, then find nearest FE node
    sensor_coords, nearest_fe_idx = _y_snap_and_find_nearest(sensor_rows, fe_coords)
    sensor_physical_ids = np.array(
        [r["physical_sensor_id"] for r in sensor_rows], dtype=np.int64
    )
    sensor_node_indices = np.arange(n_fe, n_fe + n_sensors, dtype=np.int64)

    # Build undirected FE edge set for splitting logic
    undirected_fe_edges = {
        frozenset([int(u), int(v)]) for u, v in fe_edge_index.T.tolist()
    }

    # Assign each sensor to edge-split or simple connection
    edges_to_remove: set[frozenset] = set()
    sensor_bracket: dict[int, tuple[int, int] | None] = {}
    for i in range(n_sensors):
        bracket = _find_bracketing_edge(sensor_coords[i], fe_coords, undirected_fe_edges)
        if bracket is not None:
            key = frozenset(bracket)
            if key not in edges_to_remove:
                edges_to_remove.add(key)
                sensor_bracket[i] = bracket
            else:
                sensor_bracket[i] = None  # another sensor claimed this edge
        else:
            sensor_bracket[i] = None

    # Build directed edge list: keep FE-FE edges not being split, add sensor edges
    directed_edges: list[tuple[int, int]] = []
    for u, v in fe_edge_index.T.tolist():
        if frozenset([u, v]) not in edges_to_remove:
            directed_edges.append((int(u), int(v)))

    for i in range(n_sensors):
        s_idx = int(sensor_node_indices[i])
        bracket = sensor_bracket[i]
        if bracket is not None:
            left_idx, right_idx = bracket
            directed_edges += [
                (s_idx, left_idx), (left_idx, s_idx),
                (s_idx, right_idx), (right_idx, s_idx),
            ]
        else:
            fe_idx = int(nearest_fe_idx[i])
            directed_edges += [(s_idx, fe_idx), (fe_idx, s_idx)]

    all_coords = np.vstack([fe_coords, sensor_coords])
    all_edge_index = np.array(directed_edges, dtype=np.int64).T
    all_edge_geom = np.array(
        [_edge_geom(all_coords[u], all_coords[v]) for u, v in directed_edges],
        dtype=np.float32,
    )

    snap_distances = np.array([
        float(np.linalg.norm(sensor_coords[i] - fe_coords[nearest_fe_idx[i]]))
        for i in range(n_sensors)
    ], dtype=np.float32)

    n_split = sum(1 for b in sensor_bracket.values() if b is not None)
    print(f"  {structure_name}: {n_fe} FE nodes + {n_sensors} sensor nodes "
          f"= {n_fe + n_sensors} total, {all_edge_index.shape[1]} directed edges")
    print(f"  edge-split={n_split}  simple-connect={n_sensors - n_split}")
    print(f"  snap distances (mm): min={snap_distances.min():.1f}  "
          f"max={snap_distances.max():.1f}  mean={snap_distances.mean():.1f}")

    return {
        "node_coords_xy_mm": all_coords,
        "edge_index": all_edge_index,
        "edge_attr_geom": all_edge_geom,
        "sensor_node_indices": sensor_node_indices,
        "sensor_physical_ids": sensor_physical_ids,
        "nearest_fe_node_idx": nearest_fe_idx,
        "n_fe_nodes": np.int64(n_fe),
        "structure_name": structure_name,
        "snap_distances_mm": snap_distances,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase 3 real-structure graph packages.")
    parser.add_argument(
        "--structures", nargs="+", default=list(STRUCTURES.keys()),
        choices=list(STRUCTURES.keys()),
        help="Which structures to process (default: all 5).",
    )
    parser.add_argument(
        "--output-root", type=Path, default=OUTPUT_ROOT,
        help="Root output directory (default: phase3_real_graphs/).",
    )
    args = parser.parse_args(argv)

    out_root: Path = args.output_root

    for name in args.structures:
        print(f"\nBuilding graph for {name} ...")
        pkg = build_graph_package(name, STRUCTURES[name])
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "graph_package.npz"
        np.savez(out_path, **{k: v for k, v in pkg.items() if k != "structure_name"})
        # Save structure_name as a text sidecar since np.savez can't store raw str
        (out_dir / "structure_name.txt").write_text(name, encoding="utf-8")
        print(f"  Saved -> {out_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
