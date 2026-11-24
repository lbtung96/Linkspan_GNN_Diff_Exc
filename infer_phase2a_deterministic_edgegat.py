from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict, replace as dc_replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from EdgeGATConv_Modal_GNN import (
    DEVICE,
    build_phase2a_graph_from_payload,
    load_trained_model_checkpoint,
)
from GraphSAGE_Modal_GNN import (
    compute_mode_shape_metrics,
    denormalize_frequency_predictions,
    set_global_seed,
)
from infer_phase1_unseen_structure import normalize_json_value, write_json
from infer_phase2a_unseen_structure import (
    load_frequency_normalization,
    plot_frequency_comparison,
    save_frequency_outputs,
    save_mode_shape_metrics,
    save_raw_outputs,
)
from infer_phase2a_unseen_structure_edgegat import (
    apply_input_normalization,
    load_input_normalization,
)
from generate_population_dataset_modes4 import (
    MODE_MATCH_CONFIG,
    MODE_NAMES,
    N_CANDIDATE_MODES,
    REF_DIR,
    REF_FILE_PATTERN,
    REF_FILE_PATTERN_NG2_BALCONY,
    REF_FILE_PATTERN_NG2_NO_BAL,
    REF_SENSOR_TOL_X,
    REF_X_FRACS,
    _candidate_channels_from_solve,
    _candidate_metadata_from_solve,
    assemble_phase1_sample_payload,
    compute_S,
    load_reference_library_for_ng_variant,
    run_solver,
)
from standalone_mode_matching_mmi import match_target_modes
from grid_modal_Kinematic_MPC import default_params_grid_3d_offset_upgraded_v22


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_mode_shapes_combined(
    output_dir: Path,
    coords_xy: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    mode_names: tuple[str, ...],
    structure_name: str,
) -> None:
    """3-column figure (Target | Prediction | Difference) matching Phase 2 style."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(
        nrows=len(mode_names), ncols=3,
        figsize=(12, 3.2 * len(mode_names)), squeeze=False,
    )
    for mode_idx, mode_name in enumerate(mode_names):
        pred_col = prediction[:, mode_idx]
        target_col = target[:, mode_idx]

        pred_scale = float(np.max(np.abs(pred_col))) or 1.0
        pred_norm = pred_col / pred_scale
        target_scale = float(np.max(np.abs(target_col))) or 1.0
        target_norm = target_col / target_scale

        diff_col = pred_norm - target_norm
        diff_lim = float(max(np.max(np.abs(diff_col)), 1e-8))

        panels = [
            ("Target (norm.)",     target_norm, -1.0,      1.0),
            ("Prediction (norm.)", pred_norm,   -1.0,      1.0),
            ("Difference",         diff_col,    -diff_lim, diff_lim),
        ]
        for col_idx, (title, values, vmin, vmax) in enumerate(panels):
            ax = axes[mode_idx, col_idx]
            sc = ax.scatter(coords_xy[:, 0], coords_xy[:, 1], c=values,
                            cmap=_MODE_SHAPE_CMAP, s=28, vmin=vmin, vmax=vmax,
                            edgecolors="#333333", linewidths=0.4)
            ax.set_title(f"{mode_name} {title}")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.2)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Mode Shape Comparison: {structure_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "mode_shapes_compare_all.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# SSI comparison helpers (used only when --ssi-dir is given)
# ---------------------------------------------------------------------------

_SENSOR_TXT_EXCLUDED_NODES = frozenset({1, 2, 19, 20})  # supports + slave nodes

def _parse_sensor_txt_for_ssi(txt_path: Path) -> np.ndarray:
    """Return measured sensor x,y coords in mm from grid_<structure>.txt, sorted by node_id.
    Nodes 1, 2 (supports) and 19, 20 (slaves) are always excluded."""
    rows = []
    with open(txt_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip().lstrip("\ufeff")
            if not stripped or stripped.startswith(("#", "%", "//")):
                continue
            tokens = stripped.replace(",", " ").split()
            if len(tokens) < 4:
                continue
            try:
                node_id = int(float(tokens[0]))
                x_mm = float(tokens[1]) * 1000.0
                y_mm = float(tokens[2]) * 1000.0
            except ValueError:
                continue
            if node_id in _SENSOR_TXT_EXCLUDED_NODES:
                continue
            rows.append((node_id, x_mm, y_mm))
    rows.sort(key=lambda r: r[0])
    return np.array([[x, y] for _, x, y in rows], dtype=float)


def _load_ssi_data(ssi_dir: Path, structure_name: str, ssi_stem: str | None = None) -> dict:
    """Load SSI results from <structure>_Data_picked.mat and sensor coords from grid_<structure>.txt.

    Returns dict with keys: ssi_freq_all_hz, ssi_phi_u3, sensor_xy_mm, n_sensors, n_ssi_modes.
    uz DOFs are extracted as rows [2, 5, 8, ...] (3rd component per sensor, 0-indexed).
    ssi_stem: override the mat file stem (default: structure_name), useful when the
              _Data_picked.mat filename has a different spelling than the structure name.
    """
    import scipy.io
    stem = ssi_stem if ssi_stem is not None else structure_name
    mat_path = ssi_dir / f"{stem}_Data_picked.mat"
    txt_path = ssi_dir / f"grid_{structure_name}.txt"
    if not mat_path.exists():
        raise FileNotFoundError(f"SSI mat file not found: {mat_path}")
    if not txt_path.exists():
        raise FileNotFoundError(f"Sensor grid txt not found: {txt_path}")

    mat = scipy.io.loadmat(str(mat_path))
    ssi_freq_all_hz = np.asarray(mat["Extracted_frequencies"][:, 1], dtype=float)
    phi_sorted = np.asarray(mat["phi_sorted"], dtype=float)  # (n_channels, n_ssi_modes)

    n_channels, n_ssi_modes = phi_sorted.shape
    if n_channels % 3 != 0:
        raise ValueError(f"phi_sorted has {n_channels} rows — expected n_sensors × 3 DOFs.")
    n_sensors = n_channels // 3
    uz_indices = np.arange(2, n_channels, 3)
    ssi_phi_u3 = phi_sorted[uz_indices, :]  # (n_sensors, n_ssi_modes)

    sensor_xy_mm = _parse_sensor_txt_for_ssi(txt_path)
    if sensor_xy_mm.shape[0] != n_sensors:
        raise ValueError(
            f"Sensor txt has {sensor_xy_mm.shape[0]} measured rows but phi_sorted implies {n_sensors} sensors. "
            "Check that support/slave nodes are excluded correctly."
        )
    return {
        "ssi_freq_all_hz": ssi_freq_all_hz,
        "ssi_phi_u3": ssi_phi_u3,
        "sensor_xy_mm": sensor_xy_mm,
        "n_sensors": n_sensors,
        "n_ssi_modes": n_ssi_modes,
        "pre_ordered": _ssi_file_is_pre_ordered(mat, n_ssi_modes),
        "source_mode_index": (
            np.asarray(mat["source_mode_index"], dtype=int).reshape(-1)
            if "source_mode_index" in mat else None
        ),
    }


def _ssi_file_is_pre_ordered(mat: dict, n_ssi_modes: int) -> bool:
    """True when the .mat already stores exactly B1/T1/B2/T2 in that order.

    Files written by SSI_modal_infor/build_ssi_modal_infor.py carry a
    ``mode_names`` variable declaring the curated ordering, so no MAC matching is
    needed (and must not be applied — see _assign_ssi_modes_identity). Legacy
    field files carry every identified mode and no ``mode_names``, so they fall
    back to _mac_match_ssi_modes.
    """
    if n_ssi_modes != len(MODE_NAMES) or "mode_names" not in mat:
        return False
    raw = np.asarray(mat["mode_names"])
    names = ["".join(np.asarray(row).ravel().tolist()).strip() for row in raw]
    return names == list(MODE_NAMES)


def _assign_ssi_modes_identity(
    fe_target_at_sensors: np.ndarray,
    ssi_phi_u3: np.ndarray,
    ssi_freq_all_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Take the SSI modes in the order the file already stores them.

    Same signature and return contract as _mac_match_ssi_modes, but the column
    assignment is the identity instead of a per-target argmax over MAC. Only the
    sign is still resolved against the FE target. The returned MAC is therefore
    a *diagnostic* of how well the curated mode matches the FE target, not the
    objective that chose it.

    Used for pre-ordered files (see _ssi_file_is_pre_ordered). Re-running the
    argmax matcher on a 4-mode curated file is not equivalent: when this was
    written (2026-08-26), the FE T2 target at Hellesylt and Leknes scored higher
    against the curated T1 mode than against the curated T2 mode, so the matcher
    would have duplicated T1 and dropped T2. The 2026-09-18
    mode_candidate_override changed both FE targets and removed that particular
    collision, but argmax still has no uniqueness constraint - keep the identity.
    """
    fe = np.asarray(fe_target_at_sensors, dtype=float)
    phi = np.asarray(ssi_phi_u3, dtype=float)
    freq = np.asarray(ssi_freq_all_hz, dtype=float).reshape(-1)

    n_sensors, n_target = fe.shape
    if phi.shape != (n_sensors, n_target):
        raise ValueError(
            f"Pre-ordered SSI phi has shape {phi.shape}, expected {(n_sensors, n_target)}."
        )
    if freq.size != n_target:
        raise ValueError(f"Pre-ordered SSI has {freq.size} frequencies, expected {n_target}.")

    ssi_phi_matched = np.zeros((n_sensors, n_target), dtype=float)
    ssi_mac = np.zeros(n_target, dtype=float)
    for m in range(n_target):
        fe_vec = fe[:, m]
        ssi_vec = phi[:, m].copy()
        if np.dot(fe_vec, ssi_vec) < 0:
            ssi_vec = -ssi_vec
        ssi_phi_matched[:, m] = ssi_vec
        denom = float(np.dot(fe_vec, fe_vec) * np.dot(ssi_vec, ssi_vec))
        ssi_mac[m] = float(np.dot(fe_vec, ssi_vec) ** 2 / denom) if denom > 1e-24 else 0.0

    return ssi_phi_matched, freq.copy(), ssi_mac


def resolve_ssi_modes(
    fe_target_at_sensors: np.ndarray,
    ssi_data: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to identity assignment or MAC matching based on the file itself."""
    if ssi_data.get("pre_ordered"):
        return _assign_ssi_modes_identity(
            fe_target_at_sensors, ssi_data["ssi_phi_u3"], ssi_data["ssi_freq_all_hz"]
        )
    return _mac_match_ssi_modes(
        fe_target_at_sensors, ssi_data["ssi_phi_u3"], ssi_data["ssi_freq_all_hz"]
    )


def _mac_match_ssi_modes(
    fe_target_at_sensors: np.ndarray,
    ssi_phi_u3: np.ndarray,
    ssi_freq_all_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each of 4 FE target modes, pick the SSI mode with highest MAC.

    Returns: ssi_phi_matched (n_sensors, 4), ssi_freq_matched (4,), ssi_mac (4,).
    Sign of each matched SSI mode is flipped to align with FE target.
    """
    n_sensors, n_target = fe_target_at_sensors.shape
    ssi_phi_matched = np.zeros((n_sensors, n_target), dtype=float)
    ssi_freq_matched = np.zeros(n_target, dtype=float)
    ssi_mac = np.zeros(n_target, dtype=float)

    for m in range(n_target):
        fe_vec = fe_target_at_sensors[:, m]
        fe_norm2 = np.dot(fe_vec, fe_vec)
        best_mac, best_k = -1.0, 0
        for k in range(ssi_phi_u3.shape[1]):
            ssi_vec = ssi_phi_u3[:, k]
            denom = fe_norm2 * np.dot(ssi_vec, ssi_vec)
            if denom < 1e-24:
                continue
            mac_val = float(np.dot(fe_vec, ssi_vec) ** 2 / denom)
            if mac_val > best_mac:
                best_mac, best_k = mac_val, k
        ssi_vec = ssi_phi_u3[:, best_k].copy()
        if np.dot(fe_vec, ssi_vec) < 0:
            ssi_vec = -ssi_vec
        ssi_phi_matched[:, m] = ssi_vec
        ssi_freq_matched[m] = ssi_freq_all_hz[best_k]
        ssi_mac[m] = best_mac

    return ssi_phi_matched, ssi_freq_matched, ssi_mac


def _make_mode_shape_cmap():
    import matplotlib.colors as mcolors
    # blue (-1) → white (0) → red (+1); midpoint at 0.5 = value 0
    return mcolors.LinearSegmentedColormap.from_list(
        "mode_shape",
        [(0.0, "#3B4CC0"), (0.5, "#FFFFFF"), (1.0, "#B40426")],
    )


_MODE_SHAPE_CMAP = _make_mode_shape_cmap()


def _plot_mode_shapes_with_ssi(
    output_dir: Path,
    coords_xy: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    sensor_xy_mm: np.ndarray,
    ssi_phi_matched: np.ndarray,
    mode_names: tuple[str, ...],
    structure_name: str,
    n_fe_nodes: int | None = None,
) -> None:
    """3-column figure: FE Target | Prediction (norm.) | SSI sparse sensors."""
    import matplotlib.pyplot as plt

    n_fe = n_fe_nodes if n_fe_nodes is not None else len(coords_xy)
    fe_coords = coords_xy[:n_fe]

    fig, axes = plt.subplots(
        nrows=len(mode_names), ncols=3,
        figsize=(13, 3.2 * len(mode_names)), squeeze=False,
    )
    for mode_idx, mode_name in enumerate(mode_names):
        target_col = target[:n_fe, mode_idx]
        pred_col = prediction[:, mode_idx]
        ssi_col = ssi_phi_matched[:, mode_idx]

        pred_scale = float(np.max(np.abs(pred_col))) or 1.0
        pred_norm = pred_col / pred_scale

        ssi_scale = float(np.max(np.abs(ssi_col))) or 1.0
        ssi_norm = ssi_col / ssi_scale

        panels = [
            ("FE Target",            fe_coords,    target_col, -1.0, 1.0, 28),
            ("Prediction (norm.)",   coords_xy,    pred_norm,  -1.0, 1.0, 28),
            ("SSI (sensors, norm.)", sensor_xy_mm, ssi_norm,   -1.0, 1.0, 80),
        ]
        for col_idx, (title, xy, values, vmin, vmax, dot_s) in enumerate(panels):
            ax = axes[mode_idx, col_idx]
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=_MODE_SHAPE_CMAP,
                            s=dot_s, vmin=vmin, vmax=vmax,
                            edgecolors="#333333", linewidths=0.4)
            ax.set_title(f"{mode_name} — {title}")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.2)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Mode Shape Comparison (with SSI): {structure_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "mode_shapes_compare_ssi.png", dpi=180)
    plt.close(fig)


def _plot_frequency_3bar(
    output_dir: Path,
    pred_freq_hz: np.ndarray,
    target_freq_hz: np.ndarray,
    ssi_freq_matched: np.ndarray,
    ssi_mac: np.ndarray,
    mode_names: tuple[str, ...],
    structure_name: str,
) -> None:
    """3-bar grouped frequency plot: Predicted / FE Target / SSI."""
    import matplotlib.pyplot as plt

    x = np.arange(len(mode_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width, pred_freq_hz.reshape(-1),       width, label="Predicted",  color="#1f77b4")
    ax.bar(x,          target_freq_hz.reshape(-1),     width, label="FE Target",  color="#ff7f0e")
    ax.bar(x + width,  ssi_freq_matched.reshape(-1),  width, label="SSI",        color="#2ca02c")
    for i, (mac_val, xi) in enumerate(zip(ssi_mac, x)):
        ax.text(xi + width, float(ssi_freq_matched.reshape(-1)[i]) + 0.02,
                f"MAC={mac_val:.2f}", ha="center", va="bottom", fontsize=7, color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(mode_names)
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(f"Frequency Comparison (with SSI): {structure_name}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "frequency_compare_ssi.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Structure registry
# Each entry is a partial override applied on top of
# default_params_grid_3d_offset_upgraded_v22().  Nested keys (bottom_beam,
# balcony, nsm) are shallow-merged into the corresponding default sub-dict.
#
# The GEOMETRY here is a verbatim mirror of ``build_phase3_graphs.STRUCTURES``,
# which CLAUDE.md declares canonical because it generated every stored artefact.
# Synced 2026-09-02, when Magerholm and Rykkjem were added and Hundeidvik,
# Leknes, Saebo, Liabygda and Stranda were re-parameterised.  If you edit one
# dict, edit the other in the same commit: E5/stage0_checks.py diffs them field
# by field, but only after the fact.
#
# ``mode_col_swap`` (Leknes) and ``mode_candidate_override`` (Hellesylt,
# Leknes) are the two intentional additions - mode-matching corrections, not
# geometry, so the graph builder has no counterpart for either. Anything that
# builds FE params from build_phase3_graphs.STRUCTURES must carry both across.
# An empty dict {} is NOT "use all defaults": the defaults are Magerholm's
# geometry, so an empty entry silently builds the wrong structure.
# ---------------------------------------------------------------------------
REAL_STRUCTURE_PARAMS: dict[str, dict[str, Any]] = {
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
        # The matcher's T2 (candidate #6, 11.01 Hz) twists the whole deck one
        # way; the measured T2 (9.445 Hz) reverses its twist just past the
        # support. Candidate #7 (13.14 Hz) has that shape: MAC 0.876 against
        # SSI T2 (0.093 for #6), twist-only MAC 0.929. Chosen 2026-09-18 by one
        # rule applied to all 36 cells of Table 8 - the best FE mode within
        # 0.5-2x the SSI frequency, adopted if it beats the matcher by >= 0.05 -
        # which changes exactly this cell and Leknes T2. See
        # experiments/E2_training_distribution/t2_mac_diagnostic/.
        "mode_candidate_override": {"T2": {"candidate": 7, "freq_hz": 13.137}},
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
        # The MAC matcher assigns B2 and T2 in the wrong order; swap columns
        # 2<->3 after assembly.  Not geometry, so there is no counterpart in
        # build_phase3_graphs.STRUCTURES.
        #
        # Re-verified 2026-09-02 under the new geometry.  The original reason
        # (B2 8.255 Hz and T2 8.049 Hz only 206 mHz apart) no longer applies -
        # the gap is now 891 mHz - but the swap is still right, and SSI is what
        # settles it.  Unswapped the matcher returns B2=9.327 / T2=8.437, i.e.
        # B2 above T2; SSI has B2=8.546 below T2=9.974.  Swapped, FE reads
        # B2=8.437 / T2=9.327 and the ordering agrees with SSI.
        "mode_col_swap": [2, 3],
        # Applied in final slot names, i.e. after the swap above. The swapped-in
        # T2 (candidate #5, 9.33 Hz) is a near-uniform twist of the whole deck:
        # MAC 0.011 against SSI T2. Candidate #6 (12.71 Hz) is the best FE mode
        # within 0.5-2x the SSI frequency, MAC 0.412 - still weak; the measured
        # T2 is 23 % symmetric, capping any FE torsion mode at 0.77. Same rule
        # and folder as Hellesylt's override.
        "mode_candidate_override": {"T2": {"candidate": 6, "freq_hz": 12.712}},
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


def _build_params(structure_name: str) -> dict:
    """Return a full params dict for *structure_name* by applying its partial
    override on top of the default params.  Nested sub-dicts are shallow-merged
    so that only explicitly overridden keys change."""
    base = default_params_grid_3d_offset_upgraded_v22()
    overrides = REAL_STRUCTURE_PARAMS[structure_name]
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **val}
        else:
            base[key] = val
    return base


# ---------------------------------------------------------------------------
# Core payload assembly
# ---------------------------------------------------------------------------

def apply_mode_candidate_override(match_res, override: dict | None,
                                  mode_col_swap, freq_all_hz: np.ndarray):
    """Replace the matcher's FE candidate for named slots with a declared one.

    ``override`` maps a final slot name to ``{"candidate": k, "freq_hz": f}``.
    Final means after ``mode_col_swap``: the swap is applied to the assembled
    payload later, so an override of a swapped slot is written into the other
    matcher slot here. Each mode is sign-aligned against the reference template
    of its matcher slot, so an overridden candidate inherits the sign convention
    the slot already had.

    The returned result is used only to build LABELS
    (assemble_deterministic_phase2a_payload copies the overridden slot's target
    fields from a second assembly). The simulated input keeps the matcher's
    selection, as it does under mode_col_swap, so
    E6/task2_propagation_only_control.resimulate_deterministic does not apply it.

    ``freq_hz`` guards the index: candidate numbering is eigsh's ascending
    order, and a geometry change can shift it. A mismatch beyond 0.1 % raises
    instead of silently overriding with a different mode.
    """
    if not override:
        return match_res
    sel = np.asarray(match_res.selected_idx, dtype=int).copy()
    mac = np.asarray(match_res.selected_mac, dtype=float).copy()
    scores = np.asarray(match_res.selected_scores, dtype=float).copy()
    comp = np.asarray(match_res.selected_composite, dtype=float).copy()
    applied: dict[str, dict[str, Any]] = {}
    for name, spec in override.items():
        final = list(MODE_NAMES).index(name)
        slot = final
        if mode_col_swap is not None:
            i, j = int(mode_col_swap[0]), int(mode_col_swap[1])
            slot = {i: j, j: i}.get(final, final)
        k = int(spec["candidate"])
        f_want = float(spec["freq_hz"])
        f_got = float(np.asarray(freq_all_hz, dtype=float)[k])
        if abs(f_got - f_want) > 1e-3 * f_want:
            raise ValueError(
                f"mode_candidate_override {name}: candidate {k} is {f_got:.4f} Hz, "
                f"declared {f_want:.4f} Hz. The FE model changed; re-derive the override "
                "(experiments/E2_training_distribution/t2_mac_diagnostic/).")
        others = np.delete(sel, slot)
        if k in others:
            raise ValueError(f"mode_candidate_override {name}: candidate {k} is already "
                             f"selected for another slot ({sel.tolist()}).")
        applied[name] = {"matcher_slot": int(slot), "matcher_candidate": int(sel[slot]),
                         "candidate": k, "freq_hz": f_got}
        sel[slot] = k
        # the matcher's scores describe the candidate it chose, not this one
        mac[slot] = np.nan
        scores[slot] = np.nan
        comp[slot] = np.nan
    diagnostics = dict(match_res.diagnostics)
    diagnostics["mode_candidate_override"] = applied
    warnings = list(match_res.warning_messages) + [
        f"mode_candidate_override applied: {applied}"]
    return dc_replace(match_res, selected_idx=sel, selected_mac=mac,
                      selected_scores=scores, selected_composite=comp,
                      diagnostics=diagnostics, warning_messages=warnings)


# Per-slot LABEL fields of an assembled payload: (n, 4) columns and (4,) vectors.
# Features (x_psd, edge_attr_*) are deliberately absent - a mode-matching
# correction relabels, it does not change what the network is shown.
_PAYLOAD_LABEL_COL_KEYS = (
    "y_mode_shape_abs", "y_mode_shape_signed", "phi_u3_sensor_4",
    "phi_u3_sensor_4_signed", "phi_u3_sensor_4_signed_norm",
)
_PAYLOAD_LABEL_VEC_KEYS = (
    "y_freq_hz", "matched_freq_hz", "y_mode_shape_scale", "y_mode_shape_signed_scale",
    "matched_mac", "matched_composite_score", "matched_ambiguity_flag",
    "sign_flip_factors", "sign_reference_dot_before", "sign_reference_dot_after",
    "sign_fallback_used", "sign_ambiguous_flag", "matched_mode_indices",
    "selected_mode_indices_sensor_domain", "sign_reference_sensor_index",
    "sign_reference_sensor_node_id", "sign_fallback_reason", "S4",
)


def _take_payload_label_slot(dst: dict, src: dict, slot: int) -> None:
    """Copy one mode slot's label fields from ``src`` into ``dst`` in place."""
    for key in _PAYLOAD_LABEL_COL_KEYS:
        if key in src and key in dst:
            arr = np.asarray(dst[key]).copy()
            arr[:, slot] = np.asarray(src[key])[:, slot]
            dst[key] = arr
    for key in _PAYLOAD_LABEL_VEC_KEYS:
        if key in src and key in dst and np.asarray(dst[key]).shape[:1] == (4,):
            arr = np.asarray(dst[key]).copy()
            arr[slot] = np.asarray(src[key])[slot]
            dst[key] = arr


def _swap_payload_mode_cols(payload: dict, i: int, j: int) -> None:
    """Swap mode columns i and j in-place across all mode-indexed payload fields."""
    for key in ("y_mode_shape_abs", "y_mode_shape_signed",
                "phi_u3_sensor_4", "phi_u3_sensor_4_signed", "phi_u3_sensor_4_signed_norm"):
        if key in payload:
            arr = np.asarray(payload[key]).copy()
            arr[:, [i, j]] = arr[:, [j, i]]
            payload[key] = arr
    for key in ("y_freq_hz", "matched_freq_hz", "y_mode_shape_scale",
                "y_mode_shape_signed_scale", "matched_mac", "matched_composite_score",
                "matched_ambiguity_flag", "sign_flip_factors",
                "sign_reference_dot_before", "sign_reference_dot_after",
                "sign_fallback_used", "sign_ambiguous_flag",
                "matched_mode_indices", "selected_mode_indices_sensor_domain"):
        if key in payload:
            arr = np.asarray(payload[key]).copy()
            arr[[i, j]] = arr[[j, i]]
            payload[key] = arr
    # edge_attr_cpsd: (E, 8) packed as [Re0,Im0, Re1,Im1, Re2,Im2, Re3,Im3]
    if "edge_attr_cpsd" in payload:
        arr = np.asarray(payload["edge_attr_cpsd"]).copy()
        arr[:, [2*i, 2*i+1, 2*j, 2*j+1]] = arr[:, [2*j, 2*j+1, 2*i, 2*i+1]]
        payload["edge_attr_cpsd"] = arr


def assemble_deterministic_phase2a_payload(
    params_user: dict,
    *,
    structure_name: str = "custom",
    min_mac_threshold: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the phase-2A sample payload for a fully specified FE parameter set.

    Mirrors ``assemble_unseen_phase2a_payload`` from infer_phase2a_unseen_structure.py
    but skips random sampling — the caller supplies the complete params dict.
    """
    params = deepcopy(params_user)
    params["nmodes"] = N_CANDIDATE_MODES  # ensure solver computes enough modes
    mode_col_swap = params.pop("mode_col_swap", None)  # pop before FE solver
    mode_override = params.pop("mode_candidate_override", None)

    solve_res = run_solver(params, ref_x_fracs=REF_X_FRACS, tol_x=REF_SENSOR_TOL_X)
    ref_lib = load_reference_library_for_ng_variant(REF_DIR, int(params["Ng"]), params)

    match_cfg = (
        dc_replace(MODE_MATCH_CONFIG, min_mac_threshold=min_mac_threshold)
        if min_mac_threshold is not None else MODE_MATCH_CONFIG
    )
    match_res = match_target_modes(
        phi_u3_candidates=solve_res.phi_u3_rep,
        freq_candidates_hz=solve_res.freq_all_hz,
        ref_lib=ref_lib,
        config=match_cfg,
        candidate_channels=_candidate_channels_from_solve(solve_res, match_cfg.signature_config),
        candidate_participation=solve_res.participation,
        candidate_metadata=_candidate_metadata_from_solve(solve_res),
    )

    ng = int(params["Ng"])
    if ng == 2:
        balcony_on = bool((params.get("balcony") or {}).get("enable", False))
        ref_fname = REF_FILE_PATTERN_NG2_BALCONY if balcony_on else REF_FILE_PATTERN_NG2_NO_BAL
        ref_path = REF_DIR / ref_fname
    else:
        ref_path = REF_DIR / REF_FILE_PATTERN.format(Ng=ng)

    def _assemble(res) -> dict[str, Any]:
        idx = np.asarray(res.selected_idx, dtype=int).reshape(-1)
        if idx.shape != (4,):
            raise ValueError(f"Expected 4 matched modes, got shape {idx.shape}.")
        phi_u3_4 = np.asarray(solve_res.phi_u3_rep[:, idx], dtype=float)
        return assemble_phase1_sample_payload(
            sample_id=-1,
            sampled=params,  # full params dict — has Ng, Nx, and is JSON-serialisable
            solve_res=solve_res,
            match_res=res,
            matcher_config_json=json.dumps(asdict(MODE_MATCH_CONFIG), sort_keys=True),
            reference_file=str(ref_path),
            freqs_4=np.asarray(solve_res.freq_all_hz[idx], dtype=float),
            phi_u3_4=phi_u3_4,
            S4=np.asarray(compute_S(phi_u3_4), dtype=float),
        )

    # Features come from the matcher's selection, exactly as the training
    # generator would simulate this FE model.
    payload = _assemble(match_res)

    # A candidate override is a LABEL correction, like mode_col_swap: it replaces
    # the target fields of its slot and leaves the simulated input alone. The
    # label fields are taken from a second assembly over the same eigen-solution,
    # so they carry the pipeline's own sign convention and normalisation. (Letting
    # the override change the simulated set was tried on 2026-09-18 and rejected:
    # the network had never seen that mode set and its FE-simulated "ceiling" at
    # Hellesylt fell in every mode, B1 0.870 -> 0.367 - an input artefact, not a
    # property of the structure. PROVENANCE Phase I.)
    label_res = apply_mode_candidate_override(
        match_res, mode_override, mode_col_swap, solve_res.freq_all_hz)
    if label_res is not match_res:
        applied = label_res.diagnostics["mode_candidate_override"]
        label_payload = _assemble(label_res)
        for spec in applied.values():
            _take_payload_label_slot(payload, label_payload, int(spec["matcher_slot"]))
        payload["mode_candidate_override_json"] = np.asarray(
            json.dumps(normalize_json_value(applied), sort_keys=True))
    payload["sample_identifier"] = np.asarray(structure_name)

    if mode_col_swap is not None:
        i, j = int(mode_col_swap[0]), int(mode_col_swap[1])
        _swap_payload_mode_cols(payload, i, j)

    metadata: dict[str, Any] = {
        "structure_name": structure_name,
        "params": normalize_json_value(params),
        "matched_mode_indices": np.asarray(payload["matched_mode_indices"]).tolist(),
        "target_freq_hz": np.asarray(payload["y_freq_hz"]).tolist(),
        "mode_names": list(MODE_NAMES),
        "mode_candidate_override": normalize_json_value(mode_override),
        "match_result_json": json.loads(match_res.to_json()),
        "reference_file": str(ref_path),
    }
    return payload, metadata


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------

def run_inference_case(
    *,
    run_dir: Path,
    output_root: Path,
    case_name: str,
    params_user: dict,
    structure_name: str,
    checkpoint_path: Path | None,
    shape_target_mode: str,
    edge_feature_mode: str,
    ssi_dir: Path | None = None,
    ssi_stem: str | None = None,
) -> Path:
    output_dir = output_root / case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else run_dir / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model, checkpoint = load_trained_model_checkpoint(ckpt_path, device=DEVICE)
    checkpoint_config = dict(checkpoint["config"])
    if checkpoint_config.get("shape_target_mode", "signed") != shape_target_mode:
        raise ValueError("Explicit shape_target_mode does not match checkpoint config.")
    if checkpoint_config.get("edge_feature_mode", "cpsd_full") != edge_feature_mode:
        raise ValueError("Explicit edge_feature_mode does not match checkpoint config.")

    payload, metadata = assemble_deterministic_phase2a_payload(
        params_user, structure_name=structure_name
    )
    graph, freq_target_hz = build_phase2a_graph_from_payload(
        payload,
        edge_feature_mode=edge_feature_mode,
        shape_target_mode=shape_target_mode,
    )
    apply_input_normalization(graph, load_input_normalization(run_dir))
    graph = graph.to(DEVICE)

    freq_stats = load_frequency_normalization(checkpoint_path=ckpt_path, checkpoint=checkpoint, run_dir=run_dir)
    with torch.no_grad():
        outputs = model(graph, graph.ndata["x"], graph.edata["edge_x"])
        pred_mode_shape = outputs.pred_mode_shape_abs.detach().cpu().numpy()
        pred_freq_z = outputs.pred_freq.detach().cpu().numpy()
        pred_freq_hz = denormalize_frequency_predictions(outputs.pred_freq, freq_stats).detach().cpu().numpy()

    target_mode_shape = graph.ndata["y"].detach().cpu().numpy()
    target_freq_hz = freq_target_hz.unsqueeze(0).detach().cpu().numpy()
    coords_xy = np.asarray(payload["coords_xy_mm"], dtype=np.float32)
    mode_names = tuple(str(x) for x in np.asarray(payload["target_mode_names"]).reshape(-1).tolist())
    mode_shape_metrics = compute_mode_shape_metrics(pred_mode_shape, target_mode_shape, mode_names)
    save_mode_shape_metrics(output_dir, mode_shape_metrics, mode_names)
    frequency_summary = save_frequency_outputs(
        output_dir,
        pred_freq_z=pred_freq_z,
        pred_freq_hz=pred_freq_hz,
        mode_names=mode_names,
        target_freq_hz=target_freq_hz,
    )
    save_raw_outputs(
        output_dir,
        payload=payload,
        pred_mode_shape_abs=pred_mode_shape,
        target_mode_shape_abs=target_mode_shape,
        pred_freq_z=pred_freq_z,
        pred_freq_hz=pred_freq_hz,
        target_freq_hz=target_freq_hz,
        mode_shape_metrics=mode_shape_metrics,
        frequency_summary=frequency_summary,
    )
    _plot_mode_shapes_combined(output_dir, coords_xy, pred_mode_shape, target_mode_shape, mode_names, structure_name)
    plot_frequency_comparison(
        output_dir,
        pred_freq_hz=np.asarray(pred_freq_hz, dtype=np.float32).reshape(-1),
        mode_names=mode_names,
        target_freq_hz=np.asarray(target_freq_hz, dtype=np.float32).reshape(-1),
    )

    ssi_summary: dict | None = None
    if ssi_dir is not None:
        ssi_data = _load_ssi_data(ssi_dir, structure_name, ssi_stem)
        sensor_xy = ssi_data["sensor_xy_mm"]
        # nearest FE node for each sensor position
        dists = np.sqrt(np.sum((coords_xy[None, :, :] - sensor_xy[:, None, :]) ** 2, axis=2))
        nearest_fe_idx = np.argmin(dists, axis=1)
        fe_target_at_sensors = target_mode_shape[nearest_fe_idx, :]
        ssi_phi_matched, ssi_freq_matched, ssi_mac = resolve_ssi_modes(
            fe_target_at_sensors, ssi_data
        )
        _plot_mode_shapes_with_ssi(
            output_dir, coords_xy, pred_mode_shape, target_mode_shape,
            sensor_xy, ssi_phi_matched, mode_names, structure_name,
        )
        _plot_frequency_3bar(
            output_dir,
            pred_freq_hz=np.asarray(pred_freq_hz, dtype=float).reshape(-1),
            target_freq_hz=np.asarray(target_freq_hz, dtype=float).reshape(-1),
            ssi_freq_matched=ssi_freq_matched,
            ssi_mac=ssi_mac,
            mode_names=mode_names,
            structure_name=structure_name,
        )
        ssi_summary = {
            "ssi_freq_matched_hz": ssi_freq_matched.tolist(),
            "ssi_mac_vs_fe_target": ssi_mac.tolist(),
            "n_ssi_candidate_modes": int(ssi_data["n_ssi_modes"]),
            "n_sensors": int(ssi_data["n_sensors"]),
            "ssi_mode_assignment": (
                "curated_order" if ssi_data.get("pre_ordered") else "argmax_mac"
            ),
            "ssi_source_mode_index": (
                ssi_data["source_mode_index"].tolist()
                if ssi_data.get("source_mode_index") is not None else None
            ),
        }
        print("SSI MAC vs FE target:", {m: f"{v:.3f}" for m, v in zip(mode_names, ssi_mac)})

    summary = {
        "case_name": case_name,
        "output_dir": str(output_dir),
        "run_dir": str(run_dir),
        "checkpoint_path": str(ckpt_path),
        "edge_feature_mode": edge_feature_mode,
        "shape_target_mode": shape_target_mode,
        "structure_name": structure_name,
        "mode_names": list(mode_names),
        "sample_identifier": str(payload.get("sample_identifier", structure_name)),
        "n_nodes": int(pred_mode_shape.shape[0]),
        "n_modes": int(pred_mode_shape.shape[1]),
        "pred_freq_z": np.asarray(pred_freq_z, dtype=np.float32).reshape(-1).tolist(),
        "pred_freq_hz": np.asarray(pred_freq_hz, dtype=np.float32).reshape(-1).tolist(),
        "target_freq_hz": np.asarray(target_freq_hz, dtype=np.float32).reshape(-1).tolist(),
        "mode_shape_metrics": mode_shape_metrics,
        "frequency_summary": frequency_summary,
        "ssi_summary": ssi_summary,
        "metadata": normalize_json_value(metadata),
    }
    write_json(output_dir / "summary.json", summary)
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot EdgeGAT Phase-2A inference for a deterministic (named) real structure."
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Training run directory containing checkpoints/ and normalization files.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Root output directory (default: experiments/E1_architecture_ablation/inference/<run_name>/).")
    parser.add_argument("--checkpoint-path", type=Path, default=None,
                        help="Explicit checkpoint .pt file; defaults to <run_dir>/checkpoints/best_model.pt.")
    parser.add_argument("--case-name", type=str, required=True,
                        help="Sub-directory name for this inference run's outputs.")
    parser.add_argument("--structure-name", type=str, default="Hellesylt",
                        help=f"Named structure from REAL_STRUCTURE_PARAMS. Available: {list(REAL_STRUCTURE_PARAMS)}")
    parser.add_argument("--params-json", type=Path, default=None,
                        help="Path to a JSON file containing a full params dict; bypasses REAL_STRUCTURE_PARAMS.")
    parser.add_argument("--seed", type=int, default=20260325,
                        help="Global RNG seed for white-noise simulation reproducibility.")
    parser.add_argument("--shape-target-mode", choices=["signed", "abs"], default="signed")
    parser.add_argument("--edge-feature-mode", choices=["cpsd_full", "cpsd_resonance"], default="cpsd_full")
    parser.add_argument("--ssi-dir", type=Path, default=None,
                        help="Folder containing <structure>_Data_picked.mat and grid_<structure>.txt. "
                             "Enables SSI comparison plots (mode_shapes_compare_ssi.png, frequency_compare_ssi.png).")
    parser.add_argument("--ssi-stem", type=str, default=None,
                        help="Override the filename stem used to locate <stem>_Data_picked.mat "
                             "(default: structure_name). Useful when the file has a typo, e.g. Eisdal vs Eidsdal.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    if args.params_json is not None:
        with open(args.params_json, "r", encoding="utf-8") as f:
            params_user = json.load(f)
        structure_name = args.case_name
    elif args.structure_name in REAL_STRUCTURE_PARAMS:
        params_user = _build_params(args.structure_name)
        structure_name = args.structure_name
    else:
        raise ValueError(
            f"Unknown structure '{args.structure_name}'. "
            f"Available: {list(REAL_STRUCTURE_PARAMS)}"
        )

    run_dir = Path(args.run_dir)
    output_root = args.output_dir or Path("experiments") / "E1_architecture_ablation" / "inference" / run_dir.name
    output_dir = run_inference_case(
        run_dir=run_dir,
        output_root=output_root,
        case_name=args.case_name,
        params_user=params_user,
        structure_name=structure_name,
        checkpoint_path=args.checkpoint_path,
        shape_target_mode=args.shape_target_mode,
        edge_feature_mode=args.edge_feature_mode,
        ssi_dir=args.ssi_dir,
        ssi_stem=args.ssi_stem,
    )
    print(f"Inference output directory: {output_dir}")


if __name__ == "__main__":
    main()
