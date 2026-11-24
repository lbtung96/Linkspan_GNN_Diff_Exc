from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import csd

from export_projected_graph import build_projected_graph_from_model
from grid_modal_Kinematic_MPC import (
    add_nodal_springs,
    assemble_KM_3d_with_nsm,
    build_influence_vector,
    build_grid_model_3d_upgraded_v22,
    build_rigid_mpc_T,
    default_params_grid_3d_offset_upgraded_v22,
    find_nodes_by_coordinate,
    kNpm_to_Npmm,
    modal_participation,
    reduce_modes_rz,
    reduce_modes_u2,
    reduce_modes_u3,
    run_grid_modal_with_params_and_plot,
    select_girder_sensor_nodes,
    solve_modes_3d,
)
from simulate_uniform_inertial_u3_white_noise import (
    WELCH_NFFT,
    WELCH_NOVERLAP,
    WELCH_NPERSEG,
    simulate_white_noise_from_modal_model,
)
from standalone_mode_matching_mmi import (
    ModeMatchConfig,
    ReferenceModeLibrary,
    SignatureConfig,
    load_reference_mode_library,
    match_target_modes,
)


# =========================
# User-editable config
# =========================
N_STRUCTURES_TOTAL = 800
OUT_DIR = Path("population_dataset_modes4")
SEED = 20260507
PHASE1_FREQ_BINS = 400 # The first number of bins in the PSD so that the target modes are well-resolved within the range. Deltaf = fs/nfft, so with fs=200Hz and nfft=2048, we get ~0.1Hz resolution. With target modes up to ~20Hz, 400 bins gives us coverage up to 40Hz.
PHASE1_ZETA = 0.01
PHASE1_SAMPLES_SUBDIR = "samples_phase1"
PHASE2A_SAMPLES_SUBDIR = "samples_phase2"
PHASE1_SCHEMA_VERSION = "pbshm_phase1_v1"

REF_DIR = Path(".")
REF_FILE_PATTERN = "ref_modes_Ng{Ng}.npz"
REF_FILE_PATTERN_NG2_BALCONY = "ref_modes_Ng2_Balcony.npz"
REF_FILE_PATTERN_NG2_NO_BAL  = "ref_modes_Ng2_No_bal.npz"
# REF_X_FRACS = [0.11, 0.24, 0.38, 0.50, 0.64, 0.77, 0.95, 1.00]
REF_X_FRACS = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
REF_SENSOR_TOL_X = 1e-6
MODE_NAMES = ["B1", "T1", "B2", "T2"]
N_CANDIDATE_MODES = 15
MIN_MAC_THRESHOLD = 0.85
MODE_MATCH_CONFIG = ModeMatchConfig(
    min_mac_threshold=MIN_MAC_THRESHOLD,
    use_u2_for_torsion=True,
    use_balcony_penalty=True,
    use_participation_features=True,
    use_two_stage_torsion_rerank=True,
    torsion_shortlist_k=3,
    signature_config=SignatureConfig(
        use_channel_u3=True,
        use_channel_u2=True,
        use_channel_rz=True,
        use_feature_raw=False,
        use_feature_mean=True,
        use_feature_left_right_diff=True,
        normalize_signature=True,
    ),
    use_participation_factors=True,
    use_effective_modal_mass=True,
)
@dataclass
class OutlierConfig:
    enable_outlier_filter: bool = True
    method: str = "iqr"  # options: "iqr", "mad", "zscore"
    iqr_k: float = 1.5
    mad_k: float = 3.5
    zscore_k: float = 3.0
    verbose: bool = True


OUTLIER_CONFIG = OutlierConfig()

MAX_ATTEMPTS_PER_NG = 20000
MAX_BALCONY_TRIES = 40

PLOT_BINS = 30
PLOT_DPI = 160

NG_CHOICES = [2, 3, 4, 5]
DIAG_CHOICES_ALL = ["global_x", "staggered_x", "single_partial_x", "None"]
DIAG_CHOICES_NON_NG4 = ["global_x", "staggered_x", "None"]
E_RANGE = (160000.0, 230000.0)
RHO_RANGE = (7.0e-9, 8.0e-9)



@dataclass
class SolveResult:
    freq_all_hz: np.ndarray
    phi_full: np.ndarray
    phi_u3_rep: np.ndarray
    phi_u2_rep: np.ndarray
    phi_rz_rep: np.ndarray
    sensor_node_ids: np.ndarray
    model: dict[str, Any]
    M_full: Any
    participation: dict[str, np.ndarray]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def to_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=_json_default)


def compute_ng_targets(n_total: int, ng_choices: list[int]) -> dict[int, int]:
    base = n_total // len(ng_choices)
    rem = n_total % len(ng_choices)
    targets = {}
    for i, ng in enumerate(ng_choices):
        targets[ng] = base + (1 if i < rem else 0)
    return targets


def sample_balcony_for_span(rng: np.random.Generator, L_mm: float) -> dict[str, Any]:
    enable = bool(rng.random() < 0.5)
    if not enable:
        return {
            "enable": False,
            "side": "none",
            "x_center_frac": np.nan,
            "nBeams": 0,
            "dx_mm": np.nan,
            "length_mm": np.nan,
            "edge_beam_enable": False,
            "brace_enable": False,
        }

    side = str(rng.choice(["left", "right", "both"], p=[1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]))

    for _ in range(MAX_BALCONY_TRIES):
        x_center_frac = float(rng.uniform(0.85, 0.92))
        n_beams = int(rng.integers(1, 4))
        dx_mm = float(rng.uniform(500.0, 1000.0))
        length_mm = float(rng.uniform(2000.0, 3000.0))

        x_c = x_center_frac * L_mm
        centers = x_c + (np.arange(n_beams, dtype=float) - (n_beams - 1) / 2.0) * dx_mm
        margin = 100.0
        min_ok = np.all(centers - 0.5 * length_mm >= margin)
        max_ok = np.all(centers + 0.5 * length_mm <= (L_mm - margin))
        if not (min_ok and max_ok):
            continue

        # Forced rule from spec.
        if n_beams > 1:
            edge_beam_enable = True
            brace_enable = True
        else:
            edge_beam_enable = False
            brace_enable = False

        return {
            "enable": True,
            "side": side,
            "x_center_frac": x_center_frac,
            "nBeams": n_beams,
            "dx_mm": dx_mm,
            "length_mm": length_mm,
            "edge_beam_enable": edge_beam_enable,
            "brace_enable": brace_enable,
        }

    raise ValueError("Failed balcony fit rule after retries")


def sample_params_for_ng(ng: int, rng: np.random.Generator) -> dict[str, Any]:
    L_mm = float(rng.uniform(14000.0, 22000.0))
    Nx = int(rng.integers(15, 26))

    if ng <= 1:
        raise ValueError(f"Invalid Ng={ng}")
    if ng == 2:
        dy_low, dy_high = 4000.0, 4500.0          # 2-girder spans need wider bays
    else:
        dy_low = max(2000.0, 4500.0 / (ng - 1))
        dy_high = min(2500.0, 12000.0 / (ng - 1))
        if dy_low > dy_high:
            raise ValueError(f"No valid dy range for Ng={ng}: [{dy_low}, {dy_high}]")
    dy_mm = float(rng.uniform(dy_low, dy_high))

    if ng == 4:
        diag_pattern = str(rng.choice(DIAG_CHOICES_ALL))
    else:
        diag_pattern = str(rng.choice(DIAG_CHOICES_NON_NG4))

    balcony = sample_balcony_for_span(rng, L_mm)

    sampled = {
        "L_mm": L_mm,
        "E": float(rng.uniform(*E_RANGE)),
        "rho": float(rng.uniform(*RHO_RANGE)),
        "Nx": Nx,
        "Ng": int(ng),
        "dy_mm": dy_mm,
        "diag_pattern": diag_pattern,
        "spring": {
            "kx_kN_per_m": float(rng.uniform(1750.0, 2250.0)),
            "kz_kN_per_m": float(rng.uniform(17500.0, 22500.0)),
        },
        "bottom_beam": {
            "x_frac": float(rng.uniform(0.55, 0.70)),
            "overhang_mm": float(rng.uniform(500.0, 600.0)),
        },
        "balcony": balcony,
        "nsm": {
            "enable": True,
            "total_mass_tonne": float(rng.uniform(25.0, 35.0)),
        },
    }
    return sampled


def validate_sampled_params(sampled: dict[str, Any]) -> bool:
    ng = int(sampled["Ng"])
    dy = float(sampled["dy_mm"])
    width = (ng - 1) * dy
    if ng == 2:
        if not (4000.0 <= dy <= 4500.0):          # 2-girder bay width
            return False
    else:
        if not (2000.0 <= dy <= 2500.0):
            return False
        if not (4500.0 <= width <= 12000.0):
            return False

    diag = str(sampled["diag_pattern"])
    if ng != 4 and diag == "single_partial_x":
        return False

    b = sampled["balcony"]
    if bool(b["enable"]):
        n = int(b["nBeams"])
        if n > 1 and not (bool(b["edge_beam_enable"]) and bool(b["brace_enable"])):
            return False
        if n == 1 and (bool(b["edge_beam_enable"]) or bool(b["brace_enable"])):
            return False

    return True


def build_solver_params(sampled: dict[str, Any], n_candidate_modes: int) -> dict[str, Any]:
    p = deepcopy(default_params_grid_3d_offset_upgraded_v22())

    p["L_mm"] = float(sampled["L_mm"])
    p["E"] = float(sampled["E"])
    p["rho"] = float(sampled["rho"])
    p["G"] = p["E"] / (2 * (1 + p["nu"]))
    p["Nx"] = int(sampled["Nx"])
    p["Ng"] = int(sampled["Ng"])
    p["dy_mm"] = float(sampled["dy_mm"])
    p["diag_pattern"] = str(sampled["diag_pattern"])

    p["spring"]["kx_kN_per_m"] = float(sampled["spring"]["kx_kN_per_m"])
    p["spring"]["kz_kN_per_m"] = float(sampled["spring"]["kz_kN_per_m"])
    p["spring"]["kx_N_per_mm"] = float(kNpm_to_Npmm(p["spring"]["kx_kN_per_m"]))
    p["spring"]["kz_N_per_mm"] = float(kNpm_to_Npmm(p["spring"]["kz_kN_per_m"]))

    p["bottom_beam"]["x_frac"] = float(sampled["bottom_beam"]["x_frac"])
    p["bottom_beam"]["overhang_mm"] = float(sampled["bottom_beam"]["overhang_mm"])

    p["balcony"]["enable"] = bool(sampled["balcony"]["enable"])
    if p["balcony"]["enable"]:
        p["balcony"]["side"] = str(sampled["balcony"]["side"])
        p["balcony"]["x_center_frac"] = float(sampled["balcony"]["x_center_frac"])
        p["balcony"]["nBeams"] = int(sampled["balcony"]["nBeams"])
        p["balcony"]["dx_mm"] = float(sampled["balcony"]["dx_mm"])
        p["balcony"]["length_mm"] = float(sampled["balcony"]["length_mm"])
        p["balcony"]["edge_beam_enable"] = bool(sampled["balcony"]["edge_beam_enable"])
        p["balcony"]["brace_enable"] = bool(sampled["balcony"]["brace_enable"])
    else:
        p["balcony"]["side"] = "both"

    p["nsm"]["enable"] = True
    p["nsm"]["total_mass_tonne"] = float(sampled["nsm"]["total_mass_tonne"])

    p["nmodes"] = int(n_candidate_modes)
    return p


def run_solver(p: dict[str, Any], ref_x_fracs: list[float], tol_x: float) -> SolveResult:
    model = build_grid_model_3d_upgraded_v22(p)

    K_full, M_full = assemble_KM_3d_with_nsm(model)
    K_full = add_nodal_springs(K_full, model)

    ndof_full = int(model["dof"]["ndof"])
    mpc_pairs = np.asarray(model.get("mpc_pairs", np.zeros((0, 2), dtype=int)), dtype=int)
    if mpc_pairs.size:
        T, keep_dofs = build_rigid_mpc_T(model)
        full_to_red = np.full(ndof_full, -1, dtype=int)
        full_to_red[keep_dofs] = np.arange(keep_dofs.size, dtype=int)
        K = (T.T @ K_full @ T).tocsc()
        M = (T.T @ M_full @ T).tocsc()
    else:
        T = None
        full_to_red = np.arange(ndof_full, dtype=int)
        K = K_full
        M = M_full

    freq_hz, Phi_full, _ = solve_modes_3d(
        K,
        M,
        model["bc"],
        int(p["nmodes"]),
        full_to_red=full_to_red,
        T=T,
        M_full=M_full,
    )

    sensor_node_ids = select_girder_sensor_nodes(model, x_fracs=ref_x_fracs, tol_x=tol_x)
    phi_u3_rep = reduce_modes_u3(Phi_full, sensor_node_ids)
    phi_u2_rep = reduce_modes_u2(Phi_full, sensor_node_ids)
    phi_rz_rep = reduce_modes_rz(Phi_full, sensor_node_ids)

    # Modal participation/effective-mass indicators for physics-aware matching.
    x0_nodes = find_nodes_by_coordinate(model, x=0.0, tol=1e-6)
    bb = np.asarray(model.get("bottom_node_ids", np.zeros(0, dtype=int)), dtype=int).reshape(-1)
    bb_ends = np.array([int(bb[0]), int(bb[-1])], dtype=int) if bb.size >= 2 else np.zeros(0, dtype=int)
    support_nodes = np.unique(np.r_[x0_nodes, bb_ends]).astype(int)
    if support_nodes.size == 0:
        support_nodes = np.asarray(model["nodes"][:, 0], dtype=int)[: max(1, min(4, int(model["nodes"].shape[0])))]

    ruy = build_influence_vector(model, support_nodes, dof_1based=2)
    ruz = build_influence_vector(model, support_nodes, dof_1based=3)
    gamma_uy, meff_uy, _, _ = modal_participation(np.asarray(Phi_full, dtype=float), M_full, ruy)
    gamma_uz, meff_uz, _, _ = modal_participation(np.asarray(Phi_full, dtype=float), M_full, ruz)
    participation = {
        "gamma_uy": np.asarray(gamma_uy, dtype=float),
        "gamma_uz": np.asarray(gamma_uz, dtype=float),
        "meff_uy": np.asarray(meff_uy, dtype=float),
        "meff_uz": np.asarray(meff_uz, dtype=float),
    }

    return SolveResult(
        freq_all_hz=np.asarray(freq_hz, dtype=float),
        phi_full=np.asarray(Phi_full, dtype=float),
        phi_u3_rep=np.asarray(phi_u3_rep, dtype=float),
        phi_u2_rep=np.asarray(phi_u2_rep, dtype=float),
        phi_rz_rep=np.asarray(phi_rz_rep, dtype=float),
        sensor_node_ids=np.asarray(sensor_node_ids, dtype=int),
        model=model,
        M_full=M_full,
        participation=participation,
    )


def _ref_key(ng: int, sampled: dict) -> object:
    """Hashable key for ref_by_ng; splits Ng=2 into balcony/no-balcony variants."""
    if ng == 2:
        balcony_enabled = bool((sampled.get("balcony") or {}).get("enable", False))
        return (2, balcony_enabled)
    return ng


def load_reference_library_for_ng_variant(ref_dir: Path, ng: int, sampled: dict) -> ReferenceModeLibrary:
    """Load the correct reference library for a sample, using balcony-aware variant for Ng=2."""
    if ng == 2:
        balcony_enabled = bool((sampled.get("balcony") or {}).get("enable", False))
        fname = REF_FILE_PATTERN_NG2_BALCONY if balcony_enabled else REF_FILE_PATTERN_NG2_NO_BAL
        ref_path = ref_dir / fname
    else:
        ref_path = ref_dir / REF_FILE_PATTERN.format(Ng=ng)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
    lib = load_reference_mode_library(ref_path, target_names=MODE_NAMES)
    return lib


def load_reference_library_for_ng(ref_dir: Path, ng: int) -> ReferenceModeLibrary:
    ref_path = ref_dir / REF_FILE_PATTERN.format(Ng=ng)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")

    lib = load_reference_mode_library(ref_path, target_names=MODE_NAMES)
    if int(lib.ng) != int(ng):
        raise ValueError(f"Reference Ng mismatch in {ref_path}: Ng_ref={lib.ng}, expected={ng}")
    if lib.x_fracs.shape != (len(REF_X_FRACS),):
        raise ValueError(f"Unexpected x_fracs shape in {ref_path}: {lib.x_fracs.shape}")
    if not np.allclose(lib.x_fracs, np.asarray(REF_X_FRACS, dtype=float), atol=1e-12):
        raise ValueError(f"x_fracs mismatch in {ref_path}")
    return lib


def load_reference_for_ng(ref_dir: Path, ng: int) -> np.ndarray:
    """Backward-compatible loader returning one reference vector per target."""
    lib = load_reference_library_for_ng(ref_dir, ng)
    return np.asarray(lib.primary_u3_targets, dtype=float)


def load_reference_library_from_file(reference_file: str | Path, ng: int) -> ReferenceModeLibrary:
    ref_path = Path(reference_file)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
    lib = load_reference_mode_library(ref_path, target_names=MODE_NAMES)
    if int(lib.ng) != int(ng):
        raise ValueError(f"Reference Ng mismatch in {ref_path}: Ng_ref={lib.ng}, expected={ng}")
    return lib


def normalize_cols_by_max_abs_signed(X: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(X, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"Expected 2D matrix for signed normalization, got {A.shape}.")
    scale = np.max(np.abs(A), axis=0)
    out = np.zeros_like(A, dtype=float)
    nz = scale > eps
    out[:, nz] = A[:, nz] / scale[nz]
    return out, scale.astype(float)


def compute_reference_sign_alignment(
    phi_sensor_4: np.ndarray,
    ref_sensor_4: np.ndarray,
    *,
    sensor_node_ids: np.ndarray | None = None,
    dot_eps: float = 1e-10,
    amp_eps: float = 1e-10,
) -> dict[str, Any]:
    cand = np.asarray(phi_sensor_4, dtype=float)
    ref = np.asarray(ref_sensor_4, dtype=float)
    if cand.shape != ref.shape or cand.ndim != 2:
        raise ValueError(f"Sensor/reference sign-alignment shape mismatch: {cand.shape} vs {ref.shape}.")

    n_modes = cand.shape[1]
    aligned = np.asarray(cand, dtype=float).copy()
    before = np.sum(cand * ref, axis=0)
    after = np.zeros(n_modes, dtype=float)
    flips = np.ones(n_modes, dtype=float)
    sensor_index = np.argmax(np.abs(ref), axis=0).astype(np.int64)
    fallback_used = np.zeros(n_modes, dtype=bool)
    ambiguous = np.zeros(n_modes, dtype=bool)
    reasons: list[str] = []

    if sensor_node_ids is None:
        sensor_node_id = np.full(n_modes, -1, dtype=np.int64)
    else:
        sensor_ids = np.asarray(sensor_node_ids, dtype=np.int64).reshape(-1)
        if sensor_ids.shape[0] != cand.shape[0]:
            raise ValueError(f"sensor_node_ids length {sensor_ids.shape[0]} does not match sensor rows {cand.shape[0]}.")
        sensor_node_id = sensor_ids[sensor_index]

    for j in range(n_modes):
        dot_val = float(before[j])
        flip = 1.0
        reason = 'dot_positive_kept'
        if abs(dot_val) > dot_eps:
            if dot_val < 0.0:
                flip = -1.0
                reason = 'dot_negative_flipped'
        else:
            fallback_used[j] = True
            ref_idx = int(sensor_index[j])
            ref_val = float(ref[ref_idx, j])
            cand_val = float(cand[ref_idx, j])
            if abs(ref_val) > amp_eps and abs(cand_val) > amp_eps:
                if cand_val * ref_val < 0.0:
                    flip = -1.0
                    reason = 'anchor_sensor_flipped'
                else:
                    reason = 'anchor_sensor_kept'
            else:
                ambiguous[j] = True
                reason = 'ambiguous_near_zero_kept'
        aligned[:, j] = flip * cand[:, j]
        after[j] = float(np.dot(aligned[:, j], ref[:, j]))
        flips[j] = flip
        reasons.append(reason)

    return {
        'phi_sensor_4_aligned': aligned,
        'sign_flip_factors': flips,
        'sign_reference_dot_before': before.astype(float),
        'sign_reference_dot_after': after.astype(float),
        'sign_reference_sensor_index': sensor_index,
        'sign_reference_sensor_node_id': sensor_node_id,
        'sign_fallback_used': fallback_used,
        'sign_ambiguous_flag': ambiguous,
        'sign_fallback_reason': np.asarray(reasons, dtype='<U32'),
        'sign_rule_name': 'reference_dot_then_anchor_sensor',
        'sign_rule_description': 'Use reference-template dot product sign; if near zero, use the strongest reference sensor entry when both template and candidate amplitudes are non-negligible.',
        'sign_alignment_ready': bool(np.all(~ambiguous)),
    }


def build_signed_mode_shape_payload(
    *,
    sampled: dict[str, Any],
    solve_res: SolveResult,
    reference_file: str,
    phi_u3_sensor_4: np.ndarray,
    phi_u3_full_4: np.ndarray,
    graph_node_ids: np.ndarray,
) -> dict[str, Any]:
    ref_lib = load_reference_library_from_file(reference_file, int(sampled['Ng']))
    sign_info = compute_reference_sign_alignment(
        phi_u3_sensor_4,
        np.asarray(ref_lib.primary_u3_targets, dtype=float),
        sensor_node_ids=solve_res.sensor_node_ids,
    )
    flips = np.asarray(sign_info['sign_flip_factors'], dtype=float).reshape(1, -1)
    phi_full_signed = np.asarray(phi_u3_full_4, dtype=float) * flips
    phi_full_signed_norm, phi_full_signed_scale = normalize_cols_by_max_abs_signed(phi_full_signed)
    full_rows = np.asarray(graph_node_ids, dtype=np.int64).reshape(-1) - 1
    y_mode_shape_signed = np.asarray(phi_full_signed_norm[full_rows, :], dtype=float)

    return {
        'y_mode_shape_signed': y_mode_shape_signed,
        'y_mode_shape_signed_scale': np.asarray(phi_full_signed_scale, dtype=float),
        'phi_u3_sensor_4_signed': np.asarray(sign_info['phi_sensor_4_aligned'], dtype=float),
        'phi_u3_sensor_4_signed_norm': np.asarray(
            normalize_cols_by_max_abs_signed(np.asarray(sign_info['phi_sensor_4_aligned'], dtype=float))[0],
            dtype=float,
        ),
        'sign_flip_factors': np.asarray(sign_info['sign_flip_factors'], dtype=float),
        'sign_reference_dot_before': np.asarray(sign_info['sign_reference_dot_before'], dtype=float),
        'sign_reference_dot_after': np.asarray(sign_info['sign_reference_dot_after'], dtype=float),
        'sign_reference_sensor_index': np.asarray(sign_info['sign_reference_sensor_index'], dtype=np.int64),
        'sign_reference_sensor_node_id': np.asarray(sign_info['sign_reference_sensor_node_id'], dtype=np.int64),
        'sign_fallback_used': np.asarray(sign_info['sign_fallback_used'], dtype=bool),
        'sign_ambiguous_flag': np.asarray(sign_info['sign_ambiguous_flag'], dtype=bool),
        'sign_fallback_reason': np.asarray(sign_info['sign_fallback_reason']),
        'sign_rule_name': np.asarray(str(sign_info['sign_rule_name'])),
        'sign_rule_description': np.asarray(str(sign_info['sign_rule_description'])),
        'sign_alignment_ready': np.asarray(bool(sign_info['sign_alignment_ready'])),
        'sign_reference_template_sensor_4': np.asarray(ref_lib.primary_u3_targets, dtype=float),
    }


def _candidate_channels_from_solve(solve_res: SolveResult, cfg: SignatureConfig) -> dict[str, np.ndarray]:
    ch: dict[str, np.ndarray] = {"u3": np.asarray(solve_res.phi_u3_rep, dtype=float)}
    if bool(cfg.use_channel_u2):
        ch["u2"] = np.asarray(solve_res.phi_u2_rep, dtype=float)
    if bool(cfg.use_channel_rz):
        ch["rz"] = np.asarray(solve_res.phi_rz_rep, dtype=float)
    return ch


def _candidate_metadata_from_solve(solve_res: SolveResult) -> dict[str, Any]:
    return {
        "model": solve_res.model,
        "phi_full": np.asarray(solve_res.phi_full, dtype=float),
    }


def compute_S(phi_u3_4: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi_u3_4, dtype=float)
    if phi.ndim != 2 or phi.shape[1] != 4:
        raise ValueError(f"Expected phi_u3_4 shape (N,4), got {phi.shape}")

    S = np.zeros(4, dtype=float)
    for j in range(4):
        v = phi[:, j]
        denom = float(np.max(np.abs(v)))
        if denom <= 0.0:
            S[j] = 0.0
        else:
            vn = v / denom
            S[j] = float(np.sum(vn * vn))
    return S


def build_summary_row(
    sample_id: int,
    sampled: dict[str, Any],
    freqs_4: np.ndarray,
    S4: np.ndarray,
    mac4: np.ndarray,
    selected_idx: np.ndarray,
) -> dict[str, Any]:
    b = sampled["balcony"]
    row = {
        "sample_id": int(sample_id),
        "Ng": int(sampled["Ng"]),
        "L_mm": float(sampled["L_mm"]),
        "E": float(sampled["E"]),
        "rho": float(sampled["rho"]),
        "Nx": int(sampled["Nx"]),
        "dy_mm": float(sampled["dy_mm"]),
        "diag_pattern": str(sampled["diag_pattern"]),
        "kx_kN_per_m": float(sampled["spring"]["kx_kN_per_m"]),
        "kz_kN_per_m": float(sampled["spring"]["kz_kN_per_m"]),
        "bottom_x_frac": float(sampled["bottom_beam"]["x_frac"]),
        "bottom_overhang_mm": float(sampled["bottom_beam"]["overhang_mm"]),
        "balcony_enable": int(bool(b["enable"])),
        "balcony_side": str(b["side"]),
        "balcony_nBeams": int(b["nBeams"]),
        "balcony_dx_mm": float(b["dx_mm"]) if np.isfinite(b["dx_mm"]) else np.nan,
        "balcony_length_mm": float(b["length_mm"]) if np.isfinite(b["length_mm"]) else np.nan,
        "nsm_mass_tonne": float(sampled["nsm"]["total_mass_tonne"]),
    }

    for i, name in enumerate(MODE_NAMES):
        row[f"freq_{name}_hz"] = float(freqs_4[i])
        row[f"S_{name}"] = float(S4[i])
        row[f"mac_{name}"] = float(mac4[i])
        row[f"mode_idx_{name}"] = int(selected_idx[i])
    return row


def save_sample_npz(
    out_samples_dir: Path,
    sample_id: int,
    sampled: dict[str, Any],
    freqs_4: np.ndarray,
    phi_u3_4: np.ndarray,
    S4: np.ndarray,
    selected_idx: np.ndarray,
    mac4: np.ndarray,
    match_result_json: str | None = None,
    composite_scores: np.ndarray | None = None,
    ambiguity_flags: np.ndarray | None = None,
    matcher_config_json: str | None = None,
) -> Path:
    ng = int(sampled["Ng"])
    out_path = out_samples_dir / f"sample_{sample_id:04d}_Ng{ng}.npz"

    np.savez(
        out_path,
        params_json=np.asarray(to_json(sampled)),
        Ng=np.asarray(ng, dtype=int),
        Nx=np.asarray(int(sampled["Nx"]), dtype=int),
        L_mm=np.asarray(float(sampled["L_mm"]), dtype=float),
        dy_mm=np.asarray(float(sampled["dy_mm"]), dtype=float),
        diag_pattern=np.asarray(str(sampled["diag_pattern"])),
        freqs_4=np.asarray(freqs_4, dtype=float),
        phi_u3_4=np.asarray(phi_u3_4, dtype=float),
        S4=np.asarray(S4, dtype=float),
        selected_mode_indices=np.asarray(selected_idx, dtype=int),
        mac_values=np.asarray(mac4, dtype=float),
        composite_scores=np.asarray(composite_scores if composite_scores is not None else np.asarray(mac4, dtype=float), dtype=float),
        ambiguity_flags=np.asarray(
            ambiguity_flags if ambiguity_flags is not None else np.zeros_like(np.asarray(mac4, dtype=float), dtype=bool),
            dtype=bool,
        ),
        mode_match_result_json=np.asarray(match_result_json if match_result_json is not None else ""),
        mode_match_config_json=np.asarray(matcher_config_json if matcher_config_json is not None else ""),
        REF_X_FRACS=np.asarray(REF_X_FRACS, dtype=float),
    )
    return out_path


def _validate_phase1_mapping(
    graph_node_ids: np.ndarray,
    full_node_count: int,
    x_psd: np.ndarray,
    y_mode_shape_abs: np.ndarray,
    edge_index: np.ndarray,
    edge_attr_geom: np.ndarray,
    freq_bins_hz: np.ndarray,
) -> None:
    node_ids = np.asarray(graph_node_ids, dtype=int).reshape(-1)
    full_rows = node_ids - 1
    if np.any(full_rows < 0) or np.any(full_rows >= int(full_node_count)):
        raise ValueError(
            f"Projected graph node IDs map outside full FE node range [1,{int(full_node_count)}]."
        )
    n_graph_nodes = int(node_ids.size)
    if x_psd.shape[0] != n_graph_nodes:
        raise ValueError(f"x_psd row count mismatch: expected {n_graph_nodes}, got {x_psd.shape[0]}.")
    if y_mode_shape_abs.shape[0] != n_graph_nodes:
        raise ValueError(
            f"y_mode_shape_abs row count mismatch: expected {n_graph_nodes}, got {y_mode_shape_abs.shape[0]}."
        )
    if y_mode_shape_abs.shape[1] != 4:
        raise ValueError(f"Expected 4 matched modes in y_mode_shape_abs, got {y_mode_shape_abs.shape[1]}.")
    if edge_attr_geom.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_attr_geom row count mismatch: expected {edge_index.shape[1]}, got {edge_attr_geom.shape[0]}."
        )
    if x_psd.shape[1] != freq_bins_hz.shape[0]:
        raise ValueError(
            f"Frequency bin count mismatch: x_psd has {x_psd.shape[1]}, freq_bins_hz has {freq_bins_hz.shape[0]}."
        )


def _validate_phase2a_frequency_targets(
    target_mode_names: np.ndarray,
    matched_freq_hz: np.ndarray,
    y_freq_hz: np.ndarray,
) -> None:
    expected_names = np.asarray(MODE_NAMES, dtype="<U2")
    got_names = np.asarray(target_mode_names)
    if got_names.shape != (4,):
        raise ValueError(f"target_mode_names must have shape (4,), got {got_names.shape}.")
    if not np.array_equal(got_names, expected_names):
        raise ValueError(
            f"target_mode_names must stay aligned with {expected_names.tolist()}, got {got_names.tolist()}."
        )

    matched = np.asarray(matched_freq_hz, dtype=float).reshape(-1)
    y_freq = np.asarray(y_freq_hz, dtype=np.float32).reshape(-1)
    if matched.shape != (4,):
        raise ValueError(f"matched_freq_hz must have shape (4,), got {matched.shape}.")
    if y_freq.shape != (4,):
        raise ValueError(f"y_freq_hz must have shape (4,), got {y_freq.shape}.")
    if y_freq.dtype != np.float32:
        raise ValueError(f"y_freq_hz must be float32, got {y_freq.dtype}.")
    if not np.allclose(y_freq.astype(float), matched, rtol=1e-6, atol=1e-6):
        raise ValueError("y_freq_hz must match matched_freq_hz in the fixed target mode order.")


def build_cpsd_edge_payload(
    *,
    edge_index: np.ndarray,
    matched_freq_hz: np.ndarray,
    freq_bins_hz: np.ndarray,
    a3_abs: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
    nfft: int,
) -> dict[str, Any]:
    return build_cpsd_resonance_edge_payload(
        edge_index=edge_index,
        matched_freq_hz=matched_freq_hz,
        freq_bins_hz=freq_bins_hz,
        a3_abs=a3_abs,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
    )


def _compute_edge_complex_coherency_spectra(
    *,
    edge_index: np.ndarray,
    a3_abs: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
    nfft: int,
) -> np.ndarray:
    edge_index = np.asarray(edge_index, dtype=np.int64)
    accel = np.asarray(a3_abs, dtype=float)

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E_dir), got {edge_index.shape}.")
    if accel.ndim != 2:
        raise ValueError(f"a3_abs must be 2D [N, T], got {accel.shape}.")
    if np.any(edge_index < 0) or np.any(edge_index >= accel.shape[0]):
        raise ValueError("edge_index references nodes outside the available acceleration array.")

    edge_coherency = np.zeros((edge_index.shape[1], nfft // 2 + 1), dtype=np.complex64)
    auto_cache: dict[int, np.ndarray] = {}

    for edge_col in range(edge_index.shape[1]):
        src = int(edge_index[0, edge_col])
        dst = int(edge_index[1, edge_col])

        if src not in auto_cache:
            _, pxx = csd(
                accel[src, :],
                accel[src, :],
                fs=fs,
                window="hamming",
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                detrend=False,
                return_onesided=True,
                scaling="density",
            )
            auto_cache[src] = np.asarray(pxx, dtype=np.complex128)
        if dst not in auto_cache:
            _, pyy = csd(
                accel[dst, :],
                accel[dst, :],
                fs=fs,
                window="hamming",
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                detrend=False,
                return_onesided=True,
                scaling="density",
            )
            auto_cache[dst] = np.asarray(pyy, dtype=np.complex128)

        _, pxy = csd(
            accel[src, :],
            accel[dst, :],
            fs=fs,
            window="hamming",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend=False,
            return_onesided=True,
            scaling="density",
        )
        pxy = np.asarray(pxy, dtype=np.complex128)
        denom = np.sqrt(np.maximum(np.real(auto_cache[src]), 0.0) * np.maximum(np.real(auto_cache[dst]), 0.0))
        coherency = np.zeros_like(pxy, dtype=np.complex128)
        valid = denom > 1e-18
        coherency[valid] = pxy[valid] / denom[valid]
        edge_coherency[edge_col, :] = np.asarray(coherency, dtype=np.complex64)

    return edge_coherency


def build_cpsd_full_edge_payload(
    *,
    edge_index: np.ndarray,
    freq_bins_hz: np.ndarray,
    a3_abs: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
    nfft: int,
) -> dict[str, Any]:
    freq_bins = np.asarray(freq_bins_hz, dtype=float).reshape(-1)
    if freq_bins.ndim != 1 or freq_bins.size == 0:
        raise ValueError("freq_bins_hz must be a non-empty 1D array.")

    edge_coherency = _compute_edge_complex_coherency_spectra(
        edge_index=edge_index,
        a3_abs=a3_abs,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
    )
    if edge_coherency.shape[1] < freq_bins.size:
        raise ValueError(
            f"Computed coherency spectrum has only {edge_coherency.shape[1]} bins, but freq_bins_hz requests {freq_bins.size}."
        )
    edge_coherency = edge_coherency[:, : freq_bins.size]

    feature_names: list[str] = []
    for bin_idx, freq_hz in enumerate(freq_bins.tolist()):
        feature_names.extend([f"bin_{bin_idx:03d}_{freq_hz:.6f}_hz_re", f"bin_{bin_idx:03d}_{freq_hz:.6f}_hz_im"])

    edge_attr_cpsd_full = np.zeros((edge_coherency.shape[0], freq_bins.size * 2), dtype=np.float32)
    edge_attr_cpsd_full[:, 0::2] = np.real(edge_coherency).astype(np.float32)
    edge_attr_cpsd_full[:, 1::2] = np.imag(edge_coherency).astype(np.float32)

    return {
        "edge_attr_cpsd_full": edge_attr_cpsd_full,
        "edge_attr_cpsd_full_feature_names": np.asarray(feature_names),
        "edge_attr_cpsd_full_freq_hz": np.asarray(freq_bins, dtype=float),
        "edge_attr_cpsd_full_signal_domain": np.asarray("absolute_u3_acceleration"),
        "edge_attr_cpsd_full_representation": np.asarray("complex_coherency_real_imag"),
        "edge_attr_cpsd_full_description": np.asarray(
            "Per directed edge, store real and imaginary parts of normalized CPSD (complex coherency) for every retained Welch frequency bin in freq_bins_hz."
        ),
        "edge_attr_cpsd_full_shape": np.asarray(edge_attr_cpsd_full.shape, dtype=np.int64),
        "edge_attr_cpsd_full_feature_dim": np.asarray(int(edge_attr_cpsd_full.shape[1]), dtype=np.int64),
    }


def build_cpsd_resonance_edge_payload(
    *,
    edge_index: np.ndarray,
    matched_freq_hz: np.ndarray,
    freq_bins_hz: np.ndarray,
    a3_abs: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
    nfft: int,
) -> dict[str, Any]:
    matched_freq = np.asarray(matched_freq_hz, dtype=float).reshape(-1)
    freq_bins = np.asarray(freq_bins_hz, dtype=float).reshape(-1)
    if freq_bins.ndim != 1 or freq_bins.size == 0:
        raise ValueError("freq_bins_hz must be a non-empty 1D array.")
    if matched_freq.ndim != 1 or matched_freq.size != 4:
        raise ValueError(f"matched_freq_hz must have shape (4,), got {matched_freq.shape}.")

    edge_coherency = _compute_edge_complex_coherency_spectra(
        edge_index=edge_index,
        a3_abs=a3_abs,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
    )
    if edge_coherency.shape[1] < freq_bins.size:
        raise ValueError(
            f"Computed coherency spectrum has only {edge_coherency.shape[1]} bins, but freq_bins_hz requests {freq_bins.size}."
        )

    feature_names: list[str] = []
    for mode_name in MODE_NAMES:
        feature_names.extend([f"{mode_name}_re_coherency", f"{mode_name}_im_coherency"])

    sampled_bin_idx = np.asarray(
        [int(np.argmin(np.abs(freq_bins - freq_val))) for freq_val in matched_freq],
        dtype=np.int64,
    )
    sampled_freq_hz = np.asarray(freq_bins[sampled_bin_idx], dtype=float)
    edge_attr_cpsd_resonance = np.zeros((edge_coherency.shape[0], matched_freq.size * 2), dtype=np.float32)
    for mode_idx, bin_idx in enumerate(sampled_bin_idx.tolist()):
        edge_attr_cpsd_resonance[:, 2 * mode_idx] = np.real(edge_coherency[:, bin_idx]).astype(np.float32)
        edge_attr_cpsd_resonance[:, 2 * mode_idx + 1] = np.imag(edge_coherency[:, bin_idx]).astype(np.float32)

    return {
        "edge_attr_cpsd": edge_attr_cpsd_resonance,
        "edge_attr_cpsd_feature_names": np.asarray(feature_names),
        "edge_attr_cpsd_mode_names": np.asarray(MODE_NAMES),
        "edge_attr_cpsd_matched_freq_hz": np.asarray(matched_freq, dtype=float),
        "edge_attr_cpsd_sampled_freq_hz": np.asarray(sampled_freq_hz, dtype=float),
        "edge_attr_cpsd_bin_indices": np.asarray(sampled_bin_idx, dtype=np.int64),
        "edge_attr_cpsd_signal_domain": np.asarray("absolute_u3_acceleration"),
        "edge_attr_cpsd_representation": np.asarray("complex_coherency_real_imag"),
        "edge_attr_cpsd_description": np.asarray(
            "Per directed edge, store real and imaginary parts of normalized CPSD (complex coherency) at the four matched modal frequencies using the same Welch bins as x_psd."
        ),
        "edge_attr_cpsd_shape": np.asarray(edge_attr_cpsd_resonance.shape, dtype=np.int64),
        "edge_attr_cpsd_resonance": edge_attr_cpsd_resonance,
        "edge_attr_cpsd_resonance_feature_names": np.asarray(feature_names),
        "edge_attr_cpsd_resonance_mode_names": np.asarray(MODE_NAMES),
        "edge_attr_cpsd_resonance_matched_freq_hz": np.asarray(matched_freq, dtype=float),
        "edge_attr_cpsd_resonance_sampled_freq_hz": np.asarray(sampled_freq_hz, dtype=float),
        "edge_attr_cpsd_resonance_bin_indices": np.asarray(sampled_bin_idx, dtype=np.int64),
        "edge_attr_cpsd_resonance_signal_domain": np.asarray("absolute_u3_acceleration"),
        "edge_attr_cpsd_resonance_representation": np.asarray("complex_coherency_real_imag"),
        "edge_attr_cpsd_resonance_description": np.asarray(
            "Per directed edge, store real and imaginary parts of normalized CPSD (complex coherency) at the nearest Welch bins to the four matched modal frequencies."
        ),
        "edge_attr_cpsd_resonance_shape": np.asarray(edge_attr_cpsd_resonance.shape, dtype=np.int64),
        "edge_attr_cpsd_resonance_feature_dim": np.asarray(int(edge_attr_cpsd_resonance.shape[1]), dtype=np.int64),
    }


def assemble_phase1_sample_payload(
    sample_id: int,
    sampled: dict[str, Any],
    solve_res: SolveResult,
    match_res: Any,
    *,
    matcher_config_json: str,
    reference_file: str,
    freqs_4: np.ndarray,
    phi_u3_4: np.ndarray,
    S4: np.ndarray,
    bandpass: bool = False,
) -> dict[str, Any]:
    selected_idx = np.asarray(match_res.selected_idx, dtype=int).reshape(-1)
    matched_mac = np.asarray(match_res.selected_mac, dtype=float).reshape(-1)
    matched_composite = np.asarray(match_res.selected_composite, dtype=float).reshape(-1)
    ambiguity_flags = np.asarray(match_res.ambiguity_flags, dtype=bool).reshape(-1)

    if selected_idx.shape != (4,):
        raise ValueError(f"Expected 4 matched mode indices for Phase 1, got {selected_idx.shape}.")

    graph_payload = build_projected_graph_from_model(solve_res.model, include_sensor_map=False)
    sim_model = dict(solve_res.model)
    sim_model["M_full"] = solve_res.M_full
    sim_payload = simulate_white_noise_from_modal_model(
        sim_model,
        solve_res.phi_full,
        solve_res.freq_all_hz,
        selected_idx,
        zeta=PHASE1_ZETA,
        plot=False,
        return_time_series=True,
        use_all_modes=bandpass,
        bandpass_hz=(1.5, 20.0) if bandpass else None,
    )

    graph_node_ids = np.asarray(graph_payload["graph_node_ids"], dtype=np.int64)
    full_rows = graph_node_ids - 1
    freq_bins_full = np.asarray(sim_payload["freq_bins_hz"], dtype=float).reshape(-1)
    if freq_bins_full.shape[0] < PHASE1_FREQ_BINS:
        raise ValueError(
            f"Simulation returned only {freq_bins_full.shape[0]} frequency bins; expected at least {PHASE1_FREQ_BINS}."
        )

    freq_bins_hz = np.asarray(freq_bins_full[:PHASE1_FREQ_BINS], dtype=float)
    x_psd = np.asarray(sim_payload["Paa_norm"], dtype=float)[full_rows, :PHASE1_FREQ_BINS]
    x_psd_row_scale = np.asarray(sim_payload["Paa_row_scale"], dtype=float)[full_rows]
    y_mode_shape_abs = np.asarray(sim_payload["phi_u3_abs_norm"], dtype=float)[full_rows, :]
    y_mode_shape_scale = np.asarray(sim_payload["phi_u3_abs_scale"], dtype=float).reshape(-1)
    signed_payload = build_signed_mode_shape_payload(
        sampled=sampled,
        solve_res=solve_res,
        reference_file=reference_file,
        phi_u3_sensor_4=np.asarray(phi_u3_4, dtype=float),
        phi_u3_full_4=np.asarray(sim_payload["phi_u3_all"], dtype=float),
        graph_node_ids=graph_node_ids,
    )

    _validate_phase1_mapping(
        graph_node_ids=graph_node_ids,
        full_node_count=int(np.asarray(sim_payload["Paa_abs"]).shape[0]),
        x_psd=x_psd,
        y_mode_shape_abs=y_mode_shape_abs,
        edge_index=np.asarray(graph_payload["edge_index"], dtype=np.int64),
        edge_attr_geom=np.asarray(graph_payload["edge_attr_geom"], dtype=float),
        freq_bins_hz=freq_bins_hz,
    )

    target_mode_names = np.asarray(MODE_NAMES, dtype="<U2")
    y_freq_hz = np.asarray(freqs_4, dtype=np.float32)
    cpsd_payload = build_cpsd_edge_payload(
        edge_index=np.asarray(graph_payload["edge_index"], dtype=np.int64),
        matched_freq_hz=np.asarray(freqs_4, dtype=float),
        freq_bins_hz=freq_bins_hz,
        a3_abs=np.asarray(sim_payload["a3_abs"], dtype=float)[full_rows, :],
        fs=float(np.asarray(sim_payload["fs"]).reshape(())),
        nperseg=WELCH_NPERSEG,
        noverlap=WELCH_NOVERLAP,
        nfft=WELCH_NFFT,
    )
    cpsd_full_payload = build_cpsd_full_edge_payload(
        edge_index=np.asarray(graph_payload["edge_index"], dtype=np.int64),
        freq_bins_hz=freq_bins_hz,
        a3_abs=np.asarray(sim_payload["a3_abs"], dtype=float)[full_rows, :],
        fs=float(np.asarray(sim_payload["fs"]).reshape(())),
        nperseg=WELCH_NPERSEG,
        noverlap=WELCH_NOVERLAP,
        nfft=WELCH_NFFT,
    )
    _validate_phase2a_frequency_targets(
        target_mode_names=target_mode_names,
        matched_freq_hz=np.asarray(freqs_4, dtype=float),
        y_freq_hz=y_freq_hz,
    )

    payload = {
        "schema_version": np.asarray(PHASE1_SCHEMA_VERSION),
        "sample_index": np.asarray(int(sample_id), dtype=np.int64),
        "sample_id": np.asarray(int(sample_id), dtype=np.int64),
        "Ng": np.asarray(int(sampled["Ng"]), dtype=np.int64),
        "Nx": np.asarray(int(sampled["Nx"]), dtype=np.int64),
        "n_graph_nodes": np.asarray(int(graph_node_ids.size), dtype=np.int64),
        "n_graph_edges_directed": np.asarray(int(np.asarray(graph_payload["edge_index"]).shape[1]), dtype=np.int64),
        "n_freq_bins": np.asarray(int(freq_bins_hz.shape[0]), dtype=np.int64),
        "params_json": np.asarray(to_json(sampled)),
        "matcher_config_json": np.asarray(matcher_config_json),
        "matcher_result_json": np.asarray(match_res.to_json()),
        "reference_file": np.asarray(str(reference_file)),
        "graph_node_ids": graph_node_ids,
        "coords_xy_mm": np.asarray(graph_payload["coords_xy_mm"], dtype=float),
        "coords_xyz0_mm": np.asarray(graph_payload["coords_xyz0_mm"], dtype=float),
        "edge_index": np.asarray(graph_payload["edge_index"], dtype=np.int64),
        "edge_index_undirected": np.asarray(graph_payload["edge_index_undirected"], dtype=np.int64),
        "edge_length_mm": np.asarray(graph_payload["edge_length_mm"], dtype=float),
        "edge_theta_rad": np.asarray(graph_payload["edge_theta_rad"], dtype=float),
        "edge_attr_geom": np.asarray(graph_payload["edge_attr_geom"], dtype=float),
        "edge_attr_cpsd": np.asarray(cpsd_payload["edge_attr_cpsd"], dtype=np.float32),
        "edge_attr_cpsd_feature_names": np.asarray(cpsd_payload["edge_attr_cpsd_feature_names"]),
        "edge_attr_cpsd_mode_names": np.asarray(cpsd_payload["edge_attr_cpsd_mode_names"]),
        "edge_attr_cpsd_matched_freq_hz": np.asarray(cpsd_payload["edge_attr_cpsd_matched_freq_hz"], dtype=float),
        "edge_attr_cpsd_sampled_freq_hz": np.asarray(cpsd_payload["edge_attr_cpsd_sampled_freq_hz"], dtype=float),
        "edge_attr_cpsd_bin_indices": np.asarray(cpsd_payload["edge_attr_cpsd_bin_indices"], dtype=np.int64),
        "edge_attr_cpsd_signal_domain": np.asarray(cpsd_payload["edge_attr_cpsd_signal_domain"]),
        "edge_attr_cpsd_representation": np.asarray(cpsd_payload["edge_attr_cpsd_representation"]),
        "edge_attr_cpsd_description": np.asarray(cpsd_payload["edge_attr_cpsd_description"]),
        "edge_attr_cpsd_shape": np.asarray(cpsd_payload["edge_attr_cpsd_shape"], dtype=np.int64),
        "edge_attr_cpsd_resonance": np.asarray(cpsd_payload["edge_attr_cpsd_resonance"], dtype=np.float32),
        "edge_attr_cpsd_resonance_feature_names": np.asarray(cpsd_payload["edge_attr_cpsd_resonance_feature_names"]),
        "edge_attr_cpsd_resonance_mode_names": np.asarray(cpsd_payload["edge_attr_cpsd_resonance_mode_names"]),
        "edge_attr_cpsd_resonance_matched_freq_hz": np.asarray(cpsd_payload["edge_attr_cpsd_resonance_matched_freq_hz"], dtype=float),
        "edge_attr_cpsd_resonance_sampled_freq_hz": np.asarray(cpsd_payload["edge_attr_cpsd_resonance_sampled_freq_hz"], dtype=float),
        "edge_attr_cpsd_resonance_bin_indices": np.asarray(cpsd_payload["edge_attr_cpsd_resonance_bin_indices"], dtype=np.int64),
        "edge_attr_cpsd_resonance_signal_domain": np.asarray(cpsd_payload["edge_attr_cpsd_resonance_signal_domain"]),
        "edge_attr_cpsd_resonance_representation": np.asarray(cpsd_payload["edge_attr_cpsd_resonance_representation"]),
        "edge_attr_cpsd_resonance_description": np.asarray(cpsd_payload["edge_attr_cpsd_resonance_description"]),
        "edge_attr_cpsd_resonance_shape": np.asarray(cpsd_payload["edge_attr_cpsd_resonance_shape"], dtype=np.int64),
        "edge_attr_cpsd_resonance_feature_dim": np.asarray(cpsd_payload["edge_attr_cpsd_resonance_feature_dim"], dtype=np.int64),
        "edge_attr_cpsd_full": np.asarray(cpsd_full_payload["edge_attr_cpsd_full"], dtype=np.float32),
        "edge_attr_cpsd_full_feature_names": np.asarray(cpsd_full_payload["edge_attr_cpsd_full_feature_names"]),
        "edge_attr_cpsd_full_freq_hz": np.asarray(cpsd_full_payload["edge_attr_cpsd_full_freq_hz"], dtype=float),
        "edge_attr_cpsd_full_signal_domain": np.asarray(cpsd_full_payload["edge_attr_cpsd_full_signal_domain"]),
        "edge_attr_cpsd_full_representation": np.asarray(cpsd_full_payload["edge_attr_cpsd_full_representation"]),
        "edge_attr_cpsd_full_description": np.asarray(cpsd_full_payload["edge_attr_cpsd_full_description"]),
        "edge_attr_cpsd_full_shape": np.asarray(cpsd_full_payload["edge_attr_cpsd_full_shape"], dtype=np.int64),
        "edge_attr_cpsd_full_feature_dim": np.asarray(cpsd_full_payload["edge_attr_cpsd_full_feature_dim"], dtype=np.int64),
        "group_include_names": np.asarray(graph_payload["group_include_names"]),
        "group_include_ids": np.asarray(graph_payload["group_include_ids"], dtype=np.int64),
        "group_include_enabled": np.asarray(graph_payload["group_include_enabled"], dtype=bool),
        "freq_bins_hz": freq_bins_hz,
        "x_psd": x_psd,
        "x_psd_row_scale": x_psd_row_scale,
        "target_mode_names": target_mode_names,
        "y_mode_shape_abs": y_mode_shape_abs,
        "y_mode_shape_scale": y_mode_shape_scale,
        "y_mode_shape_signed": np.asarray(signed_payload["y_mode_shape_signed"], dtype=float),
        "y_mode_shape_signed_scale": np.asarray(signed_payload["y_mode_shape_signed_scale"], dtype=float),
        "y_freq_hz": y_freq_hz,
        "matched_mode_indices": selected_idx.astype(np.int64),
        "matched_freq_hz": np.asarray(freqs_4, dtype=float),
        "matched_mac": matched_mac,
        "matched_composite_score": matched_composite,
        "matched_ambiguity_flag": ambiguity_flags,
        "zeta": np.asarray(PHASE1_ZETA, dtype=float),
        "selected_mode_indices_sensor_domain": selected_idx.astype(np.int64),
        "phi_u3_sensor_4": np.asarray(phi_u3_4, dtype=float),
        "phi_u3_sensor_4_signed": np.asarray(signed_payload["phi_u3_sensor_4_signed"], dtype=float),
        "phi_u3_sensor_4_signed_norm": np.asarray(signed_payload["phi_u3_sensor_4_signed_norm"], dtype=float),
        "sign_flip_factors": np.asarray(signed_payload["sign_flip_factors"], dtype=float),
        "sign_reference_dot_before": np.asarray(signed_payload["sign_reference_dot_before"], dtype=float),
        "sign_reference_dot_after": np.asarray(signed_payload["sign_reference_dot_after"], dtype=float),
        "sign_reference_sensor_index": np.asarray(signed_payload["sign_reference_sensor_index"], dtype=np.int64),
        "sign_reference_sensor_node_id": np.asarray(signed_payload["sign_reference_sensor_node_id"], dtype=np.int64),
        "sign_fallback_used": np.asarray(signed_payload["sign_fallback_used"], dtype=bool),
        "sign_ambiguous_flag": np.asarray(signed_payload["sign_ambiguous_flag"], dtype=bool),
        "sign_fallback_reason": np.asarray(signed_payload["sign_fallback_reason"]),
        "sign_rule_name": np.asarray(signed_payload["sign_rule_name"]),
        "sign_rule_description": np.asarray(signed_payload["sign_rule_description"]),
        "sign_alignment_ready": np.asarray(signed_payload["sign_alignment_ready"]),
        "sign_reference_template_sensor_4": np.asarray(signed_payload["sign_reference_template_sensor_4"], dtype=float),
        "S4": np.asarray(S4, dtype=float),
    }
    return payload


def save_phase1_sample_npz(
    out_phase1_dir: Path,
    sample_id: int,
    sampled: dict[str, Any],
    payload: dict[str, Any],
) -> Path:
    ng = int(sampled["Ng"])
    out_path = out_phase1_dir / f"sample_{sample_id:04d}_Ng{ng}_phase1_gnn.npz"
    np.savez(out_path, **payload)
    return out_path


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_probability_histograms(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No rows available for plotting.")
        return

    S_by_mode = {mode: np.asarray([float(r[f"S_{mode}"]) for r in rows], dtype=float) for mode in MODE_NAMES}
    F_by_mode = {mode: np.asarray([float(r[f"freq_{mode}_hz"]) for r in rows], dtype=float) for mode in MODE_NAMES}

    all_S = np.concatenate([S_by_mode[mode] for mode in MODE_NAMES])
    bins_S = np.histogram_bin_edges(all_S, bins=PLOT_BINS)

    plt.figure("Probability Histogram of S (Modes 4)")
    for mode in MODE_NAMES:
        plt.hist(S_by_mode[mode], bins=bins_S, density=True, alpha=0.5, label=mode)
    plt.title("Probability Histogram of S by Target Mode")
    plt.xlabel("S = sum((phi/maxabs)^2)")
    plt.ylabel("Probability density")
    plt.grid(True)
    plt.legend()

    all_F = np.concatenate([F_by_mode[mode] for mode in MODE_NAMES])
    bins_F = np.histogram_bin_edges(all_F, bins=PLOT_BINS)

    plt.figure("Probability Histogram of Frequency (Modes 4)")
    for mode in MODE_NAMES:
        plt.hist(F_by_mode[mode], bins=bins_F, density=True, alpha=0.5, label=mode)
    plt.title("Probability Histogram of Natural Frequency by Target Mode")
    plt.xlabel("Natural frequency (Hz)")
    plt.ylabel("Probability density")
    plt.grid(True)
    plt.legend()

    plt.show()


def _detect_outlier_mask_1d(x: np.ndarray, config: OutlierConfig) -> tuple[np.ndarray, float, float, str]:
    arr = np.asarray(x, dtype=float).reshape(-1)
    method = str(config.method).lower().strip()
    if arr.size == 0:
        return np.zeros(0, dtype=bool), -np.inf, np.inf, method

    if method == "iqr":
        q1 = float(np.percentile(arr, 25.0))
        q3 = float(np.percentile(arr, 75.0))
        iqr = float(q3 - q1)
        low = q1 - float(config.iqr_k) * iqr
        high = q3 + float(config.iqr_k) * iqr
        return (arr < low) | (arr > high), low, high, method

    if method == "mad":
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        if mad <= 1e-12:
            return np.zeros(arr.size, dtype=bool), -np.inf, np.inf, method
        sigma = mad / 0.6745
        low = med - float(config.mad_k) * sigma
        high = med + float(config.mad_k) * sigma
        return (arr < low) | (arr > high), low, high, method

    if method == "zscore":
        mu = float(np.mean(arr))
        sd = float(np.std(arr))
        if sd <= 1e-12:
            return np.zeros(arr.size, dtype=bool), -np.inf, np.inf, method
        low = mu - float(config.zscore_k) * sd
        high = mu + float(config.zscore_k) * sd
        return (arr < low) | (arr > high), low, high, method

    raise ValueError(f"Unsupported outlier method '{config.method}'. Use one of: iqr, mad, zscore.")


def detect_frequency_outliers(rows: list[dict[str, Any]], config: OutlierConfig) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "sample_outlier_mask": np.zeros(0, dtype=bool),
            "mode_outlier_masks": {m: np.zeros(0, dtype=bool) for m in MODE_NAMES},
            "outlier_sample_ids": [],
            "outlier_details": [],
            "filtered_rows": [],
            "n_outliers_total": 0,
            "n_outlier_samples": 0,
            "counts_by_mode": {m: 0 for m in MODE_NAMES},
        }

    if not bool(config.enable_outlier_filter):
        return {
            "sample_outlier_mask": np.zeros(n, dtype=bool),
            "mode_outlier_masks": {m: np.zeros(n, dtype=bool) for m in MODE_NAMES},
            "outlier_sample_ids": [],
            "outlier_details": [],
            "filtered_rows": list(rows),
            "n_outliers_total": 0,
            "n_outlier_samples": 0,
            "counts_by_mode": {m: 0 for m in MODE_NAMES},
        }

    sample_mask = np.zeros(n, dtype=bool)
    mode_masks: dict[str, np.ndarray] = {}
    details: list[dict[str, Any]] = []
    counts_by_mode: dict[str, int] = {}

    sample_ids = [int(r.get("sample_id", i)) for i, r in enumerate(rows)]

    for mode in MODE_NAMES:
        key = f"freq_{mode}_hz"
        vals = np.asarray([float(r[key]) for r in rows], dtype=float)
        mask, low, high, method_used = _detect_outlier_mask_1d(vals, config)
        mode_masks[mode] = mask
        counts_by_mode[mode] = int(np.sum(mask))
        sample_mask |= mask

        idxs = np.where(mask)[0]
        for i in idxs:
            details.append(
                {
                    "sample_id": int(sample_ids[int(i)]),
                    "mode_name": str(mode),
                    "value": float(vals[int(i)]),
                    "lower_bound": float(low),
                    "upper_bound": float(high),
                    "method": str(method_used),
                }
            )

    filtered_rows = [r for i, r in enumerate(rows) if not bool(sample_mask[i])]
    outlier_sample_ids = [int(sample_ids[i]) for i in np.where(sample_mask)[0]]

    return {
        "sample_outlier_mask": sample_mask,
        "mode_outlier_masks": mode_masks,
        "outlier_sample_ids": outlier_sample_ids,
        "outlier_details": details,
        "filtered_rows": filtered_rows,
        "n_outliers_total": int(sum(counts_by_mode.values())),
        "n_outlier_samples": int(np.sum(sample_mask)),
        "counts_by_mode": counts_by_mode,
    }


def save_outlier_report_json(report: dict[str, Any], config: OutlierConfig, out_path: Path) -> None:
    payload = {
        "config": asdict(config),
        "n_outliers_total": int(report["n_outliers_total"]),
        "n_outlier_samples": int(report["n_outlier_samples"]),
        "counts_by_mode": {k: int(v) for k, v in report["counts_by_mode"].items()},
        "outlier_sample_ids": [int(x) for x in report["outlier_sample_ids"]],
        "outlier_details": report["outlier_details"],
    }
    out_path.write_text(to_json(payload), encoding="utf-8")


def plot_probability_histograms_before_after(
    rows_before: list[dict[str, Any]],
    rows_after: list[dict[str, Any]],
) -> None:
    if not rows_before:
        print("No rows available for plotting.")
        return

    s_before = {mode: np.asarray([float(r[f"S_{mode}"]) for r in rows_before], dtype=float) for mode in MODE_NAMES}
    f_before = {mode: np.asarray([float(r[f"freq_{mode}_hz"]) for r in rows_before], dtype=float) for mode in MODE_NAMES}

    all_s_before = np.concatenate([s_before[m] for m in MODE_NAMES])
    bins_s_before = np.histogram_bin_edges(all_s_before, bins=PLOT_BINS)
    plt.figure("Probability Histogram of S (Before outlier removal)")
    for mode in MODE_NAMES:
        plt.hist(s_before[mode], bins=bins_s_before, density=True, alpha=0.5, label=mode)
    plt.title("Probability Histogram of S by Target Mode (Before outlier removal)")
    plt.xlabel("S = sum((phi/maxabs)^2)")
    plt.ylabel("Probability density")
    plt.grid(True)
    plt.legend()

    all_f_before = np.concatenate([f_before[m] for m in MODE_NAMES])
    bins_f_before = np.histogram_bin_edges(all_f_before, bins=PLOT_BINS)
    plt.figure("Probability Histogram of Frequency (Before outlier removal)")
    for mode in MODE_NAMES:
        plt.hist(f_before[mode], bins=bins_f_before, density=True, alpha=0.5, label=mode)
    plt.title("Probability Histogram of Natural Frequency by Target Mode (Before outlier removal)")
    plt.xlabel("Natural frequency (Hz)")
    plt.ylabel("Probability density")
    plt.grid(True)
    plt.legend()

    if not rows_after:
        plt.figure("Probability Histogram of S (After outlier removal)")
        plt.title("Probability Histogram of S by Target Mode (After outlier removal) - no data")
        plt.xlabel("S = sum((phi/maxabs)^2)")
        plt.ylabel("Probability density")
        plt.grid(True)
        plt.figure("Probability Histogram of Frequency (After outlier removal)")
        plt.title("Probability Histogram of Natural Frequency by Target Mode (After outlier removal) - no data")
        plt.xlabel("Natural frequency (Hz)")
        plt.ylabel("Probability density")
        plt.grid(True)
        plt.show()
        return

    s_after = {mode: np.asarray([float(r[f"S_{mode}"]) for r in rows_after], dtype=float) for mode in MODE_NAMES}
    f_after = {mode: np.asarray([float(r[f"freq_{mode}_hz"]) for r in rows_after], dtype=float) for mode in MODE_NAMES}

    all_s_after = np.concatenate([s_after[m] for m in MODE_NAMES])
    bins_s_after = np.histogram_bin_edges(all_s_after, bins=PLOT_BINS)
    plt.figure("Probability Histogram of S (After outlier removal)")
    for mode in MODE_NAMES:
        plt.hist(s_after[mode], bins=bins_s_after, density=True, alpha=0.5, label=mode)
    plt.title("Probability Histogram of S by Target Mode (After outlier removal)")
    plt.xlabel("S = sum((phi/maxabs)^2)")
    plt.ylabel("Probability density")
    plt.grid(True)
    plt.legend()

    all_f_after = np.concatenate([f_after[m] for m in MODE_NAMES])
    bins_f_after = np.histogram_bin_edges(all_f_after, bins=PLOT_BINS)
    plt.figure("Probability Histogram of Frequency (After outlier removal)")
    for mode in MODE_NAMES:
        plt.hist(f_after[mode], bins=bins_f_after, density=True, alpha=0.5, label=mode)
    plt.title("Probability Histogram of Natural Frequency by Target Mode (After outlier removal)")
    plt.xlabel("Natural frequency (Hz)")
    plt.ylabel("Probability density")
    plt.grid(True)
    plt.legend()

    plt.show()


def plot_selected_mode_shapes_for_inspection(
    model: dict[str, Any],
    phi_full: np.ndarray,
    freq_all_hz: np.ndarray,
    selected_idx: np.ndarray,
    selected_mac: np.ndarray,
    selected_composite: np.ndarray | None = None,
) -> None:
    nodes = np.asarray(model["nodes"], dtype=float)
    x_all = np.asarray(nodes[:, 1], dtype=float)
    nn = int(nodes.shape[0])
    ndof_expect = 6 * nn
    phi = np.asarray(phi_full, dtype=float)
    freq = np.asarray(freq_all_hz, dtype=float).reshape(-1)
    idx4 = np.asarray(selected_idx, dtype=int).reshape(-1)
    mac4 = np.asarray(selected_mac, dtype=float).reshape(-1)
    comp4 = (
        np.asarray(selected_composite, dtype=float).reshape(-1)
        if selected_composite is not None
        else np.full(idx4.size, np.nan, dtype=float)
    )

    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for j, mode_name in enumerate(MODE_NAMES):
        ax = axs.ravel()[j]
        i_mode = int(idx4[j])
        if phi.shape[0] < ndof_expect or i_mode < 0 or i_mode >= phi.shape[1]:
            ax.set_title(f"{mode_name} | unavailable")
            ax.grid(True)
            continue
        u3 = phi[2:ndof_expect:6, i_mode]
        u3n = _normalize_max_abs(u3)
        order = np.argsort(x_all)
        x = x_all[order]
        y = u3n[order]
        ax.plot(x, np.zeros_like(x), "k--", linewidth=0.8, label="Undeformed baseline")
        ax.plot(x, y, "-", linewidth=1.1, label="Selected mode U3 (normalized)")
        ax.set_title(
            f"{mode_name} | idx={i_mode}, f={float(freq[i_mode]):.3f} Hz, "
            f"MAC={float(mac4[j]):.4f}, score={float(comp4[j]):.4f}"
        )
        ax.set_xlabel("x coordinate (mm)")
        ax.set_ylabel("Normalized U3")
        ax.grid(True)
        ax.legend()
    fig.suptitle("Selected Full-Field Mode Shapes (U3 vs x) for Inspection")


def clean_output_directory(out_dir: Path) -> None:
    samples = out_dir / "samples"
    phase1_samples = out_dir / PHASE1_SAMPLES_SUBDIR
    phase2a_samples = out_dir / PHASE2A_SAMPLES_SUBDIR
    plots = out_dir / "plots"

    if samples.exists():
        for p in samples.glob("*"):
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    if plots.exists():
        for p in plots.glob("*"):
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    if phase1_samples.exists():
        for p in phase1_samples.glob("*"):
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    if phase2a_samples.exists():
        for p in phase2a_samples.glob("*"):
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    summary = out_dir / "summary.csv"
    if summary.exists():
        summary.unlink()
    filtered_summary = out_dir / "filtered_summary.csv"
    if filtered_summary.exists():
        filtered_summary.unlink()
    outlier_report = out_dir / "outlier_report.json"
    if outlier_report.exists():
        outlier_report.unlink()

    fail_log = out_dir / "failures.log"
    if fail_log.exists():
        fail_log.unlink()


def find_sample_npz(sample_id: int) -> Path:
    samples_dir = Path(OUT_DIR) / "samples"
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    pattern = f"sample_{sample_id:04d}_Ng*.npz"
    matches = sorted(samples_dir.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No sample file found for sample_id={sample_id} using pattern {pattern} in {samples_dir}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple sample files found for sample_id={sample_id}: {[str(m) for m in matches]}"
        )
    return matches[0]


def _parse_params_json(npz_obj: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "params_json" not in npz_obj:
        raise KeyError("Sample file missing required key: params_json")
    raw = npz_obj["params_json"]
    if np.ndim(raw) == 0:
        s = str(raw.item())
    else:
        s = str(raw)
    return json.loads(s)


def _normalize_max_abs(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(-1)
    den = float(np.max(np.abs(a)))
    if den <= eps:
        return np.zeros_like(a, dtype=float)
    return a / den


def _print_match_diagnostics(match_res: Any, topk: int = 5) -> None:
    print(f"\nMatcher signature type: {match_res.signature_type_used}")
    active_switches = match_res.diagnostics.get("active_switches", {})
    if active_switches:
        print(
            "Active switches: "
            f"two_stage_torsion={bool(active_switches.get('use_two_stage_torsion_rerank', False))}, "
            f"u2_for_torsion={bool(active_switches.get('use_u2_for_torsion', False))}, "
            f"balcony_penalty={bool(active_switches.get('use_balcony_penalty', False))}, "
            f"participation={bool(active_switches.get('use_participation_features', False))}"
        )
    torsion_stage2 = match_res.diagnostics.get("torsion_stage2", {})
    if active_switches.get("use_two_stage_torsion_rerank", False):
        print(f"Torsion shortlist_k={int(match_res.diagnostics.get('torsion_shortlist_k', 0))}")
    for j, mode_name in enumerate(MODE_NAMES):
        sel = match_res.score_breakdown.get(mode_name, {})
        stage_label = str(sel.get("selection_stage", "base_u3_mmi"))
        print(
            f"Selected {mode_name}: idx={int(sel.get('mode_idx', -1))}, "
            f"f={float(sel.get('freq_hz', np.nan)):.6f} Hz, "
            f"base_mmi={float(sel.get('base_mmi_u3', np.nan)):.6f}, "
            f"stage={stage_label}, "
            f"stage2_score={float(sel.get('stage2_score', np.nan)):.6f}, "
            f"u2_term={float(sel.get('u2_term', np.nan)):.6f}, "
            f"bal_pen={float(sel.get('balcony_penalty', np.nan)):.6f}, "
            f"part_term={float(sel.get('participation_term', np.nan)):.6f}, "
            f"total={float(sel.get('final_total', sel.get('total', np.nan))):.6f}"
        )
        rows = match_res.top_candidates_per_target.get(mode_name, [])
        print(f"Stage-1 top-{min(topk, len(rows))} base candidates for {mode_name}:")
        for row in rows[:topk]:
            print(
                f"  mode_idx={int(row['mode_idx']):>3d}, base={float(row.get('base_mmi_u3', np.nan)):.6f}, "
                f"shortlist={bool(row.get('shortlisted_flag', False))}, "
                f"stage2={float(row.get('stage2_score', np.nan)):.6f}, "
                f"final={float(row.get('final_total', row['total'])):.6f}, "
                f"stage2_rank={_diag_fmt(row.get('stage2_rank', np.nan), '.0f')}, "
                f"base={float(row.get('base_mmi_u3', np.nan)):.6f}, "
                f"freq={float(row['freq_hz']):.6f} Hz"
            )
        if mode_name in torsion_stage2:
            shortlist = torsion_stage2[mode_name].get("shortlist_mode_indices", [])
            print(f"Stage-2 shortlist for {mode_name}: {shortlist}")
            for row in torsion_stage2[mode_name].get("shortlist_rows", []):
                print(
                    f"  mode_idx={int(row.get('mode_idx', -1)):>3d}, "
                    f"stage2_rank={int(row.get('stage2_rank', -1)):>2d}, "
                    f"stage2_score={float(row.get('stage2_score', np.nan)):.6f}, "
                    f"u2_norm={float(row.get('u2_norm_contamination', np.nan)):.6f}, "
                    f"u2_peak={float(row.get('u2_peak_contamination', np.nan)):.6f}, "
                    f"part_gamma={float(row.get('participation_gamma_ratio', np.nan)):.6f}, "
                    f"part_meff={float(row.get('participation_meff_ratio', np.nan)):.6f}, "
                    f"girder_bal={float(row.get('girder_to_balcony_ratio', np.nan)):.6f}, "
                    f"basis={row.get('stage2_basis', '')}, "
                    f"selected={bool(row.get('final_selected_flag', False))}"
                )
    if match_res.warning_messages:
        print("Ambiguity warnings:")
        for w in match_res.warning_messages:
            print(f"  - {w}")


def _inspect_csv_paths(sample_id: int) -> dict[str, Path]:
    stem = f"inspect_{int(sample_id):04d}"
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "score_breakdown": out_dir / f"{stem}_score_breakdown.csv",
        "top_candidates": out_dir / f"{stem}_top_candidates.csv",
        "torsion_stage2": out_dir / f"{stem}_torsion_stage2.csv",
        "score_matrix": out_dir / f"{stem}_score_matrix.csv",
        "base_score_matrix": out_dir / f"{stem}_base_score_matrix.csv",
    }


def _write_inspect_matcher_csvs(sample_id: int, match_res: Any) -> dict[str, Path]:
    paths = _inspect_csv_paths(sample_id)

    with paths["score_breakdown"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target",
                "mode_idx",
                "freq_hz",
                "template_index",
                "template_freq_hz",
                "mac_u3",
                "base_freq_penalty",
                "base_mmi_u3",
                "mac_u2",
                "u2_term",
                "balcony_metric",
                "balcony_penalty",
                "participation_metric",
                "participation_gamma_ratio",
                "participation_meff_ratio",
                "participation_term",
                "selection_stage",
                "stage1_rank",
                "shortlisted_flag",
                "stage2_score",
                "stage2_rank",
                "stage2_basis",
                "girder_u3_metric",
                "balcony_combined_metric",
                "girder_to_balcony_ratio",
                "u2_norm_contamination",
                "u2_peak_contamination",
                "final_total",
            ],
        )
        writer.writeheader()
        for target in MODE_NAMES:
            row = dict(match_res.score_breakdown.get(target, {}))
            writer.writerow(
                {
                    "target": target,
                    "mode_idx": row.get("mode_idx", np.nan),
                    "freq_hz": row.get("freq_hz", np.nan),
                    "template_index": row.get("template_index", np.nan),
                    "template_freq_hz": row.get("template_freq_hz", np.nan),
                    "mac_u3": row.get("mac_u3", row.get("mac", np.nan)),
                    "base_freq_penalty": row.get("base_freq_penalty", row.get("freq_penalty", np.nan)),
                    "base_mmi_u3": row.get("base_mmi_u3", np.nan),
                    "mac_u2": row.get("mac_u2", row.get("u2_mac", np.nan)),
                    "u2_term": row.get("u2_term", np.nan),
                    "balcony_metric": row.get("balcony_metric", row.get("balcony_ratio", np.nan)),
                    "balcony_penalty": row.get("balcony_penalty", np.nan),
                    "participation_metric": row.get("participation_metric", row.get("participation_score", np.nan)),
                    "participation_gamma_ratio": row.get("participation_gamma_ratio", np.nan),
                    "participation_meff_ratio": row.get("participation_meff_ratio", np.nan),
                    "participation_term": row.get("participation_term", np.nan),
                    "selection_stage": row.get("selection_stage", ""),
                    "stage1_rank": row.get("stage1_rank", np.nan),
                    "shortlisted_flag": row.get("shortlisted_flag", False),
                    "stage2_score": row.get("stage2_score", np.nan),
                    "stage2_rank": row.get("stage2_rank", np.nan),
                    "stage2_basis": row.get("stage2_basis", ""),
                    "girder_u3_metric": row.get("girder_u3_metric", np.nan),
                    "balcony_combined_metric": row.get("balcony_combined_metric", np.nan),
                    "girder_to_balcony_ratio": row.get("girder_to_balcony_ratio", np.nan),
                    "u2_norm_contamination": row.get("u2_norm_contamination", np.nan),
                    "u2_peak_contamination": row.get("u2_peak_contamination", np.nan),
                    "final_total": row.get("final_total", row.get("total", np.nan)),
                }
            )

    with paths["top_candidates"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target",
                "rank",
                "stage1_rank",
                "mode_idx",
                "freq_hz",
                "template_index",
                "template_freq_hz",
                "mac_u3",
                "base_freq_penalty",
                "base_mmi_u3",
                "mac_u2",
                "u2_term",
                "balcony_metric",
                "balcony_penalty",
                "participation_metric",
                "participation_gamma_ratio",
                "participation_meff_ratio",
                "participation_term",
                "shortlisted_flag",
                "selection_stage",
                "stage2_score",
                "stage2_rank",
                "stage2_basis",
                "girder_u3_metric",
                "balcony_combined_metric",
                "girder_to_balcony_ratio",
                "u2_norm_contamination",
                "u2_peak_contamination",
                "final_total",
            ],
        )
        writer.writeheader()
        for target in MODE_NAMES:
            rows = list(match_res.top_candidates_per_target.get(target, []))
            for rank, row in enumerate(rows[:5], start=1):
                writer.writerow(
                    {
                        "target": target,
                        "rank": rank,
                        "stage1_rank": row.get("stage1_rank", rank),
                        "mode_idx": row.get("mode_idx", np.nan),
                        "freq_hz": row.get("freq_hz", np.nan),
                        "template_index": row.get("template_index", np.nan),
                        "template_freq_hz": row.get("template_freq_hz", np.nan),
                        "mac_u3": row.get("mac_u3", row.get("mac", row.get("primary_mac", np.nan))),
                        "base_freq_penalty": row.get("base_freq_penalty", row.get("freq_penalty", np.nan)),
                        "base_mmi_u3": row.get("base_mmi_u3", np.nan),
                        "mac_u2": row.get("mac_u2", row.get("u2_mac", np.nan)),
                        "u2_term": row.get("u2_term", np.nan),
                        "balcony_metric": row.get("balcony_metric", row.get("balcony_ratio", np.nan)),
                        "balcony_penalty": row.get("balcony_penalty", np.nan),
                        "participation_metric": row.get("participation_metric", row.get("participation_score", np.nan)),
                        "participation_gamma_ratio": row.get("participation_gamma_ratio", np.nan),
                        "participation_meff_ratio": row.get("participation_meff_ratio", np.nan),
                        "participation_term": row.get("participation_term", np.nan),
                        "shortlisted_flag": row.get("shortlisted_flag", False),
                        "selection_stage": row.get("selection_stage", ""),
                        "stage2_score": row.get("stage2_score", np.nan),
                        "stage2_rank": row.get("stage2_rank", np.nan),
                        "stage2_basis": row.get("stage2_basis", ""),
                        "girder_u3_metric": row.get("girder_u3_metric", np.nan),
                        "balcony_combined_metric": row.get("balcony_combined_metric", np.nan),
                        "girder_to_balcony_ratio": row.get("girder_to_balcony_ratio", np.nan),
                        "u2_norm_contamination": row.get("u2_norm_contamination", np.nan),
                        "u2_peak_contamination": row.get("u2_peak_contamination", np.nan),
                        "final_total": row.get("final_total", row.get("total", np.nan)),
                    }
                )

    torsion_stage2 = match_res.diagnostics.get("torsion_stage2", {})
    with paths["torsion_stage2"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target",
                "shortlist_rank_from_stage1",
                "mode_idx",
                "freq_hz",
                "base_mmi_u3",
                "u2_metric",
                "u2_norm_contamination",
                "u2_peak_contamination",
                "participation_metric",
                "participation_gamma_ratio",
                "participation_meff_ratio",
                "girder_u3_metric",
                "balcony_metric",
                "balcony_contamination",
                "girder_to_balcony_ratio",
                "stage2_basis",
                "stage2_score",
                "stage2_rank",
                "final_selected_flag",
                "final_total",
            ],
        )
        writer.writeheader()
        for target in ("T1", "T2"):
            for row in torsion_stage2.get(target, {}).get("shortlist_rows", []):
                writer.writerow({k: row.get(k, np.nan) for k in writer.fieldnames})

    score_matrix = np.asarray(match_res.diagnostics.get("score_matrix", []), dtype=float)
    base_score_matrix = np.asarray(match_res.diagnostics.get("base_score_matrix", []), dtype=float)
    candidate_mode_indices = np.asarray(
        match_res.diagnostics.get("candidate_mode_indices_used", np.arange(score_matrix.shape[0], dtype=int)),
        dtype=int,
    ).reshape(-1)
    with paths["score_matrix"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_mode_idx", *MODE_NAMES])
        for i in range(score_matrix.shape[0]):
            mode_idx = int(candidate_mode_indices[i]) if i < candidate_mode_indices.size else int(i)
            writer.writerow([mode_idx, *score_matrix[i, :].tolist()])

    with paths["base_score_matrix"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_mode_idx", *MODE_NAMES])
        for i in range(base_score_matrix.shape[0]):
            mode_idx = int(candidate_mode_indices[i]) if i < candidate_mode_indices.size else int(i)
            writer.writerow([mode_idx, *base_score_matrix[i, :].tolist()])

    return paths


def _diag_fmt(v: Any, fmt: str) -> str:
    try:
        val = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(val):
        return "nan"
    return format(val, fmt)


def _plot_matcher_score_breakdown_table(sample_id: int, match_res: Any) -> None:
    fig, ax = plt.subplots(figsize=(21, 3.4))
    ax.axis("off")
    rows: list[list[str]] = []
    for target in MODE_NAMES:
        row = match_res.score_breakdown.get(target, {})
        rows.append(
            [
                target,
                _diag_fmt(row.get("mode_idx", np.nan), ".0f"),
                _diag_fmt(row.get("freq_hz", np.nan), ".4f"),
                _diag_fmt(row.get("mac_u3", row.get("mac", np.nan)), ".4f"),
                _diag_fmt(row.get("base_freq_penalty", row.get("freq_penalty", np.nan)), ".4f"),
                _diag_fmt(row.get("base_mmi_u3", np.nan), ".4f"),
                _diag_fmt(row.get("mac_u2", row.get("u2_mac", np.nan)), ".4f"),
                _diag_fmt(row.get("u2_term", np.nan), ".4f"),
                _diag_fmt(row.get("balcony_metric", row.get("balcony_ratio", np.nan)), ".4f"),
                _diag_fmt(row.get("balcony_penalty", np.nan), ".4f"),
                _diag_fmt(row.get("participation_metric", row.get("participation_score", np.nan)), ".4f"),
                _diag_fmt(row.get("participation_term", np.nan), ".4f"),
                str(row.get("selection_stage", "")),
                _diag_fmt(row.get("stage1_rank", np.nan), ".0f"),
                _diag_fmt(row.get("stage2_score", np.nan), ".4f"),
                _diag_fmt(row.get("stage2_rank", np.nan), ".0f"),
                _diag_fmt(row.get("final_total", row.get("total", np.nan)), ".4f"),
                _diag_fmt(row.get("template_index", np.nan), ".0f"),
                _diag_fmt(row.get("template_freq_hz", np.nan), ".4f"),
            ]
        )
    table = ax.table(
        cellText=rows,
        colLabels=[
            "target",
            "mode_idx",
            "freq_hz",
            "mac_u3",
            "base_freq_pen",
            "base_mmi_u3",
            "mac_u2",
            "u2_term",
            "balcony_metric",
            "balcony_penalty",
            "participation_metric",
            "participation_term",
            "selection_stage",
            "stage1_rank",
            "stage2_score",
            "stage2_rank",
            "final_total",
            "template_idx",
            "template_freq_hz",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)
    fig.suptitle(f"Matcher Final Selection Summary - Sample {int(sample_id)}")


def _plot_matcher_top_candidates_tables(sample_id: int, match_res: Any, topk: int = 5) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(28, 10), constrained_layout=True)
    for ax, target in zip(axs.ravel(), MODE_NAMES):
        ax.axis("off")
        rows = []
        for rank, row in enumerate(match_res.top_candidates_per_target.get(target, [])[:topk], start=1):
            rows.append(
                [
                    str(rank),
                    _diag_fmt(row.get("stage1_rank", rank), ".0f"),
                    _diag_fmt(row.get("mode_idx", np.nan), ".0f"),
                    _diag_fmt(row.get("freq_hz", np.nan), ".4f"),
                    _diag_fmt(row.get("mac_u3", row.get("mac", row.get("primary_mac", np.nan))), ".4f"),
                    _diag_fmt(row.get("base_freq_penalty", row.get("freq_penalty", np.nan)), ".4f"),
                    _diag_fmt(row.get("base_mmi_u3", np.nan), ".4f"),
                    str(bool(row.get("shortlisted_flag", False))),
                    _diag_fmt(row.get("stage2_score", np.nan), ".4f"),
                    _diag_fmt(row.get("stage2_rank", np.nan), ".0f"),
                    _diag_fmt(row.get("u2_norm_contamination", np.nan), ".4f"),
                    _diag_fmt(row.get("u2_peak_contamination", np.nan), ".4f"),
                    _diag_fmt(row.get("participation_gamma_ratio", np.nan), ".4f"),
                    _diag_fmt(row.get("participation_meff_ratio", np.nan), ".4f"),
                    _diag_fmt(row.get("girder_to_balcony_ratio", np.nan), ".4f"),
                    _diag_fmt(row.get("final_total", row.get("total", np.nan)), ".4f"),
                    _diag_fmt(row.get("template_index", np.nan), ".0f"),
                    _diag_fmt(row.get("template_freq_hz", np.nan), ".4f"),
                ]
            )
        table = ax.table(
            cellText=rows,
            colLabels=[
                "rank",
                "stage1_rank",
                "mode_idx",
                "freq_hz",
                "mac_u3",
                "base_freq_pen",
                "base_mmi_u3",
                "shortlisted",
                "stage2_score",
                "stage2_rank",
                "u2_norm_contam",
                "u2_peak_contam",
                "part_gamma",
                "part_meff",
                "girder/bal",
                "final_total",
                "template_idx",
                "template_freq_hz",
            ],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.0)
        table.scale(1.0, 1.25)
        ax.set_title(f"{target} Stage-1 Base Ranking")
    fig.suptitle(f"Matcher Stage-1 Base Rankings - Sample {int(sample_id)}")


def _plot_matcher_torsion_stage2_tables(sample_id: int, match_res: Any) -> None:
    torsion_stage2 = match_res.diagnostics.get("torsion_stage2", {})
    fig, axs = plt.subplots(1, 2, figsize=(24, 6), constrained_layout=True)
    for ax, target in zip(axs.ravel(), ("T1", "T2")):
        ax.axis("off")
        rows = []
        for row in torsion_stage2.get(target, {}).get("shortlist_rows", []):
            rows.append(
                [
                    _diag_fmt(row.get("shortlist_rank_from_stage1", np.nan), ".0f"),
                    _diag_fmt(row.get("stage2_rank", np.nan), ".0f"),
                    _diag_fmt(row.get("mode_idx", np.nan), ".0f"),
                    _diag_fmt(row.get("freq_hz", np.nan), ".4f"),
                    _diag_fmt(row.get("base_mmi_u3", np.nan), ".4f"),
                    _diag_fmt(row.get("u2_norm_contamination", np.nan), ".4f"),
                    _diag_fmt(row.get("u2_peak_contamination", np.nan), ".4f"),
                    _diag_fmt(row.get("participation_gamma_ratio", np.nan), ".4f"),
                    _diag_fmt(row.get("participation_meff_ratio", np.nan), ".4f"),
                    _diag_fmt(row.get("girder_u3_metric", np.nan), ".4f"),
                    _diag_fmt(row.get("balcony_metric", np.nan), ".4f"),
                    _diag_fmt(row.get("girder_to_balcony_ratio", np.nan), ".4f"),
                    _diag_fmt(row.get("stage2_score", np.nan), ".4f"),
                    str(bool(row.get("final_selected_flag", False))),
                ]
            )
        table = ax.table(
            cellText=rows,
            colLabels=[
                "stage1_rank",
                "stage2_rank",
                "mode_idx",
                "freq_hz",
                "base_mmi_u3",
                "u2_norm_contam",
                "u2_peak_contam",
                "part_gamma",
                "part_meff",
                "girder_u3",
                "balcony_metric",
                "girder/bal",
                "stage2_score",
                "selected",
            ],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.scale(1.0, 1.35)
        ax.set_title(f"{target} Stage-2 Torsion Rerank")
    fig.suptitle(f"Matcher Stage-2 Torsion Discriminators - Sample {int(sample_id)}")


def _plot_matcher_score_matrix(
    sample_id: int,
    score_matrix: np.ndarray,
    candidate_mode_indices: np.ndarray,
    selected_idx: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
    shortlist_by_target: dict[str, list[int]] | None = None,
) -> None:
    if score_matrix.ndim != 2 or score_matrix.shape[1] != len(MODE_NAMES):
        raise ValueError(f"Missing or invalid score_matrix in matcher diagnostics: shape={score_matrix.shape}")
    candidate_mode_indices = np.asarray(
        candidate_mode_indices if candidate_mode_indices.size else np.arange(score_matrix.shape[0], dtype=int),
        dtype=int,
    ).reshape(-1)

    fig, ax = plt.subplots(figsize=(7.5, 10))
    masked = np.ma.masked_invalid(score_matrix)
    im = ax.imshow(masked, aspect="auto", cmap="viridis", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_title(f"{title} - Sample {int(sample_id)}")
    ax.set_xlabel("Target")
    ax.set_ylabel("Candidate mode index")
    ax.set_xticks(np.arange(len(MODE_NAMES)))
    ax.set_xticklabels(MODE_NAMES)

    y_ticks = np.arange(score_matrix.shape[0], dtype=int)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(
        [
            str(int(candidate_mode_indices[v])) if v < candidate_mode_indices.size else str(int(v))
            for v in y_ticks
        ]
    )

    selected_idx = np.asarray(selected_idx, dtype=int).reshape(-1)
    row_by_mode_idx = {int(mode_idx): i for i, mode_idx in enumerate(candidate_mode_indices.tolist())}
    for j, mode_idx in enumerate(selected_idx):
        i_sel = row_by_mode_idx.get(int(mode_idx))
        if i_sel is None:
            continue
        ax.scatter(j, i_sel, s=180, facecolors="none", edgecolors="red", linewidths=2.0)
        ax.text(j, i_sel, "x", color="white", ha="center", va="center", fontsize=9, fontweight="bold")

    for j in range(len(MODE_NAMES) + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.3, alpha=0.4)
    for i in range(score_matrix.shape[0] + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.2, alpha=0.2)

    if shortlist_by_target:
        for j, target in enumerate(MODE_NAMES):
            for mode_idx in shortlist_by_target.get(target, []):
                i_short = row_by_mode_idx.get(int(mode_idx))
                if i_short is None:
                    continue
                ax.scatter(j, i_short, s=90, facecolors="none", edgecolors="gold", linewidths=1.5, marker="s")


def _plot_matcher_score_heatmaps(sample_id: int, match_res: Any) -> None:
    score_matrix = np.asarray(match_res.diagnostics.get("score_matrix", []), dtype=float)
    base_score_matrix = np.asarray(match_res.diagnostics.get("base_score_matrix", []), dtype=float)
    candidate_mode_indices = np.asarray(
        match_res.diagnostics.get("candidate_mode_indices_used", np.arange(score_matrix.shape[0], dtype=int)),
        dtype=int,
    ).reshape(-1)
    selected_idx = np.asarray(match_res.selected_idx, dtype=int).reshape(-1)
    torsion_stage2 = match_res.diagnostics.get("torsion_stage2", {})
    shortlist_by_target = {
        target: list(info.get("shortlist_mode_indices", [])) for target, info in torsion_stage2.items()
    }
    _plot_matcher_score_matrix(
        sample_id,
        score_matrix,
        candidate_mode_indices,
        selected_idx,
        title="Matcher Final Score Matrix",
        colorbar_label="Final total score",
    )
    if base_score_matrix.shape == score_matrix.shape:
        _plot_matcher_score_matrix(
            sample_id,
            base_score_matrix,
            candidate_mode_indices,
            selected_idx,
            title="Matcher Base Score Matrix",
            colorbar_label="Base MMI U3",
            shortlist_by_target=shortlist_by_target,
        )


def inspect_sample(sample_id: int) -> None:
    sample_path = find_sample_npz(sample_id)
    print(f"\nInspecting sample_id={sample_id}")
    print(f"File: {sample_path}")

    sample_npz = np.load(sample_path, allow_pickle=False)
    sampled = _parse_params_json(sample_npz)
    p = build_solver_params(sampled, n_candidate_modes=N_CANDIDATE_MODES)
    print("Parameters:")
    print(f"  Ng={int(sampled['Ng'])}")
    print(f"  L_mm={float(sampled['L_mm'])}")
    print(f"  Nx={int(sampled['Nx'])}")
    print(f"  dy_mm={float(sampled['dy_mm'])}")
    print(f"  diag_pattern={str(sampled['diag_pattern'])}")
    print(f"  kx_kN_per_m={float(sampled['spring']['kx_kN_per_m'])}")
    print(f"  kz_kN_per_m={float(sampled['spring']['kz_kN_per_m'])}")
    print(f"  bottom_x_frac={float(sampled['bottom_beam']['x_frac'])}")
    print(f"  bottom_overhang_mm={float(sampled['bottom_beam']['overhang_mm'])}")
    print(f"  balcony_enable={bool(sampled['balcony']['enable'])}")
    print(f"  balcony_side={str(sampled['balcony']['side'])}")
    print(f"  balcony_nBeams={int(sampled['balcony']['nBeams'])}")
    print(f"  balcony_dx_mm={float(sampled['balcony']['dx_mm'])}")
    print(f"  balcony_length_mm={float(sampled['balcony']['length_mm'])}")
    print(f"  nsm_mass_tonne={float(sampled['nsm']['total_mass_tonne'])}")

    run_grid_modal_with_params_and_plot(p, do_plot=True, run_modal=True)

    solve_res = run_solver(p, ref_x_fracs=REF_X_FRACS, tol_x=REF_SENSOR_TOL_X)
    ref_lib = load_reference_library_for_ng_variant(REF_DIR, int(sampled["Ng"]), sampled)
    match_res = match_target_modes(
        phi_u3_candidates=solve_res.phi_u3_rep,
        freq_candidates_hz=solve_res.freq_all_hz,
        ref_lib=ref_lib,
        config=MODE_MATCH_CONFIG,
        candidate_channels=_candidate_channels_from_solve(solve_res, MODE_MATCH_CONFIG.signature_config),
        candidate_participation=solve_res.participation,
        candidate_metadata=_candidate_metadata_from_solve(solve_res),
    )

    print("\nMatcher summary:")
    for name, idx in zip(MODE_NAMES, np.asarray(match_res.selected_idx, dtype=int).tolist()):
        freq = float(solve_res.freq_all_hz[int(idx)])
        print(f"  {name}: mode_idx={idx}, freq_hz={freq:.6f}")
    _print_match_diagnostics(match_res, topk=5)

    csv_paths = _write_inspect_matcher_csvs(sample_id, match_res)
    print("Inspection matcher CSVs:")
    print(f"  {csv_paths['score_breakdown']}")
    print(f"  {csv_paths['top_candidates']}")
    print(f"  {csv_paths['torsion_stage2']}")
    print(f"  {csv_paths['score_matrix']}")
    print(f"  {csv_paths['base_score_matrix']}")

    _plot_matcher_score_breakdown_table(sample_id, match_res)
    _plot_matcher_top_candidates_tables(sample_id, match_res, topk=5)
    _plot_matcher_torsion_stage2_tables(sample_id, match_res)
    _plot_matcher_score_heatmaps(sample_id, match_res)
    print("Opened Stage-1, Stage-2, final summary, and score-matrix matcher diagnostic matplotlib windows.")
    plt.show()


def main(bandpass: bool = False) -> None:
    out_dir = Path(OUT_DIR)

    samples_subdir = "samples_phase2_bandpass" if bandpass else PHASE2A_SAMPLES_SUBDIR
    if bandpass:
        # Only clean the bandpass output subdirectory; leave samples_phase2/ untouched.
        bp_dir = out_dir / samples_subdir
        if bp_dir.exists():
            import shutil as _shutil
            for p in bp_dir.glob("*"):
                p.unlink() if (p.is_file() or p.is_symlink()) else _shutil.rmtree(p)
    else:
        print(f"Cleaning output directory: {OUT_DIR}")
        clean_output_directory(Path(OUT_DIR))
    out_samples_dir = out_dir / "samples"
    out_phase1_dir = out_dir / samples_subdir
    out_plots_dir = out_dir / "plots"
    out_samples_dir.mkdir(parents=True, exist_ok=True)
    out_phase1_dir.mkdir(parents=True, exist_ok=True)
    out_plots_dir.mkdir(parents=True, exist_ok=True)

    failure_log_path = out_dir / "failures.log"
    summary_csv_path = out_dir / "summary.csv"
    filtered_summary_csv_path = out_dir / "filtered_summary.csv"
    outlier_report_json_path = out_dir / "outlier_report.json"

    seed_seq = np.random.SeedSequence(SEED)
    child_ss = seed_seq.spawn(len(NG_CHOICES))
    rng_by_ng = {ng: np.random.default_rng(ss) for ng, ss in zip(NG_CHOICES, child_ss)}

    ref_by_ng: dict = {}
    for ng in NG_CHOICES:
        if ng == 2:
            ref_by_ng[(2, True)]  = load_reference_library_for_ng_variant(REF_DIR, 2, {"balcony": {"enable": True}})
            ref_by_ng[(2, False)] = load_reference_library_for_ng_variant(REF_DIR, 2, {"balcony": {"enable": False}})
        else:
            ref_by_ng[ng] = load_reference_library_for_ng(REF_DIR, ng)

    targets = compute_ng_targets(N_STRUCTURES_TOTAL, NG_CHOICES)
    accepted = {ng: 0 for ng in NG_CHOICES}
    rejected = {ng: 0 for ng in NG_CHOICES}
    attempts = {ng: 0 for ng in NG_CHOICES}

    summary_rows: list[dict[str, Any]] = []
    sample_id = 0

    with failure_log_path.open("w", encoding="utf-8") as fail_log:
        fail_log.write(f"SEED={SEED}\n")
        fail_log.write(f"targets={targets}\n")

        for ng in NG_CHOICES:
            rng = rng_by_ng[ng]
            target_n = targets[ng]
            print(f"[Ng={ng}] target={target_n}")

            while accepted[ng] < target_n:
                attempts[ng] += 1
                if attempts[ng] > MAX_ATTEMPTS_PER_NG:
                    raise RuntimeError(
                        f"Exceeded MAX_ATTEMPTS_PER_NG={MAX_ATTEMPTS_PER_NG} for Ng={ng}. "
                        f"accepted={accepted[ng]}, rejected={rejected[ng]}"
                    )

                try:
                    sampled = sample_params_for_ng(ng, rng)
                    if not validate_sampled_params(sampled):
                        rejected[ng] += 1
                        continue

                    p = build_solver_params(sampled, n_candidate_modes=N_CANDIDATE_MODES)
                    solve_res = run_solver(p, ref_x_fracs=REF_X_FRACS, tol_x=REF_SENSOR_TOL_X)

                    ref_lib = ref_by_ng[_ref_key(ng, sampled)]
                    phi_ref_targets = np.asarray(ref_lib.primary_u3_targets, dtype=float)
                    if solve_res.phi_u3_rep.shape[0] != phi_ref_targets.shape[0]:
                        raise ValueError(
                            f"Reduced vector dimension mismatch for Ng={ng}: "
                            f"new={solve_res.phi_u3_rep.shape[0]}, ref={phi_ref_targets.shape[0]}"
                        )

                    match_res = match_target_modes(
                        phi_u3_candidates=solve_res.phi_u3_rep,
                        freq_candidates_hz=solve_res.freq_all_hz,
                        ref_lib=ref_lib,
                        config=MODE_MATCH_CONFIG,
                        candidate_channels=_candidate_channels_from_solve(solve_res, MODE_MATCH_CONFIG.signature_config),
                        candidate_participation=solve_res.participation,
                        candidate_metadata=_candidate_metadata_from_solve(solve_res),
                    )
                    selected_idx = np.asarray(match_res.selected_idx, dtype=int)
                    mac4 = np.asarray(match_res.selected_mac, dtype=float)

                    freqs_4 = solve_res.freq_all_hz[selected_idx]
                    phi_u3_4 = solve_res.phi_u3_rep[:, selected_idx]
                    S4 = compute_S(phi_u3_4)
                    matcher_config_json = json.dumps(asdict(MODE_MATCH_CONFIG), sort_keys=True)
                    matcher_result_json = match_res.to_json()

                    save_sample_npz(
                        out_samples_dir=out_samples_dir,
                        sample_id=sample_id,
                        sampled=sampled,
                        freqs_4=freqs_4,
                        phi_u3_4=phi_u3_4,
                        S4=S4,
                        selected_idx=selected_idx,
                        mac4=mac4,
                        composite_scores=np.asarray(match_res.selected_composite, dtype=float),
                        ambiguity_flags=np.asarray(match_res.ambiguity_flags, dtype=bool),
                        match_result_json=matcher_result_json,
                        matcher_config_json=matcher_config_json,
                    )

                    phase1_payload = assemble_phase1_sample_payload(
                        sample_id=sample_id,
                        sampled=sampled,
                        solve_res=solve_res,
                        match_res=match_res,
                        matcher_config_json=matcher_config_json,
                        reference_file=str(ref_lib.source_file),
                        freqs_4=freqs_4,
                        phi_u3_4=phi_u3_4,
                        S4=S4,
                        bandpass=bandpass,
                    )
                    save_phase1_sample_npz(
                        out_phase1_dir=out_phase1_dir,
                        sample_id=sample_id,
                        sampled=sampled,
                        payload=phase1_payload,
                    )

                    row = build_summary_row(
                        sample_id=sample_id,
                        sampled=sampled,
                        freqs_4=freqs_4,
                        S4=S4,
                        mac4=mac4,
                        selected_idx=selected_idx,
                    )
                    summary_rows.append(row)

                    accepted[ng] += 1
                    sample_id += 1

                    if accepted[ng] % 5 == 0 or accepted[ng] == target_n:
                        print(
                            f"[Ng={ng}] accepted={accepted[ng]}/{target_n}, "
                            f"rejected={rejected[ng]}, attempts={attempts[ng]}"
                        )

                except Exception as exc:
                    rejected[ng] += 1
                    fail_log.write(
                        f"Ng={ng} attempt={attempts[ng]} error={type(exc).__name__}: {exc}\n"
                    )
                    fail_log.write(f"sampled={to_json(sampled) if 'sampled' in locals() else '<none>'}\n")
                    fail_log.write("---\n")
                    continue

    write_summary_csv(summary_rows, summary_csv_path)

    outlier_report = detect_frequency_outliers(summary_rows, OUTLIER_CONFIG)
    filtered_rows = outlier_report["filtered_rows"]
    write_summary_csv(filtered_rows, filtered_summary_csv_path)
    save_outlier_report_json(outlier_report, OUTLIER_CONFIG, outlier_report_json_path)

    if OUTLIER_CONFIG.verbose:
        print("\nOutlier detection report")
        print(f"Method: {OUTLIER_CONFIG.method}")
        print(f"Total samples: {len(summary_rows)}")
        print(f"Outlier samples removed: {outlier_report['n_outlier_samples']}")
        print(f"Filtered samples kept: {len(filtered_rows)}")
        print(f"Outlier counts by mode: {outlier_report['counts_by_mode']}")
        print(f"Outlier sample IDs: {outlier_report['outlier_sample_ids']}")
        for d in outlier_report["outlier_details"]:
            print(
                f"sample_id={int(d['sample_id'])}, mode={d['mode_name']}, "
                f"freq={float(d['value']):.6f} Hz, "
                f"bounds=[{float(d['lower_bound']):.6f}, {float(d['upper_bound']):.6f}]"
            )

    plot_probability_histograms_before_after(summary_rows, filtered_rows)

    print("\nGeneration complete.")
    print(f"Total accepted: {len(summary_rows)} / {N_STRUCTURES_TOTAL}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Filtered CSV: {filtered_summary_csv_path}")
    print(f"Outlier report JSON: {outlier_report_json_path}")
    print(f"Samples dir:  {out_samples_dir}")
    print(f"Phase-2A samples dir: {out_phase1_dir}")
    for ng in NG_CHOICES:
        print(
            f"[Ng={ng}] accepted={accepted[ng]} target={targets[ng]} "
            f"rejected={rejected[ng]} attempts={attempts[ng]}"
        )

    # Only prompt when someone is actually there to answer.  `except EOFError`
    # is not enough: it catches a *closed* stdin, but under a detached or
    # backgrounded run stdin is an open pipe that never delivers a line, so
    # input() blocks for ever and the generator never exits - after writing all
    # its samples.  That hung a 16-hour background run of this script.
    if not sys.stdin.isatty():
        print("Enter sample_id to inspect: (skipped - not an interactive terminal)")
        user_in = ""
    else:
        try:
            user_in = input("Enter sample_id to inspect (or press Enter to skip): ").strip()
        except EOFError:
            user_in = ""
    if user_in:
        try:
            inspect_sample(int(user_in))
        except Exception as exc:
            print(f"Inspection failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic population dataset and inspect samples.")
    parser.add_argument(
        "--inspect",
        type=int,
        default=None,
        help="If provided, skip generation and inspect the specified sample_id.",
    )
    parser.add_argument(
        "--bandpass",
        action="store_true",
        help=(
            "Simulate with all FE modes and apply a 1.5–20 Hz Butterworth bandpass to "
            "a3_abs before computing PSD/CPSD. Produces samples_phase2_bandpass/ instead "
            "of samples_phase2/, leaving the existing dataset untouched."
        ),
    )
    args = parser.parse_args()

    if args.inspect is not None:
        inspect_sample(int(args.inspect))
    else:
        main(bandpass=args.bandpass)
