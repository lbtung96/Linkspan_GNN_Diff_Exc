from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


TARGETS = ("B1", "T1", "B2", "T2")
DEFAULT_GAMMA = 0.5


@dataclass
class SignatureConfig:
    use_channel_u3: bool = True
    use_channel_u2: bool = True
    use_channel_rz: bool = False
    use_feature_raw: bool = False
    use_feature_mean: bool = True
    use_feature_left_right_diff: bool = True
    normalize_signature: bool = True


@dataclass
class ModeMatchConfig:
    use_mac: bool = True
    use_subspace_library: bool = True
    use_torsion_bending_ratio: bool = True
    use_frequency_penalty: bool = True
    use_global_assignment: bool = True
    use_ambiguity_flagging: bool = True
    use_more_stations_if_available: bool = True

    use_participation_factors: bool = True
    use_effective_modal_mass: bool = True

    min_mac_threshold: float = 0.70
    ambiguity_gap_threshold: float = 0.02
    torsion_ratio_soft_low: float = 0.45
    torsion_ratio_soft_high: float = 1.15

    freq_penalty_weight: float = 0.15
    mac_weight: float = 0.45
    shape_type_weight: float = 0.12
    subspace_weight: float = 0.16
    ambiguity_penalty_weight: float = 0.07
    participation_weight: float = 0.10
    effective_mass_weight: float = 0.10

    top_k_per_target: int = 8
    library_topk_avg: int = 3
    mmi_gamma: float = DEFAULT_GAMMA
    use_u2_for_torsion: bool = False
    use_balcony_penalty: bool = False
    use_participation_features: bool = False
    use_two_stage_torsion_rerank: bool = False
    torsion_shortlist_k: int = 3
    u2_torsion_weight: float = 0.35
    balcony_penalty_weight: float = 0.20
    participation_feature_weight: float = 0.15

    signature_config: SignatureConfig = field(default_factory=SignatureConfig)


@dataclass
class ReferenceModeLibrary:
    target_names: list[str]
    x_fracs: np.ndarray
    ng: int
    sensor_xy: np.ndarray
    u3_library_by_target: dict[str, np.ndarray]
    u2_library_by_target: dict[str, np.ndarray] | None
    freq_library_by_target: dict[str, np.ndarray]
    primary_u3_targets: np.ndarray
    primary_freq_targets: np.ndarray
    primary_u2_targets: np.ndarray | None = None
    sig_library_by_target: dict[str, np.ndarray] | None = None
    signature_meta: dict[str, Any] | None = None
    source_file: str | None = None


@dataclass
class ModeMatchResult:
    selected_idx: np.ndarray
    selected_scores: np.ndarray
    selected_mac: np.ndarray
    selected_composite: np.ndarray
    ambiguity_flags: np.ndarray
    signature_type_used: str
    top_candidates_per_target: dict[str, list[dict[str, float]]]
    score_breakdown: dict[str, dict[str, float]]
    warning_messages: list[str]
    diagnostics: dict[str, Any]
    richer_signature_available: bool
    candidate_mean_u3: np.ndarray | None
    candidate_diff_u3: np.ndarray | None
    ref_mean_u3_targets: np.ndarray | None
    ref_diff_u3_targets: np.ndarray | None

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "selected_idx": self.selected_idx.astype(int).tolist(),
            "selected_scores": self.selected_scores.astype(float).tolist(),
            "selected_mac": self.selected_mac.astype(float).tolist(),
            "selected_composite": self.selected_composite.astype(float).tolist(),
            "ambiguity_flags": self.ambiguity_flags.astype(bool).tolist(),
            "signature_type_used": str(self.signature_type_used),
            "top_candidates_per_target": self.top_candidates_per_target,
            "score_breakdown": self.score_breakdown,
            "warning_messages": self.warning_messages,
            "diagnostics": self.diagnostics,
            "richer_signature_available": bool(self.richer_signature_available),
        }
        return json.dumps(payload, sort_keys=True)


def signature_meta_dict(sig_cfg: SignatureConfig, x_fracs: np.ndarray | list[float]) -> dict[str, Any]:
    return {
        "channels": {
            "u3": bool(sig_cfg.use_channel_u3),
            "u2": bool(sig_cfg.use_channel_u2),
            "rz": bool(sig_cfg.use_channel_rz),
        },
        "features": {
            "raw": bool(sig_cfg.use_feature_raw),
            "mean": bool(sig_cfg.use_feature_mean),
            "left_right_diff": bool(sig_cfg.use_feature_left_right_diff),
        },
        "normalize_signature": bool(sig_cfg.normalize_signature),
        "x_fracs": np.asarray(x_fracs, dtype=float).reshape(-1).tolist(),
        "sign_alignment": "dot_with_anchor_positive",
    }


def _norm_cols(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = np.asarray(X, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={A.shape}")
    den = np.linalg.norm(A, axis=0)
    out = np.zeros_like(A, dtype=float)
    nz = den > eps
    out[:, nz] = A[:, nz] / den[nz]
    return out


def align_sign_to_anchor(v: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(-1)
    b = np.asarray(anchor, dtype=float).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"Sign alignment size mismatch: {a.size} vs {b.size}")
    if float(a @ b) < 0.0:
        return -a
    return a


def _station_decompose(X: np.ndarray, ng: int, n_stations: int) -> tuple[np.ndarray, np.ndarray, bool]:
    P = np.asarray(X, dtype=float)
    n_sensors, n_modes = P.shape
    ok = (ng >= 1) and (n_stations >= 1) and (n_sensors == ng * n_stations)
    if not ok:
        return np.zeros((0, n_modes), dtype=float), np.zeros((0, n_modes), dtype=float), False
    block = P.reshape(ng, n_stations, n_modes)
    mean = np.mean(block, axis=0)
    diff = block[-1, :, :] - block[0, :, :] if ng >= 2 else np.zeros((0, n_modes), dtype=float)
    return mean, diff, True


def build_mode_signature_matrix(
    channels: dict[str, np.ndarray],
    ng: int,
    n_stations: int,
    sig_cfg: SignatureConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str], bool]:
    mats: list[np.ndarray] = []
    feats: dict[str, np.ndarray] = {}
    warnings: list[str] = []

    channel_specs = [
        ("u3", bool(sig_cfg.use_channel_u3)),
        ("u2", bool(sig_cfg.use_channel_u2)),
        ("rz", bool(sig_cfg.use_channel_rz)),
    ]

    n_modes = None
    for cname, enabled in channel_specs:
        if not enabled:
            continue
        X = channels.get(cname)
        if X is None:
            warnings.append(f"Requested channel '{cname}' is unavailable; skipping.")
            continue
        A = np.asarray(X, dtype=float)
        if A.ndim != 2:
            warnings.append(f"Channel '{cname}' has invalid shape {A.shape}; skipping.")
            continue
        if n_modes is None:
            n_modes = int(A.shape[1])
        if int(A.shape[1]) != int(n_modes):
            warnings.append(f"Channel '{cname}' mode count mismatch; skipping.")
            continue

        if sig_cfg.use_feature_raw:
            mats.append(A)
            feats[f"{cname}_raw"] = A

        mean, diff, ok = _station_decompose(A, ng=ng, n_stations=n_stations)
        if not ok:
            warnings.append(
                f"Channel '{cname}' cannot be reshaped into (Ng={ng}, stations={n_stations}); using raw only if enabled."
            )
            continue

        if sig_cfg.use_feature_mean:
            mats.append(mean)
            feats[f"{cname}_mean"] = mean
        if sig_cfg.use_feature_left_right_diff and diff.size:
            mats.append(diff)
            feats[f"{cname}_diff"] = diff

    if n_modes is None:
        raise ValueError("No valid channel matrix available for signature building.")
    if not mats:
        if "u3" in channels and np.asarray(channels["u3"]).ndim == 2:
            u3 = np.asarray(channels["u3"], dtype=float)
            mats = [u3]
            feats["u3_raw_fallback"] = u3
            warnings.append("No enabled signature component available; falling back to U3 raw.")
        else:
            raise ValueError("No signature components available.")

    sig = np.vstack(mats)
    richer_ok = any((k.endswith("_mean") or k.endswith("_diff")) for k in feats.keys())
    if bool(sig_cfg.normalize_signature):
        sig = _norm_cols(sig)
    return sig, feats, warnings, richer_ok


def _mac_vector_to_library(phi: np.ndarray, ref_library: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(phi, dtype=float).reshape(-1)
    R = np.asarray(ref_library, dtype=float)
    if R.ndim != 2:
        raise ValueError(f"Reference library must be 2D, got shape={R.shape}")
    if R.shape[0] != v.size:
        raise ValueError(f"MAC size mismatch: candidate={v.size}, reference={R.shape[0]}")

    vv = float(v @ v)
    rr = np.sum(R * R, axis=0)
    den = vv * rr
    num = np.abs(v @ R) ** 2

    out = np.zeros(R.shape[1], dtype=float)
    valid = den > eps
    out[valid] = num[valid] / den[valid]
    return np.clip(out, 0.0, 1.0)


def _relative_frequency_error(f_ref: np.ndarray, f_cand: float, eps: float = 1e-12) -> np.ndarray:
    ref = np.asarray(f_ref, dtype=float).reshape(-1)
    out = np.zeros(ref.size, dtype=float)
    valid = np.isfinite(ref) & (np.abs(ref) > eps)
    out[valid] = np.abs(float(f_cand) - ref[valid]) / np.abs(ref[valid])
    return out


def _resolve_gamma(config: ModeMatchConfig | None) -> float:
    if config is None:
        return DEFAULT_GAMMA
    val = getattr(config, "mmi_gamma", DEFAULT_GAMMA)
    try:
        gamma = float(val)
    except Exception as exc:
        raise ValueError(f"Invalid mmi_gamma value: {val}") from exc
    return float(np.clip(gamma, 0.0, 1.0))


def _group_node_ids(model: dict[str, Any], group_ids: set[int]) -> np.ndarray:
    elems = np.asarray(model.get("elems", np.zeros((0, 7))), dtype=float)
    if elems.size == 0:
        return np.zeros(0, dtype=int)
    grp = elems[:, 3].astype(int)
    mask = np.isin(grp, np.asarray(sorted(group_ids), dtype=int))
    if not np.any(mask):
        return np.zeros(0, dtype=int)
    node_ids = np.unique(np.r_[elems[mask, 1].astype(int), elems[mask, 2].astype(int)])
    return np.asarray(node_ids, dtype=int)


def _node_energy_from_mode(phi_full: np.ndarray, node_ids: np.ndarray, dof_1based: tuple[int, ...] = (2, 3)) -> float:
    if node_ids.size == 0:
        return 0.0
    phi = np.asarray(phi_full, dtype=float).reshape(-1)
    total = 0.0
    for dof in dof_1based:
        idx = 6 * (np.asarray(node_ids, dtype=int) - 1) + (int(dof) - 1)
        idx = idx[(idx >= 0) & (idx < phi.size)]
        if idx.size:
            total += float(np.sum(phi[idx] ** 2))
    return total


def _compute_balcony_contamination(
    phi_full_candidates: np.ndarray | None,
    mode_indices: np.ndarray,
    candidate_metadata: dict[str, Any] | None,
) -> np.ndarray:
    if phi_full_candidates is None:
        raise ValueError("use_balcony_penalty=True requires full candidate mode shapes.")
    if candidate_metadata is None or "model" not in candidate_metadata:
        raise ValueError("use_balcony_penalty=True requires candidate_metadata['model'].")

    model = candidate_metadata["model"]
    balcony_nodes = _group_node_ids(model, {7, 8, 9})
    if balcony_nodes.size == 0:
        return np.zeros(mode_indices.size, dtype=float)

    all_struct_nodes = _group_node_ids(model, {1, 2, 3, 5, 6, 7, 8, 9})
    deck_nodes = np.setdiff1d(all_struct_nodes, balcony_nodes, assume_unique=False)
    if deck_nodes.size == 0:
        return np.zeros(mode_indices.size, dtype=float)

    phi_full = np.asarray(phi_full_candidates, dtype=float)
    ratios = np.zeros(mode_indices.size, dtype=float)
    for k, i_mode in enumerate(np.asarray(mode_indices, dtype=int)):
        v = phi_full[:, int(i_mode)]
        e_bal = _node_energy_from_mode(v, balcony_nodes, dof_1based=(2, 3))
        e_deck = _node_energy_from_mode(v, deck_nodes, dof_1based=(2, 3))
        den = e_bal + e_deck
        ratios[k] = 0.0 if den <= 1e-12 else float(e_bal / den)
    return ratios


def _node_channel_norm_from_mode(
    phi_full: np.ndarray,
    node_ids: np.ndarray,
    dof_1based: tuple[int, ...],
) -> float:
    if node_ids.size == 0:
        return 0.0
    phi = np.asarray(phi_full, dtype=float).reshape(-1)
    vals: list[np.ndarray] = []
    for dof in dof_1based:
        idx = 6 * (np.asarray(node_ids, dtype=int) - 1) + (int(dof) - 1)
        idx = idx[(idx >= 0) & (idx < phi.size)]
        if idx.size:
            vals.append(phi[idx])
    if not vals:
        return 0.0
    return float(np.linalg.norm(np.concatenate(vals)))


def _compute_girder_balcony_metrics(
    phi_full_candidates: np.ndarray | None,
    mode_indices: np.ndarray,
    candidate_metadata: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if phi_full_candidates is None:
        raise ValueError("Two-stage T1 reranking requires full candidate mode shapes.")
    if candidate_metadata is None or "model" not in candidate_metadata:
        raise ValueError("Two-stage T1 reranking requires candidate_metadata['model'].")

    model = candidate_metadata["model"]
    girder_nodes = _group_node_ids(model, {1})
    balcony_nodes = _group_node_ids(model, {7, 8, 9})
    if girder_nodes.size == 0:
        raise ValueError("Two-stage T1 reranking requires girder nodes (group 1).")

    phi_full = np.asarray(phi_full_candidates, dtype=float)
    girder_u3 = np.zeros(mode_indices.size, dtype=float)
    balcony_combined = np.full(mode_indices.size, np.nan, dtype=float)
    girder_to_balcony_ratio = np.full(mode_indices.size, np.nan, dtype=float)
    for k, i_mode in enumerate(np.asarray(mode_indices, dtype=int)):
        v = phi_full[:, int(i_mode)]
        girder_u3[k] = _node_channel_norm_from_mode(v, girder_nodes, dof_1based=(3,))
        if balcony_nodes.size:
            balcony_combined[k] = _node_channel_norm_from_mode(v, balcony_nodes, dof_1based=(2, 3))
            girder_to_balcony_ratio[k] = float(girder_u3[k] / max(float(balcony_combined[k]), 1e-12))
    return girder_u3, balcony_combined, girder_to_balcony_ratio


def _compute_u2_u3_contamination_metrics(
    phi_u2_candidates: np.ndarray | None,
    phi_u3_candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    phi_u3 = np.asarray(phi_u3_candidates, dtype=float)
    if phi_u2_candidates is None:
        shape = (phi_u3.shape[1],)
        return np.full(shape, np.nan, dtype=float), np.full(shape, np.nan, dtype=float)

    phi_u2 = np.asarray(phi_u2_candidates, dtype=float)
    if phi_u2.shape != phi_u3.shape:
        raise ValueError(f"Candidate U2 shape mismatch: expected {phi_u3.shape}, got {phi_u2.shape}")

    norm_u2 = np.linalg.norm(phi_u2, axis=0)
    norm_u3 = np.linalg.norm(phi_u3, axis=0)
    peak_u2 = np.max(np.abs(phi_u2), axis=0)
    peak_u3 = np.max(np.abs(phi_u3), axis=0)

    norm_ratio = np.full(phi_u3.shape[1], np.nan, dtype=float)
    peak_ratio = np.full(phi_u3.shape[1], np.nan, dtype=float)
    den_norm = norm_u2 + norm_u3
    den_peak = peak_u2 + peak_u3
    valid_norm = den_norm > 1e-12
    valid_peak = den_peak > 1e-12
    norm_ratio[valid_norm] = norm_u2[valid_norm] / den_norm[valid_norm]
    peak_ratio[valid_peak] = peak_u2[valid_peak] / den_peak[valid_peak]
    return norm_ratio, peak_ratio


def _participation_torsion_metrics(
    candidate_participation: dict[str, np.ndarray] | None, mode_index: int
) -> tuple[float, float, float]:
    if candidate_participation is None:
        raise ValueError("use_participation_features=True requires candidate_participation.")

    def _pick(key: str) -> float:
        arr = np.asarray(candidate_participation.get(key, np.zeros(0, dtype=float)), dtype=float).reshape(-1)
        if mode_index < 0 or mode_index >= arr.size:
            return np.nan
        return float(arr[mode_index])

    gamma_uy = abs(_pick("gamma_uy"))
    gamma_uz = abs(_pick("gamma_uz"))
    meff_uy = max(0.0, _pick("meff_uy")) if np.isfinite(_pick("meff_uy")) else np.nan
    meff_uz = max(0.0, _pick("meff_uz")) if np.isfinite(_pick("meff_uz")) else np.nan

    gamma_ratio = np.nan
    meff_ratio = np.nan
    vals: list[float] = []
    if np.isfinite(gamma_uy) and np.isfinite(gamma_uz) and (gamma_uy + gamma_uz) > 1e-12:
        gamma_ratio = gamma_uy / (gamma_uy + gamma_uz)
        vals.append(gamma_ratio)
    if np.isfinite(meff_uy) and np.isfinite(meff_uz) and (meff_uy + meff_uz) > 1e-12:
        meff_ratio = meff_uy / (meff_uy + meff_uz)
        vals.append(meff_ratio)
    if not vals:
        raise ValueError("Participation features requested but gamma/meff arrays are unavailable or invalid.")
    score = float(np.clip(np.mean(vals), 0.0, 1.0))
    return score, float(gamma_ratio), float(meff_ratio)


def _signature_type_label(config: ModeMatchConfig) -> str:
    parts = ["u3_mac_mmi"]
    if bool(config.use_two_stage_torsion_rerank):
        parts.append("two_stage_torsion")
    if bool(config.use_u2_for_torsion):
        parts.append("u2_torsion")
    if bool(config.use_balcony_penalty):
        parts.append("balcony_penalty")
    if bool(config.use_participation_features):
        parts.append("participation")
    return "+".join(parts)


def load_reference_mode_library(npz_path: str | Path, target_names: list[str]) -> ReferenceModeLibrary:
    p = Path(npz_path)
    z = np.load(p, allow_pickle=False)

    ng = int(np.asarray(z["Ng"]).item())
    x_fracs = np.asarray(z["x_fracs"], dtype=float).reshape(-1)
    sensor_xy = np.asarray(z["sensor_xy"], dtype=float)
    phi_u3_ref = np.asarray(z["Phi_u3_ref"], dtype=float)
    phi_u2_ref = np.asarray(z["Phi_u2_ref"], dtype=float) if "Phi_u2_ref" in z.files else None
    freq_hz_ref = np.asarray(z["freq_hz_ref"], dtype=float).reshape(-1)
    ref_mode_idx = np.asarray(z["ref_mode_indices_0based"], dtype=int).reshape(-1)

    if ref_mode_idx.size != len(target_names):
        raise ValueError(f"Expected {len(target_names)} reference mode indices in {p}, got {ref_mode_idx.size}")
    if np.any(ref_mode_idx < 0) or np.any(ref_mode_idx >= phi_u3_ref.shape[1]):
        raise ValueError(f"Invalid ref_mode_indices_0based in {p}: {ref_mode_idx.tolist()}")

    primary_u3_targets = np.asarray(phi_u3_ref[:, ref_mode_idx], dtype=float)
    primary_u2_targets = np.asarray(phi_u2_ref[:, ref_mode_idx], dtype=float) if phi_u2_ref is not None else None
    primary_freq_targets = np.asarray(freq_hz_ref[ref_mode_idx], dtype=float)

    if "Phi_u3_target_library" in z.files:
        lib_u3 = np.asarray(z["Phi_u3_target_library"], dtype=float)
        if lib_u3.ndim != 3 or lib_u3.shape[1] != len(target_names):
            raise ValueError(f"Unexpected Phi_u3_target_library shape in {p}: {lib_u3.shape}")
    else:
        lib_u3 = primary_u3_targets[:, :, None]

    if "freq_hz_target_library" in z.files:
        lib_f = np.asarray(z["freq_hz_target_library"], dtype=float)
        if lib_f.ndim != 2 or lib_f.shape[0] != len(target_names):
            raise ValueError(f"Unexpected freq_hz_target_library shape in {p}: {lib_f.shape}")
    else:
        lib_f = primary_freq_targets[:, None]

    u3_library_by_target: dict[str, np.ndarray] = {}
    u2_library_by_target: dict[str, np.ndarray] | None = None
    freq_library_by_target: dict[str, np.ndarray] = {}
    if "Phi_u2_target_library" in z.files:
        lib_u2 = np.asarray(z["Phi_u2_target_library"], dtype=float)
        if lib_u2.ndim != 3 or lib_u2.shape[1] != len(target_names):
            raise ValueError(f"Unexpected Phi_u2_target_library shape in {p}: {lib_u2.shape}")
        u2_library_by_target = {}
    elif primary_u2_targets is not None:
        lib_u2 = primary_u2_targets[:, :, None]
        u2_library_by_target = {}
    else:
        lib_u2 = None
    for j, name in enumerate(target_names):
        u3_library_by_target[name] = np.asarray(lib_u3[:, j, :], dtype=float)
        if u2_library_by_target is not None and lib_u2 is not None:
            u2_library_by_target[name] = np.asarray(lib_u2[:, j, :], dtype=float)
        freq_library_by_target[name] = np.asarray(lib_f[j, :], dtype=float).reshape(-1)

    signature_meta = None
    if "signature_meta_json" in z.files:
        raw = z["signature_meta_json"]
        s = str(raw.item()) if np.ndim(raw) == 0 else str(raw)
        if s.strip():
            signature_meta = json.loads(s)

    return ReferenceModeLibrary(
        target_names=list(target_names),
        x_fracs=x_fracs,
        ng=ng,
        sensor_xy=sensor_xy,
        u3_library_by_target=u3_library_by_target,
        u2_library_by_target=u2_library_by_target,
        freq_library_by_target=freq_library_by_target,
        primary_u3_targets=primary_u3_targets,
        primary_freq_targets=primary_freq_targets,
        primary_u2_targets=primary_u2_targets,
        sig_library_by_target=None,
        signature_meta=signature_meta,
        source_file=str(p),
    )


def _solve_assignment_max(score_matrix: np.ndarray) -> np.ndarray:
    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f"score_matrix must be 2D, got shape={scores.shape}")

    n_candidates, n_targets = scores.shape
    if n_candidates < n_targets:
        raise ValueError(f"Need at least {n_targets} candidate modes, got {n_candidates}")

    try:
        from scipy.optimize import linear_sum_assignment

        finite = np.isfinite(scores)
        if not np.any(finite):
            raise ValueError("No finite MMI scores available for assignment.")

        floor = float(np.nanmin(scores[finite])) - 1e6
        profit = np.where(finite, scores, floor)
        row_idx, col_idx = linear_sum_assignment(-profit)

        assigned = np.full(n_targets, -1, dtype=int)
        for r, c in zip(row_idx, col_idx):
            if c < n_targets:
                assigned[int(c)] = int(r)

        if np.any(assigned < 0):
            raise ValueError("Global assignment returned an incomplete mapping.")
        return assigned
    except ImportError:
        from itertools import combinations, permutations

        best_sum = -np.inf
        best_perm: tuple[int, ...] | None = None
        for subset in combinations(range(n_candidates), n_targets):
            for perm in permutations(subset, n_targets):
                total = 0.0
                valid = True
                for j, i in enumerate(perm):
                    score = float(scores[i, j])
                    if not np.isfinite(score):
                        valid = False
                        break
                    total += score
                if valid and total > best_sum:
                    best_sum = total
                    best_perm = perm

        if best_perm is None:
            raise ValueError("Fallback global assignment failed: no feasible mapping.")
        return np.asarray(best_perm, dtype=int)


def match_target_modes(
    phi_u3_candidates: np.ndarray,
    freq_candidates_hz: np.ndarray,
    ref_lib: ReferenceModeLibrary,
    config: ModeMatchConfig,
    *,
    candidate_channels: dict[str, np.ndarray] | None = None,
    candidate_participation: dict[str, np.ndarray] | None = None,
    candidate_metadata: dict[str, Any] | None = None,
) -> ModeMatchResult:
    phi_u3 = np.asarray(phi_u3_candidates, dtype=float)
    freq_hz = np.asarray(freq_candidates_hz, dtype=float).reshape(-1)

    if phi_u3.ndim != 2:
        raise ValueError(f"phi_u3_candidates must be 2D, got shape={phi_u3.shape}")
    if phi_u3.shape[1] != freq_hz.size:
        raise ValueError(f"Candidate shape mismatch: modes={phi_u3.shape[1]} vs frequencies={freq_hz.size}")

    target_names = list(ref_lib.target_names)
    n_targets = len(target_names)
    if n_targets != 4:
        raise ValueError(f"Expected 4 targets, got {n_targets}")
    if phi_u3.shape[1] <= 2:
        raise ValueError(f"Need candidate modes beyond rigid-body modes 0 and 1, got {phi_u3.shape[1]}")

    gamma = _resolve_gamma(config)
    excluded_mode_indices = np.asarray([0, 1], dtype=int)
    candidate_mode_indices = np.arange(phi_u3.shape[1], dtype=int)
    valid_mode_mask = candidate_mode_indices >= 2
    candidate_mode_indices = candidate_mode_indices[valid_mode_mask]
    if candidate_mode_indices.size < n_targets:
        raise ValueError(
            f"Need at least {n_targets} non-rigid candidates after excluding modes [0, 1], got {candidate_mode_indices.size}"
        )

    phi_u3_f = phi_u3[:, candidate_mode_indices]
    freq_hz_f = freq_hz[candidate_mode_indices]
    phi_u2_f = None
    if candidate_channels is not None and "u2" in candidate_channels:
        u2 = np.asarray(candidate_channels["u2"], dtype=float)
        if u2.shape != phi_u3.shape:
            raise ValueError(f"Candidate U2 shape mismatch: expected {phi_u3.shape}, got {u2.shape}")
        phi_u2_f = u2[:, candidate_mode_indices]

    phi_full = None
    if candidate_metadata is not None and "phi_full" in candidate_metadata:
        phi_full = np.asarray(candidate_metadata["phi_full"], dtype=float)
        if phi_full.ndim != 2 or phi_full.shape[1] != freq_hz.size:
            raise ValueError(f"candidate_metadata['phi_full'] shape mismatch: {phi_full.shape}")

    balcony_ratio = None
    if bool(config.use_balcony_penalty):
        balcony_ratio = _compute_balcony_contamination(phi_full, candidate_mode_indices, candidate_metadata)
    girder_u3_metric = None
    balcony_combined_metric = None
    girder_to_balcony_ratio = None
    if bool(config.use_two_stage_torsion_rerank) and bool(config.use_balcony_penalty):
        girder_u3_metric, balcony_combined_metric, girder_to_balcony_ratio = _compute_girder_balcony_metrics(
            phi_full,
            candidate_mode_indices,
            candidate_metadata,
        )
    u2_norm_ratio, u2_peak_ratio = _compute_u2_u3_contamination_metrics(phi_u2_f, phi_u3_f)

    n_candidates = phi_u3_f.shape[1]
    use_two_stage_torsion = bool(config.use_two_stage_torsion_rerank)
    shortlist_k = max(1, int(getattr(config, "torsion_shortlist_k", 3)))

    best_mac = np.full((n_candidates, n_targets), -np.inf, dtype=float)
    best_u2_mac = np.full((n_candidates, n_targets), np.nan, dtype=float)
    best_combined_mac = np.full((n_candidates, n_targets), -np.inf, dtype=float)
    best_freq_penalty = np.full((n_candidates, n_targets), np.inf, dtype=float)
    best_base_mmi_u3 = np.full((n_candidates, n_targets), -np.inf, dtype=float)
    best_u2_term = np.zeros((n_candidates, n_targets), dtype=float)
    best_mmi = np.full((n_candidates, n_targets), -np.inf, dtype=float)
    best_balcony_penalty = np.zeros((n_candidates, n_targets), dtype=float)
    best_participation_score = np.full((n_candidates, n_targets), np.nan, dtype=float)
    best_participation_gamma_ratio = np.full((n_candidates, n_targets), np.nan, dtype=float)
    best_participation_meff_ratio = np.full((n_candidates, n_targets), np.nan, dtype=float)
    best_participation_term = np.zeros((n_candidates, n_targets), dtype=float)
    best_template_idx = np.full((n_candidates, n_targets), -1, dtype=int)
    best_template_freq = np.full((n_candidates, n_targets), np.nan, dtype=float)

    for j, name in enumerate(target_names):
        ref_modes = np.asarray(ref_lib.u3_library_by_target[name], dtype=float)
        ref_freqs = np.asarray(ref_lib.freq_library_by_target[name], dtype=float).reshape(-1)

        if ref_modes.ndim != 2 or ref_modes.shape[0] != phi_u3_f.shape[0]:
            raise ValueError(f"Invalid reference U3 library shape for {name}: {ref_modes.shape}")

        ref_u2 = None
        use_u2_here = bool(config.use_u2_for_torsion) and name.startswith("T")
        if use_u2_here:
            if phi_u2_f is None:
                raise ValueError("use_u2_for_torsion=True requires candidate U2 reduced modes.")
            if ref_lib.u2_library_by_target is None or name not in ref_lib.u2_library_by_target:
                raise ValueError(
                    f"use_u2_for_torsion=True requires U2 reference library for target {name}. "
                    f"Re-capture references with Phi_u2_ref/Phi_u2_target_library."
                )
            ref_u2 = np.asarray(ref_lib.u2_library_by_target[name], dtype=float)
            if ref_u2.shape != ref_modes.shape:
                raise ValueError(f"Reference U2 library shape mismatch for {name}: {ref_u2.shape} vs {ref_modes.shape}")

        for i in range(n_candidates):
            mac_vals = _mac_vector_to_library(phi_u3_f[:, i], ref_modes)
            if use_u2_here and ref_u2 is not None and phi_u2_f is not None:
                u2_mac_vals = _mac_vector_to_library(phi_u2_f[:, i], ref_u2)
                combined_mac_vals = (1.0 - float(config.u2_torsion_weight)) * mac_vals + float(config.u2_torsion_weight) * u2_mac_vals
            else:
                u2_mac_vals = np.full(mac_vals.shape, np.nan, dtype=float)
                combined_mac_vals = mac_vals.copy()

            freq_pen = _relative_frequency_error(ref_freqs, freq_hz_f[i])
            base_mmi_vals = (1.0 - gamma) * mac_vals - gamma * freq_pen
            u2_term_vals = (1.0 - gamma) * (combined_mac_vals - mac_vals)
            use_stage2_here = use_two_stage_torsion and name.startswith("T")
            mmi_vals = base_mmi_vals.copy()
            if not use_stage2_here:
                mmi_vals = mmi_vals + u2_term_vals

            participation_score = np.nan
            participation_gamma_ratio = np.nan
            participation_meff_ratio = np.nan
            participation_term = 0.0
            if bool(config.use_participation_features) and name.startswith("T"):
                participation_score, participation_gamma_ratio, participation_meff_ratio = _participation_torsion_metrics(
                    candidate_participation,
                    int(candidate_mode_indices[i]),
                )
                participation_term = float(config.participation_feature_weight) * (participation_score - 0.5)
                if not use_stage2_here:
                    mmi_vals = mmi_vals + participation_term

            balcony_penalty = 0.0
            if bool(config.use_balcony_penalty) and name.startswith("T") and balcony_ratio is not None:
                balcony_penalty = float(config.balcony_penalty_weight) * float(balcony_ratio[i])
                if not use_stage2_here:
                    mmi_vals = mmi_vals - balcony_penalty

            template_scores = base_mmi_vals if use_stage2_here else mmi_vals
            k = int(np.argmax(template_scores))
            best_mac[i, j] = float(mac_vals[k])
            best_u2_mac[i, j] = float(u2_mac_vals[k]) if np.isfinite(u2_mac_vals[k]) else np.nan
            best_combined_mac[i, j] = float(combined_mac_vals[k])
            best_freq_penalty[i, j] = float(freq_pen[k])
            best_base_mmi_u3[i, j] = float(base_mmi_vals[k])
            best_u2_term[i, j] = float(u2_term_vals[k])
            best_mmi[i, j] = float(mmi_vals[k])
            best_balcony_penalty[i, j] = float(balcony_penalty)
            best_participation_score[i, j] = float(participation_score) if np.isfinite(participation_score) else np.nan
            best_participation_gamma_ratio[i, j] = (
                float(participation_gamma_ratio) if np.isfinite(participation_gamma_ratio) else np.nan
            )
            best_participation_meff_ratio[i, j] = (
                float(participation_meff_ratio) if np.isfinite(participation_meff_ratio) else np.nan
            )
            best_participation_term[i, j] = float(participation_term)
            best_template_idx[i, j] = k
            if k < ref_freqs.size:
                best_template_freq[i, j] = float(ref_freqs[k])

    feasible = np.isfinite(best_mmi)
    if config.min_mac_threshold > 0:
        feasible &= best_mac >= float(config.min_mac_threshold)
    base_score_matrix = np.where(feasible, best_base_mmi_u3, -np.inf)
    legacy_score_matrix = np.where(feasible, best_mmi, -np.inf)
    final_score_matrix = legacy_score_matrix.copy()

    torsion_stage2: dict[str, Any] = {}
    if use_two_stage_torsion:
        for torsion_name in ("T1", "T2"):
            j = target_names.index(torsion_name)
            base_col = base_score_matrix[:, j]
            finite_rows = np.flatnonzero(np.isfinite(base_col))
            order = finite_rows[np.argsort(base_col[finite_rows])[::-1]] if finite_rows.size else np.zeros(0, dtype=int)
            shortlist_rows = order[: min(shortlist_k, order.size)]
            final_score_matrix[:, j] = -np.inf

            stage2_rows: list[dict[str, float | bool | str]] = []
            for shortlist_rank, i in enumerate(shortlist_rows, start=1):
                row: dict[str, float | bool | str] = {
                    "target": torsion_name,
                    "shortlist_rank_from_stage1": float(shortlist_rank),
                    "mode_idx": float(candidate_mode_indices[i]),
                    "freq_hz": float(freq_hz_f[i]),
                    "base_mmi_u3": float(best_base_mmi_u3[i, j]),
                    "u2_metric": float(best_u2_mac[i, j]) if np.isfinite(best_u2_mac[i, j]) else np.nan,
                    "u2_norm_contamination": float(u2_norm_ratio[i]) if np.isfinite(u2_norm_ratio[i]) else np.nan,
                    "u2_peak_contamination": float(u2_peak_ratio[i]) if np.isfinite(u2_peak_ratio[i]) else np.nan,
                    "participation_metric": float(best_participation_score[i, j]) if np.isfinite(best_participation_score[i, j]) else np.nan,
                    "participation_gamma_ratio": float(best_participation_gamma_ratio[i, j]) if np.isfinite(best_participation_gamma_ratio[i, j]) else np.nan,
                    "participation_meff_ratio": float(best_participation_meff_ratio[i, j]) if np.isfinite(best_participation_meff_ratio[i, j]) else np.nan,
                    "girder_u3_metric": float(girder_u3_metric[i]) if girder_u3_metric is not None and np.isfinite(girder_u3_metric[i]) else np.nan,
                    "balcony_metric": float(balcony_combined_metric[i]) if balcony_combined_metric is not None and np.isfinite(balcony_combined_metric[i]) else np.nan,
                    "balcony_contamination": float(balcony_ratio[i]) if balcony_ratio is not None and np.isfinite(balcony_ratio[i]) else np.nan,
                    "girder_to_balcony_ratio": float(girder_to_balcony_ratio[i]) if girder_to_balcony_ratio is not None and np.isfinite(girder_to_balcony_ratio[i]) else np.nan,
                }

                components: list[float] = []
                component_labels: list[str] = []
                if torsion_name == "T2":
                    if bool(config.use_u2_for_torsion):
                        if np.isfinite(u2_norm_ratio[i]):
                            components.append(1.0 - float(np.clip(u2_norm_ratio[i], 0.0, 1.0)))
                            component_labels.append("lower_u2_norm_contamination")
                        if np.isfinite(u2_peak_ratio[i]):
                            components.append(1.0 - float(np.clip(u2_peak_ratio[i], 0.0, 1.0)))
                            component_labels.append("lower_u2_peak_contamination")
                    if bool(config.use_participation_features):
                        part_terms = [
                            float(best_participation_gamma_ratio[i, j]) if np.isfinite(best_participation_gamma_ratio[i, j]) else np.nan,
                            float(best_participation_meff_ratio[i, j]) if np.isfinite(best_participation_meff_ratio[i, j]) else np.nan,
                        ]
                        part_terms = [v for v in part_terms if np.isfinite(v)]
                        if part_terms:
                            components.append(1.0 - float(np.clip(np.mean(part_terms), 0.0, 1.0)))
                            component_labels.append("lower_u2_participation_contamination")
                else:
                    if bool(config.use_balcony_penalty):
                        if girder_to_balcony_ratio is not None and np.isfinite(girder_to_balcony_ratio[i]):
                            ratio_score = float(girder_to_balcony_ratio[i] / (1.0 + girder_to_balcony_ratio[i]))
                            components.append(ratio_score)
                            component_labels.append("higher_girder_to_balcony_ratio")
                        if balcony_ratio is not None and np.isfinite(balcony_ratio[i]):
                            components.append(1.0 - float(np.clip(balcony_ratio[i], 0.0, 1.0)))
                            component_labels.append("lower_balcony_contamination")

                if components:
                    stage2_score = float(np.mean(components))
                    stage2_basis = ",".join(component_labels)
                else:
                    stage2_score = 0.0
                    stage2_basis = "fallback_stage1_base_order"
                final_col_score = stage2_score + 1e-3 * float(best_base_mmi_u3[i, j]) - 1e-6 * float(shortlist_rank)
                final_score_matrix[i, j] = final_col_score
                row["stage2_score"] = stage2_score
                row["final_total"] = final_col_score
                row["stage2_basis"] = stage2_basis
                stage2_rows.append(row)

            stage2_rows.sort(
                key=lambda r: (
                    float(r.get("stage2_score", -np.inf)),
                    float(r.get("base_mmi_u3", -np.inf)),
                    -float(r.get("shortlist_rank_from_stage1", np.inf)),
                ),
                reverse=True,
            )
            top_stage1_mode_idx = (
                int(candidate_mode_indices[shortlist_rows[0]]) if shortlist_rows.size else None
            )
            top_stage2_mode_idx = int(stage2_rows[0]["mode_idx"]) if stage2_rows else None
            for rank, row in enumerate(stage2_rows, start=1):
                row["stage2_rank"] = float(rank)
                row["final_selected_flag"] = False
            torsion_stage2[torsion_name] = {
                "shortlist_mode_indices": [int(candidate_mode_indices[i]) for i in shortlist_rows.tolist()],
                "shortlist_rows": stage2_rows,
                "top_stage1_mode_idx": top_stage1_mode_idx,
                "top_stage2_mode_idx": top_stage2_mode_idx,
                "rerank_changed_top_choice": bool(
                    top_stage1_mode_idx is not None and top_stage2_mode_idx is not None and top_stage1_mode_idx != top_stage2_mode_idx
                ),
            }

    score_matrix = final_score_matrix

    for j, name in enumerate(target_names):
        if not np.any(np.isfinite(score_matrix[:, j])):
            raise ValueError(
                f"No feasible candidate survives MMI scoring for target {name} "
                f"with min_mac_threshold={config.min_mac_threshold:.3f}."
            )

    selected_idx = _solve_assignment_max(score_matrix)
    selected_scores = np.asarray([score_matrix[selected_idx[j], j] for j in range(n_targets)], dtype=float)
    selected_mac = np.asarray([best_mac[selected_idx[j], j] for j in range(n_targets)], dtype=float)

    top_candidates_per_target: dict[str, list[dict[str, float]]] = {}
    score_breakdown: dict[str, dict[str, float]] = {}
    score_rows_by_target: dict[str, list[dict[str, float]]] = {}
    warnings: list[str] = []

    for j, name in enumerate(target_names):
        order_source = base_score_matrix[:, j] if (use_two_stage_torsion and name.startswith("T")) else score_matrix[:, j]
        finite_order_rows = np.flatnonzero(np.isfinite(order_source))
        order = finite_order_rows[np.argsort(order_source[finite_order_rows])[::-1]] if finite_order_rows.size else np.zeros(0, dtype=int)
        shortlist_mode_set = set(torsion_stage2.get(name, {}).get("shortlist_mode_indices", []))
        stage2_row_lookup = {
            int(row["mode_idx"]): row for row in torsion_stage2.get(name, {}).get("shortlist_rows", [])
        }
        rows: list[dict[str, float]] = []
        for i in order:
            if not np.isfinite(base_score_matrix[i, j] if (use_two_stage_torsion and name.startswith("T")) else score_matrix[i, j]):
                continue
            row_data = {
                "target": name,
                "mode_idx": float(candidate_mode_indices[i]),
                "freq_hz": float(freq_hz_f[i]),
                "template_index": float(best_template_idx[i, j]),
                "template_freq_hz": float(best_template_freq[i, j]) if np.isfinite(best_template_freq[i, j]) else np.nan,
                "mac": float(best_mac[i, j]),
                "mac_u3": float(best_mac[i, j]),
                "base_freq_penalty": float(best_freq_penalty[i, j]),
                "freq_penalty": float(best_freq_penalty[i, j]),
                "base_mmi_u3": float(best_base_mmi_u3[i, j]),
                "u2_mac": float(best_u2_mac[i, j]) if np.isfinite(best_u2_mac[i, j]) else np.nan,
                "mac_u2": float(best_u2_mac[i, j]) if np.isfinite(best_u2_mac[i, j]) else np.nan,
                "u2_term": float(best_u2_term[i, j]),
                "combined_mac": float(best_combined_mac[i, j]),
                "balcony_metric": float(balcony_ratio[i]) if balcony_ratio is not None else np.nan,
                "balcony_ratio": float(balcony_ratio[i]) if balcony_ratio is not None else np.nan,
                "balcony_penalty": float(best_balcony_penalty[i, j]),
                "participation_metric": float(best_participation_score[i, j]) if np.isfinite(best_participation_score[i, j]) else np.nan,
                "participation_score": float(best_participation_score[i, j]) if np.isfinite(best_participation_score[i, j]) else np.nan,
                "participation_gamma_ratio": float(best_participation_gamma_ratio[i, j]) if np.isfinite(best_participation_gamma_ratio[i, j]) else np.nan,
                "participation_meff_ratio": float(best_participation_meff_ratio[i, j]) if np.isfinite(best_participation_meff_ratio[i, j]) else np.nan,
                "participation_term": float(best_participation_term[i, j]),
                "girder_u3_metric": float(girder_u3_metric[i]) if girder_u3_metric is not None and np.isfinite(girder_u3_metric[i]) else np.nan,
                "balcony_combined_metric": float(balcony_combined_metric[i]) if balcony_combined_metric is not None and np.isfinite(balcony_combined_metric[i]) else np.nan,
                "girder_to_balcony_ratio": float(girder_to_balcony_ratio[i]) if girder_to_balcony_ratio is not None and np.isfinite(girder_to_balcony_ratio[i]) else np.nan,
                "u2_norm_contamination": float(u2_norm_ratio[i]) if np.isfinite(u2_norm_ratio[i]) else np.nan,
                "u2_peak_contamination": float(u2_peak_ratio[i]) if np.isfinite(u2_peak_ratio[i]) else np.nan,
                "mmi": float(score_matrix[i, j]),
                "total": float(score_matrix[i, j]),
                "final_total": float(score_matrix[i, j]),
                "primary_mac": float(best_mac[i, j]),
                "stage1_rank": float(len(rows) + 1),
                "shortlisted_flag": bool(int(candidate_mode_indices[i]) in shortlist_mode_set),
                "selection_stage": "stage2_torsion_rerank" if (use_two_stage_torsion and name.startswith("T")) else "base_u3_mmi",
            }
            stage2_row = stage2_row_lookup.get(int(candidate_mode_indices[i]))
            if stage2_row is not None:
                row_data["stage2_score"] = float(stage2_row.get("stage2_score", np.nan))
                row_data["stage2_rank"] = float(stage2_row.get("stage2_rank", np.nan))
                row_data["stage2_basis"] = str(stage2_row.get("stage2_basis", ""))
            rows.append(row_data)
            if len(rows) >= max(5, int(getattr(config, "top_k_per_target", 8))):
                break
        top_candidates_per_target[name] = rows
        all_rows: list[dict[str, float]] = []
        for i in order:
            if not np.isfinite(score_matrix[i, j]):
                continue
            all_rows.append(
                {
                    "target": name,
                    "mode_idx": float(candidate_mode_indices[i]),
                    "freq_hz": float(freq_hz_f[i]),
                    "template_index": float(best_template_idx[i, j]),
                    "template_freq_hz": float(best_template_freq[i, j]) if np.isfinite(best_template_freq[i, j]) else np.nan,
                    "mac_u3": float(best_mac[i, j]),
                    "base_freq_penalty": float(best_freq_penalty[i, j]),
                    "base_mmi_u3": float(best_base_mmi_u3[i, j]),
                    "mac_u2": float(best_u2_mac[i, j]) if np.isfinite(best_u2_mac[i, j]) else np.nan,
                    "u2_term": float(best_u2_term[i, j]),
                    "balcony_metric": float(balcony_ratio[i]) if balcony_ratio is not None else np.nan,
                    "balcony_penalty": float(best_balcony_penalty[i, j]),
                    "participation_metric": float(best_participation_score[i, j]) if np.isfinite(best_participation_score[i, j]) else np.nan,
                    "participation_gamma_ratio": float(best_participation_gamma_ratio[i, j]) if np.isfinite(best_participation_gamma_ratio[i, j]) else np.nan,
                    "participation_meff_ratio": float(best_participation_meff_ratio[i, j]) if np.isfinite(best_participation_meff_ratio[i, j]) else np.nan,
                    "participation_term": float(best_participation_term[i, j]),
                    "girder_u3_metric": float(girder_u3_metric[i]) if girder_u3_metric is not None and np.isfinite(girder_u3_metric[i]) else np.nan,
                    "balcony_combined_metric": float(balcony_combined_metric[i]) if balcony_combined_metric is not None and np.isfinite(balcony_combined_metric[i]) else np.nan,
                    "girder_to_balcony_ratio": float(girder_to_balcony_ratio[i]) if girder_to_balcony_ratio is not None and np.isfinite(girder_to_balcony_ratio[i]) else np.nan,
                    "u2_norm_contamination": float(u2_norm_ratio[i]) if np.isfinite(u2_norm_ratio[i]) else np.nan,
                    "u2_peak_contamination": float(u2_peak_ratio[i]) if np.isfinite(u2_peak_ratio[i]) else np.nan,
                    "shortlisted_flag": bool(int(candidate_mode_indices[i]) in shortlist_mode_set),
                    "final_total": float(score_matrix[i, j]),
                }
            )
        score_rows_by_target[name] = all_rows

        sel = int(selected_idx[j])
        sel_mode_idx = int(candidate_mode_indices[sel])
        if name in torsion_stage2:
            for row in torsion_stage2[name].get("shortlist_rows", []):
                row["final_selected_flag"] = int(row["mode_idx"]) == sel_mode_idx
        score_breakdown[name] = {
            "mode_idx": float(sel_mode_idx),
            "freq_hz": float(freq_hz_f[sel]),
            "template_index": float(best_template_idx[sel, j]),
            "template_freq_hz": float(best_template_freq[sel, j]) if np.isfinite(best_template_freq[sel, j]) else np.nan,
            "mac": float(best_mac[sel, j]),
            "mac_u3": float(best_mac[sel, j]),
            "base_freq_penalty": float(best_freq_penalty[sel, j]),
            "freq_penalty": float(best_freq_penalty[sel, j]),
            "base_mmi_u3": float(best_base_mmi_u3[sel, j]),
            "u2_mac": float(best_u2_mac[sel, j]) if np.isfinite(best_u2_mac[sel, j]) else np.nan,
            "mac_u2": float(best_u2_mac[sel, j]) if np.isfinite(best_u2_mac[sel, j]) else np.nan,
            "u2_term": float(best_u2_term[sel, j]),
            "combined_mac": float(best_combined_mac[sel, j]),
            "balcony_metric": float(balcony_ratio[sel]) if balcony_ratio is not None else np.nan,
            "balcony_penalty": float(best_balcony_penalty[sel, j]),
            "balcony_ratio": float(balcony_ratio[sel]) if balcony_ratio is not None else np.nan,
            "participation_metric": float(best_participation_score[sel, j]) if np.isfinite(best_participation_score[sel, j]) else np.nan,
            "participation_score": float(best_participation_score[sel, j]) if np.isfinite(best_participation_score[sel, j]) else np.nan,
            "participation_gamma_ratio": float(best_participation_gamma_ratio[sel, j]) if np.isfinite(best_participation_gamma_ratio[sel, j]) else np.nan,
            "participation_meff_ratio": float(best_participation_meff_ratio[sel, j]) if np.isfinite(best_participation_meff_ratio[sel, j]) else np.nan,
            "participation_term": float(best_participation_term[sel, j]),
            "girder_u3_metric": float(girder_u3_metric[sel]) if girder_u3_metric is not None and np.isfinite(girder_u3_metric[sel]) else np.nan,
            "balcony_combined_metric": float(balcony_combined_metric[sel]) if balcony_combined_metric is not None and np.isfinite(balcony_combined_metric[sel]) else np.nan,
            "girder_to_balcony_ratio": float(girder_to_balcony_ratio[sel]) if girder_to_balcony_ratio is not None and np.isfinite(girder_to_balcony_ratio[sel]) else np.nan,
            "u2_norm_contamination": float(u2_norm_ratio[sel]) if np.isfinite(u2_norm_ratio[sel]) else np.nan,
            "u2_peak_contamination": float(u2_peak_ratio[sel]) if np.isfinite(u2_peak_ratio[sel]) else np.nan,
            "mmi": float(score_matrix[sel, j]),
            "total": float(score_matrix[sel, j]),
            "final_total": float(score_matrix[sel, j]),
            "primary_mac": float(best_mac[sel, j]),
            "stage1_rank": float(next((r["stage1_rank"] for r in top_candidates_per_target[name] if int(r["mode_idx"]) == sel_mode_idx), np.nan)),
            "shortlisted_flag": bool(sel_mode_idx in shortlist_mode_set),
            "selection_stage": "stage2_torsion_rerank" if (use_two_stage_torsion and name.startswith("T")) else "base_u3_mmi",
        }
        stage2_row = stage2_row_lookup.get(sel_mode_idx)
        if stage2_row is not None:
            score_breakdown[name]["stage2_score"] = float(stage2_row.get("stage2_score", np.nan))
            score_breakdown[name]["stage2_rank"] = float(stage2_row.get("stage2_rank", np.nan))
            score_breakdown[name]["stage2_basis"] = str(stage2_row.get("stage2_basis", ""))

    diagnostics = {
        "target_names": target_names,
        "active_switches": {
            "use_u2_for_torsion": bool(config.use_u2_for_torsion),
            "use_balcony_penalty": bool(config.use_balcony_penalty),
            "use_participation_features": bool(config.use_participation_features),
            "use_two_stage_torsion_rerank": bool(config.use_two_stage_torsion_rerank),
        },
        "score_matrix_shape": list(score_matrix.shape),
        "score_matrix": np.asarray(score_matrix, dtype=float).tolist(),
        "base_score_matrix": np.asarray(base_score_matrix, dtype=float).tolist(),
        "legacy_score_matrix": np.asarray(legacy_score_matrix, dtype=float).tolist(),
        "u2_term_matrix": np.asarray(best_u2_term, dtype=float).tolist(),
        "balcony_penalty_matrix": np.asarray(best_balcony_penalty, dtype=float).tolist(),
        "participation_term_matrix": np.asarray(best_participation_term, dtype=float).tolist(),
        "u2_norm_contamination": np.asarray(u2_norm_ratio, dtype=float).tolist(),
        "u2_peak_contamination": np.asarray(u2_peak_ratio, dtype=float).tolist(),
        "torsion_stage2": torsion_stage2,
        "torsion_shortlist_k": shortlist_k,
        "score_rows_by_target": score_rows_by_target,
        "candidate_mode_indices_used": candidate_mode_indices.astype(int).tolist(),
        "excluded_candidate_mode_indices": excluded_mode_indices.astype(int).tolist(),
        "global_assignment_used": True,
        "gamma": gamma,
        "config": asdict(config),
        "reference_file": ref_lib.source_file,
    }

    return ModeMatchResult(
        selected_idx=np.asarray(candidate_mode_indices[selected_idx], dtype=int),
        selected_scores=selected_scores,
        selected_mac=selected_mac,
        selected_composite=selected_scores,
        ambiguity_flags=np.zeros(n_targets, dtype=bool),
        signature_type_used=_signature_type_label(config),
        top_candidates_per_target=top_candidates_per_target,
        score_breakdown=score_breakdown,
        warning_messages=warnings,
        diagnostics=diagnostics,
        richer_signature_available=False,
        candidate_mean_u3=None,
        candidate_diff_u3=None,
        ref_mean_u3_targets=None,
        ref_diff_u3_targets=None,
    )


def load_reference_library(npz_path: str | Path) -> ReferenceModeLibrary:
    return load_reference_mode_library(npz_path, target_names=list(TARGETS))


def _print_reference_summary(ref_lib: ReferenceModeLibrary) -> None:
    print(f"Reference file: {ref_lib.source_file}")
    print(f"Ng: {ref_lib.ng}")
    for name in ref_lib.target_names:
        n_templates = int(ref_lib.u3_library_by_target[name].shape[1])
        freqs = np.asarray(ref_lib.freq_library_by_target[name], dtype=float).reshape(-1)
        print(f"{name}: templates={n_templates}, freqs={freqs.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone U3 MAC/MMI matcher for B1/T1/B2/T2.")
    parser.add_argument("--ref", required=True, help="Path to ref_modes_Ng*.npz")
    parser.add_argument("--cand", default=None, help="Optional candidate .npz for a quick check")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="MMI gamma weight")
    args = parser.parse_args()

    ref_lib = load_reference_library(args.ref)
    _print_reference_summary(ref_lib)

    if args.cand is None:
        return

    z = np.load(args.cand, allow_pickle=False)
    freq_hz = np.asarray(z["freq_hz"], dtype=float).reshape(-1)
    phi_u3 = np.asarray(z["Phi_u3"], dtype=float)

    cfg = ModeMatchConfig(min_mac_threshold=0.0, mmi_gamma=float(args.gamma))
    result = match_target_modes(phi_u3, freq_hz, ref_lib, cfg)
    print(result.to_json())


if __name__ == "__main__":
    main()
