"""Uniform inertial base-shaking simulation with full-node U3 outputs and PSDs."""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
from scipy.signal import get_window, welch

from grid_modal_Kinematic_MPC import (
    build_influence_matrix,
    build_influence_vector,
    find_nodes_by_coordinate,
    gdof0,
    get_girder_node_ids,
    grid_modal_demo_3d_offset_upgraded_v22,
    mac_matrix,
    match_modes_by_mac,
    modal_participation,
    plot_model_3d,
    reduce_modes_u2,
    reduce_modes_u3,
    select_girder_sensor_nodes,
)


def normalize_rows_by_max_abs(X: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    row_max = np.max(np.abs(X), axis=1)
    scale = np.maximum(row_max, eps)
    return X / scale[:, None], row_max


def normalize_cols_by_max_abs(X: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    col_max = np.max(np.abs(X), axis=0)
    scale = np.maximum(col_max, eps)
    return X / scale[None, :], col_max


DO_PLOT = True
PLOT_GEOMETRY = True
PLOT_EXCITATION_CHANNELS = True
ENABLE_TORSION_PROXY = True
TORSION_NODE_PAIR = [117, 22]
GEOM_LABEL_EVERY = 1
GEOM_LABEL_FONTSIZE = 7
GEOM_VIEW_ELEV = 20
GEOM_VIEW_AZIM = 35
PLOT_GROUPS = {
    "G_GIRDER": True,
    "G_CROSS": False,
    "G_DIAG": False,
    "G_BOTTOM": False,
    "G_FRONT": False,
    "G_BALCONY": False,
    "G_EDGE": False,
    "G_BBRACE": False,
}
SELECTED_NODE_IDS = [104, 9, 107, 12, 108, 13, 228, 94, 46, 226, 117, 22]
AUTO_SELECT = "supports_plus_midspan"

SEED = 12345
ZETA = 0.01
SIGMA_AG_MMPS2 = 100.0
T_TOTAL = 60.0
DT = 0.005
FS = 200.0
WELCH_NPERSEG = 2048
WELCH_NOVERLAP = 1024
WELCH_NFFT = 2048

CAPTURE_REF = True
CONTINUE_AFTER_CAPTURE = False
REF_MODE_INDICES_0BASED = [2, 3, 5, 6]
REF_X_FRACS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
REF_SENSOR_TOL_X = 1e-6

GROUP_NAME_TO_ID = {
    "G_GIRDER": 1,
    "G_CROSS": 2,
    "G_DIAG": 3,
    "G_RIGID": 4,
    "G_BOTTOM": 5,
    "G_FRONT": 6,
    "G_BALCONY": 7,
    "G_EDGE": 8,
    "G_BBRACE": 9,
}


def get_nodes_for_groups(model: dict, group_ids: list[int]) -> np.ndarray:
    elems = np.asarray(model["elems"])
    if elems.size == 0 or len(group_ids) == 0:
        return np.zeros(0, dtype=int)
    grp = elems[:, 3].astype(int)
    mask = np.isin(grp, np.asarray(group_ids, dtype=int))
    if not np.any(mask):
        return np.zeros(0, dtype=int)
    return np.unique(np.r_[elems[mask, 1].astype(int), elems[mask, 2].astype(int)]).astype(int)


def plot_undeformed_with_node_ids(model: dict, params: dict, label_every: int = 1, label_node_ids: Optional[np.ndarray] = None) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure("Undeformed FE Model with Node IDs")
    ax = fig.add_subplot(111, projection="3d")
    plot_model_3d(model, ax, params=params)
    ids = model["nodes"][:, 0].astype(int)
    xyz = model["nodes"][:, 1:4]
    label_set = None if label_node_ids is None else set(np.asarray(label_node_ids, dtype=int).tolist())
    step = max(int(label_every), 1)
    for k, (nid, (x, y, z)) in enumerate(zip(ids, xyz)):
        if (k % step) == 0 and (label_set is None or int(nid) in label_set):
            ax.text(float(x), float(y), float(z), str(int(nid)), fontsize=GEOM_LABEL_FONTSIZE)
    ax.view_init(elev=GEOM_VIEW_ELEV, azim=GEOM_VIEW_AZIM)
    ax.set_title("Undeformed FE Model with Node IDs")


def choose_default_nodes(model: dict, support_nodes: np.ndarray) -> np.ndarray:
    ids = model["nodes"][:, 0].astype(int)
    xy = model["nodes"][:, 1:3]
    if AUTO_SELECT == "supports_plus_midspan":
        x_target = 0.5 * (float(np.min(xy[:, 0])) + float(np.max(xy[:, 0])))
        y_target = 0.5 * (float(np.min(xy[:, 1])) + float(np.max(xy[:, 1])))
        d2 = (xy[:, 0] - x_target) ** 2 + (xy[:, 1] - y_target) ** 2
        mid_id = int(ids[int(np.argmin(d2))])
        return np.unique(np.r_[np.asarray(support_nodes, dtype=int), np.array([mid_id], dtype=int)]).astype(int)
    return np.array([int(ids[0])], dtype=int)


def integrate_acc_to_disp(acc: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    nt = int(acc.size)
    v = np.zeros(nt, dtype=float)
    u = np.zeros(nt, dtype=float)
    for i in range(nt - 1):
        v[i + 1] = v[i] + dt * acc[i]
        u[i + 1] = u[i] + dt * v[i]
    return v, u


def newmark_sdof(omega: float, zeta: float, p_t: np.ndarray, dt: float, beta: float = 0.25, gamma: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nt = int(p_t.size)
    q = np.zeros(nt, dtype=float)
    qd = np.zeros(nt, dtype=float)
    qdd = np.zeros(nt, dtype=float)
    k = float(omega**2)
    c = float(2.0 * zeta * omega)
    if nt == 0:
        return q, qd, qdd
    qdd[0] = p_t[0] - c * qd[0] - k * q[0]
    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    k_eff = k + a0 + a1 * c
    for i in range(nt - 1):
        p_eff = p_t[i + 1] + (a0 * q[i] + a2 * qd[i] + a3 * qdd[i]) + c * (a1 * q[i] + a4 * qd[i] + a5 * qdd[i])
        q[i + 1] = p_eff / k_eff
        qdd[i + 1] = a0 * (q[i + 1] - q[i]) - a2 * qd[i] - a3 * qdd[i]
        qd[i + 1] = qd[i] + dt * ((1.0 - gamma) * qdd[i] + gamma * qdd[i + 1])
    return q, qd, qdd


def welch_all_nodes(a_nodes: np.ndarray, fs: float, window: np.ndarray, *, nperseg: int, noverlap: int, nfft: int) -> tuple[np.ndarray, np.ndarray]:
    nn = int(a_nodes.shape[0])
    f, p0 = welch(a_nodes[0, :], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=False, return_onesided=True, scaling="density")
    Paa = np.zeros((nn, p0.size), dtype=float)
    Paa[0, :] = p0
    for i in range(1, nn):
        _, Paa[i, :] = welch(a_nodes[i, :], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=False, return_onesided=True, scaling="density")
    return f, Paa


def _resolve_support_nodes(model: dict) -> np.ndarray:
    x0_nodes = find_nodes_by_coordinate(model, x=0.0, tol=1e-6)
    bb = np.asarray(model.get("bottom_node_ids", np.zeros(0, dtype=int)), dtype=int).reshape(-1)
    bb_ends = np.array([int(bb[0]), int(bb[-1])], dtype=int) if bb.size >= 2 else np.zeros(0, dtype=int)
    support_nodes = np.unique(np.r_[x0_nodes, bb_ends]).astype(int)
    if support_nodes.size == 0:
        nn = int(np.asarray(model["nodes"]).shape[0])
        support_nodes = np.arange(1, max(1, min(4, nn)) + 1, dtype=int)
    return support_nodes


def _resolve_selected_nodes(model: dict, support_nodes: np.ndarray) -> np.ndarray:
    sel = choose_default_nodes(model, support_nodes) if SELECTED_NODE_IDS is None else np.asarray(SELECTED_NODE_IDS, dtype=int)
    nn = int(np.asarray(model["nodes"]).shape[0])
    sel = np.unique(sel[(sel >= 1) & (sel <= nn)]).astype(int)
    if sel.size == 0:
        raise ValueError("No valid selected nodes after filtering.")
    return sel


def _compute_torsion_proxy(u3_abs: np.ndarray, a3_abs: np.ndarray, *, fs: float, window: np.ndarray, nperseg: int, noverlap: int, nfft: int) -> dict[str, np.ndarray]:
    empty = {
        "torsion_pair": np.zeros(0, dtype=int),
        "torsion_u3_abs": np.zeros(0, dtype=float),
        "torsion_a3_abs": np.zeros(0, dtype=float),
        "torsion_f": np.zeros(0, dtype=float),
        "torsion_Paa": np.zeros(0, dtype=float),
    }
    if not ENABLE_TORSION_PROXY or TORSION_NODE_PAIR is None:
        return empty
    nn = int(u3_abs.shape[0])
    left, right = int(TORSION_NODE_PAIR[0]), int(TORSION_NODE_PAIR[1])
    if not (1 <= left <= nn and 1 <= right <= nn):
        warnings.warn(f"TORSION_NODE_PAIR={TORSION_NODE_PAIR} out of valid range [1,{nn}]. Skipping torsion proxy.", RuntimeWarning)
        return empty
    torsion_u3_abs = u3_abs[left - 1, :] - u3_abs[right - 1, :]
    torsion_a3_abs = a3_abs[left - 1, :] - a3_abs[right - 1, :]
    torsion_f, torsion_Paa = welch(torsion_a3_abs, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=False, return_onesided=True, scaling="density")
    return {
        "torsion_pair": np.array([left, right], dtype=int),
        "torsion_u3_abs": torsion_u3_abs,
        "torsion_a3_abs": torsion_a3_abs,
        "torsion_f": torsion_f,
        "torsion_Paa": torsion_Paa,
    }


def _save_reference_capture(ref_path: str, Ng: int, Nx: int, L_mm: float, sensor_node_ids: np.ndarray, sensor_xy: np.ndarray, freq_hz: np.ndarray, phi_u3_ref: np.ndarray, phi_u2_ref: np.ndarray, ref_mode_idx: np.ndarray) -> None:
    np.savez(
        ref_path,
        Ng=np.asarray(Ng, dtype=int),
        Nx=np.asarray(Nx, dtype=int),
        L_mm=np.asarray(L_mm, dtype=float),
        x_fracs=np.asarray(REF_X_FRACS, dtype=float),
        sensor_node_ids=np.asarray(sensor_node_ids, dtype=int),
        sensor_xy=np.asarray(sensor_xy, dtype=float),
        freq_hz_ref=np.asarray(freq_hz, dtype=float),
        Phi_u3_ref=np.asarray(phi_u3_ref, dtype=float),
        Phi_u2_ref=np.asarray(phi_u2_ref, dtype=float),
        ref_mode_indices_0based=np.asarray(ref_mode_idx, dtype=int),
        Phi_u3_target_library=np.asarray(phi_u3_ref[:, ref_mode_idx][:, :, None], dtype=float),
        Phi_u2_target_library=np.asarray(phi_u2_ref[:, ref_mode_idx][:, :, None], dtype=float),
        freq_hz_target_library=np.asarray(freq_hz[ref_mode_idx][:, None], dtype=float),
        library_sample_ids=np.asarray([-1], dtype=int),
        library_mode_indices=np.asarray(ref_mode_idx[:, None], dtype=int),
        Phi_sig_target_library=np.zeros((0, 4, 0), dtype=float),
        signature_meta_json=np.asarray(""),
    )


def _validate_simulation_payload(payload: dict[str, Any]) -> None:
    idx = np.asarray(payload["matched_mode_indices"], dtype=int).reshape(-1)
    fsel = np.asarray(payload["matched_freq_hz"], dtype=float).reshape(-1)
    Paa_abs = np.asarray(payload["Paa_abs"], dtype=float)
    Paa_norm = np.asarray(payload["Paa_norm"], dtype=float)
    Paa_row_scale = np.asarray(payload["Paa_row_scale"], dtype=float).reshape(-1)
    phi_u3_all = np.asarray(payload["phi_u3_all"], dtype=float)
    phi_u3_abs = np.asarray(payload["phi_u3_abs"], dtype=float)
    phi_u3_abs_norm = np.asarray(payload["phi_u3_abs_norm"], dtype=float)
    phi_u3_abs_scale = np.asarray(payload["phi_u3_abs_scale"], dtype=float).reshape(-1)
    if fsel.size != idx.size:
        raise ValueError(f"matched_freq_hz length must match matched_mode_indices length, got {fsel.size} vs {idx.size}.")
    if Paa_abs.ndim != 2 or Paa_norm.shape != Paa_abs.shape:
        raise ValueError(f"Paa_abs/Paa_norm shape mismatch: {Paa_abs.shape} vs {Paa_norm.shape}.")
    if Paa_row_scale.shape != (Paa_abs.shape[0],):
        raise ValueError(f"Paa_row_scale must have shape ({Paa_abs.shape[0]},), got {Paa_row_scale.shape}.")
    if phi_u3_all.shape != phi_u3_abs.shape or phi_u3_all.shape != phi_u3_abs_norm.shape:
        raise ValueError("phi_u3_all, phi_u3_abs, and phi_u3_abs_norm must have identical shapes.")
    if phi_u3_all.shape[0] != Paa_abs.shape[0] or phi_u3_all.shape[1] != idx.size:
        raise ValueError("Full-node array dimensions are inconsistent with matched_mode_indices.")
    if phi_u3_abs_scale.shape != (idx.size,):
        raise ValueError(f"phi_u3_abs_scale must have shape ({idx.size},), got {phi_u3_abs_scale.shape}.")
    if np.any(Paa_row_scale < 0.0) or np.any(phi_u3_abs_scale < 0.0):
        raise ValueError("Normalization scales must be nonnegative.")


def simulate_white_noise_from_modal_model(
    model,
    Phi_full,
    freq_hz,
    matched_mode_indices,
    *,
    zeta=0.01,
    sigma_ag_mmps2=None,
    dt=None,
    fs=None,
    t_total=None,
    nperseg=None,
    noverlap=None,
    normalize_psd_rows=True,
    normalize_mode_shapes=True,
    plot=False,
    return_time_series=False,
    use_all_modes: bool = False,
    bandpass_hz: tuple[float, float] | None = None,
) -> dict[str, Any]:
    Phi_full = np.asarray(Phi_full, dtype=float)
    freq_hz = np.asarray(freq_hz, dtype=float).reshape(-1)
    matched_mode_indices = np.asarray(matched_mode_indices, dtype=int).reshape(-1)
    if matched_mode_indices.size == 0:
        raise ValueError("matched_mode_indices must not be empty.")
    if np.any(matched_mode_indices < 0) or np.any(matched_mode_indices >= freq_hz.size):
        raise ValueError(f"matched_mode_indices out of range for freq_hz size {freq_hz.size}: {matched_mode_indices.tolist()}")
    sigma_ag_mmps2 = float(SIGMA_AG_MMPS2 if sigma_ag_mmps2 is None else sigma_ag_mmps2)
    dt = float(DT if dt is None else dt)
    fs = float(FS if fs is None else fs)
    t_total = float(T_TOTAL if t_total is None else t_total)
    nperseg = int(WELCH_NPERSEG if nperseg is None else nperseg)
    noverlap = int(WELCH_NOVERLAP if noverlap is None else noverlap)
    if dt <= 0.0 or fs <= 0.0 or t_total <= 0.0:
        raise ValueError(f"dt, fs, and t_total must be positive; got dt={dt}, fs={fs}, t_total={t_total}.")
    if nperseg <= 0 or noverlap < 0 or noverlap >= nperseg:
        raise ValueError(f"Welch settings invalid: nperseg={nperseg}, noverlap={noverlap}.")

    M_full = model.get("M_full")
    if M_full is None:
        raise ValueError("model must include 'M_full' for white-noise simulation.")

    nn = int(np.asarray(model["nodes"]).shape[0])
    matched_freq_hz = np.asarray(freq_hz[matched_mode_indices], dtype=float)
    PhiN = np.asarray(Phi_full[:, matched_mode_indices], dtype=float)
    n_sel = int(matched_mode_indices.size)
    # Simulation basis: all FE modes when use_all_modes=True, else just the 4 matched ones.
    # PhiN (4 matched modes) is always used for labels (phi_u3_all). Only the response
    # time series changes — so PSD/CPSD reflect the full structural response.
    if use_all_modes:
        PhiN_sim = np.asarray(Phi_full, dtype=float)
        sim_freq_hz = np.asarray(freq_hz, dtype=float)
    else:
        PhiN_sim = PhiN
        sim_freq_hz = matched_freq_hz
    n_sim = int(sim_freq_hz.size)
    support_nodes = _resolve_support_nodes(model)
    selected_nodes = _resolve_selected_nodes(model, support_nodes)
    selected_idx = selected_nodes - 1

    rux = build_influence_vector(model, support_nodes, dof_1based=1)
    ruy = build_influence_vector(model, support_nodes, dof_1based=2)
    ruz = build_influence_vector(model, support_nodes, dof_1based=3)
    R, input_map = build_influence_matrix(model, support_nodes, [1, 2, 3])
    Gamma_ux_all, Meff_ux_all, cum_ux, _ = modal_participation(Phi_full, M_full, rux)
    Gamma_uy_all, Meff_uy_all, cum_uy, _ = modal_participation(Phi_full, M_full, ruy)
    Gamma_uz_all, Meff_uz_all, cum_uz, _ = modal_participation(Phi_full, M_full, ruz)

    nt = int(round(t_total / dt))
    if nt < 2:
        raise ValueError(f"Time discretization too short: nt={nt} from t_total={t_total}, dt={dt}.")
    t = np.arange(nt, dtype=float) * dt
    rng = np.random.default_rng(SEED)
    ag_inputs = sigma_ag_mmps2 * rng.standard_normal((int(R.shape[1]), nt))
    v_inputs = np.zeros_like(ag_inputs)
    u_inputs = np.zeros_like(ag_inputs)
    for i in range(ag_inputs.shape[0]):
        v_inputs[i, :], u_inputs[i, :] = integrate_acc_to_disp(ag_inputs[i, :], dt)

    dof_map = input_map[:, 1].astype(int) if input_map.size else np.zeros(0, dtype=int)
    mask_x = dof_map == 1
    mask_y = dof_map == 2
    mask_z = dof_map == 3
    a_gx = np.mean(ag_inputs[mask_x, :], axis=0) if np.any(mask_x) else np.zeros(nt, dtype=float)
    a_gy = np.mean(ag_inputs[mask_y, :], axis=0) if np.any(mask_y) else np.zeros(nt, dtype=float)
    a_gz = np.mean(ag_inputs[mask_z, :], axis=0) if np.any(mask_z) else np.zeros(nt, dtype=float)
    u_gx = np.mean(u_inputs[mask_x, :], axis=0) if np.any(mask_x) else np.zeros(nt, dtype=float)
    u_gy = np.mean(u_inputs[mask_y, :], axis=0) if np.any(mask_y) else np.zeros(nt, dtype=float)
    u_gz = np.mean(u_inputs[mask_z, :], axis=0) if np.any(mask_z) else np.zeros(nt, dtype=float)

    zeta_arr = np.full(n_sel, float(zeta), dtype=float) if np.isscalar(zeta) else np.asarray(zeta, dtype=float).reshape(-1)
    if zeta_arr.shape != (n_sel,):
        raise ValueError(f"zeta must be scalar or shape ({n_sel},), got {zeta_arr.shape}.")
    zeta_scalar = float(zeta_arr[0])
    q_sim = np.zeros((n_sim, nt), dtype=float)
    qd_sim = np.zeros((n_sim, nt), dtype=float)
    qdd_sim = np.zeros((n_sim, nt), dtype=float)
    Gamma_mat_sim = np.asarray(PhiN_sim.T @ (M_full @ R), dtype=float)
    p_all_sim = -Gamma_mat_sim @ ag_inputs
    for k, omega in enumerate(2.0 * np.pi * sim_freq_hz):
        q_sim[k, :], qd_sim[k, :], qdd_sim[k, :] = newmark_sdof(omega, zeta_scalar, p_all_sim[k, :], dt, beta=0.25, gamma=0.5)

    u_rel_full = PhiN_sim @ q_sim
    udd_rel_full = PhiN_sim @ qdd_sim
    u_abs_full = u_rel_full + R @ u_inputs
    a_abs_full = udd_rel_full + R @ ag_inputs
    u3_rel = np.zeros((nn, nt), dtype=float)
    u3_abs = np.zeros((nn, nt), dtype=float)
    a3_rel = np.zeros((nn, nt), dtype=float)
    a3_abs = np.zeros((nn, nt), dtype=float)
    phi_u3_all = np.zeros((nn, n_sel), dtype=float)
    for nid in range(1, nn + 1):
        idx = gdof0(nid, 3)
        u3_rel[nid - 1, :] = u_rel_full[idx, :]
        u3_abs[nid - 1, :] = u_abs_full[idx, :]
        a3_rel[nid - 1, :] = udd_rel_full[idx, :]
        a3_abs[nid - 1, :] = a_abs_full[idx, :]
        phi_u3_all[nid - 1, :] = PhiN[idx, :]

    if bandpass_hz is not None:
        from phase3_utils import bandpass_filter
        a3_abs = bandpass_filter(a3_abs, fs, f_low=float(bandpass_hz[0]), f_high=float(bandpass_hz[1]))
        a3_rel = bandpass_filter(a3_rel, fs, f_low=float(bandpass_hz[0]), f_high=float(bandpass_hz[1]))

    window = get_window("hamming", nperseg, fftbins=True)
    f, Paa_rel = welch_all_nodes(a3_rel, fs, window, nperseg=nperseg, noverlap=noverlap, nfft=WELCH_NFFT)
    _, Paa_abs = welch_all_nodes(a3_abs, fs, window, nperseg=nperseg, noverlap=noverlap, nfft=WELCH_NFFT)
    Paa_norm, Paa_row_scale = normalize_rows_by_max_abs(Paa_abs) if normalize_psd_rows else (np.asarray(Paa_abs, dtype=float), np.max(np.abs(Paa_abs), axis=1))
    phi_u3_abs = np.abs(phi_u3_all)
    phi_u3_abs_norm, phi_u3_abs_scale = normalize_cols_by_max_abs(phi_u3_abs) if normalize_mode_shapes else (np.asarray(phi_u3_abs, dtype=float), np.max(np.abs(phi_u3_abs), axis=0))

    payload: dict[str, Any] = {
        "matched_mode_indices": matched_mode_indices.astype(np.int64),
        "matched_freq_hz": matched_freq_hz.astype(float),
        "freq_bins_hz": np.asarray(f, dtype=float),
        "Paa_abs": np.asarray(Paa_abs, dtype=float),
        "Paa_norm": np.asarray(Paa_norm, dtype=float),
        "Paa_row_scale": np.asarray(Paa_row_scale, dtype=float),
        "phi_u3_all": np.asarray(phi_u3_all, dtype=float),
        "phi_u3_abs": np.asarray(phi_u3_abs, dtype=float),
        "phi_u3_abs_norm": np.asarray(phi_u3_abs_norm, dtype=float),
        "phi_u3_abs_scale": np.asarray(phi_u3_abs_scale, dtype=float),
        "zeta": np.asarray(float(zeta), dtype=float) if np.isscalar(zeta) else zeta_arr.astype(float),
        "fs": np.asarray(fs, dtype=float),
        "dt": np.asarray(dt, dtype=float),
        "t_total": np.asarray(t_total, dtype=float),
        "support_nodes": support_nodes.astype(np.int64),
        "selected_nodes": selected_nodes.astype(np.int64),
        "Gamma_ux": np.asarray(Gamma_ux_all[matched_mode_indices], dtype=float),
        "Gamma_uy": np.asarray(Gamma_uy_all[matched_mode_indices], dtype=float),
        "Gamma_uz": np.asarray(Gamma_uz_all[matched_mode_indices], dtype=float),
        "Gamma_mat": np.asarray(Gamma_mat_sim, dtype=float),
        "Meff_ux": np.asarray(Meff_ux_all[matched_mode_indices], dtype=float),
        "Meff_uy": np.asarray(Meff_uy_all[matched_mode_indices], dtype=float),
        "Meff_uz": np.asarray(Meff_uz_all[matched_mode_indices], dtype=float),
        "Meff_ux_all": np.asarray(Meff_ux_all, dtype=float),
        "Meff_uy_all": np.asarray(Meff_uy_all, dtype=float),
        "Meff_uz_all": np.asarray(Meff_uz_all, dtype=float),
        "cum_ratio_ux": np.asarray(cum_ux, dtype=float),
        "cum_ratio_uy": np.asarray(cum_uy, dtype=float),
        "cum_ratio_uz": np.asarray(cum_uz, dtype=float),
        "input_map": np.asarray(input_map, dtype=np.int64),
        "Paa_abs_sel": np.asarray(Paa_abs[selected_idx, :], dtype=float),
        "Paa_sel_norm": np.asarray(Paa_norm[selected_idx, :], dtype=float),
        "Paa_sel_row_scale": np.asarray(Paa_row_scale[selected_idx], dtype=float),
        "phi_u3_sel": np.asarray(phi_u3_all[selected_idx, :], dtype=float),
        "phi_u3_sel_abs": np.asarray(phi_u3_abs[selected_idx, :], dtype=float),
        "phi_u3_sel_abs_norm": np.asarray(phi_u3_abs_norm[selected_idx, :], dtype=float),
    }
    payload.update(_compute_torsion_proxy(u3_abs, a3_abs, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=WELCH_NFFT))
    if return_time_series:
        payload.update(
            {
                "t": np.asarray(t, dtype=float),
                "ag_inputs": np.asarray(ag_inputs, dtype=float),
                "u_inputs": np.asarray(u_inputs, dtype=float),
                "v_inputs": np.asarray(v_inputs, dtype=float),
                "a_gx": np.asarray(a_gx, dtype=float),
                "a_gy": np.asarray(a_gy, dtype=float),
                "a_gz": np.asarray(a_gz, dtype=float),
                "u_gx": np.asarray(u_gx, dtype=float),
                "u_gy": np.asarray(u_gy, dtype=float),
                "u_gz": np.asarray(u_gz, dtype=float),
                "u3_rel": np.asarray(u3_rel, dtype=float),
                "u3_abs": np.asarray(u3_abs, dtype=float),
                "a3_rel": np.asarray(a3_rel, dtype=float),
                "a3_abs": np.asarray(a3_abs, dtype=float),
                "u3_rel_sel": np.asarray(u3_rel[selected_idx, :], dtype=float),
                "u3_abs_sel": np.asarray(u3_abs[selected_idx, :], dtype=float),
                "a3_rel_sel": np.asarray(a3_rel[selected_idx, :], dtype=float),
                "a3_abs_sel": np.asarray(a3_abs[selected_idx, :], dtype=float),
                "Paa_rel": np.asarray(Paa_rel, dtype=float),
                "Paa_rel_sel": np.asarray(Paa_rel[selected_idx, :], dtype=float),
            }
        )
    _validate_simulation_payload(payload)
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator

        if not return_time_series:
            raise ValueError("plot=True requires return_time_series=True.")
        ag_rms = np.sqrt(np.mean(ag_inputs**2, axis=0))
        if PLOT_EXCITATION_CHANNELS:
            plt.figure("RMS support acceleration inputs")
            plt.plot(t, ag_rms, lw=0.9, label="RMS across all support inputs")
            plt.xlabel("Time [s]")
            plt.ylabel("a_g,rms [mm/s^2]")
            plt.grid(True, alpha=0.3)
            plt.legend()
        plot_count = min(int(selected_nodes.size), 12)
        if selected_nodes.size > 12:
            warnings.warn("More than 12 selected nodes; plotting first 12 only.", RuntimeWarning)
        plt.figure("Selected node U3 displacement time series (absolute)")
        for j in range(plot_count):
            plt.plot(t, u3_abs[selected_idx[j], :], lw=1.0, label=f"node {int(selected_nodes[j])}")
        plt.xlabel("Time [s]")
        plt.ylabel("u3_abs [mm]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        fig, ax = plt.subplots(num="Selected node U3 acceleration PSD (absolute)")
        for j in range(plot_count):
            ax.semilogy(f, np.maximum(Paa_abs[selected_idx[j], :], 1e-30), lw=1.0, label=f"node {int(selected_nodes[j])}")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD [(mm/s^2)^2/Hz]")
        ax.minorticks_on()
        ax.grid(True, which="major", alpha=0.45, linestyle="-")
        ax.grid(True, which="minor", alpha=0.25, linestyle=":")
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.legend()
        fig_n, ax_n = plt.subplots(num="Selected node U3 acceleration PSD (Normalized)")
        for j in range(plot_count):
            ax_n.semilogy(f, np.maximum(Paa_norm[selected_idx[j], :], 1e-30), lw=1.0, label=f"node {int(selected_nodes[j])}")
        ax_n.set_xlabel("Frequency [Hz]")
        ax_n.set_ylabel("Normalized PSD [-]")
        ax_n.minorticks_on()
        ax_n.grid(True, which="major", alpha=0.45, linestyle="-")
        ax_n.grid(True, which="minor", alpha=0.25, linestyle=":")
        ax_n.xaxis.set_major_locator(MultipleLocator(5))
        ax_n.xaxis.set_minor_locator(MultipleLocator(1))
        ax_n.legend()
        plt.show()
    return payload


def main() -> None:
    res = grid_modal_demo_3d_offset_upgraded_v22(do_plot=False, run_modal=True)
    params = res["params"]
    model = dict(res["model"])
    Phi_full = np.asarray(res["Phi"], dtype=float)
    freq_hz = np.asarray(res["freq_hz"], dtype=float)
    model["M_full"] = res["M_full"] if "M_full" in res else res["M"]
    nn = int(model["nodes"].shape[0])
    Ng = int(model.get("Ng", params["Ng"]))
    Nx = int(model.get("Nx", params["Nx"]))
    L_mm = float(model.get("L_mm", params["L_mm"]))
    ref_path = f"ref_modes_Ng{Ng}.npz"

    sensor_node_ids = select_girder_sensor_nodes(model, x_fracs=REF_X_FRACS, tol_x=REF_SENSOR_TOL_X)
    phi_u3_new = reduce_modes_u3(Phi_full, sensor_node_ids)
    phi_u2_new = reduce_modes_u2(Phi_full, sensor_node_ids)
    sensor_xy = np.asarray(model["nodes"], dtype=float)[sensor_node_ids - 1, 1:3]
    girder_ids = get_girder_node_ids(model)
    print(f"Selected sensor nodes (len={sensor_node_ids.size}): {sensor_node_ids.tolist()}")
    print(f"Self-check: nSensors={sensor_node_ids.size}, reduced_vector_length(U3)={phi_u3_new.shape[0]}, girder_nodes={girder_ids.size}")

    ref_mode_idx = np.asarray(REF_MODE_INDICES_0BASED, dtype=int).reshape(-1)
    if np.any(ref_mode_idx < 0) or np.any(ref_mode_idx >= freq_hz.size):
        raise ValueError(f"REF_MODE_INDICES_0BASED={ref_mode_idx.tolist()} out of range for {freq_hz.size} available modes.")
    if CAPTURE_REF:
        _save_reference_capture(ref_path, Ng, Nx, L_mm, sensor_node_ids, sensor_xy, freq_hz, phi_u3_new, phi_u2_new, ref_mode_idx)
        print(f"Saved reference to {ref_path} (nSensors={sensor_node_ids.size}, nModes={phi_u3_new.shape[1]})")
        if not CONTINUE_AFTER_CAPTURE:
            return

    if DO_PLOT and PLOT_GEOMETRY:
        enabled_group_names = [name for name, enabled in PLOT_GROUPS.items() if bool(enabled)]
        group_ids = [GROUP_NAME_TO_ID[name] for name in enabled_group_names if name in GROUP_NAME_TO_ID]
        label_node_ids = None
        if len(group_ids) > 0:
            label_node_ids = get_nodes_for_groups(model, group_ids)
            label_node_ids = np.asarray(label_node_ids, dtype=int)
            label_node_ids = label_node_ids[(label_node_ids >= 1) & (label_node_ids <= nn)]
            label_node_ids = np.unique(label_node_ids).astype(int)
            if label_node_ids.size == 0:
                warnings.warn("No nodes found for selected PLOT_GROUPS; falling back to labeling all nodes.", RuntimeWarning)
                label_node_ids = None
        print(f"Geometry label groups: names={enabled_group_names}, ids={group_ids}, labeled_nodes={nn if label_node_ids is None else int(label_node_ids.size)}")
        plot_undeformed_with_node_ids(model, params, label_every=GEOM_LABEL_EVERY, label_node_ids=label_node_ids)

    ref_npz = np.load(ref_path, allow_pickle=False)
    if int(np.asarray(ref_npz["Ng"]).item()) != Ng:
        raise ValueError(f"Reference Ng mismatch: file has Ng={int(np.asarray(ref_npz['Ng']).item())}, current model has Ng={Ng}.")
    phi_u3_ref = np.asarray(ref_npz["Phi_u3_ref"], dtype=float)
    freq_hz_ref = np.asarray(ref_npz["freq_hz_ref"], dtype=float)
    ref_mode_idx_loaded = np.asarray(ref_npz["ref_mode_indices_0based"], dtype=int) if "ref_mode_indices_0based" in ref_npz else ref_mode_idx
    phi_ref_targets = phi_u3_ref[:, ref_mode_idx_loaded]
    if phi_u3_new.shape[0] != phi_ref_targets.shape[0]:
        raise ValueError(f"Reduced DOF mismatch within same Ng={Ng}: new={phi_u3_new.shape[0]}, ref={phi_ref_targets.shape[0]}.")

    mac_full = mac_matrix(phi_u3_new, phi_ref_targets)
    topk = min(5, mac_full.shape[0])
    for j_ref in range(mac_full.shape[1]):
        order = np.argsort(mac_full[:, j_ref])[::-1][:topk]
        print(f"Ref mode {int(ref_mode_idx_loaded[j_ref])} top-{topk} MAC candidates: {[(int(i), float(mac_full[i, j_ref])) for i in order]}")
    matched_mode_indices, matched_mac = match_modes_by_mac(phi_u3_new, phi_ref_targets)
    matched_freqs = freq_hz[matched_mode_indices]
    print(f"Matched mode indices (0-based): {matched_mode_indices.tolist()}")
    print(f"MAC scores: {np.array2string(matched_mac, precision=6, separator=', ')}")
    print(f"Reference frequencies [Hz]: {np.array2string(freq_hz_ref[ref_mode_idx_loaded], precision=6, separator=', ')}")
    print(f"Matched frequencies [Hz]:   {np.array2string(matched_freqs, precision=6, separator=', ')}")

    sim_res = simulate_white_noise_from_modal_model(model, Phi_full, freq_hz, matched_mode_indices, zeta=ZETA, sigma_ag_mmps2=SIGMA_AG_MMPS2, dt=DT, fs=FS, t_total=T_TOTAL, nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP, normalize_psd_rows=True, normalize_mode_shapes=True, plot=DO_PLOT, return_time_series=True)
    print("PSD normalization check (first selected node max):", np.max(sim_res["Paa_sel_norm"][0, :]))
    print("Mode normalization check (first mode max):", np.max(np.abs(sim_res["phi_u3_abs_norm"][:, 0])))

    np.savez(
        "sim_u3_white_noise_outputs.npz",
        dt=sim_res["dt"], fs=sim_res["fs"], t=sim_res["t"], support_nodes=sim_res["support_nodes"], selected_nodes=sim_res["selected_nodes"],
        N=np.asarray(matched_mode_indices.size, dtype=int), freq_hz=sim_res["matched_freq_hz"], matched_mode_indices=sim_res["matched_mode_indices"], matched_mac=matched_mac, ref_freqs_hz=freq_hz_ref[ref_mode_idx_loaded], Phi=Phi_full,
        a_gx=sim_res["a_gx"], a_gy=sim_res["a_gy"], a_gz=sim_res["a_gz"], u_gx=sim_res["u_gx"], u_gy=sim_res["u_gy"], u_gz=sim_res["u_gz"], ag_inputs=sim_res["ag_inputs"], input_map=sim_res["input_map"],
        Gamma_ux=sim_res["Gamma_ux"], Gamma_uy=sim_res["Gamma_uy"], Gamma_uz=sim_res["Gamma_uz"], Gamma_mat=sim_res["Gamma_mat"], Meff_ux=sim_res["Meff_ux"], Meff_uy=sim_res["Meff_uy"], Meff_uz=sim_res["Meff_uz"], cum_ratio_ux=sim_res["cum_ratio_ux"], cum_ratio_uy=sim_res["cum_ratio_uy"], cum_ratio_uz=sim_res["cum_ratio_uz"],
        u3_rel=sim_res["u3_rel"], u3_abs=sim_res["u3_abs"], a3_rel=sim_res["a3_rel"], a3_abs=sim_res["a3_abs"], u3_rel_sel=sim_res["u3_rel_sel"], u3_abs_sel=sim_res["u3_abs_sel"], a3_rel_sel=sim_res["a3_rel_sel"], a3_abs_sel=sim_res["a3_abs_sel"],
        f=sim_res["freq_bins_hz"], Paa_rel=sim_res["Paa_rel"], Paa_abs=sim_res["Paa_abs"], Paa_rel_sel=sim_res["Paa_rel_sel"], Paa_abs_sel=sim_res["Paa_abs_sel"], Paa_norm=sim_res["Paa_norm"], Paa_row_max=sim_res["Paa_row_scale"], Paa_sel_norm=sim_res["Paa_sel_norm"], Paa_sel_row_max=sim_res["Paa_sel_row_scale"],
        phi_u3_all=sim_res["phi_u3_all"], phi_u3_all_norm=sim_res["phi_u3_abs_norm"], phi_u3_all_col_max=sim_res["phi_u3_abs_scale"], phi_u3_sel=sim_res["phi_u3_sel"], phi_u3_sel_norm=sim_res["phi_u3_sel_abs_norm"], phi_u3_sel_col_max=sim_res["phi_u3_abs_scale"],
        torsion_pair=sim_res["torsion_pair"], torsion_u3_abs=sim_res["torsion_u3_abs"], torsion_a3_abs=sim_res["torsion_a3_abs"], torsion_Paa=sim_res["torsion_Paa"], torsion_f=sim_res["torsion_f"],
    )

    selected_mass_ratio = np.array([np.sum(sim_res["Meff_ux"]) / np.sum(sim_res["Meff_ux_all"]), np.sum(sim_res["Meff_uy"]) / np.sum(sim_res["Meff_uy_all"]), np.sum(sim_res["Meff_uz"]) / np.sum(sim_res["Meff_uz_all"])], dtype=float)
    print(f"Selected modes by MAC: N={matched_mode_indices.size}, indices={matched_mode_indices.tolist()}, selected_mass_ratio={selected_mass_ratio}")
    print(f"Selected output nodes: {np.asarray(sim_res['selected_nodes'], dtype=int).tolist()}")
    print(f"Support nodes: {np.asarray(sim_res['support_nodes']).size}, inputs: {np.asarray(sim_res['input_map']).shape[0]}")
    print(f"Input map preview (nid,dof): {np.asarray(sim_res['input_map'])[: min(8, np.asarray(sim_res['input_map']).shape[0])].tolist()}")
    print(f"Shapes: u3_abs={np.asarray(sim_res['u3_abs']).shape}, a3_abs={np.asarray(sim_res['a3_abs']).shape}, Paa_abs={np.asarray(sim_res['Paa_abs']).shape}")


if __name__ == "__main__":
    main()
