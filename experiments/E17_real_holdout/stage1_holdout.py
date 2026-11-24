"""E17 stage 1 - fill in real sensors that were really measured.

    python tools/run_stage.py experiments/E17_real_holdout/stage1_holdout.py --workers 9

E14 and E16 judged every fill-in against simulated ground truth.  On the nine real
bridges the ground truth is free: each has 14-16 recorded sensors.  Hide some,
fill them in from the rest over the hybrid graph the network reads (FE nodes plus
sensor nodes, phase3_real_graphs/), and compare with what the hidden sensors
recorded.  The same test runs on each bridge's FE twin - s6's broadband
simulation of the real FE model, taken at the same sensors - so real and
simulated decks face the same graph, the same sensors and the same operator.

Sources
  real          raw_u3_timeseries_clean, 128 Hz, 1.5-20 Hz bandpass (E5's builder)
  twin_matched  FE twin at the sensors' nearest FE nodes, 200 Hz, bandpassed, cut to
                the real record's number of Welch segments (same estimation noise)
  twin_full     the same twin at its full length (s6's propagation-only input)

Layouts (kept sensors; everything else is hidden), from the two most instrumented
girder lines, a (y = 0) and b, front = largest x:
  single      the front sensor on line a                       (1 sensor)
  front_pair  the front sensor on each line                    (~ s8 outer n = 1)
  front_mid   front_pair + each line's sensor nearest x = L/2  (~ s8 outer n = 2)
  alternate   every other sensor along each line, front kept   (about half)
  one_line    every sensor on line a                           (fill across the deck)
  loso        leave one sensor out, every sensor in turn
  full        every sensor (network only: the s5 / s6 reproduction gate)

Imputers: production FP (phase3_utils, 50 iterations) and E16's FP-rw, Harmonic,
Tikhonov, PCFI, NeighborMean, Mean.  All are linear in the sensor records (PCFI's
channel refinement runs on PSD bins only), so every fill is W @ records, and every
spectral quantity of a filled signal follows exactly from the sensors' Welch
cross-spectral matrix G (the segment average is bilinear).

At each mode (SSI frequency on real data, FE target frequency on the twin; the
nearest Welch bin) and each hidden sensor h:
  deflection    d(u) = G(ref, u) / sqrt(G(ref, ref)), ref = the kept sensor with
                the largest PSD at that bin; filled d_fill(h) = sum_k W[h,k] d(k)
  coherence     |G(fill_h, h)|^2 / (G(fill_h, fill_h) G(h, h)) - the user's criterion
and per layout: MAC of the hidden deflections (>= 2 hidden), amplitude-weighted
sign agreement, median amplitude ratio, mean coherence; MAC of the whole sensor
vector (kept measured + hidden filled); the same fill applied to the label shape
(SSI on real, FE target on the twin - E16's ceiling, on real decks); and the
row-normalised PSD error at the hidden sensors, as the network would see it.

Network: edgegat_N{MAIN_N}, best_model, on real and twin_full features built with
E5's corrected builder from the kept sensors only (FP), scored as s5 / s6 do.

Outputs: fill.csv, fill_sensors.csv, psd.csv, network.csv, checks.json.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from copy import deepcopy  # noqa: E402
from dataclasses import replace as dc_replace  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.signal import csd, get_window, welch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "experiments" / "E16_fp_imputers"))

import fr_common as C  # noqa: E402
import imputers as IM  # noqa: E402
import s6_decomposition as S6  # noqa: E402  (also puts E5_real_transfer on sys.path)
from build_phase3_features_corrected import (  # noqa: E402
    corrected_edge_features,
    corrected_node_features,
    propagate,
    to_training_grid,
)
from EdgeGATConv_Modal_GNN import DEVICE, load_trained_model_checkpoint  # noqa: E402
from infer_phase2a_unseen_structure import load_frequency_normalization  # noqa: E402
from phase3_utils import BANDPASS_F_HIGH, BANDPASS_F_LOW, bandpass_filter, load_graph_package  # noqa: E402
from simulate_uniform_inertial_u3_white_noise import (  # noqa: E402
    WELCH_NFFT,
    WELCH_NOVERLAP,
    WELCH_NPERSEG,
    normalize_rows_by_max_abs,
)

FIELD_ROOT = REPO / "data" / "legacy_preP" / "field_modal_filtering_outputs"  # pre-P, as in the archive
RUN = C.RUNS / C.sweep_run_name(C.MAIN_N)
IMPUTERS = ("FP", "FP-rw", "Harmonic", "Tikhonov", "PCFI", "NeighborMean", "Mean")
FILL_LAYOUTS = ("single", "front_pair", "front_mid", "alternate", "one_line", "loso")
NET_LAYOUTS = ("full", "single", "front_pair", "front_mid", "alternate", "one_line")
SOURCES = ("real", "twin_matched", "twin_full")
MODES = C.MODE_NAMES
S6_FS = 200.0                                          # the generator's sampling rate
_M: dict = {}


def mac_c(u, v) -> float:
    """MAC for complex (or real) vectors."""
    u = np.asarray(u).reshape(-1)
    v = np.asarray(v).reshape(-1)
    return float(np.abs(np.vdot(u, v)) ** 2 / (np.vdot(u, u).real * np.vdot(v, v).real + 1e-30))


# ---- sensor layouts ---------------------------------------------------------------------------
def layouts(sxy: np.ndarray, L: float) -> tuple[dict, tuple[int, int]]:
    ys = np.round(sxy[:, 1]).astype(int)
    vals, counts = np.unique(ys, return_counts=True)
    ya, yb = sorted(vals[np.argsort(-counts, kind="stable")[:2]])
    line = {y: np.where(ys == y)[0] for y in (ya, yb)}
    a, b = (line[y][np.argsort(-sxy[line[y], 0], kind="stable")] for y in (ya, yb))
    mid = lambda idx: int(idx[np.argmin(np.abs(sxy[idx, 0] - L / 2))])  # noqa: E731
    out = {"full": np.arange(len(sxy)),
           "single": np.array([a[0]]),
           "front_pair": np.sort([a[0], b[0]]),
           "front_mid": np.unique([a[0], b[0], mid(a), mid(b)]),
           "alternate": np.sort(np.concatenate([a[0::2], b[0::2]])),
           "one_line": np.sort(line[ya])}
    return out, (int(ya), int(yb))


# ---- spectra ----------------------------------------------------------------------------------
def csd_matrix(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Welch cross-spectral matrix G[i, j, f] = <conj(X_i) X_j>, the pipeline's settings
    (Hamming 2048, overlap 1024, nfft 2048, no detrend, one-sided density)."""
    win = get_window("hamming", WELCH_NPERSEG, fftbins=True)
    step = WELCH_NPERSEG - WELCH_NOVERLAP
    nseg = (x.shape[1] - WELCH_NPERSEG) // step + 1
    idx = np.arange(nseg)[:, None] * step + np.arange(WELCH_NPERSEG)[None, :]
    X = np.fft.rfft(x[:, idx] * win, n=WELCH_NFFT, axis=-1)            # (s, nseg, F)
    w = np.full(X.shape[-1], 2.0)
    w[0] = 1.0
    if WELCH_NFFT % 2 == 0:
        w[-1] = 1.0
    X = X * np.sqrt(w / (fs * (win ** 2).sum()))
    G = np.einsum("igf,jgf->ijf", np.conj(X), X) / nseg
    return G, np.fft.rfftfreq(WELCH_NFFT, 1.0 / fs), nseg


def csd_check(x: np.ndarray, fs: float, G: np.ndarray) -> float:
    kw = dict(fs=fs, window=get_window("hamming", WELCH_NPERSEG, fftbins=True), nperseg=WELCH_NPERSEG,
              noverlap=WELCH_NOVERLAP, nfft=WELCH_NFFT, detrend=False, return_onesided=True, scaling="density")
    _, p0 = welch(x[0], **kw)
    _, p01 = csd(x[0], x[1], **kw)
    return float(max(np.abs(G[0, 0] - p0).max() / p0.max(), np.abs(G[0, 1] - p01).max() / np.abs(p01).max()))


# ---- operators --------------------------------------------------------------------------------
def operator(name: str, ei: np.ndarray, n: int, known: np.ndarray):
    if name == "FP":                                   # production operator on the unit inputs
        X = np.zeros((n, known.size), dtype=np.float32)
        X[known, np.arange(known.size)] = 1.0
        return propagate(X, known, ei, n).astype(np.float64), None
    return IM.operator(name, ei, n, known)


def psd_rows(P: np.ndarray, f_native: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return normalize_rows_by_max_abs(to_training_grid(f_native, P, grid))[0]


# ---- one source on one structure ---------------------------------------------------------------
def fill_source(src: str, s: str, x: np.ndarray, fs: float, Y: np.ndarray, freqs: np.ndarray,
                sens_nodes: np.ndarray, sxy: np.ndarray, lay: dict, ei: np.ndarray, n: int,
                grid: np.ndarray) -> tuple[list, list, list, dict]:
    G, f, nseg = csd_matrix(x, fs)
    P = G[np.arange(len(x)), np.arange(len(x))].real                   # (s, F) auto-PSDs
    bins = [int(np.argmin(np.abs(f - fm))) for fm in freqs]
    band = (f >= BANDPASS_F_LOW) & (f <= BANDPASS_F_HIGH)
    Pn = psd_rows(P, f, grid)
    ns = len(x)
    rows, srows, prows = [], [], []
    ops: dict = {}

    def W_for(name: str, kept: np.ndarray):
        key = (name, tuple(kept))
        if key not in ops:
            W, xi = operator(name, ei, n, sens_nodes[kept])
            ops[key] = (W, xi)
        return ops[key]

    def per_hidden(name, kept, hidden, m, ref):
        """filled and measured deflection and coherence at each hidden sensor, mode m."""
        W, _ = W_for(name, kept)
        Wh = W[sens_nodes[hidden]]                                     # (H, K)
        Gb = G[:, :, bins[m]]
        sc = np.sqrt(Gb[ref, ref].real)
        d_meas = Gb[ref, hidden] / sc
        d_fill = Wh @ Gb[ref, kept] / sc
        g_fh = np.einsum("hk,kh->h", Wh, Gb[np.ix_(kept, hidden)])
        g_ff = np.einsum("hk,kl,hl->h", Wh, Gb[np.ix_(kept, kept)], Wh).real
        coh = np.abs(g_fh) ** 2 / np.maximum(g_ff * Gb[hidden, hidden].real, 1e-300)
        y_fill = Wh @ Y[kept, m]
        return d_meas, d_fill, coh, y_fill, Gb, sc

    def summarise(lname, name, kept, hidden, m, ref, d_meas, d_fill, coh, y_fill, d_all, holdout=""):
        a = np.abs(d_meas)
        agree = (np.real(d_fill * np.conj(d_meas)) > 0).astype(float)
        v_fill = d_all.copy()
        v_fill[hidden] = d_fill
        y = Y[:, m].copy()
        y[hidden] = y_fill
        return {"source": src, "structure": s, "layout": lname, "holdout": holdout, "imputer": name,
                "mode": MODES[m], "freq_hz": float(freqs[m]), "bin_hz": float(f[bins[m]]), "nseg": nseg,
                "n_kept": len(kept), "n_hidden": len(hidden), "ref_sensor": int(ref),
                "defl_mac_hidden": mac_c(d_fill, d_meas) if len(hidden) >= 2 else "",
                "sign_agree_w": float((a * agree).sum() / max(a.sum(), 1e-300)),
                "amp_ratio_median": float(np.median(np.abs(d_fill) / np.maximum(a, 1e-300))),
                "coh_mean": float(np.mean(coh)),
                "defl_mac_all": mac_c(v_fill, d_all),
                "label_mac_all": mac_c(y, Y[:, m]),
                "label_mac_hidden": mac_c(y_fill, Y[hidden, m]) if len(hidden) >= 2 else ""}

    def sensor_rows(lname, name, hidden, m, ref, d_meas, d_fill, coh):
        return [{"source": src, "structure": s, "layout": lname, "imputer": name, "mode": MODES[m],
                 "sensor": int(h), "x_mm": float(sxy[h, 0]), "y_mm": float(sxy[h, 1]), "ref_sensor": int(ref),
                 "d_meas_re": float(dm.real), "d_meas_im": float(dm.imag),
                 "d_fill_re": float(df.real), "d_fill_im": float(df.imag), "coh": float(c)}
                for h, dm, df, c in zip(hidden, d_meas, d_fill, coh)]

    for name in IMPUTERS:
        # fixed layouts
        for lname in FILL_LAYOUTS[:-1]:
            kept = lay[lname]
            hidden = np.setdiff1d(np.arange(ns), kept)
            for m in range(4):
                ref = int(kept[np.argmax(P[kept, bins[m]])])
                d_meas, d_fill, coh, y_fill, Gb, sc = per_hidden(name, kept, hidden, m, ref)
                d_all = Gb[ref, :] / sc
                rows.append(summarise(lname, name, kept, hidden, m, ref, d_meas, d_fill, coh, y_fill, d_all))
                srows += sensor_rows(lname, name, hidden, m, ref, d_meas, d_fill, coh)
            W, xi = W_for(name, kept)
            Pf = IM.apply(name, W, xi, sens_nodes[kept], P[kept], refine=True)[sens_nodes[hidden]]
            prows.append({"source": src, "structure": s, "layout": lname, "imputer": name,
                          "n_kept": len(kept), "n_hidden": len(hidden),
                          "psd_rel_err_hidden": float(np.linalg.norm(psd_rows(Pf, f, grid) - Pn[hidden])
                                                      / np.linalg.norm(Pn[hidden]))})
        # leave one out: one reference per mode for every hold-out, so the hidden values form one vector
        for m in range(4):
            ref = int(np.argmax(P[:, bins[m]]))
            hs = [h for h in range(ns) if h != ref]
            dm, dfl, ch, yf = [], [], [], []
            for h in hs:
                kept = np.setdiff1d(np.arange(ns), [h])
                d_meas, d_fill, coh, y_fill, Gb, sc = per_hidden(name, kept, np.array([h]), m, ref)
                dm.append(d_meas[0]); dfl.append(d_fill[0]); ch.append(coh[0]); yf.append(y_fill[0])
            hs = np.array(hs)
            d_all = G[ref, :, bins[m]] / np.sqrt(G[ref, ref, bins[m]].real)
            r = summarise("loso", name, np.setdiff1d(np.arange(ns), hs), hs, m, ref,
                          np.array(dm), np.array(dfl), np.array(ch), np.array(yf), d_all, holdout="each")
            r["n_kept"] = ns - 1
            r["n_hidden"] = 1
            r["label_mac_all"] = ""                                      # the ref sensor is never held out
            rows.append(r)
            srows += sensor_rows("loso", name, hs, m, ref, np.array(dm), np.array(dfl), np.array(ch))
        Pf = []
        for h in range(ns):
            kept = np.setdiff1d(np.arange(ns), [h])
            W, xi = W_for(name, kept)
            Pf.append(IM.apply(name, W, xi, sens_nodes[kept], P[kept], refine=True)[sens_nodes[h]])
        prows.append({"source": src, "structure": s, "layout": "loso", "imputer": name,
                      "n_kept": ns - 1, "n_hidden": 1,
                      "psd_rel_err_hidden": float(np.linalg.norm(psd_rows(np.array(Pf), f, grid) - Pn)
                                                  / np.linalg.norm(Pn))})
    info = {"nseg": nseg, "n_samples": int(x.shape[1]), "fs": fs, "csd_vs_scipy": csd_check(x, fs, G)}
    return rows, srows, prows, info


def simulate_twin(s: str, n_samples: int) -> tuple[np.ndarray, np.ndarray, float]:
    """s6's simulate_real, unfiltered branch only: the generator's 60 s record (s6's
    propagation-only input), and one long enough to cut n_samples from."""
    params = deepcopy(S6._build_params(s))
    params["nmodes"] = S6.N_CANDIDATE_MODES
    params.pop("mode_col_swap", None)
    params.pop("mode_candidate_override", None)
    solve_res = S6.run_solver(params, ref_x_fracs=S6.REF_X_FRACS, tol_x=S6.REF_SENSOR_TOL_X)
    ref_lib = S6.load_reference_library_for_ng_variant(S6.REF_DIR, int(params["Ng"]), params)
    thr = S6.MAC_THR.get(s)
    cfg = dc_replace(S6.MODE_MATCH_CONFIG, min_mac_threshold=thr) if thr else S6.MODE_MATCH_CONFIG
    match_res = S6.match_target_modes(
        phi_u3_candidates=solve_res.phi_u3_rep, freq_candidates_hz=solve_res.freq_all_hz,
        ref_lib=ref_lib, config=cfg,
        candidate_channels=S6._candidate_channels_from_solve(solve_res, cfg.signature_config),
        candidate_participation=solve_res.participation,
        candidate_metadata=S6._candidate_metadata_from_solve(solve_res))
    selected_idx = np.asarray(match_res.selected_idx, dtype=int).reshape(-1)
    graph_payload = S6.build_projected_graph_from_model(solve_res.model, include_sensor_map=False)
    sim_model = dict(solve_res.model)
    sim_model["M_full"] = solve_res.M_full
    rows = np.asarray(graph_payload["graph_node_ids"], dtype=np.int64) - 1
    out = []
    for t_total in (None, (n_samples + 400) / S6_FS):
        sim = S6.simulate_white_noise_from_modal_model(
            sim_model, solve_res.phi_full, solve_res.freq_all_hz, selected_idx,
            zeta=S6.PHASE1_ZETA, plot=False, return_time_series=True, use_all_modes=True,
            bandpass_hz=None, t_total=t_total)
        out.append(np.asarray(sim["a3_abs"], dtype=np.float64)[rows, :])
        fs = float(np.asarray(sim["fs"]).reshape(()))
    assert fs == S6_FS
    return out[0], out[1], fs


def init_worker() -> None:
    torch.set_num_threads(1)
    ck = RUN / "checkpoints" / "best_model.pt"
    model, checkpoint = load_trained_model_checkpoint(ck, device=DEVICE)
    model.eval()
    _M["model"] = model
    _M["freq_stats"] = load_frequency_normalization(checkpoint_path=ck, checkpoint=checkpoint, run_dir=RUN)


def network_rows(src, s, x, fs, kept_sets, sens_nodes, ei, n, n_fe, grid, y_ssi, fe_target, labels_f):
    rows, feats = [], {}
    for lname in NET_LAYOUTS:
        kept = kept_sets[lname]
        known = sens_nodes[kept]
        xn = corrected_node_features(x[kept], fs, grid, n_nodes=n, sensor_idx=known, edge_index=ei,
                                     order="A", norm_range="grid")
        ts = np.zeros((n, x.shape[1]), dtype=np.float32)
        ts[known] = x[kept].astype(np.float32)
        ea = corrected_edge_features(propagate(ts, known, ei, n).astype(np.float64), fs, grid, ei)
        if lname == "full":
            feats = {"x": xn, "ea": ea}
        p, fr = S6.infer(_M["model"], _M["freq_stats"], ei, xn, ea, n)
        pz, _ = S6.infer(_M["model"], _M["freq_stats"], ei, np.zeros_like(xn), np.zeros_like(ea), n)
        for m, name in enumerate(MODES):
            rows.append({"source": src, "structure": s, "layout": lname, "n_kept": len(kept), "mode": name,
                         "mac_ssi_sensor_nodes": S6.mac(p[sens_nodes, m], y_ssi[:, m]),
                         "mac_fe_fe_nodes": S6.mac(p[:n_fe, m], fe_target[:, m]),
                         "mac_pred_vs_zeroinput": S6.mac(p[:, m], pz[:, m]),
                         "pred_freq_hz": float(fr[m]), "ssi_freq_hz": float(labels_f["ssi"][m]),
                         "fe_freq_hz": float(labels_f["fe"][m])})
    return rows, feats


def run_structure(s: str) -> dict:
    t0 = time.time()
    grid = np.load(next(iter(sorted(C.POP_BROADBAND.glob("*.npz")))))["freq_bins_hz"].astype(np.float64)
    pkg = load_graph_package(s, graph_root=str(S6.GRAPH_ROOT))
    ei = pkg["edge_index"].astype(np.int64)
    coords = pkg["node_coords_xy_mm"].astype(np.float64)
    n, n_fe = coords.shape[0], int(pkg["n_fe_nodes"])
    sens_nodes = pkg["sensor_node_indices"].astype(int)
    nearest_fe = pkg["nearest_fe_node_idx"].astype(int)
    sxy = coords[sens_nodes]
    lay, lines = layouts(sxy, coords[:n_fe, 0].max())
    raw6 = np.load(RUN / "decomposition_raw" / f"{s}.npz")             # s6's labels, label-corrected
    fe_target, fe_freq = raw6["fe_target"].astype(np.float64), raw6["fe_freq_hz"].astype(np.float64)
    y_ssi, ssi_freq = raw6["y_ssi"].astype(np.float64), raw6["ssi_freq_hz"].astype(np.float64)
    assert np.array_equal(raw6["sensor_idx"], sens_nodes) and np.array_equal(raw6["nearest_fe"], nearest_fe)
    labels_f = {"ssi": ssi_freq, "fe": fe_freq}

    # real recordings, as E5's builder reads them
    field = np.load(FIELD_ROOT / s / "data" / "field_modal_filtering_outputs.npz", allow_pickle=False)
    u3 = field["raw_u3_timeseries_clean"].astype(np.float64).T
    fs_r = float(2.0 * field["freq_hz"].astype(np.float64)[-1])
    x_real = bandpass_filter(u3, fs_r, f_low=BANDPASS_F_LOW, f_high=BANDPASS_F_HIGH)

    # FE twin, as s6's propagation-only control builds it (60 s), and again with as many
    # Welch segments as the real record
    nseg_real = (x_real.shape[1] - WELCH_NPERSEG) // (WELCH_NPERSEG - WELCH_NOVERLAP) + 1
    t_match = (nseg_real - 1) * (WELCH_NPERSEG - WELCH_NOVERLAP) + WELCH_NPERSEG
    a3_std, a3_long, fs_t = simulate_twin(s, t_match)
    x_twin = bandpass_filter(a3_std[nearest_fe, :], fs_t, f_low=BANDPASS_F_LOW, f_high=BANDPASS_F_HIGH)
    x_match = bandpass_filter(a3_long[nearest_fe, :], fs_t, f_low=BANDPASS_F_LOW, f_high=BANDPASS_F_HIGH)[:, :t_match]
    assert x_match.shape[1] == t_match

    fill, sens, psd, info = [], [], [], {}
    y_twin = fe_target[nearest_fe]
    for src, x, fs, Y, fq in (("real", x_real, fs_r, y_ssi, ssi_freq),
                              ("twin_matched", x_match, fs_t, y_twin, fe_freq),
                              ("twin_full", x_twin, fs_t, y_twin, fe_freq)):
        r, sr, pr, inf = fill_source(src, s, x, fs, Y, fq, sens_nodes, sxy, lay, ei, n, grid)
        fill += r; sens += sr; psd += pr; info[src] = inf

    # checks: production FP is linear, so W @ x == propagate(x)
    rng = np.random.default_rng(0)
    kept = lay["front_mid"]
    xr = np.zeros((n, 7), dtype=np.float32)
    xr[sens_nodes[kept]] = rng.random((len(kept), 7)).astype(np.float32)
    W, _ = operator("FP", ei, n, sens_nodes[kept])
    direct = propagate(xr, sens_nodes[kept], ei, n).astype(np.float64)
    info["fp_operator_vs_direct_rel"] = float(np.abs(W @ xr[sens_nodes[kept]] - direct).max() / np.abs(direct).max())

    net_r, feat_r = network_rows("real", s, x_real, fs_r, lay, sens_nodes, ei, n, n_fe, grid, y_ssi, fe_target, labels_f)
    net_t, _ = network_rows("twin_full", s, x_twin, fs_t, lay, sens_nodes, ei, n, n_fe, grid, y_ssi, fe_target, labels_f)
    d = np.load(S6.CORRECTED / s / "phase3_dataset_corrected.npz", allow_pickle=False)
    info["real_full_vs_stored_node_maxdiff"] = float(np.abs(feat_r["x"] - d["x_psd"]).max())
    info["real_full_vs_stored_edge_maxdiff"] = float(np.abs(feat_r["ea"] - d["edge_attr_cpsd_full"]).max())
    info["lines_y_mm"] = list(lines)
    info["layouts"] = {k: [int(i) for i in v] for k, v in lay.items()}
    info["seconds"] = round(time.time() - t0, 1)
    print(f"[{s}] {info['seconds']:.0f}s  real T={x_real.shape[1]} nseg={nseg_real}  twin T={x_twin.shape[1]}  "
          f"csd {info['real']['csd_vs_scipy']:.1e}  FP-op {info['fp_operator_vs_direct_rel']:.1e}  "
          f"stored x {info['real_full_vs_stored_node_maxdiff']:.1e} ea {info['real_full_vs_stored_edge_maxdiff']:.1e}",
          flush=True)
    return {"structure": s, "fill": fill, "sensors": sens, "psd": psd, "network": net_r + net_t, "info": info}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def gates(net: list[dict], infos: dict) -> dict:
    tol = 1e-5
    s5 = {(r["structure"], r["mode"]): r for r in csv.DictReader((C.RESULTS / "s5_zeroshot.csv").open(encoding="utf-8"))}
    s6 = {(r["structure"], r["mode"]): r for r in csv.DictReader((C.RESULTS / "s6_decomposition.csv").open(encoding="utf-8"))
          if r["variant"] == "prop_hybrid"}
    d5, d6 = 0.0, 0.0
    for r in net:
        if r["layout"] != "full":
            continue
        k = (r["structure"], r["mode"])
        if r["source"] == "real":
            ref = s5[k]
            d5 = max(d5, abs(r["mac_ssi_sensor_nodes"] - float(ref["mac_ssi"])),
                     abs(r["mac_fe_fe_nodes"] - float(ref["mac_fe_det"])),
                     abs(r["pred_freq_hz"] - float(ref["pred_freq_hz"])))
        else:
            ref = s6[k]
            d6 = max(d6, abs(r["mac_ssi_sensor_nodes"] - float(ref["mac_vs_ssi_sensor_nodes"])),
                     abs(r["mac_fe_fe_nodes"] - float(ref["mac_vs_fe_all_fe_nodes"])),
                     abs(r["pred_freq_hz"] - float(ref["pred_freq_hz"])))
    feat = max(max(i["real_full_vs_stored_node_maxdiff"], i["real_full_vs_stored_edge_maxdiff"]) for i in infos.values())
    csdx = max(i[src]["csd_vs_scipy"] for i in infos.values() for src in SOURCES)
    fpop = max(i["fp_operator_vs_direct_rel"] for i in infos.values())
    g = {"real_full_reproduces_s5": {"max_abs_diff": d5, "tol": tol, "pass": d5 <= tol},
         "twin_full_reproduces_s6_prop_hybrid": {"max_abs_diff": d6, "tol": tol, "pass": d6 <= tol},
         "real_full_features_vs_stored": {"max_abs_diff": feat, "tol": 1e-6, "pass": feat <= 1e-6},
         "csd_matrix_vs_scipy": {"max_rel_diff": csdx, "tol": 1e-9, "pass": csdx <= 1e-9},
         "fp_operator_linear": {"max_rel_diff": fpop, "tol": 1e-5, "pass": fpop <= 1e-5}}
    g["pass"] = all(v["pass"] for v in g.values() if isinstance(v, dict))
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures", nargs="+", default=list(C.LOOCV), choices=list(C.LOOCV))
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as ex:
        res = list(ex.map(run_structure, args.structures))
    out = {k: [r for x in res for r in x[k]] for k in ("fill", "sensors", "psd", "network")}
    infos = {x["structure"]: x["info"] for x in res}
    write_csv(HERE / "fill.csv", out["fill"])
    write_csv(HERE / "fill_sensors.csv", out["sensors"])
    write_csv(HERE / "psd.csv", out["psd"])
    write_csv(HERE / "network.csv", out["network"])
    g = gates(out["network"], infos) if set(args.structures) == set(C.LOOCV) else {"pass": None, "note": "subset run"}
    (HERE / "checks.json").write_text(json.dumps({"gates": g, "per_structure": infos}, indent=2), encoding="utf-8")
    print(json.dumps(g, indent=2))
    print(f"done in {time.time() - t0:.0f}s")
    return 0 if g["pass"] in (True, None) else 1


if __name__ == "__main__":
    sys.exit(main())
