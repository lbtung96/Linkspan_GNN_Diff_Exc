"""T18 - excitation realism: which synthetic excitation makes the nine bridges' FE twins look like their field recordings?

    python tools/run_stage.py pipeline/r_excitation_realism.py [--workers 9]

No network.  For each LOOCV bridge: the FE twin (s6_decomposition's solve: the deterministic parameters,
all 15 candidate modes, PHASE1_ZETA), simulated for exactly the field record's length, read at the field
sensors (graph_package nearest_fe_node_idx; the field columns follow the SSI sensor order, checked: FDD
vectors vs SSI shapes), under six excitations:

  supports       the paper's model: independent white base acceleration at every support DOF (x, y, z)
  front_common   one 3-DOF white base acceleration shared by the two front (bottom-beam-end) supports
  front_indep    independent 3-DOF white base acceleration at each front support
  deck_noise     independent white vertical forces at every deck graph node (the classical OMA assumption)
  walkers        three 70 kg walkers on random straight-line paths at 1.1-1.5 m/s, pacing 1.6-2.4 Hz with
                 harmonics (DLF 0.4 / 0.1 / 0.1), 4 % slow pacing jitter; vertical force shared between the
                 three nearest deck nodes by inverse distance
  walkers_amb    walkers + deck_noise scaled to 30 % of the walkers' in-band RMS

Modal simulation here: q'' + 2 zeta w q' + w^2 q = Phi^T F (mass-normalised modes, checked), Newmark from
the simulator; base excitation as the simulator (F = -M R a_g, absolute acceleration + R a_g).  Gate: the
'supports' case with the simulator's record draw reproduces simulate_white_noise_from_modal_model's a3_abs.

Both field and simulation: bandpass 1.5-20 Hz (phase3_utils, the training filter), Welch/CSD with a Hamming
window, 0.1 Hz resolution (field 1280 samples at 128 Hz, simulation 2000 at 200 Hz), 50 % overlap.  No
measurement noise is added to the simulation.  Statistics over the 1.5-20 Hz band:
  band_*          share of the summed sensor PSD in 1.5-4, 4-8, 8-12, 12-20 Hz
  share_*         share of the four target modes' peak heights (summed sensor PSD, max within +-0.2 Hz of the
                  mode: SSI frequency for the field, FE target frequency for the simulation)
  peak_to_bg      median over the four modes of peak height / median of the summed PSD
  sensor_spread   max / min of the in-band power over sensors; sensor_corr_vs_field: correlation over sensors
                  of log in-band power with the field's
  coh_mode, coh_bg  median |coherency| over all sensor pairs at the mode bins / at background bins (summed
                  PSD below its in-band median)
  t1_antiphase    at T1: fraction of sensor pairs on opposite sides of the T1 shape (|phi| >= 0.3; SSI shape for
                  the field, FE target for the simulation) whose Re coherency is negative
Writes results/T18_excitation_realism/.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from copy import deepcopy  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import h_common as H  # noqa: E402

EC, C = H.EC, H.C
OUT = EC.EC_RESULTS / "T18_excitation_realism"
EXCITATIONS = ("supports", "front_common", "front_indep", "deck_noise", "walkers", "walkers_amb")
BASE = 20261140
BANDS = ((1.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 20.0))
DF = 0.1
MODES = C.MODE_NAMES


# ------------------------------------------------------------------ helpers
def spectra(a: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """a (n_ch, n_t) -> f, S (n_ch, n_ch, n_f) one-sided CSD, Hamming, 0.1 Hz, 50 % overlap."""
    from scipy.signal import csd
    nps = int(round(fs / DF))
    n = a.shape[0]
    f, _ = csd(a[0], a[0], fs=fs, window="hamming", nperseg=nps, noverlap=nps // 2, detrend=False)
    S = np.zeros((n, n, f.size), dtype=complex)
    for i in range(n):
        for j in range(i, n):
            _, S[i, j] = csd(a[i], a[j], fs=fs, window="hamming", nperseg=nps, noverlap=nps // 2, detrend=False)
            S[j, i] = np.conj(S[i, j])
    return f, S


def stats(f: np.ndarray, S: np.ndarray, fmodes: np.ndarray, phi_t1: np.ndarray) -> dict:
    band = (f >= 1.5) & (f <= 20.0)
    P = np.real(np.einsum("iif->if", S))                       # (n_ch, n_f)
    tot = P.sum(0)
    out = {}
    e_band = tot[band].sum()
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi) if hi < 20 else (f >= lo) & (f <= hi)
        out[f"band_{lo:g}_{hi:g}"] = float(tot[m].sum() / e_band)
    peaks, pbins = [], []
    for fk in fmodes:
        w = band & (np.abs(f - fk) <= 0.2)
        if not w.any():
            peaks.append(np.nan)
            pbins.append(None)
            continue
        idx = np.where(w)[0]
        b = idx[np.argmax(tot[idx])]
        peaks.append(tot[b])
        pbins.append(b)
    peaks = np.array(peaks, float)
    sh = peaks / np.nansum(peaks)
    for k, m in enumerate(MODES):
        out[f"share_{m}"] = float(sh[k])
    out["peak_to_bg"] = float(np.nanmedian(peaks / np.median(tot[band])))
    pw = P[:, band].sum(1)
    out["sensor_spread"] = float(pw.max() / pw.min())
    out["_log_sensor_power"] = np.log10(pw / pw.max())
    n = P.shape[0]
    iu = np.triu_indices(n, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        G = S / np.sqrt(np.einsum("if,jf->ijf", P, P))
    mode_bins = sorted({b for b in pbins if b is not None})
    bg = band & (tot < np.median(tot[band]))
    out["coh_mode"] = float(np.median(np.abs(G[iu[0], iu[1]][:, mode_bins])))
    out["coh_bg"] = float(np.median(np.abs(G[iu[0], iu[1]][:, bg])))
    b1 = pbins[1]
    pi, pj = phi_t1[iu[0]], phi_t1[iu[1]]
    use = (np.abs(pi) >= 0.3 * np.abs(phi_t1).max()) & (np.abs(pj) >= 0.3 * np.abs(phi_t1).max()) & (pi * pj < 0)
    out["t1_antiphase_pairs"] = int(use.sum())
    out["t1_antiphase"] = float(np.mean(np.real(G[iu[0][use], iu[1][use], b1]) < 0)) if (use.any() and b1 is not None) else np.nan
    out["_tot_norm"] = tot / e_band
    return out


def walker_force(nt: int, dt: float, xy_nodes: np.ndarray, rng: np.random.Generator, n_walkers: int = 3) -> np.ndarray:
    """(n_nodes, nt) vertical nodal forces from n_walkers random walkers (N; 70 kg each)."""
    from scipy.signal import butter, sosfiltfilt
    W = 70.0 * 9.81
    lo, hi = xy_nodes.min(0), xy_nodes.max(0)
    F = np.zeros((xy_nodes.shape[0], nt))
    t = np.arange(nt) * dt
    sos = butter(2, 0.3, fs=1.0 / dt, output="sos")
    for _ in range(n_walkers):
        v = rng.uniform(1100.0, 1500.0)                     # mm/s
        f0 = rng.uniform(1.6, 2.4)
        xi = sosfiltfilt(sos, rng.standard_normal(nt))
        xi /= xi.std()
        phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.04 * xi)) * dt
        th = rng.uniform(0, 2 * np.pi, 3)
        force = W * (1 + 0.4 * np.sin(phase + th[0]) + 0.1 * np.sin(2 * phase + th[1]) + 0.1 * np.sin(3 * phase + th[2]))
        pos = np.empty((nt, 2))
        p = rng.uniform(lo, hi)
        tgt = rng.uniform(lo, hi)
        step = v * dt
        for k in range(nt):
            d = tgt - p
            dist = np.hypot(*d)
            if dist < step:
                p = tgt
                tgt = rng.uniform(lo, hi)
            else:
                p = p + d / dist * step
            pos[k] = p
        dd = np.hypot(pos[:, None, 0] - xy_nodes[None, :, 0], pos[:, None, 1] - xy_nodes[None, :, 1])  # (nt, n)
        near = np.argsort(dd, axis=1)[:, :3]
        w = 1.0 / (np.take_along_axis(dd, near, 1) + 100.0)
        w /= w.sum(1, keepdims=True)
        for c in range(3):
            np.add.at(F, (near[:, c], np.arange(nt)), -force * w[:, c])
    return F


# ------------------------------------------------------------------ one bridge
def bridge_job(s: str) -> dict:
    import s6_decomposition as S6
    import simulate_uniform_inertial_u3_white_noise as SIM
    from export_projected_graph import build_projected_graph_from_model
    from grid_modal_Kinematic_MPC import build_influence_matrix, gdof0
    from phase3_utils import bandpass_filter

    t0 = time.time()
    pos = list(C.LOOCV).index(s)
    d = np.load(C.REAL_FEATURES / s / "phase3_dataset_corrected.npz", allow_pickle=False)
    fo = np.load(C.FIELD_OUTPUTS / s / "data" / "field_modal_filtering_outputs.npz", allow_pickle=False)
    fs_field = float(d["fs_hz"])
    field = np.asarray(fo["raw_u3_timeseries_clean"], float).T          # (n_sensors, n_t)
    T = field.shape[1] / fs_field
    pkg = S6.load_graph_package(s, graph_root=str(S6.GRAPH_ROOT))
    near = pkg["nearest_fe_node_idx"].astype(int)
    ssi_f = d["y_freq_hz"].astype(float)
    fe_f = d["fe_target_freq_hz"].astype(float)
    y_ssi = d["y_mode_shape_ssi"].astype(float)
    fe_t1 = d["fe_target_mode_shape"].astype(float)[near, 1]

    # column order of the field record = SSI sensor order (FDD singular vector at each SSI frequency)
    fr, U = fo["freq_hz"], fo["first_singular_vectors"]
    order_mac = []
    for k, fk in enumerate(ssi_f):
        b = int(np.argmin(np.abs(fr - fk)))
        u = U[:, b] * np.exp(-1j * np.angle(U[np.argmax(np.abs(U[:, b])), b]))
        u = np.real(u)
        order_mac.append(float((u @ y_ssi[:, k]) ** 2 / ((u @ u) * (y_ssi[:, k] @ y_ssi[:, k]))))

    params = deepcopy(S6._build_params(s))
    params["nmodes"] = S6.N_CANDIDATE_MODES
    params.pop("mode_col_swap", None)
    params.pop("mode_candidate_override", None)
    sr = S6.run_solver(params, ref_x_fracs=S6.REF_X_FRACS, tol_x=S6.REF_SENSOR_TOL_X)
    model, Phi, M, freqs = sr.model, np.asarray(sr.phi_full, float), sr.M_full, np.asarray(sr.freq_all_hz, float)
    gp = build_projected_graph_from_model(model, include_sensor_map=False)
    gids = np.asarray(gp["graph_node_ids"], dtype=np.int64)
    if not np.allclose(np.asarray(gp["coords_xy_mm"], float), pkg["node_coords_xy_mm"][: gids.size].astype(float)):
        raise RuntimeError(f"{s}: FE graph coordinates differ from the graph package")
    sens_ids = gids[near]
    zrow = np.array([gdof0(int(n), 3) for n in sens_ids])
    mass_check = float(np.abs(Phi.T @ (M @ Phi) - np.eye(Phi.shape[1])).max())
    dt = float(SIM.DT)
    nt = int(round(T / dt))
    zeta = float(S6.PHASE1_ZETA)
    sup = SIM._resolve_support_nodes(model)
    hinge = set(int(v) for v in SIM.find_nodes_by_coordinate(model, x=0.0, tol=1e-6)) if hasattr(SIM, "find_nodes_by_coordinate") else set()
    if not hinge:
        from grid_modal_Kinematic_MPC import find_nodes_by_coordinate
        hinge = set(int(v) for v in find_nodes_by_coordinate(model, x=0.0, tol=1e-6))
    front = np.array([n for n in sup if int(n) not in hinge], dtype=int)
    deck_nodes = np.array([int(n) for n in gids if int(n) not in set(int(v) for v in sup)], dtype=int)
    xy = {int(n): c for n, c in zip(gids, np.asarray(gp["coords_xy_mm"], float))}
    deck_xy = np.array([xy[n] for n in deck_nodes])
    deck_z = np.array([gdof0(n, 3) for n in deck_nodes])

    def modal_response(P: np.ndarray, base_add: np.ndarray | None = None) -> np.ndarray:
        qdd = np.zeros_like(P)
        for k, om in enumerate(2 * np.pi * freqs):
            _, _, qdd[k] = SIM.newmark_sdof(om, zeta, P[k], dt, beta=0.25, gamma=0.5)
        a = Phi[zrow] @ qdd
        if base_add is not None:
            a = a + base_add
        return a

    def base(nodes: np.ndarray, ag: np.ndarray) -> np.ndarray:
        R, _ = build_influence_matrix(model, nodes, [1, 2, 3])
        P = -np.asarray(Phi.T @ (M @ R), dtype=float) @ ag
        return modal_response(P, np.asarray(R.tocsr()[zrow] @ ag, dtype=float))

    rng = lambda k: np.random.default_rng(np.random.SeedSequence([BASE, pos, k]))  # noqa: E731
    sig = float(SIM.SIGMA_AG_MMPS2)
    sims = {}
    # supports (paper model) + gate against the simulator
    R_all, _ = build_influence_matrix(model, sup, [1, 2, 3])
    ss = np.random.SeedSequence([BASE, pos, 0])
    ag = sig * np.random.default_rng(ss).standard_normal((R_all.shape[1], nt))
    sims["supports"] = base(sup, ag)
    sim_model = dict(model)
    sim_model["M_full"] = M
    ref = SIM.simulate_white_noise_from_modal_model(sim_model, Phi, freqs, np.arange(4), zeta=zeta, plot=False,
                                                    return_time_series=True, use_all_modes=True, bandpass_hz=None,
                                                    t_total=T, seed=ss)   # (archive: SIM.SEED set for the call)
    ref_a = np.asarray(ref["a3_abs"], float)[sens_ids - 1]
    gate_rel = float(np.abs(ref_a - sims["supports"]).max() / np.abs(ref_a).max())
    g1 = rng(1).standard_normal((3, nt)) * sig
    sims["front_common"] = base(front, np.vstack([g1] * front.size))
    sims["front_indep"] = base(front, sig * rng(2).standard_normal((3 * front.size, nt)))
    Fd = rng(3).standard_normal((deck_nodes.size, nt))
    sims["deck_noise"] = modal_response(Phi[deck_z].T @ Fd)
    Fw = walker_force(nt, dt, deck_xy, rng(4))
    sims["walkers"] = modal_response(Phi[deck_z].T @ Fw)

    fs_sim = 1.0 / dt
    bp = {k: bandpass_filter(v, fs_sim, f_low=1.5, f_high=20.0) for k, v in sims.items()}
    rw = np.sqrt(np.mean(bp["walkers"] ** 2))
    rn = np.sqrt(np.mean(bp["deck_noise"] ** 2))
    bp["walkers_amb"] = bp["walkers"] + (0.3 * rw / rn) * bp["deck_noise"]
    rows, curves = [], {}
    fb = bandpass_filter(field, fs_field, f_low=1.5, f_high=20.0)
    f, S = spectra(fb, fs_field)
    st = stats(f, S, ssi_f, y_ssi[:, 1])
    ref_logp = st.pop("_log_sensor_power")
    curves["field"] = (f, st.pop("_tot_norm"))
    rows.append({"structure": s, "source": "field", **st, "sensor_corr_vs_field": 1.0})
    for k in EXCITATIONS:
        f2, S2 = spectra(bp[k], fs_sim)
        st = stats(f2, S2, fe_f, fe_t1)
        lp = st.pop("_log_sensor_power")
        curves[k] = (f2, st.pop("_tot_norm"))
        rows.append({"structure": s, "source": k, **st, "sensor_corr_vs_field": float(np.corrcoef(lp, ref_logp)[0, 1])})
    check = {"structure": s, "record_s": round(T, 1), "n_sensors": int(field.shape[0]),
             "field_order_mac_min": min(order_mac), "field_order_mac_median": float(np.median(order_mac)),
             "field_order_mac_per_mode": [round(v, 3) for v in order_mac], "mass_normalisation_maxdev": mass_check,
             "supports_vs_simulator_max_rel": gate_rel, "n_supports": int(len(sup)), "n_front": int(front.size),
             "n_deck_nodes": int(deck_nodes.size), "fe_freqs_hz": freqs.round(2).tolist(),
             "seconds": round(time.time() - t0, 1)}
    return {"rows": rows, "check": check, "curves": {k: (v[0].tolist(), v[1].tolist()) for k, v in curves.items()},
            "ssi_f": ssi_f.tolist(), "fe_f": fe_f.tolist()}


# ------------------------------------------------------------------ report
def figures(res: list[dict], summ: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = {"field": "k", "supports": "#c44e52", "front_common": "#dd8452", "front_indep": "#937860",
            "deck_noise": "#4c72b0", "walkers": "#55a868", "walkers_amb": "#8172b2"}
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, r in zip(axes.flat, res):
        for k, (f, y) in r["curves"].items():
            f, y = np.array(f), np.array(y)
            m = (f >= 1.5) & (f <= 20)
            ax.semilogy(f[m], y[m], color=cols[k], lw=1.6 if k == "field" else 0.9, label=k, zorder=3 if k == "field" else 2)
        for fk in r["ssi_f"]:
            ax.axvline(fk, color="k", ls=":", lw=0.7)
        for fk in r["fe_f"]:
            ax.axvline(fk, color="#c44e52", ls=":", lw=0.7)
        ax.set_title(r["check"]["structure"], fontsize=9)
        ax.set_ylim(1e-5, 1)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("frequency (Hz)")
    fig.suptitle("Summed sensor PSD, normalised to unit in-band energy (dotted: SSI black, FE red)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_spectra_per_bridge.png", dpi=140)
    plt.close(fig)

    import pandas as pd
    d = pd.DataFrame(summ).set_index("source")
    keys = ["band_1.5_4", "band_4_8", "band_8_12", "band_12_20", "share_B1", "share_T1", "share_B2", "share_T2",
            "coh_mode", "coh_bg", "t1_antiphase", "sensor_corr_vs_field"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    order = ["field"] + list(EXCITATIONS)
    for ax, grp, title in ((axes[0], keys[:4], "share of in-band energy"), (axes[1], keys[4:8], "share of the four mode peaks"),
                           (axes[2], keys[8:], "coherency, torsion cue, sensor pattern")):
        w = 0.8 / len(order)
        for i, src in enumerate(order):
            ax.bar(np.arange(len(grp)) + (i - len(order) / 2 + 0.5) * w, [d.loc[src, k] for k in grp], w,
                   color=cols[src], label=src)
        ax.set_xticks(range(len(grp)), [k.replace("band_", "").replace("share_", "") for k in grp], rotation=20, fontsize=8)
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=7)
    fig.suptitle("Field recordings vs FE twins under six excitations (mean over the nine bridges)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_statistics.png", dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        res = list(pool.imap(bridge_job, list(C.LOOCV)))
    rows = [r for x in res for r in x["rows"]]
    checks = [x["check"] for x in res]
    H.write_csv(OUT / "per_bridge.csv", rows)
    H.write_csv(OUT / "checks.csv", checks)
    num = [k for k in rows[0] if k not in ("structure", "source")]
    summ = []
    for src in ["field"] + list(EXCITATIONS):
        g = [r for r in rows if r["source"] == src]
        summ.append({"source": src, **{k: float(np.nanmean([float(r[k]) for r in g])) for k in num}})
    # distance to the field, per bridge then averaged
    dist = []
    for src in EXCITATIONS:
        dd = {"source": src}
        per = []
        for s in C.LOOCV:
            fr_ = next(r for r in rows if r["structure"] == s and r["source"] == "field")
            sm = next(r for r in rows if r["structure"] == s and r["source"] == src)
            per.append({
                "band_L1": sum(abs(sm[k] - fr_[k]) for k in fr_ if k.startswith("band_")),
                "mode_share_L1": sum(abs(sm[f"share_{m}"] - fr_[f"share_{m}"]) for m in MODES),
                "log_peak_to_bg": abs(np.log10(sm["peak_to_bg"] / fr_["peak_to_bg"])),
                "coh_bg_diff": abs(sm["coh_bg"] - fr_["coh_bg"]),
                "coh_mode_diff": abs(sm["coh_mode"] - fr_["coh_mode"]),
                "sensor_corr": sm["sensor_corr_vs_field"],
                "log_sensor_spread": abs(np.log10(sm["sensor_spread"] / fr_["sensor_spread"]))})
        for k in per[0]:
            dd[k] = float(np.nanmean([p[k] for p in per]))
        dist.append(dd)
    H.write_csv(OUT / "summary.csv", summ)
    H.write_csv(OUT / "distance_to_field.csv", dist)
    gate = {"supports_reproduces_simulator_max_rel": max(c["supports_vs_simulator_max_rel"] for c in checks),
            "field_column_order_mac_median_min": min(c["field_order_mac_median"] for c in checks),
            "field_column_order_mac_min": min(c["field_order_mac_min"] for c in checks),
            "mass_normalisation_maxdev": max(c["mass_normalisation_maxdev"] for c in checks)}
    gate["pass"] = gate["supports_reproduces_simulator_max_rel"] < 1e-8 and gate["field_column_order_mac_median_min"] > 0.9 \
        and gate["mass_normalisation_maxdev"] < 1e-6
    H.write_json(OUT / "gates.json", gate)
    figures(res, summ)
    import pandas as pd
    pd.set_option("display.width", 250)
    print(pd.DataFrame(checks).drop(columns=["fe_freqs_hz"]).to_string(index=False))  # noqa
    print(pd.DataFrame(summ).round(3).to_string(index=False))
    print(pd.DataFrame(dist).round(3).to_string(index=False))
    print(gate, f"{time.time() - t0:.0f} s")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
