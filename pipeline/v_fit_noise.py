"""T20 step 1 - fit the measurement-noise floor ALPHA for the two deck excitations on the nine FE twins.

    python tools/run_stage.py pipeline/v_fit_noise.py

T18's setting: each LOOCV bridge's FE twin (s6's deterministic solve, all 15 modes), simulated for the field
record's length, read at the field sensors.  For each excitation (v_common: deck, walk) the response is
computed at every graph node; white noise sigma = ALPHA x (in-band RMS over deck nodes) is added before the
1.5-20 Hz bandpass, as v_common.simulate does for the population.  ALPHA is found by bisection so that the
median over the nine bridges of T18's peak_to_bg (median over the four modes of the summed sensor PSD's peak
within +-0.2 Hz of the FE frequency, over the in-band median of that PSD) equals the field's median.
One fitted number per excitation; the field's labels (SSI) are not used.

Draws: forces exactly as T18 (SeedSequence([20261140, bridge, k]), k = 3 deck noise, k = 4 walkers); noise
SeedSequence([20261189, bridge, e]).  Gates (noise_fit.json):
  t18     with ALPHA = 0 and walkers alone / deck noise alone, peak_to_bg equals T18's per_bridge.csv (1e-6 rel.;
          measured 1.3e-9: the response is computed for every graph node here and for the sensor rows in T18,
          so the matrix products round differently)
  fit     the fitted median is within 1 % of the field's
At the fitted ALPHA, T18's full statistics are recomputed (noise_fit_stats.csv) beside the field's.  One number
cannot match both: at the ALPHA that matches peak_to_bg, the background coherency falls well below the field's.
Writes results/T20_deck_forcing/noise_fit.json, noise_fit_bridges.csv, noise_fit_stats.csv.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from copy import deepcopy  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import numpy as np  # noqa: E402

import v_common as V  # noqa: E402

K, C = V.K, V.C
T18_BASE = 20261140
T18_DIR = V.EC.EC_RESULTS / "T18_excitation_realism"
E_IDX = {"deck": 0, "walk": 1}


def peak_to_bg(f: np.ndarray, P: np.ndarray, fmodes: np.ndarray) -> float:
    """T18's stats()['peak_to_bg'] from the PSD diagonal only (same bins, same rule)."""
    band = (f >= 1.5) & (f <= 20.0)
    tot = P.sum(0)
    peaks = []
    for fk in fmodes:
        w = band & (np.abs(f - fk) <= 0.2)
        if not w.any():
            peaks.append(np.nan)
            continue
        idx = np.where(w)[0]
        peaks.append(tot[idx[np.argmax(tot[idx])]])
    return float(np.nanmedian(np.array(peaks, float) / np.median(tot[band])))


def psd(a: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import csd

    nps = int(round(fs / 0.1))
    out = []
    for i in range(a.shape[0]):
        f, s = csd(a[i], a[i], fs=fs, window="hamming", nperseg=nps, noverlap=nps // 2, detrend=False)
        out.append(np.real(s))
    return f, np.array(out)


def bridge_job(s: str) -> dict:
    import s6_decomposition as S6
    import simulate_uniform_inertial_u3_white_noise as SIM
    from export_projected_graph import build_projected_graph_from_model
    from grid_modal_Kinematic_MPC import gdof0
    from r_excitation_realism import walker_force

    pos = list(C.LOOCV).index(s)
    d = np.load(C.REAL_FEATURES / s / "phase3_dataset_corrected.npz", allow_pickle=False)
    fo = np.load(C.FIELD_OUTPUTS / s / "data" / "field_modal_filtering_outputs.npz", allow_pickle=False)
    T = np.asarray(fo["raw_u3_timeseries_clean"]).shape[0] / float(d["fs_hz"])
    pkg = S6.load_graph_package(s, graph_root=str(S6.GRAPH_ROOT))
    near = pkg["nearest_fe_node_idx"].astype(int)
    params = deepcopy(S6._build_params(s))
    params["nmodes"] = S6.N_CANDIDATE_MODES
    params.pop("mode_col_swap", None)
    params.pop("mode_candidate_override", None)
    sr = S6.run_solver(params, ref_x_fracs=S6.REF_X_FRACS, tol_x=S6.REF_SENSOR_TOL_X)
    gp = build_projected_graph_from_model(sr.model, include_sensor_map=False)
    gids = np.asarray(gp["graph_node_ids"], dtype=np.int64)
    sup = set(int(v) for v in SIM._resolve_support_nodes(sr.model))
    is_sup = np.array([int(g) in sup for g in gids])
    if is_sup[near].any():
        raise RuntimeError(f"{s}: a field sensor maps to a support node")
    S = {"Phi": np.asarray(sr.phi_full, float), "freqs": np.asarray(sr.freq_all_hz, float),
         "zrow": np.array([gdof0(int(g), 3) for g in gids]), "is_sup": is_sup,
         "deck_xy": np.asarray(gp["coords_xy_mm"], float)[~is_sup]}
    deck = ~is_sup
    dz = S["zrow"][deck]
    dt = float(SIM.DT)
    nt = int(round(T / dt))
    rng = lambda k: np.random.default_rng(np.random.SeedSequence([T18_BASE, pos, k]))  # noqa: E731
    a_noise = V._modal(S, S["Phi"][dz].T @ rng(3).standard_normal((int(deck.sum()), nt)))
    a_walk = V._modal(S, S["Phi"][dz].T @ walker_force(nt, dt, S["deck_xy"], rng(4)))
    c = V.AMBIENT_SHARE * V._rms(V._bp(a_walk)[deck]) / V._rms(V._bp(a_noise)[deck])
    sims = {"deck": a_noise, "walk": a_walk + c * a_noise}
    out = {"structure": s, "T": T, "fe_f": d["fe_target_freq_hz"].astype(float), "ambient_scale": c,
           "t1_fe": d["fe_target_mode_shape"].astype(float)[near, 1], "walkers_only": a_walk[near]}
    for e, a in sims.items():
        out[e] = {"sens": a[near], "ref": V._rms(V._bp(a)[deck]),
                  "noise": np.random.default_rng(np.random.SeedSequence([V.FIT_NOISE_BASE, pos, E_IDX[e]]))
                  .standard_normal((near.size, nt))}
    return out


def ptb(b: dict, e: str, alpha: float) -> float:
    import simulate_uniform_inertial_u3_white_noise as SIM

    x = b[e]
    a = V._bp(x["sens"] + alpha * x["ref"] * x["noise"])
    f, P = psd(a, float(SIM.FS))
    return peak_to_bg(f, P, b["fe_f"])


def main() -> int:
    import simulate_uniform_inertial_u3_white_noise as SIM
    import r_excitation_realism as R18

    t0 = time.time()
    with get_context("spawn").Pool(9) as pool:
        br = pool.map(bridge_job, list(C.LOOCV))
    t18 = {(r["structure"], r["source"]): r for r in K.read_csv(T18_DIR / "per_bridge.csv")}
    target = float(np.median([float(t18[(s, "field")]["peak_to_bg"]) for s in C.LOOCV]))
    # gate: alpha 0 reproduces T18 for deck noise alone and walkers alone
    g = {}
    for b in br:
        s = b["structure"]
        mine = {"deck_noise": ptb(b, "deck", 0.0),
                "walkers": peak_to_bg(*psd(V._bp(b["walkers_only"]), float(SIM.FS)), b["fe_f"])}
        for src, v in mine.items():
            ref = float(t18[(s, src)]["peak_to_bg"])
            g[f"{s}/{src}"] = abs(v - ref) / ref
    gates = {"t18_reproduced": {"max_rel_diff": max(g.values()), "pass": max(g.values()) < 1e-6}}

    fit, rows = {}, []
    for e in V.EXCS:
        med = lambda al: float(np.median([ptb(b, e, al) for b in br]))  # noqa: E731
        lo, hi = 0.0, 0.25
        m0 = med(lo)
        while med(hi) > target and hi < 64:
            hi *= 2
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if med(mid) > target:
                lo = mid
            else:
                hi = mid
        al = 0.5 * (lo + hi)
        fit[e] = {"alpha": al, "median_at_alpha0": m0, "median_at_alpha": med(al)}
        for b in br:
            rows.append({"structure": b["structure"], "exc": e, "alpha": al, "peak_to_bg_alpha0": ptb(b, e, 0.0),
                         "peak_to_bg_fitted": ptb(b, e, al), "peak_to_bg_field": float(t18[(b["structure"], "field")]["peak_to_bg"]),
                         "record_s": round(b["T"], 1), "ambient_scale": b["ambient_scale"]})
        print(f"[fit] {e}: alpha {al:.4f}  median peak_to_bg {m0:.1f} -> {fit[e]['median_at_alpha']:.2f} (field {target:.2f})",
              flush=True)
    gates["fit"] = {"max_rel_miss": max(abs(fit[e]["median_at_alpha"] - target) / target for e in V.EXCS),
                    "pass": all(abs(fit[e]["median_at_alpha"] - target) / target < 0.01 for e in V.EXCS)}
    # T18's full statistics at the fitted alpha (info)
    stats_rows = []
    for b in br:
        s = b["structure"]
        for e in V.EXCS:
            x = b[e]
            a = V._bp(x["sens"] + fit[e]["alpha"] * x["ref"] * x["noise"])
            f2, S2 = R18.spectra(a, float(SIM.FS))
            st = R18.stats(f2, S2, b["fe_f"], b["t1_fe"])
            st.pop("_log_sensor_power")
            st.pop("_tot_norm")
            stats_rows.append({"structure": s, "source": f"{e}+noise", **st})
    V.T20_RESULTS.mkdir(parents=True, exist_ok=True)
    K.write_csv(V.T20_RESULTS / "noise_fit_bridges.csv", rows)
    K.write_csv(V.T20_RESULTS / "noise_fit_stats.csv", stats_rows)
    out = {"target_field_median_peak_to_bg": target, "alpha": {e: fit[e]["alpha"] for e in V.EXCS}, "fit": fit,
           "gates": gates, "pass": all(v["pass"] for v in gates.values()), "seconds": round(time.time() - t0, 1)}
    K.write_json(V.NOISE_FIT, out)
    print(json.dumps(out, indent=1))
    cols = ["share_B1", "share_T1", "share_B2", "share_T2", "band_12_20", "coh_bg", "t1_antiphase"]
    for e in V.EXCS:
        rr = [r for r in stats_rows if r["source"] == f"{e}+noise"]
        print(f"[stats] {e}+noise " + " ".join(f"{c} {np.nanmean([r[c] for r in rr]):.2f}" for c in cols))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
