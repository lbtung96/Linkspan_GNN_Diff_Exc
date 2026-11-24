"""T19 - where the real-bridge gap of the T17 networks comes from (s6's decomposition, own records).

    python tools/run_stage.py Final_results_for_paper/EC_comments/pipeline/u_decomposition.py [--workers 8]

s6_decomposition.py's layers, for the six T17 networks (q_common.RUNS).  s6's own code is imported, not
copied: simulate_real, the deterministic FE payload, the corrected real-data builders, propagate.
  ceiling           the bridge's FE twin, all 15 modes, every FE node, the generator's broadband path
  prop_fegraph      the same (unfiltered) response read at the sensors only -> real-data pipeline
  prop_hybrid         (bandpass -> PSD -> propagate -> coherency), on the FE-only and the hybrid graph
  zeroshot_hybrid   the Phase P field inputs (as s5, s6 and T17), hybrid graph and FE-only subgraph
  zeroshot_fegraph
Two changes from s6, both needed by the T17 networks:
  excitation  three own records per bridge, SIM.SEED = SeedSequence([20261150, bridge index, k]), k = 0..2,
              60 s as in training.  s6 reads the shared record (12345), which the T17 networks never saw.
  edges       by the network's edge mode: coherency (800), orientation (2) or both (802).  Orientation is
              fr_edge_modes.orientation of the graph package's stored angle on the hybrid graph and its FE
              subgraph (as T16/T17), and the unit vector source -> destination in the deck plane on s6's
              deterministic FE graph, which stores no angle (the two agree to 1e-7 on the hybrid graphs).
Reading: ceiling -> prop      what measuring only the sensors and propagating costs;
         prop -> zeroshot     what the real recordings cost beyond that (excitation, noise, FE-vs-real).
Gates (gates.json):
  s6      the paper network (edgegat_N2000) under the shared record reproduces s6_decomposition.csv, every
          row (1e-5: the FE eigensolve is re-run, CLAUDE.md's tolerance for that path)
  real    the T17 networks' zeroshot_hybrid MAC vs SSI equals T17's real_zeroshot.csv exactly.  Each network
          runs at its reference's thread count (paper network 2 as s6, T17 networks torch's default as
          q_eval): at 2 threads the T17 rows differ by 2e-7, the last-float32-bit effect CLAUDE.md records.
  orient  orientation from coordinates equals fr_edge_modes.orientation of the graph package (1e-6)
  record  the four records of a bridge are not the same record (max |difference| > half the max |a|);
          their correlations are recorded only: narrowband responses correlate by chance over 10 s
  sim     s6's own per-bridge checks (refilter difference 0; builder node/edge max difference recorded)
Writes results/T19_decomposition/: decomposition.csv, summary.csv, summary_arms.csv, gates.json,
sim_checks.csv; data/T19/sims/<bridge>__<record>.npz.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import q_common as Q
import p_common as P

K, C, EC = Q.K, Q.C, Q.EC
P.ALL.update({k: {kk: v[kk] for kk in ("edge", "loss", "amp", "label")} for k, v in Q.RUNS.items()})
MODES = list(C.MODE_NAMES)
T19_BASE = 20261150
OWN = ("r0", "r1", "r2")
RECS = ("paper",) + OWN
PAPER_SEED = 12345
PAPER_NET = "edgegat_N2000"
T19_DATA = EC.EC_DATA / "T19" / "sims"
T19_RESULTS = EC.EC_RESULTS / "T19_decomposition"
VARIANTS = ["ceiling", "prop_fegraph", "prop_hybrid", "zeroshot_hybrid", "zeroshot_fegraph"]
SIM_VARIANTS = VARIANTS[:3]


def record_seed(s: str, rec: str):
    if rec == "paper":
        return PAPER_SEED
    return np.random.SeedSequence([T19_BASE, C.LOOCV.index(s), int(rec[1:])])


def orient_from_coords(ei: np.ndarray, xy: np.ndarray) -> np.ndarray:
    d = np.asarray(xy, np.float64)[ei[1]] - np.asarray(xy, np.float64)[ei[0]]
    return (d / np.linalg.norm(d, axis=1, keepdims=True)).astype(np.float32)


def sim_job(task: tuple) -> dict:
    """One bridge, one record: s6's simulation block, line for line, with SIM.SEED set for the call."""
    s, rec = task
    import s6_decomposition as S6
    import simulate_uniform_inertial_u3_white_noise as SIM

    if S6.simulate_white_noise_from_modal_model.__globals__ is not SIM.__dict__:
        raise RuntimeError("s6's simulate function does not read the simulator module's SEED")
    if SIM.SEED != PAPER_SEED:
        raise RuntimeError(f"simulator SEED is {SIM.SEED!r} on entry")
    t0 = time.time()
    grid = np.load(next(iter(sorted(C.POP_BROADBAND.glob("*.npz")))))["freq_bins_hz"].astype(np.float64)
    params = S6._build_params(s)
    payload, _ = S6.assemble_deterministic_phase2a_payload(params, structure_name=s, min_mac_threshold=S6.MAC_THR.get(s))
    det_ei = np.asarray(payload["edge_index"], dtype=np.int64)
    fe_target = np.asarray(payload["y_mode_shape_signed"], dtype=np.float64)
    fe_freq = np.asarray(payload["y_freq_hz"], dtype=np.float64).reshape(-1)
    n_fe = fe_target.shape[0]
    pkg = S6.load_graph_package(s, graph_root=str(S6.GRAPH_ROOT))
    hyb_ei = pkg["edge_index"].astype(np.int64)
    n_hyb = int(pkg["node_coords_xy_mm"].shape[0])
    sensor_idx = pkg["sensor_node_indices"].astype(int)
    nearest_fe = pkg["nearest_fe_node_idx"].astype(int)

    SIM.SEED = record_seed(s, rec)
    try:
        bp, raw = S6.simulate_real(s, params)
    finally:
        SIM.SEED = PAPER_SEED
    if not np.array_equal(bp["edge_index"], det_ei):
        raise RuntimeError(f"{s}: simulated graph differs from the deterministic payload")
    fs = bp["fs"]
    x_ceiling = bp["Paa_norm"][:, :S6.PHASE1_FREQ_BINS].astype(np.float32)
    e_ceiling = S6.build_cpsd_full_edge_payload(
        edge_index=det_ei, freq_bins_hz=bp["freq_bins"][:S6.PHASE1_FREQ_BINS], a3_abs=bp["a3"],
        fs=fs, nperseg=S6.WELCH_NPERSEG, noverlap=S6.WELCH_NOVERLAP, nfft=S6.WELCH_NFFT)["edge_attr_cpsd_full"]
    filt_diff = float(np.abs(S6.bandpass_filter(raw["a3"], fs, f_low=S6.BANDPASS_F_LOW, f_high=S6.BANDPASS_F_HIGH)
                             - bp["a3"]).max())
    x_chk = S6.corrected_node_features(bp["a3"], fs, grid, n_nodes=n_fe, sensor_idx=np.arange(n_fe),
                                       edge_index=det_ei, order="A", norm_range="grid")
    e_chk = S6.corrected_edge_features(bp["a3"], fs, grid, det_ei)
    a3_bp_sens = S6.bandpass_filter(raw["a3"][nearest_fe, :], fs, f_low=S6.BANDPASS_F_LOW, f_high=S6.BANDPASS_F_HIGH)
    x_prop = S6.corrected_node_features(a3_bp_sens, fs, grid, n_nodes=n_hyb, sensor_idx=sensor_idx,
                                        edge_index=hyb_ei, order="A", norm_range="grid")
    ts = np.zeros((n_hyb, a3_bp_sens.shape[1]), dtype=np.float32)
    ts[sensor_idx] = a3_bp_sens.astype(np.float32)
    ts_prop = S6.propagate(ts, sensor_idx, hyb_ei, n_hyb).astype(np.float64)
    e_prop_fe = S6.corrected_edge_features(ts_prop[:n_fe], fs, grid, det_ei)
    e_prop_hyb = S6.corrected_edge_features(ts_prop, fs, grid, hyb_ei)
    T19_DATA.mkdir(parents=True, exist_ok=True)
    np.savez(T19_DATA / f"{s}__{rec}.npz", det_ei=det_ei, fe_coords=np.asarray(payload["coords_xy_mm"], np.float64),
             fe_target=fe_target, fe_freq=fe_freq, x_ceiling=x_ceiling, e_ceiling=e_ceiling, x_prop=x_prop,
             e_prop_fe=e_prop_fe, e_prop_hyb=e_prop_hyb, a3_first_sensor=raw["a3"][nearest_fe[0], :2000])
    return {"structure": s, "record": rec, "n_samples": int(raw["a3"].shape[1]), "fs": fs,
            "refilter_maxdiff": filt_diff,
            "ceiling_vs_builder_node_maxdiff": float(np.abs(x_chk - x_ceiling).max()),
            "ceiling_vs_builder_edge_maxdiff": float(np.abs(e_chk - e_ceiling).max()),
            "fe_coords_vs_pkg_maxdiff_mm": float(np.abs(np.asarray(payload["coords_xy_mm"], np.float64)
                                                        - pkg["node_coords_xy_mm"][:n_fe].astype(np.float64)).max()),
            "wall_s": round(time.time() - t0, 1)}


def edges(mode: str, coh: np.ndarray, o: np.ndarray) -> np.ndarray:
    if mode == "cpsd_full":
        return coh
    if mode == "geom_orient":
        return o
    if mode == P.COH_ORIENT:
        return np.hstack([coh, o]).astype(np.float32)
    raise ValueError(mode)


def infer_job(net: str) -> list[dict]:
    import torch
    import fr_edge_modes
    import s6_decomposition as S6

    # thread count moves inference in the last float32 bit (CLAUDE.md): each network runs at its reference's
    # count - the paper network at s6's 2, the T17 networks at torch's default, as q_eval's real_zeroshot ran
    if net == PAPER_NET:
        torch.set_num_threads(2)
    if net == PAPER_NET:
        rd, mode, recs, arm_name, seed = C.RUNS / C.sweep_run_name(C.MAIN_N), "cpsd_full", ("paper",), "paper", C.SEED
    else:
        arm = P.ALL[net]
        P.install_for(arm)
        rd, mode, recs = K.EC_RUNS / net, arm["edge"], OWN
        arm_name, seed = Q.RUNS[net]["base_arm"], Q.RUNS[net]["seed"]
    ck = rd / "checkpoints" / "best_model.pt"
    model, checkpoint = S6.load_trained_model_checkpoint(ck, device=S6.DEVICE)
    model.eval()
    fstats = S6.load_frequency_normalization(checkpoint_path=ck, checkpoint=checkpoint, run_dir=rd)
    rows = []
    for s in C.LOOCV:
        d = np.load(S6.CORRECTED / s / "phase3_dataset_corrected.npz", allow_pickle=False)
        pkg = S6.load_graph_package(s, graph_root=str(S6.GRAPH_ROOT))
        o_hyb = fr_edge_modes.orientation({"edge_attr_geom": pkg["edge_attr_geom"]})
        sensor_idx = pkg["sensor_node_indices"].astype(int)
        nearest_fe = pkg["nearest_fe_node_idx"].astype(int)
        y_ssi = d["y_mode_shape_ssi"].astype(np.float64)
        ssi_freq = d["y_freq_hz"].astype(np.float64).reshape(-1)
        ei_r, x_r, ea_r = d["edge_index"].astype(np.int64), d["x_psd"], d["edge_attr_cpsd_full"]
        cases = []
        for rec in recs:
            z = np.load(T19_DATA / f"{s}__{rec}.npz")
            det_ei, fe_xy, n_fe = z["det_ei"], z["fe_coords"], int(z["fe_target"].shape[0])
            fe_target, fe_freq = z["fe_target"], z["fe_freq"]
            o_fe = orient_from_coords(det_ei, fe_xy)
            cases += [(rec, "ceiling", det_ei, z["x_ceiling"], edges(mode, z["e_ceiling"], o_fe), n_fe, "fe"),
                      (rec, "prop_fegraph", det_ei, z["x_prop"][:n_fe], edges(mode, z["e_prop_fe"], o_fe), n_fe, "fe"),
                      (rec, "prop_hybrid", ei_r, z["x_prop"], edges(mode, z["e_prop_hyb"], o_hyb),
                       int(z["x_prop"].shape[0]), "hybrid")]
            if not np.array_equal(ei_r, pkg["edge_index"].astype(np.int64)):
                raise RuntimeError(f"{s}: real-feature edge_index differs from the graph package")
        m_fe = (ei_r[0] < n_fe) & (ei_r[1] < n_fe)
        cases += [("field", "zeroshot_hybrid", ei_r, x_r, edges(mode, ea_r, o_hyb), int(x_r.shape[0]), "hybrid"),
                  ("field", "zeroshot_fegraph", ei_r[:, m_fe], x_r[:n_fe], edges(mode, ea_r[m_fe], o_hyb[m_fe]),
                   n_fe, "fe")]
        for rec, v, ei, x, ea, n, topo in cases:
            p, f = S6.infer(model, fstats, ei, x, ea, n)
            p_fe = p[:n_fe]
            p_sens = p[sensor_idx] if topo == "hybrid" else p[nearest_fe]
            for m, name in enumerate(MODES):
                rows.append({"net": net, "arm": arm_name, "seed": seed, "edge_mode": mode, "structure": s,
                             "record": rec, "variant": v, "graph": topo, "mode": name,
                             "mac_vs_fe_all_fe_nodes": S6.mac(p_fe[:, m], fe_target[:, m]),
                             "mac_vs_ssi_sensor_nodes": S6.mac(p_sens[:, m], y_ssi[:, m]),
                             "pred_freq_hz": float(f[m]), "fe_target_freq_hz": float(fe_freq[m]),
                             "ssi_freq_hz": float(ssi_freq[m])})
    return rows


def orient_gate() -> dict:
    import fr_edge_modes
    import s6_decomposition as S6

    out = {}
    for s in C.LOOCV:
        pkg = S6.load_graph_package(s, graph_root=str(S6.GRAPH_ROOT))
        ei = pkg["edge_index"].astype(np.int64)
        a = orient_from_coords(ei, pkg["node_coords_xy_mm"])
        b = fr_edge_modes.orientation({"edge_attr_geom": pkg["edge_attr_geom"]})
        out[s] = float(np.abs(a.astype(np.float64) - np.asarray(b, np.float64)).max())
    return {"max_abs_diff": out, "pass": max(out.values()) < 1e-6}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    T19_RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = [(s, r) for s in C.LOOCV for r in RECS]
    checks = []
    with ProcessPoolExecutor(max_workers=a.workers, max_tasks_per_child=1) as ex:
        for c in ex.map(sim_job, jobs):
            checks.append(c)
            print(f"[sim] {c['structure']:<11} {c['record']:<5} {c['wall_s']:.0f} s  refilter {c['refilter_maxdiff']:.1e}  "
                  f"builder node {c['ceiling_vs_builder_node_maxdiff']:.1e} edge {c['ceiling_vs_builder_edge_maxdiff']:.1e}",
                  flush=True)
    K.write_csv(T19_RESULTS / "sim_checks.csv", checks)
    # the four records of a bridge are different records (first sensor, first 2000 samples)
    rec_diff, rec_corr = {}, {}
    for s in C.LOOCV:
        sig = {r: np.load(T19_DATA / f"{s}__{r}.npz")["a3_first_sensor"] for r in RECS}
        pairs = [(r1, r2) for i, r1 in enumerate(RECS) for r2 in RECS[i + 1:]]
        rec_diff[s] = min(float(np.abs(sig[a] - sig[b]).max() / max(np.abs(sig[a]).max(), np.abs(sig[b]).max()))
                          for a, b in pairs)
        rec_corr[s] = max(abs(float(np.corrcoef(sig[a], sig[b])[0, 1])) for a, b in pairs)
    nets = [PAPER_NET] + list(Q.RUNS)
    rows = []
    with ProcessPoolExecutor(max_workers=min(a.workers, len(nets)), max_tasks_per_child=1) as ex:
        for rr in ex.map(infer_job, nets):
            rows += rr
            print(f"[infer] {rr[0]['net']:<38} ({time.time() - t0:.0f} s)", flush=True)
    K.write_csv(T19_RESULTS / "decomposition.csv", rows)

    # ---- gates ------------------------------------------------------------------------------
    s6 = {(r["structure"], r["variant"], r["mode"]): r for r in K.read_csv(C.RESULTS / "s6_decomposition.csv")}
    mine = [r for r in rows if r["net"] == PAPER_NET]
    cols = ("mac_vs_fe_all_fe_nodes", "mac_vs_ssi_sensor_nodes", "pred_freq_hz", "fe_target_freq_hz", "ssi_freq_hz")
    d6 = max(abs(float(r[c]) - float(s6[(r["structure"], r["variant"], r["mode"])][c])) for r in mine for c in cols)
    t17 = {(r["arm"], r["structure"], r["mode"]): float(r["mac_ssi"]) for r in K.read_csv(Q.T17_RESULTS / "real_zeroshot.csv")}
    zr = [r for r in rows if r["net"] in Q.RUNS and r["variant"] == "zeroshot_hybrid"]
    dr = max(abs(r["mac_vs_ssi_sensor_nodes"] - t17[(r["net"], r["structure"], r["mode"])]) for r in zr)
    z6 = max(abs(float(r["mac_vs_ssi_sensor_nodes"]) - float(s6[(r["structure"], r["variant"], r["mode"])]["mac_vs_ssi_sensor_nodes"]))
             for r in mine if r["variant"].startswith("zeroshot"))
    gates = {"s6_reproduced": {"rows": len(mine), "rows_s6": len(s6), "max_abs_diff": d6, "zeroshot_rows_max_abs_diff": z6,
                               "pass": len(mine) == len(s6) and d6 < 1e-5 and z6 < 1e-12},
             "real_equals_t17": {"rows": len(zr), "max_abs_diff": dr, "pass": len(zr) == 6 * 36 and dr == 0.0},
             "orientation": orient_gate(),
             "records_distinct": {"min_rel_max_diff": rec_diff, "max_abs_corr_info": rec_corr,
                                  "pass": min(rec_diff.values()) > 0.5},
             "sim_checks": {"refilter_max": max(c["refilter_maxdiff"] for c in checks),
                            "builder_node_max": max(c["ceiling_vs_builder_node_maxdiff"] for c in checks),
                            "builder_edge_max": max(c["ceiling_vs_builder_edge_maxdiff"] for c in checks),
                            "fe_coords_max_mm": max(c["fe_coords_vs_pkg_maxdiff_mm"] for c in checks),
                            "pass": all(c["refilter_maxdiff"] == 0.0 for c in checks)}}
    gates["pass"] = all(v["pass"] for v in gates.values())
    K.write_json(T19_RESULTS / "gates.json", gates)

    # ---- summaries ---------------------------------------------------------------------------
    frames = (("FE target, all FE nodes", "mac_vs_fe_all_fe_nodes"), ("SSI, sensor nodes", "mac_vs_ssi_sensor_nodes"))
    summ = []
    for net in nets:
        rn = [r for r in rows if r["net"] == net]
        for frame, col in frames:
            out = {"net": net, "arm": rn[0]["arm"], "seed": rn[0]["seed"], "frame": frame}
            for v in VARIANTS:
                # sim variants: mean over records first, then over bridges x modes
                vv = [r for r in rn if r["variant"] == v]
                out[v] = float(np.mean([r[col] for r in vv]))
                for m in MODES:
                    out[f"{v}_{m}"] = float(np.mean([r[col] for r in vv if r["mode"] == m]))
            gap = out["ceiling"] - out["zeroshot_hybrid"]
            for pv in ("prop_fegraph", "prop_hybrid"):
                out[f"share_sensors[{pv}]"] = (out["ceiling"] - out[pv]) / gap if gap else float("nan")
            out["freq_mae_vs_ssi_zeroshot_hybrid"] = float(np.mean([abs(r["pred_freq_hz"] - r["ssi_freq_hz"])
                                                                   for r in rn if r["variant"] == "zeroshot_hybrid"]))
            out["freq_mae_vs_fe_ceiling"] = float(np.mean([abs(r["pred_freq_hz"] - r["fe_target_freq_hz"])
                                                          for r in rn if r["variant"] == "ceiling"]))
            summ.append(out)
    K.write_csv(T19_RESULTS / "summary.csv", summ)
    arms = []
    for arm in Q.BASE_ARMS:
        for frame, _ in frames:
            ss = [r for r in summ if r["arm"] == arm and r["frame"] == frame]
            out = {"arm": arm, "frame": frame, "n_seeds": len(ss)}
            for k in [k for k in ss[0] if k not in ("net", "arm", "seed", "frame")]:
                v = np.array([r[k] for r in ss], dtype=float)
                out[k], out[f"{k}_sd"] = float(v.mean()), float(v.std(ddof=1))
            arms.append(out)
    K.write_csv(T19_RESULTS / "summary_arms.csv", arms)
    print(json.dumps({k: v.get("pass") for k, v in gates.items() if isinstance(v, dict)}), "PASS" if gates["pass"] else "FAIL")
    for r in summ:
        print(f"{r['net']:<38} {r['frame']:<24} " + " ".join(f"{v} {r[v]:.3f}" for v in VARIANTS)
              + f"  sensors share fe {r['share_sensors[prop_fegraph]']:.2f} hyb {r['share_sensors[prop_hybrid]']:.2f}")
    for r in arms:
        print(f"[arm] {r['arm']:<18} {r['frame']:<24} " + " ".join(f"{v} {r[v]:.3f}+-{r[v + '_sd']:.3f}" for v in VARIANTS))
    print(f"total {time.time() - t0:.0f} s")
    return 0 if gates["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
