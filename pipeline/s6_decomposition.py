"""Stage 6 - where the transfer gap comes from (paper §7.3-7.5), broadband.

    python tools/run_stage.py pipeline/s6_decomposition.py

One network (edgegat_N{MAIN_N}, best_model), nine real geometries, three layers:

  ceiling           FE-simulated response on the real FE model, at every FE node,
                    synthesised EXACTLY as the training population is: all 15 solved
                    modes, 1.5-20 Hz bandpass, Welch, row-normalised (the generator's
                    broadband path).  No measurement, no propagation, no domain gap.
  propagation-only  the same simulated response, unfiltered, taken at the sensor
                    locations only, then passed through the real-data pipeline stage
                    for stage: bandpass -> PSD -> propagate -> interpolate ->
                    row-normalise, and propagated time series -> coherency (the
                    corrected builder of experiments/E5_real_transfer).  Scored on the
                    FE-only graph and on the hybrid graph.
  zero-shot         the corrected real features (the recordings, sensor rows 6/8 in the
                    SSI labels' order: fr_common.REAL_FEATURES, s5a, PROVENANCE Phase P),
                    on the hybrid graph and on its FE-only subgraph.

The previous decomposition could only use narrowband checkpoints, because its
simulated features held just the four target modes; with the broadband ceiling
the whole chain runs on the broadband network the paper reports.  The FE target
is the label-corrected one (mode_col_swap, the two T2 overrides); both are
label-only, and with all modes simulated the response does not depend on the
matcher's selection at all.

Writes results/s6_decomposition.csv (per structure x variant x mode),
results/s6_decomposition_summary.csv and results/s6_checks.json, and, for the
figures, runs/<main>/decomposition_raw/<structure>.npz: every variant's predicted
shapes and frequencies with the graph coordinates, FE target and SSI labels they
were scored against (written since 2026-10-01; the CSVs are unchanged by it).
"""

from __future__ import annotations

import csv
import json
import sys
import time
from copy import deepcopy
from dataclasses import replace as dc_replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import fr_common as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "experiments" / "E5_real_transfer"))

import dgl  # noqa: E402

from EdgeGATConv_Modal_GNN import DEVICE, load_trained_model_checkpoint  # noqa: E402
from GraphSAGE_Modal_GNN import denormalize_frequency_predictions  # noqa: E402
from infer_phase2a_unseen_structure import load_frequency_normalization  # noqa: E402
from infer_phase2a_deterministic_edgegat import (  # noqa: E402
    _build_params,
    assemble_deterministic_phase2a_payload,
)
from generate_population_dataset_modes4 import (  # noqa: E402
    MODE_MATCH_CONFIG,
    N_CANDIDATE_MODES,
    PHASE1_FREQ_BINS,
    PHASE1_ZETA,
    REF_DIR,
    REF_SENSOR_TOL_X,
    REF_X_FRACS,
    _candidate_channels_from_solve,
    _candidate_metadata_from_solve,
    build_cpsd_full_edge_payload,
    load_reference_library_for_ng_variant,
    run_solver,
)
from export_projected_graph import build_projected_graph_from_model  # noqa: E402
from simulate_uniform_inertial_u3_white_noise import (  # noqa: E402
    WELCH_NFFT,
    WELCH_NOVERLAP,
    WELCH_NPERSEG,
    simulate_white_noise_from_modal_model,
)
from standalone_mode_matching_mmi import match_target_modes  # noqa: E402
from phase3_utils import BANDPASS_F_HIGH, BANDPASS_F_LOW, bandpass_filter, load_graph_package  # noqa: E402
from build_phase3_features_corrected import (  # noqa: E402
    corrected_edge_features,
    corrected_node_features,
    propagate,
)

CORRECTED = C.REAL_FEATURES                       # s5a_field_inputs.py (Phase P sensor order)
GRAPH_ROOT = C.REPO / "phase3_real_graphs"
E6_RAW = C.REPO / "experiments" / "E6_controls" / "task2_raw"
MAC_THR = {"Eidsdal": 0.70}                       # as every deterministic-path stage uses
VARIANTS = ["ceiling", "prop_fegraph", "prop_hybrid", "zeroshot_hybrid", "zeroshot_fegraph"]


def mac(u, v) -> float:
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return float(np.abs(np.dot(u, v)) ** 2 / (np.dot(u, u) * np.dot(v, v) + 1e-30))


def simulate_real(structure: str, params_user: dict) -> tuple[dict, dict]:
    """E6/task2's resimulate_deterministic, with all FE modes simulated (broadband).

    Solve, match and graph exactly as there; the change is use_all_modes=True.
    One eigensolve, two simulations of the same excitation realisation: with the
    training bandpass inside the simulator (the ceiling) and unfiltered (the
    'measurement' the propagation-only control starts from)."""
    params = deepcopy(params_user)
    params["nmodes"] = N_CANDIDATE_MODES
    params.pop("mode_col_swap", None)                 # label-only corrections: not simulation inputs
    params.pop("mode_candidate_override", None)
    solve_res = run_solver(params, ref_x_fracs=REF_X_FRACS, tol_x=REF_SENSOR_TOL_X)
    ref_lib = load_reference_library_for_ng_variant(REF_DIR, int(params["Ng"]), params)
    thr = MAC_THR.get(structure)
    cfg = dc_replace(MODE_MATCH_CONFIG, min_mac_threshold=thr) if thr else MODE_MATCH_CONFIG
    match_res = match_target_modes(
        phi_u3_candidates=solve_res.phi_u3_rep, freq_candidates_hz=solve_res.freq_all_hz,
        ref_lib=ref_lib, config=cfg,
        candidate_channels=_candidate_channels_from_solve(solve_res, cfg.signature_config),
        candidate_participation=solve_res.participation,
        candidate_metadata=_candidate_metadata_from_solve(solve_res))
    selected_idx = np.asarray(match_res.selected_idx, dtype=int).reshape(-1)
    graph_payload = build_projected_graph_from_model(solve_res.model, include_sensor_map=False)
    sim_model = dict(solve_res.model)
    sim_model["M_full"] = solve_res.M_full
    rows = np.asarray(graph_payload["graph_node_ids"], dtype=np.int64) - 1
    out = []
    for bandpass in (True, False):
        sim = simulate_white_noise_from_modal_model(
            sim_model, solve_res.phi_full, solve_res.freq_all_hz, selected_idx,
            zeta=PHASE1_ZETA, plot=False, return_time_series=True, use_all_modes=True,
            bandpass_hz=(BANDPASS_F_LOW, BANDPASS_F_HIGH) if bandpass else None)
        out.append({"a3": np.asarray(sim["a3_abs"], dtype=np.float64)[rows, :],
                    "Paa_norm": np.asarray(sim["Paa_norm"], dtype=np.float64)[rows, :],
                    "freq_bins": np.asarray(sim["freq_bins_hz"], dtype=np.float64).reshape(-1),
                    "fs": float(np.asarray(sim["fs"]).reshape(())),
                    "edge_index": np.asarray(graph_payload["edge_index"], dtype=np.int64)})
    return out[0], out[1]


def infer(model, freq_stats, ei, x, ea, n):
    g = dgl.graph((ei[0], ei[1]), num_nodes=n)
    g.ndata["x"] = torch.tensor(np.asarray(x, dtype=np.float32))
    g.edata["edge_x"] = torch.tensor(np.asarray(ea, dtype=np.float32))
    g = g.to(DEVICE)
    with torch.no_grad():
        o = model(g, g.ndata["x"], g.edata["edge_x"])
        return (o.pred_mode_shape_abs.cpu().numpy(),
                denormalize_frequency_predictions(o.pred_freq, freq_stats).cpu().numpy().reshape(-1))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=C.RUNS / C.sweep_run_name(C.MAIN_N))
    ap.add_argument("--structures", nargs="+", default=list(C.LOOCV), choices=list(C.LOOCV))
    ap.add_argument("--tag", default="", help="suffix for the output files (dry runs only)")
    args = ap.parse_args()
    torch.set_num_threads(2)
    run = args.run_dir
    ck = run / "checkpoints" / "best_model.pt"
    model, checkpoint = load_trained_model_checkpoint(ck, device=DEVICE)
    model.eval()
    freq_stats = load_frequency_normalization(checkpoint_path=ck, checkpoint=checkpoint, run_dir=run)
    grid = np.load(next(iter(sorted(C.POP_BROADBAND.glob("*.npz")))))["freq_bins_hz"].astype(np.float64)
    assert grid.shape == (PHASE1_FREQ_BINS,)

    raw_dir = run / f"decomposition_raw{args.tag}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows, checks = [], {}
    for s in args.structures:
        t0 = time.time()
        params = _build_params(s)
        payload, _ = assemble_deterministic_phase2a_payload(params, structure_name=s,
                                                            min_mac_threshold=MAC_THR.get(s))
        det_ei = np.asarray(payload["edge_index"], dtype=np.int64)
        fe_target = np.asarray(payload["y_mode_shape_signed"], dtype=np.float64)
        fe_freq = np.asarray(payload["y_freq_hz"], dtype=np.float64).reshape(-1)
        n_fe = fe_target.shape[0]

        d = np.load(CORRECTED / s / "phase3_dataset_corrected.npz", allow_pickle=False)
        pkg = load_graph_package(s, graph_root=str(GRAPH_ROOT))
        hyb_ei = pkg["edge_index"].astype(np.int64)
        n_hyb = int(pkg["node_coords_xy_mm"].shape[0])
        sensor_idx = pkg["sensor_node_indices"].astype(int)
        nearest_fe = pkg["nearest_fe_node_idx"].astype(int)
        y_ssi = d["y_mode_shape_ssi"].astype(np.float64)
        ssi_freq = d["y_freq_hz"].astype(np.float64).reshape(-1)
        assert np.allclose(d["freq_hz_grid"], grid), "training grid mismatch"

        # ---- simulated response, broadband -------------------------------------------------
        bp, raw = simulate_real(s, params)
        assert np.array_equal(bp["edge_index"], det_ei)
        fs = bp["fs"]
        x_ceiling = bp["Paa_norm"][:, :PHASE1_FREQ_BINS].astype(np.float32)
        e_ceiling = build_cpsd_full_edge_payload(
            edge_index=det_ei, freq_bins_hz=bp["freq_bins"][:PHASE1_FREQ_BINS], a3_abs=bp["a3"],
            fs=fs, nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP, nfft=WELCH_NFFT)["edge_attr_cpsd_full"]
        # checks: same excitation realisation; generator path == corrected builder with every node measured
        filt_diff = float(np.abs(bandpass_filter(raw["a3"], fs, f_low=BANDPASS_F_LOW, f_high=BANDPASS_F_HIGH)
                                 - bp["a3"]).max())
        all_idx = np.arange(n_fe)
        x_chk = corrected_node_features(bp["a3"], fs, grid, n_nodes=n_fe, sensor_idx=all_idx,
                                        edge_index=det_ei, order="A", norm_range="grid")
        e_chk = corrected_edge_features(bp["a3"], fs, grid, det_ei)
        e6 = np.load(E6_RAW / f"{s}.npz")
        checks[s] = {"refilter_maxdiff": filt_diff,
                     "ceiling_vs_corrected_builder_node_maxdiff": float(np.abs(x_chk - x_ceiling).max()),
                     "ceiling_vs_corrected_builder_edge_maxdiff": float(np.abs(e_chk - e_ceiling).max()),
                     "fe_target_vs_E6_mac": [round(mac(fe_target[:, m], e6["fe_target"][:, m]), 8) for m in range(4)],
                     # the hybrid graph's first n_fe nodes are scored against the deterministic
                     # payload's FE target row for row: the two node orders must be one
                     "fe_coords_vs_graph_package_maxdiff_mm": float(np.abs(
                         np.asarray(payload["coords_xy_mm"], dtype=np.float64)
                         - pkg["node_coords_xy_mm"][:n_fe].astype(np.float64)).max())}

        # ---- propagation-only: sensors -> real pipeline ------------------------------------------
        a3_bp_sens = bandpass_filter(raw["a3"][nearest_fe, :], fs, f_low=BANDPASS_F_LOW, f_high=BANDPASS_F_HIGH)
        x_prop = corrected_node_features(a3_bp_sens, fs, grid, n_nodes=n_hyb, sensor_idx=sensor_idx,
                                         edge_index=hyb_ei, order="A", norm_range="grid")
        ts = np.zeros((n_hyb, a3_bp_sens.shape[1]), dtype=np.float32)
        ts[sensor_idx] = a3_bp_sens.astype(np.float32)
        ts_prop = propagate(ts, sensor_idx, hyb_ei, n_hyb).astype(np.float64)
        e_prop_fe = corrected_edge_features(ts_prop[:n_fe], fs, grid, det_ei)
        e_prop_hyb = corrected_edge_features(ts_prop, fs, grid, hyb_ei)

        # ---- the real recordings (corrected features) ---------------------------------------------
        ei_r, x_r, ea_r = d["edge_index"].astype(np.int64), d["x_psd"], d["edge_attr_cpsd_full"]
        m_fe = (ei_r[0] < n_fe) & (ei_r[1] < n_fe)

        cases = {
            "ceiling": (det_ei, x_ceiling, e_ceiling, n_fe, "fe"),
            "prop_fegraph": (det_ei, x_prop[:n_fe], e_prop_fe, n_fe, "fe"),
            "prop_hybrid": (hyb_ei, x_prop, e_prop_hyb, n_hyb, "hybrid"),
            "zeroshot_hybrid": (ei_r, x_r, ea_r, int(x_r.shape[0]), "hybrid"),
            "zeroshot_fegraph": (ei_r[:, m_fe], x_r[:n_fe], ea_r[m_fe], n_fe, "fe"),
        }
        raw = {"coords_xy_mm": pkg["node_coords_xy_mm"].astype(np.float64), "n_fe": np.asarray(n_fe),
               "sensor_idx": sensor_idx, "nearest_fe": nearest_fe, "fe_target": fe_target, "fe_freq_hz": fe_freq,
               "y_ssi": y_ssi, "ssi_freq_hz": ssi_freq, "mode_names": np.asarray(C.MODE_NAMES)}
        for v in VARIANTS:
            ei, x, ea, n, topo = cases[v]
            p, f = infer(model, freq_stats, ei, x, ea, n)
            raw[f"pred_{v}"], raw[f"freq_{v}"] = p, f
            p_fe = p[:n_fe]
            p_sens = p[sensor_idx] if topo == "hybrid" else p[nearest_fe]
            for m, name in enumerate(C.MODE_NAMES):
                rows.append({"structure": s, "variant": v, "graph": topo, "mode": name,
                             "mac_vs_fe_all_fe_nodes": mac(p_fe[:, m], fe_target[:, m]),
                             "mac_vs_ssi_sensor_nodes": mac(p_sens[:, m], y_ssi[:, m]),
                             "pred_freq_hz": float(f[m]), "fe_target_freq_hz": float(fe_freq[m]),
                             "ssi_freq_hz": float(ssi_freq[m])})
        np.savez(raw_dir / f"{s}.npz", **raw)
        print(f"[{s}] {time.time() - t0:.0f}s  refilter {filt_diff:.1e}  builder node "
              f"{checks[s]['ceiling_vs_corrected_builder_node_maxdiff']:.1e} edge "
              f"{checks[s]['ceiling_vs_corrected_builder_edge_maxdiff']:.1e}", flush=True)

    with (C.RESULTS / f"s6_decomposition{args.tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summ = []
    for frame, col in (("FE target, all FE nodes", "mac_vs_fe_all_fe_nodes"),
                       ("SSI, sensor nodes", "mac_vs_ssi_sensor_nodes")):
        means = {v: float(np.mean([r[col] for r in rows if r["variant"] == v])) for v in VARIANTS}
        for v in VARIANTS:
            per = {m: float(np.mean([r[col] for r in rows if r["variant"] == v and r["mode"] == m]))
                   for m in C.MODE_NAMES}
            summ.append({"frame": frame, "variant": v, **per, "mean": means[v]})
        gap = means["ceiling"] - means["zeroshot_hybrid"]
        for pv in ("prop_fegraph", "prop_hybrid"):
            summ.append({"frame": frame, "variant": f"share_of_gap_from_propagation[{pv}]",
                         **{m: "" for m in C.MODE_NAMES},
                         "mean": (means["ceiling"] - means[pv]) / gap if gap else float("nan")})
    with (C.RESULTS / f"s6_decomposition_summary{args.tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)
    ok = all(c["refilter_maxdiff"] == 0.0 and min(c["fe_target_vs_E6_mac"]) > 0.999999 for c in checks.values())
    (C.RESULTS / f"s6_checks{args.tag}.json").write_text(json.dumps({"pass": ok, "per_structure": checks}, indent=2),
                                              encoding="utf-8")
    for r in summ:
        print(r)
    print("CHECKS PASS" if ok else "CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
