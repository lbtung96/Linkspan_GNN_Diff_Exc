"""E14 stage 1 - two front-corner sensors, the rest filled by feature propagation.

Synthetic domain only. No real recording, graph package or SSI label is read.

The question
------------
The broadband N=600 network (`sensitivity_N600_ep500`, seed 123) predicts mode
shapes and frequencies on its own held-out synthetic structures at MAC ~0.9-0.98,
with every FE graph node carrying its own simulated response. What if only the
two front corners are measured - the intersection of the front beam (x = L) with
the two outermost girders - and every other node is filled in by the production
feature-propagation operator before the same network sees it?

Pipeline per held-out structure
-------------------------------
  1. params_json of the stored sample -> build_solver_params -> run_solver
     (the generator's own FE path, imported unmodified).
  2. simulate_white_noise_from_modal_model with the generator's broadband
     arguments: all 15 candidate modes, bandpass 1.5-20 Hz, zeta = PHASE1_ZETA.
     The excitation RNG is the module-level SEED, so the realisation is the one
     the stored sample was built from; step 3 checks that it is.
  3. Reproduction check: re-simulated node PSD and edge coherency against the
     stored x_psd / edge_attr_cpsd_full.
  4. Keep the bandpassed a3_abs at the two corner nodes only.
  5. Feature propagation, stage-for-stage as the real pipeline does it
     (E5/build_phase3_features_corrected.py, order 'A'):
       node features : Welch PSD at the corners -> propagate spectra ->
                       row-normalise by max-abs -> first 400 bins
       edge features : propagate the time series -> complex coherency on every
                       graph edge (build_cpsd_full_edge_payload, imported)
     Operator: phase3_utils.feature_propagation, 50 iterations, clamped - the
     production operator, on the sample's own FE graph. The two corners ARE FE
     nodes, so no hybrid graph and no bandpass beyond the generator's own.
  6. Inference with last_model.pt (primary - the repo's test metrics are
     last-epoch) and best_model.pt (robustness), CPU.

Variants (each row of the CSV)
------------------------------
  full           stored features - exactly what the model was tested on
  corners_prop   PRIMARY: two front corners measured, the rest propagated
  corner_copy    control: corner y=0's record copied to every node. Realistic
                 spectra, no spatial information at all (coherency = 1 on every
                 edge). What the network can do from topology + spectra alone.
  zeros          control: all features zero - the network's zero-input response

Diagnostic
----------
  prop_shape_mac  MAC(propagate(true mode shape | corners), true mode shape).
                  How much of each mode shape the propagation operator itself can
                  represent from two corner values. Not a network output.

Outputs (this folder only)
--------------------------
  stage1_per_sample.csv    one row per structure x checkpoint x variant x mode
  stage1_diagnostics.csv   one row per structure: reproduction, corners, reach
  raw/<sample>.npz         coords, targets and every prediction, for figures
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import get_window

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import dgl  # noqa: E402

from EdgeGATConv_Modal_GNN import DEVICE, load_trained_model_checkpoint  # noqa: E402
from GraphSAGE_Modal_GNN import denormalize_frequency_predictions, set_global_seed  # noqa: E402
from infer_phase2a_unseen_structure import load_frequency_normalization  # noqa: E402
from infer_phase2a_unseen_structure_edgegat import (  # noqa: E402
    apply_input_normalization,
    load_input_normalization,
)
from generate_population_dataset_modes4 import (  # noqa: E402
    N_CANDIDATE_MODES,
    PHASE1_FREQ_BINS,
    PHASE1_ZETA,
    REF_SENSOR_TOL_X,
    REF_X_FRACS,
    _parse_params_json,
    build_cpsd_full_edge_payload,
    build_solver_params,
    run_solver,
)
from export_projected_graph import build_projected_graph_from_model  # noqa: E402
from simulate_uniform_inertial_u3_white_noise import (  # noqa: E402
    WELCH_NFFT,
    WELCH_NOVERLAP,
    WELCH_NPERSEG,
    normalize_rows_by_max_abs,
    simulate_white_noise_from_modal_model,
    welch_all_nodes,
)
from phase3_utils import feature_propagation  # noqa: E402

MODE_NAMES = ("B1", "T1", "B2", "T2")
N_PROP_ITER = 50                       # production value (E5, E6, E9 diffusion_50)
BANDPASS_HZ = (1.5, 20.0)              # the generator's literal broadband band
RUN_DIR = REPO / "runs_phase2" / "EdgeGATConv" / "sensitivity_N600_ep500"
CHECKPOINTS = ("last", "best")
VARIANTS = ("full", "corners_prop", "corner_copy", "zeros")
CORNER_TOL_MM = 1.0
OUT_DIR = Path(__file__).resolve().parent


def mac(u: np.ndarray, v: np.ndarray) -> float:
    """Same formula as EdgeGATConv_Modal_GNN._compute_sample_metrics."""
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return float(abs(np.dot(u, v)) ** 2 / (np.dot(u, u) * np.dot(v, v) + 1e-30))


def propagate(x: np.ndarray, known_idx: np.ndarray, edge_index: np.ndarray,
              n_nodes: int) -> np.ndarray:
    """Production clamped harmonic extension (phase3_utils.feature_propagation)."""
    mask = torch.zeros(n_nodes, dtype=torch.bool)
    mask[known_idx] = True
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    return feature_propagation(xt, mask, torch.from_numpy(edge_index),
                               n_nodes, n_iter=N_PROP_ITER).numpy()


def find_front_corners(coords_xy: np.ndarray, sampled: dict) -> np.ndarray:
    """Graph indices of (L, 0) and (L, (Ng-1)*dy): front beam x outermost girders."""
    L = float(sampled["L_mm"])
    y_far = (int(sampled["Ng"]) - 1) * float(sampled["dy_mm"])
    idx = []
    for y in (0.0, y_far):
        d = np.hypot(coords_xy[:, 0] - L, coords_xy[:, 1] - y)
        i = int(np.argmin(d))
        if d[i] > CORNER_TOL_MM:
            raise ValueError(f"no graph node within {CORNER_TOL_MM} mm of ({L:.1f}, {y:.1f}); "
                             f"nearest is {d[i]:.1f} mm away")
        idx.append(i)
    if idx[0] == idx[1]:
        raise ValueError("both corners resolved to the same node")
    return np.asarray(idx, dtype=np.int64)


def node_psd_features(sig: np.ndarray, fs: float) -> np.ndarray:
    """(n, 400) row-normalised Welch PSD, the synthetic pipeline's convention."""
    window = get_window("hamming", WELCH_NPERSEG, fftbins=True)
    _, paa = welch_all_nodes(sig, fs, window, nperseg=WELCH_NPERSEG,
                             noverlap=WELCH_NOVERLAP, nfft=WELCH_NFFT)
    return normalize_rows_by_max_abs(paa)[0][:, :PHASE1_FREQ_BINS]


def edge_features(edge_index: np.ndarray, sig: np.ndarray, fs: float,
                  freq_bins: np.ndarray) -> np.ndarray:
    return build_cpsd_full_edge_payload(
        edge_index=edge_index, freq_bins_hz=freq_bins, a3_abs=sig, fs=fs,
        nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP, nfft=WELCH_NFFT,
    )["edge_attr_cpsd_full"]


def resimulate(sampled: dict, selected_idx: np.ndarray) -> dict:
    """The generator's broadband simulation, returning the time series it discards."""
    p = build_solver_params(sampled, n_candidate_modes=N_CANDIDATE_MODES)
    solve_res = run_solver(p, ref_x_fracs=REF_X_FRACS, tol_x=REF_SENSOR_TOL_X)
    graph_payload = build_projected_graph_from_model(solve_res.model, include_sensor_map=False)
    sim_model = dict(solve_res.model)
    sim_model["M_full"] = solve_res.M_full
    sim = simulate_white_noise_from_modal_model(
        sim_model, solve_res.phi_full, solve_res.freq_all_hz, selected_idx,
        zeta=PHASE1_ZETA, plot=False, return_time_series=True,
        use_all_modes=True, bandpass_hz=BANDPASS_HZ,
    )
    graph_node_ids = np.asarray(graph_payload["graph_node_ids"], dtype=np.int64)
    rows = graph_node_ids - 1
    return {
        "a3": np.asarray(sim["a3_abs"], dtype=np.float64)[rows, :],
        "fs": float(np.asarray(sim["fs"]).reshape(())),
        "graph_node_ids": graph_node_ids,
        "edge_index": np.asarray(graph_payload["edge_index"], dtype=np.int64),
        "freq_selected_hz": np.asarray(solve_res.freq_all_hz, dtype=float)[selected_idx],
    }


def infer(model, freq_stats, norm_stats, edge_index, x, ea, n_nodes):
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=n_nodes)
    g.ndata["x"] = torch.tensor(np.asarray(x, dtype=np.float32))
    g.edata["edge_x"] = torch.tensor(np.asarray(ea, dtype=np.float32))
    apply_input_normalization(g, norm_stats)
    g = g.to(DEVICE)
    with torch.no_grad():
        out = model(g, g.ndata["x"], g.edata["edge_x"])
        shapes = out.pred_mode_shape_abs.detach().cpu().numpy()
        freq = denormalize_frequency_predictions(out.pred_freq, freq_stats)
    return shapes, freq.detach().cpu().numpy().reshape(-1)


def run_sample(npz_path: Path, models: dict) -> tuple[list[dict], dict, dict]:
    t0 = time.time()
    with np.load(npz_path, allow_pickle=True) as d:
        sampled = _parse_params_json(d)
        stored = {k: d[k] for k in ("graph_node_ids", "edge_index", "coords_xy_mm", "x_psd",
                                    "edge_attr_cpsd_full", "y_mode_shape_signed", "y_freq_hz",
                                    "matched_mode_indices", "matched_freq_hz", "freq_bins_hz")}
    edge_index = stored["edge_index"].astype(np.int64)
    coords = stored["coords_xy_mm"].astype(np.float64)
    target = stored["y_mode_shape_signed"].astype(np.float64)
    true_freq = stored["y_freq_hz"].astype(np.float64).reshape(-1)
    freq_bins = stored["freq_bins_hz"].astype(np.float64)
    n = int(stored["graph_node_ids"].size)

    sim = resimulate(sampled, stored["matched_mode_indices"].astype(int))
    assert np.array_equal(sim["graph_node_ids"], stored["graph_node_ids"]), "node id drift"
    assert np.array_equal(sim["edge_index"], edge_index), "edge_index drift"
    a3, fs = sim["a3"], sim["fs"]

    # ---- reproduction of the stored features ---------------------------------
    x_resim = node_psd_features(a3, fs)
    ea_resim = edge_features(edge_index, a3, fs, freq_bins)
    diag = {
        "sample": npz_path.stem,
        "Ng": int(sampled["Ng"]), "Nx": int(sampled["Nx"]),
        "L_mm": float(sampled["L_mm"]), "dy_mm": float(sampled["dy_mm"]),
        "balcony": bool(sampled["balcony"]["enable"]),
        "diag_pattern": str(sampled["diag_pattern"]),
        "n_nodes": n, "n_edges_dir": int(edge_index.shape[1]),
        "freq_resim_maxdiff_hz": float(np.abs(sim["freq_selected_hz"]
                                              - stored["matched_freq_hz"]).max()),
        "x_psd_resim_maxdiff": float(np.abs(x_resim - stored["x_psd"]).max()),
        "edge_resim_maxdiff": float(np.abs(ea_resim - stored["edge_attr_cpsd_full"]).max()),
    }

    # ---- two front corners ----------------------------------------------------
    corners = find_front_corners(coords, sampled)
    a3_c = a3[corners]                                                  # (2, nt)

    # node features: PSD at the corners, propagate the spectra, then normalise
    window = get_window("hamming", WELCH_NPERSEG, fftbins=True)
    _, psd_c = welch_all_nodes(a3_c, fs, window, nperseg=WELCH_NPERSEG,
                               noverlap=WELCH_NOVERLAP, nfft=WELCH_NFFT)
    xp = np.zeros((n, psd_c.shape[1]), dtype=np.float32)
    xp[corners] = psd_c
    psd_prop = propagate(xp, corners, edge_index, n).astype(np.float64)
    x_prop = normalize_rows_by_max_abs(psd_prop)[0][:, :PHASE1_FREQ_BINS]

    # edge features: propagate the time series, then coherency
    xt = np.zeros((n, a3_c.shape[1]), dtype=np.float32)
    xt[corners] = a3_c
    ts_prop = propagate(xt, corners, edge_index, n).astype(np.float64)
    ea_prop = edge_features(edge_index, ts_prop, fs, freq_bins)

    # reach: how much signal the farthest node receives, relative to the corners
    rms = np.sqrt(np.mean(ts_prop ** 2, axis=1))
    diag.update({
        "corner_idx": f"{corners[0]};{corners[1]}",
        "prop_min_rms_ratio": float(rms.min() / rms[corners].mean()),
        "prop_n_unreached": int(np.sum(rms == 0.0)),
    })

    # control: one corner's record at every node
    a3_copy = np.repeat(a3_c[:1], n, axis=0)
    x_copy = node_psd_features(a3_copy, fs)
    ea_copy = edge_features(edge_index, a3_copy, fs, freq_bins)

    inputs = {
        "full": (stored["x_psd"], stored["edge_attr_cpsd_full"]),
        "corners_prop": (x_prop, ea_prop),
        "corner_copy": (x_copy, ea_copy),
        "zeros": (np.zeros_like(stored["x_psd"]), np.zeros_like(stored["edge_attr_cpsd_full"])),
    }
    diag["corners_prop_x_rowmax_min"] = float(np.abs(x_prop).max(axis=1).min())
    diag["corners_prop_edge_absmax"] = float(np.abs(ea_prop).max())

    # operator ceiling: propagate the true mode shape from its two corner values
    ys = np.zeros_like(target, dtype=np.float32)
    ys[corners] = target[corners]
    prop_shape = propagate(ys, corners, edge_index, n).astype(np.float64)

    rows: list[dict] = []
    preds: dict[str, np.ndarray] = {}
    for ck, (model, freq_stats, norm_stats) in models.items():
        for v in VARIANTS:
            x, ea = inputs[v]
            shapes, freq = infer(model, freq_stats, norm_stats, edge_index, x, ea, n)
            preds[f"{ck}_{v}"] = shapes
            preds[f"{ck}_{v}_freq"] = freq
            for m, name in enumerate(MODE_NAMES):
                rows.append({
                    "sample": npz_path.stem, "checkpoint": ck, "variant": v, "mode": name,
                    "mac": mac(shapes[:, m], target[:, m]),
                    "pred_freq_hz": float(freq[m]), "true_freq_hz": float(true_freq[m]),
                    "abs_freq_err_hz": float(abs(freq[m] - true_freq[m])),
                    "prop_shape_mac": mac(prop_shape[:, m], target[:, m]),
                })
    diag["elapsed_s"] = round(time.time() - t0, 1)
    raw = {"coords": coords, "target": target, "true_freq": true_freq,
           "corners": corners, "prop_shape": prop_shape, "edge_index": edge_index, **preds}
    return rows, diag, raw


def load_models() -> dict:
    models = {}
    norm_stats = load_input_normalization(RUN_DIR)
    for ck in CHECKPOINTS:
        path = RUN_DIR / "checkpoints" / f"{ck}_model.pt"
        model, checkpoint = load_trained_model_checkpoint(path, device=DEVICE)
        model.eval()
        freq_stats = load_frequency_normalization(checkpoint_path=path, checkpoint=checkpoint,
                                                  run_dir=RUN_DIR)
        models[ck] = (model, freq_stats, norm_stats)
    return models


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N test samples only")
    ap.add_argument("--dataset-root", type=Path, default=None,
                    help="default: dataset_root from the run's config.json")
    args = ap.parse_args()

    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible - this repo runs CPU torch only (CLAUDE.md rule 2)")
    set_global_seed(20260924)

    cfg = json.loads((RUN_DIR / "config.json").read_text(encoding="utf-8"))
    root = args.dataset_root or Path(cfg["dataset_root"])
    test_ids = [s.strip() for s in (RUN_DIR / "test_split.txt").read_text().splitlines() if s.strip()]
    if args.limit:
        test_ids = test_ids[: args.limit]
    print(f"run: {RUN_DIR.name}  dataset: {root}  test structures: {len(test_ids)}")

    models = load_models()
    (OUT_DIR / "raw").mkdir(exist_ok=True)
    all_rows: list[dict] = []
    diags: list[dict] = []
    for i, sid in enumerate(test_ids, 1):
        rows, diag, raw = run_sample(root / sid, models)
        all_rows.extend(rows)
        diags.append(diag)
        np.savez_compressed(OUT_DIR / "raw" / f"{Path(sid).stem}.npz", **raw)
        prim = {r["mode"]: r["mac"] for r in rows
                if r["checkpoint"] == "last" and r["variant"] == "corners_prop"}
        full = {r["mode"]: r["mac"] for r in rows
                if r["checkpoint"] == "last" and r["variant"] == "full"}
        print(f"[{i:>2}/{len(test_ids)}] {Path(sid).stem}  "
              f"resim dx={diag['x_psd_resim_maxdiff']:.1e} de={diag['edge_resim_maxdiff']:.1e}  "
              f"full " + " ".join(f"{full[m]:.3f}" for m in MODE_NAMES) +
              "  corners " + " ".join(f"{prim[m]:.3f}" for m in MODE_NAMES) +
              f"  ({diag['elapsed_s']}s)", flush=True)

    write_csv(OUT_DIR / "stage1_per_sample.csv", all_rows)
    write_csv(OUT_DIR / "stage1_diagnostics.csv", diags)
    print(f"\nwrote {OUT_DIR / 'stage1_per_sample.csv'} and stage1_diagnostics.csv")
    for ck in CHECKPOINTS:
        print(f"\n--- {ck}_model.pt: mean MAC over {len(test_ids)} structures ---")
        for v in VARIANTS:
            vals = [np.mean([r["mac"] for r in all_rows if r["checkpoint"] == ck
                             and r["variant"] == v and r["mode"] == m]) for m in MODE_NAMES]
            print(f"  {v:<13} " + " ".join(f"{m}={x:.3f}" for m, x in zip(MODE_NAMES, vals)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
