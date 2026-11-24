"""E14 stage 2 - the N = 10 000 network (all four girder counts), two sensor layouts.

Stage 1 used `sensitivity_N600_ep500`, which has only ever seen Ng = 2 decks
(the 10k dataset is ordered by girder count; `max_samples = 600` takes the first
600 files). This stage repeats the test on `sensitivity_N10000_ep500` - broadband,
seed 123, 500 epochs, trained on all 10 000 samples, Ng = 2-5 - over its own
1 500 held-out structures, and adds a second sensor layout.

Layouts
-------
  corners2       the two front corners: (L, 0) and (L, (Ng-1)*dy), i.e. the
                 front beam x the two outermost girders (as stage 1)
  corners2_mid2  those two plus, on each outermost girder, the girder node
                 nearest midspan x = L/2 (ties to the smaller x)

Everything else is stage 1's path, imported from `stage1_corner_propagation`
unchanged: the generator's FE solve and broadband simulation rebuilt from
`params_json`, the reproduction check against the stored features, the
production operator (`feature_propagation`, 50 iterations) in E5's order 'A'
(node PSD at the sensors -> propagate -> row-normalise; time series propagated ->
coherency), and inference on CPU with `last_model.pt` (primary) and
`best_model.pt` (robustness).

Variants per structure: `full` (stored test input), `corners2_prop`,
`corners2_mid2_prop`, `zeros` (the flat floor). `corner_copy` is dropped: in
stage 1 it scored identically to `zeros` for mode shapes (both give a spatially
flat prediction), and the frequency baseline it served is replaced in the report
by the constant training-mean guess, which needs no network.

Structures run in parallel worker processes (one BLAS/torch thread each); the
per-structure work is identical to a serial run, and results are written in the
order of `test_split.txt`.

Outputs (this folder only)
--------------------------
  stage2_per_sample.csv   structure x checkpoint x variant x mode: MAC, frequency
  stage2_layouts.csv      structure x layout x mode: sensor amplitude / peak,
                          operator ceiling MAC, sensor node indices
  stage2_diagnostics.csv  structure: reproduction, reach, chosen midspan x/L
  raw_n10000/<sample>.npz coords, targets, predictions, sensor nodes (figures)
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.signal import get_window  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from generate_population_dataset_modes4 import PHASE1_FREQ_BINS, _parse_params_json  # noqa: E402
from simulate_uniform_inertial_u3_white_noise import (  # noqa: E402
    WELCH_NFFT,
    WELCH_NOVERLAP,
    WELCH_NPERSEG,
    normalize_rows_by_max_abs,
    welch_all_nodes,
)
from GraphSAGE_Modal_GNN import set_global_seed  # noqa: E402
from EdgeGATConv_Modal_GNN import DEVICE, load_trained_model_checkpoint  # noqa: E402
from infer_phase2a_unseen_structure import load_frequency_normalization  # noqa: E402
from infer_phase2a_unseen_structure_edgegat import load_input_normalization  # noqa: E402
from stage1_corner_propagation import (  # noqa: E402
    CORNER_TOL_MM,
    MODE_NAMES,
    edge_features,
    find_front_corners,
    infer,
    mac,
    node_psd_features,
    propagate,
    resimulate,
    write_csv,
)

RUN_DIR = REPO / "runs_phase2" / "EdgeGATConv" / "sensitivity_N10000_ep500"
CHECKPOINTS = ("last", "best")
LAYOUTS = ("corners2", "corners2_mid2")
VARIANTS = ("full",) + tuple(f"{lay}_prop" for lay in LAYOUTS) + ("zeros",)
RAW_DIR = HERE / "raw_n10000"


def find_midspan_nodes(coords_xy: np.ndarray, sampled: dict) -> tuple[np.ndarray, np.ndarray]:
    """On each outermost girder, the node nearest x = L/2 (ties to the smaller x)."""
    L = float(sampled["L_mm"])
    y_far = (int(sampled["Ng"]) - 1) * float(sampled["dy_mm"])
    idx = []
    for y in (0.0, y_far):
        on = np.where((np.abs(coords_xy[:, 1] - y) < CORNER_TOL_MM)
                      & (coords_xy[:, 0] > 0.0) & (coords_xy[:, 0] < L))[0]
        if on.size == 0:
            raise ValueError(f"no interior node on the girder at y = {y:.1f}")
        # odd Nx puts two nodes exactly equidistant from L/2; without rounding,
        # float error decides the tie (it went to the larger x in 192 of 836 cases)
        dist = np.round(np.abs(coords_xy[on, 0] - 0.5 * L) / L, 6)
        order = np.lexsort((coords_xy[on, 0], dist))
        idx.append(int(on[order[0]]))
    idx = np.asarray(idx, dtype=np.int64)
    return idx, coords_xy[idx, 0] / L


def propagated_inputs(a3: np.ndarray, fs: float, sensors: np.ndarray,
                      edge_index: np.ndarray, freq_bins: np.ndarray) -> tuple:
    """Stage 1's corner path, for any sensor set: E5 order 'A'."""
    n = a3.shape[0]
    a3_s = a3[sensors]
    window = get_window("hamming", WELCH_NPERSEG, fftbins=True)
    _, psd_s = welch_all_nodes(a3_s, fs, window, nperseg=WELCH_NPERSEG,
                               noverlap=WELCH_NOVERLAP, nfft=WELCH_NFFT)
    xp = np.zeros((n, psd_s.shape[1]), dtype=np.float32)
    xp[sensors] = psd_s
    psd_prop = propagate(xp, sensors, edge_index, n).astype(np.float64)
    x = normalize_rows_by_max_abs(psd_prop)[0][:, :PHASE1_FREQ_BINS]

    xt = np.zeros((n, a3_s.shape[1]), dtype=np.float32)
    xt[sensors] = a3_s
    ts_prop = propagate(xt, sensors, edge_index, n).astype(np.float64)
    ea = edge_features(edge_index, ts_prop, fs, freq_bins)
    rms = np.sqrt(np.mean(ts_prop ** 2, axis=1))
    return x, ea, float(rms.min() / rms[sensors].mean()), int(np.sum(rms == 0.0))


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------
_MODELS: dict = {}


def _init_worker() -> None:
    torch.set_num_threads(1)
    set_global_seed(20260924)
    norm_stats = load_input_normalization(RUN_DIR)
    for ck in CHECKPOINTS:
        path = RUN_DIR / "checkpoints" / f"{ck}_model.pt"
        model, checkpoint = load_trained_model_checkpoint(path, device=DEVICE)
        model.eval()
        freq_stats = load_frequency_normalization(checkpoint_path=path, checkpoint=checkpoint,
                                                  run_dir=RUN_DIR)
        _MODELS[ck] = (model, freq_stats, norm_stats)


def run_sample(npz_path: str) -> tuple[list[dict], list[dict], dict]:
    t0 = time.time()
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as d:
        sampled = _parse_params_json(d)
        stored = {k: d[k] for k in ("graph_node_ids", "edge_index", "coords_xy_mm", "x_psd",
                                    "edge_attr_cpsd_full", "y_mode_shape_signed", "y_freq_hz",
                                    "matched_mode_indices", "matched_freq_hz", "freq_bins_hz")}
    sample = npz_path.stem
    ng = int(sampled["Ng"])
    balcony = bool(sampled["balcony"]["enable"])
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

    x_resim = node_psd_features(a3, fs)
    ea_resim = edge_features(edge_index, a3, fs, freq_bins)
    corners = find_front_corners(coords, sampled)
    mids, mid_xfrac = find_midspan_nodes(coords, sampled)
    sensors = {"corners2": corners, "corners2_mid2": np.concatenate([corners, mids])}
    assert np.unique(sensors["corners2_mid2"]).size == 4, "sensor nodes coincide"

    diag = {
        "sample": sample, "Ng": ng, "Nx": int(sampled["Nx"]), "balcony": balcony,
        "diag_pattern": str(sampled["diag_pattern"]),
        "n_nodes": n, "n_edges_dir": int(edge_index.shape[1]),
        "freq_resim_maxdiff_hz": float(np.abs(sim["freq_selected_hz"]
                                              - stored["matched_freq_hz"]).max()),
        "x_psd_resim_maxdiff": float(np.abs(x_resim - stored["x_psd"]).max()),
        "edge_resim_maxdiff": float(np.abs(ea_resim - stored["edge_attr_cpsd_full"]).max()),
        "corner_idx": f"{corners[0]};{corners[1]}",
        "mid_idx": f"{mids[0]};{mids[1]}",
        "mid_x_over_L_0": float(mid_xfrac[0]), "mid_x_over_L_1": float(mid_xfrac[1]),
    }

    inputs = {"full": (stored["x_psd"], stored["edge_attr_cpsd_full"]),
              "zeros": (np.zeros_like(stored["x_psd"]),
                        np.zeros_like(stored["edge_attr_cpsd_full"]))}
    lay_rows: list[dict] = []
    ceilings: dict[str, np.ndarray] = {}
    peak = np.abs(target).max(axis=0)
    for lay in LAYOUTS:
        s = sensors[lay]
        x, ea, reach, unreached = propagated_inputs(a3, fs, s, edge_index, freq_bins)
        inputs[f"{lay}_prop"] = (x, ea)
        diag[f"{lay}_min_rms_ratio"] = reach
        diag[f"{lay}_n_unreached"] = unreached
        ys = np.zeros_like(target, dtype=np.float32)
        ys[s] = target[s]
        ceil = propagate(ys, s, edge_index, n).astype(np.float64)
        ceilings[lay] = ceil
        for m, name in enumerate(MODE_NAMES):
            lay_rows.append({
                "sample": sample, "Ng": ng, "balcony": balcony, "layout": lay, "mode": name,
                "sensor_amp_over_peak": float(np.abs(target[s, m]).max() / peak[m]),
                "operator_ceiling_mac": mac(ceil[:, m], target[:, m]),
                "sensor_idx": ";".join(str(int(i)) for i in s),
            })

    rows: list[dict] = []
    preds: dict[str, np.ndarray] = {}
    for ck, (model, freq_stats, norm_stats) in _MODELS.items():
        for v in VARIANTS:
            x, ea = inputs[v]
            shapes, freq = infer(model, freq_stats, norm_stats, edge_index, x, ea, n)
            preds[f"{ck}_{v}"] = shapes
            preds[f"{ck}_{v}_freq"] = freq
            for m, name in enumerate(MODE_NAMES):
                rows.append({
                    "sample": sample, "Ng": ng, "balcony": balcony, "checkpoint": ck,
                    "variant": v, "mode": name,
                    "mac": mac(shapes[:, m], target[:, m]),
                    "pred_freq_hz": float(freq[m]), "true_freq_hz": float(true_freq[m]),
                    "abs_freq_err_hz": float(abs(freq[m] - true_freq[m])),
                })
    diag["elapsed_s"] = round(time.time() - t0, 1)
    np.savez_compressed(RAW_DIR / f"{sample}.npz", coords=coords, target=target,
                        true_freq=true_freq, edge_index=edge_index, corners=corners,
                        mids=mids, **{f"ceiling_{k}": v for k, v in ceilings.items()}, **preds)
    return rows, lay_rows, diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N test samples only")
    ap.add_argument("--ng", type=int, nargs="*", default=None, help="restrict to these Ng")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dataset-root", type=Path, default=None,
                    help="default: dataset_root from the run's config.json")
    args = ap.parse_args()

    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible - this repo runs CPU torch only (CLAUDE.md rule 2)")

    cfg = json.loads((RUN_DIR / "config.json").read_text(encoding="utf-8"))
    root = args.dataset_root or Path(cfg["dataset_root"])
    test_ids = [s.strip() for s in (RUN_DIR / "test_split.txt").read_text().splitlines() if s.strip()]
    if args.ng:
        test_ids = [s for s in test_ids if any(f"_Ng{g}_" in s for g in args.ng)]
    if args.limit:
        test_ids = test_ids[: args.limit]
    print(f"run: {RUN_DIR.name}  dataset: {root}  test structures: {len(test_ids)}  "
          f"workers: {args.workers}", flush=True)

    RAW_DIR.mkdir(exist_ok=True)
    all_rows, all_lay, diags = [], [], []
    t0 = time.time()
    paths = [str(root / sid) for sid in test_ids]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        for i, (rows, lay_rows, diag) in enumerate(pool.map(run_sample, paths, chunksize=1), 1):
            all_rows.extend(rows)
            all_lay.extend(lay_rows)
            diags.append(diag)
            if i % 25 == 0 or i == len(paths):
                el = time.time() - t0
                print(f"[{i:>4}/{len(paths)}] {el / 60:.1f} min elapsed, "
                      f"~{el / i * (len(paths) - i) / 60:.1f} min left  "
                      f"(last: {diag['sample']}, resim dx={diag['x_psd_resim_maxdiff']:.1e} "
                      f"de={diag['edge_resim_maxdiff']:.1e})", flush=True)

    write_csv(HERE / "stage2_per_sample.csv", all_rows)
    write_csv(HERE / "stage2_layouts.csv", all_lay)
    write_csv(HERE / "stage2_diagnostics.csv", diags)
    print(f"\nwrote stage2_per_sample.csv, stage2_layouts.csv, stage2_diagnostics.csv")
    for ng in sorted({d["Ng"] for d in diags}) + [None]:
        sel = [r for r in all_rows if r["checkpoint"] == "last" and (ng is None or r["Ng"] == ng)]
        print(f"\n--- last_model.pt, {'all Ng' if ng is None else f'Ng = {ng}'}: mean MAC ---")
        for v in VARIANTS:
            vals = [np.mean([r["mac"] for r in sel if r["variant"] == v and r["mode"] == m])
                    for m in MODE_NAMES]
            print(f"  {v:<20} " + " ".join(f"{m}={x:.3f}" for m, x in zip(MODE_NAMES, vals)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
