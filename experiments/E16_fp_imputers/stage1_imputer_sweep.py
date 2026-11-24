"""E16 stage 1 - would another feature-propagation algorithm need fewer sensors?

    python tools/run_stage.py experiments/E16_fp_imputers/stage1_imputer_sweep.py --workers 12

Final_results_for_paper's s8 measures the main network (edgegat_N{MAIN_N},
best_model) on the fixed 500-structure test set when only n stations per girder
line are measured and every other node is filled by the production operator -
Rossi et al.'s Feature Propagation, D^-1/2 A D^-1/2, 50 iterations.  This stage
repeats s8 node for node and swaps that operator for each model-free imputer
benchmarked in github.com/lbtung96/Feature-propagation (imputers.py here):

  Zero, Mean, NeighborMean   naive baselines
  FP                         production operator - the s8 reference
  FP-rw                      D^-1 A, 50 iterations
  Harmonic                   exact harmonic extension (FP-rw run to convergence)
  Tikhonov                   (M + L_sym) x = M y, known values not pinned
  PCFI                       pseudo-confidence FP (Um et al. 2023)

Nothing else changes: the generator's FE solve and broadband simulation rebuilt
from params_json, the station layouts (`outer`: the two outermost girders, 2n
sensors; `all`: every girder, Ng x n), n = 1..12, E5's feature order (Welch PSD
at the sensors -> impute -> row-normalise; time series imputed -> coherency),
the frozen network, one thread per worker.

Two measurements per structure x family x level x imputer:
  per_sample.csv  the network's MAC and |frequency error| on the imputed inputs
                  (what fig3_sensor_sweep plots), and how far those inputs sit from
                  the stored full ones (relative Frobenius error: node PSD over the
                  unmeasured nodes, edge coherency over every edge)
  ceiling.csv     MAC of the imputed TRUE mode shape - the imputer alone, no network
                  (the FP repository's modal benchmark), with the hinge line left
                  free (as for the network's inputs) and clamped to zero (the
                  repository's "supports known at rest").  The clamp is for mode
                  shapes only: the inputs are absolute accelerations, and the hinge
                  nodes carry the base excitation (their PSD is ~100x the deck's).

`full` and `zeros` (every node measured / nothing) are scored once per structure.
make_report.py checks the FP rows against s8_sensor_sweep_<family>.csv.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from functools import partial  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.signal import get_window  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "experiments" / "E14_corner_sensors"))
sys.path.insert(0, str(HERE))

import fr_common as C  # noqa: E402
import imputers as IM  # noqa: E402
import stage2_n10000_layouts as S2  # noqa: E402
import stage3_sensor_sweep as S3  # noqa: E402
from generate_population_dataset_modes4 import PHASE1_FREQ_BINS, _parse_params_json  # noqa: E402
from simulate_uniform_inertial_u3_white_noise import (  # noqa: E402
    WELCH_NFFT,
    WELCH_NOVERLAP,
    WELCH_NPERSEG,
    normalize_rows_by_max_abs,
    welch_all_nodes,
)
from stage1_corner_propagation import (  # noqa: E402
    CORNER_TOL_MM,
    MODE_NAMES,
    edge_features,
    infer,
    mac,
    node_psd_features,
    propagate,
    resimulate,
)

IMPUTERS = ("Zero", "Mean", "NeighborMean", "FP") + IM.METHODS[3:]
LEVELS = (1, 2, 3, 4, 5, 6, 8, 10, 12)
FAMILIES = ("outer", "all")


def init_worker(run_dir: str) -> None:
    """As s8: E14's stage 2 pointed at our network, best checkpoint, one thread."""
    S2.RUN_DIR = Path(run_dir)
    S2.CHECKPOINTS = ("best",)
    S2._init_worker()


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def run_sample(npz_path: str, levels: tuple[int, ...], families: tuple[str, ...],
               imputers: tuple[str, ...]) -> tuple:
    t0 = time.time()
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as d:
        sampled = _parse_params_json(d)
        stored = {k: d[k] for k in ("graph_node_ids", "edge_index", "coords_xy_mm", "x_psd",
                                    "edge_attr_cpsd_full", "y_mode_shape_signed", "y_freq_hz",
                                    "matched_mode_indices", "matched_freq_hz", "freq_bins_hz")}
    sample, ng = npz_path.stem, int(sampled["Ng"])
    edge_index = stored["edge_index"].astype(np.int64)
    coords = stored["coords_xy_mm"].astype(np.float64)
    target = stored["y_mode_shape_signed"].astype(np.float64)
    true_freq = stored["y_freq_hz"].astype(np.float64).reshape(-1)
    freq_bins = stored["freq_bins_hz"].astype(np.float64)
    x_full, ea_full = stored["x_psd"], stored["edge_attr_cpsd_full"]
    n = int(stored["graph_node_ids"].size)
    hinge = np.where(np.abs(coords[:, 0]) < CORNER_TOL_MM)[0]
    assert hinge.size == ng and not np.any(target[hinge]), f"{sample}: hinge line is not at rest"

    sim = resimulate(sampled, stored["matched_mode_indices"].astype(int))
    assert np.array_equal(sim["graph_node_ids"], stored["graph_node_ids"]), "node id drift"
    assert np.array_equal(sim["edge_index"], edge_index), "edge_index drift"
    a3, fs = sim["a3"], sim["fs"]
    diag = {"sample": sample, "Ng": ng, "n_nodes": n, "n_samples": int(a3.shape[1]),
            "x_psd_resim_maxdiff": float(np.abs(node_psd_features(a3, fs) - x_full).max()),
            "freq_resim_maxdiff_hz": float(np.abs(sim["freq_selected_hz"]
                                                  - stored["matched_freq_hz"]).max())}

    model, freq_stats, norm_stats = S2._MODELS["best"]
    rows, ceil_rows = [], []

    def score(family, imputer, variant, level, ns, x, ea, extra):
        shapes, freq = infer(model, freq_stats, norm_stats, edge_index, x, ea, n)
        r = {"sample": sample, "Ng": ng, "family": family, "imputer": imputer,
             "variant": variant, "level": level, "n_sensors": ns, "n_nodes": n}
        r.update({f"mac_{m}": mac(shapes[:, i], target[:, i]) for i, m in enumerate(MODE_NAMES)})
        r.update({f"ferr_{m}": float(abs(freq[i] - true_freq[i])) for i, m in enumerate(MODE_NAMES)})
        r.update(extra)
        rows.append(r)

    score("-", "-", "full", 0, n, x_full, ea_full, {})
    score("-", "-", "zeros", 0, 0, np.zeros_like(x_full), np.zeros_like(ea_full), {})

    window = get_window("hamming", WELCH_NPERSEG, fftbins=True)
    for fam in families:
        for lv in levels:
            s = S3.station_nodes(coords, sampled, lv, fam)
            unk = np.setdiff1d(np.arange(n), s)
            clamp = np.concatenate([s, hinge])
            # exactly S2.propagated_inputs up to the operator: float32 rows, sensors only
            a3_s = a3[s]
            _, psd_s = welch_all_nodes(a3_s, fs, window, nperseg=WELCH_NPERSEG,
                                       noverlap=WELCH_NOVERLAP, nfft=WELCH_NFFT)
            xp = np.zeros((n, psd_s.shape[1]), dtype=np.float32)
            xp[s] = psd_s
            xt = np.zeros((n, a3_s.shape[1]), dtype=np.float32)
            xt[s] = a3_s
            ys = np.zeros_like(target, dtype=np.float32)
            ys[s] = target[s]
            ys_c = ys.copy()                               # hinge rows stay 0: at rest
            for imp in imputers:
                if imp == "FP":
                    psd_imp = propagate(xp, s, edge_index, n).astype(np.float64)
                    ts_imp = propagate(xt, s, edge_index, n).astype(np.float64)
                    ceil = {"free": propagate(ys, s, edge_index, n),
                            "clamped": propagate(ys_c, clamp, edge_index, n)}
                else:
                    W, xi = IM.operator(imp, edge_index, n, s)
                    psd_imp = IM.apply(imp, W, xi, s, xp[s], refine=True)
                    ts_imp = IM.apply(imp, W, xi, s, xt[s], refine=False)
                    Wc, xic = IM.operator(imp, edge_index, n, clamp)
                    ceil = {"free": IM.apply(imp, W, xi, s, ys[s], refine=True),
                            "clamped": IM.apply(imp, Wc, xic, clamp, ys_c[clamp], refine=True)}
                x = normalize_rows_by_max_abs(psd_imp)[0][:, :PHASE1_FREQ_BINS]
                ea = edge_features(edge_index, ts_imp, fs, freq_bins)
                rms = np.sqrt(np.mean(ts_imp ** 2, axis=1))
                if imp == "FP":
                    assert np.all(rms > 0.0), f"{sample} n={lv}: unreached nodes"
                score(fam, imp, f"n{lv}", lv, int(s.size), x, ea,
                      {"x_err": rel_err(x[unk], x_full[unk]), "ea_err": rel_err(ea, ea_full),
                       "n_silent": int(np.sum(rms == 0.0))})
                for sup, cs in ceil.items():
                    cs = np.asarray(cs, dtype=np.float64)
                    r = {"sample": sample, "Ng": ng, "family": fam, "imputer": imp,
                         "variant": f"n{lv}", "level": lv, "n_sensors": int(s.size),
                         "supports": sup}
                    r.update({f"mac_{m}": mac(cs[:, i], target[:, i])
                              for i, m in enumerate(MODE_NAMES)})
                    ceil_rows.append(r)
    diag["elapsed_s"] = round(time.time() - t0, 1)
    return rows, ceil_rows, diag


def write(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    ap.add_argument("--levels", type=int, nargs="+", default=list(LEVELS))
    ap.add_argument("--imputers", nargs="+", default=list(IMPUTERS), choices=IMPUTERS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--per-ng", type=int, default=None, help="first K test structures per Ng (pilot)")
    ap.add_argument("--tag", default="", help="prefix for the outputs, so a pilot cannot overwrite the run")
    args = ap.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible - this repo runs CPU torch only (CLAUDE.md rule 2)")

    run = C.RUNS / C.sweep_run_name(C.MAIN_N)
    test = [s.strip() for s in (C.SPLITS / "test.txt").read_text(encoding="utf-8").splitlines() if s.strip()]
    if args.per_ng:
        test = [s for g in C.NG_VALUES for s in [t for t in test if f"_Ng{g}_" in t][: args.per_ng]]
    paths = [str(C.POP_BROADBAND / s) for s in test]
    job = partial(run_sample, levels=tuple(args.levels), families=tuple(args.families),
                  imputers=tuple(args.imputers))
    print(f"{run.name}: {len(paths)} structures, families {args.families}, levels {args.levels}, "
          f"{len(args.imputers)} imputers, {args.workers} workers", flush=True)
    rows, ceil_rows, diags, t0 = [], [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(run),)) as pool:
        for i, (r, cr, dg) in enumerate(pool.map(job, paths), 1):
            rows.extend(r)
            ceil_rows.extend(cr)
            diags.append(dg)
            if i % 25 == 0 or i == len(paths):
                el = time.time() - t0
                print(f"  [{i:>4}/{len(paths)}] {el / 60:.1f} min, "
                      f"~{el / i * (len(paths) - i) / 60:.1f} min left", flush=True)
    pre = f"{args.tag}_" if args.tag else ""
    write(HERE / f"{pre}per_sample.csv", rows)
    write(HERE / f"{pre}ceiling.csv", ceil_rows)
    write(HERE / f"{pre}diagnostics.csv", diags)
    print(f"re-simulation reproduces stored node features to "
          f"{max(d['x_psd_resim_maxdiff'] for d in diags):.2e}")
    for fam in args.families:
        print(f"\n[{fam}] mean MAC over B1, T1, B2, T2 (network)")
        print("  imputer       " + " ".join(f"{f'n{lv}':>6}" for lv in args.levels))
        for imp in args.imputers:
            vals = [np.mean([np.mean([r[f"mac_{m}"] for m in MODE_NAMES]) for r in rows
                             if r["family"] == fam and r["imputer"] == imp and r["level"] == lv])
                    for lv in args.levels]
            print(f"  {imp:<13} " + " ".join(f"{v:6.3f}" for v in vals), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
