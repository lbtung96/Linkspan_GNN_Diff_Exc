"""E14 stage 3 - how many measured nodes does the N = 10 000 network need?

Sweeps the number of measured nodes on `sensitivity_N10000_ep500`'s own held-out
structures, every other node filled by the production propagation operator
exactly as in stages 1-2 (helpers imported from `stage2_n10000_layouts`, which
imports stage 1's).

Layouts: n stations per girder line, evenly spaced back from the front beam,

    x_k = L * (1 - k / n),   k = 0 .. n-1,

each snapped to the nearest node on that girder line (0 < x <= L; ties to the
smaller x, distances rounded to 1e-6 L first). The front beam (x = L) is always
measured and the hinge line (x = 0, the supports) never is - the deck moves least
there. `--family outer` puts the stations on the two outermost girders only
(2n sensors at every Ng); `--family all` on every girder (Ng * n sensors).

n = 1 and n = 2 on the outer girders are stage 2's `corners2` and
`corners2_mid2` node for node, so those two levels must reproduce stage 2 exactly;
the report checks it.

Outputs (this folder only; `--tag` prefixes them, so a pilot cannot overwrite the
recorded run):
  <tag>_per_sample.csv    one row per structure x checkpoint x variant, wide:
                          mac_<mode>, ferr_<mode> (Hz), n_sensors, n_nodes
  <tag>_diagnostics.csv   one row per structure: reproduction, station x/L
  raw_<tag>/<sample>.npz  coords, edges, the sensor set of every level (figures)
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
from functools import partial  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import stage2_n10000_layouts as S2  # noqa: E402
from generate_population_dataset_modes4 import _parse_params_json  # noqa: E402
from stage1_corner_propagation import (  # noqa: E402
    CORNER_TOL_MM,
    MODE_NAMES,
    edge_features,
    infer,
    mac,
    node_psd_features,
    resimulate,
    write_csv,
)

RUN_DIR = S2.RUN_DIR


def girder_ys(sampled: dict, family: str) -> list[float]:
    ng, dy = int(sampled["Ng"]), float(sampled["dy_mm"])
    js = [0, ng - 1] if family == "outer" else list(range(ng))
    return [j * dy for j in js]


def station_nodes(coords_xy: np.ndarray, sampled: dict, n: int, family: str) -> np.ndarray:
    """Nodes at x = L(1 - k/n), k = 0..n-1, on each selected girder line."""
    L = float(sampled["L_mm"])
    out = []
    for y in girder_ys(sampled, family):
        on = np.where((np.abs(coords_xy[:, 1] - y) < CORNER_TOL_MM)
                      & (coords_xy[:, 0] > 0.0) & (coords_xy[:, 0] <= L + CORNER_TOL_MM))[0]
        for k in range(n):
            x_target = L * (1.0 - k / n)
            dist = np.round(np.abs(coords_xy[on, 0] - x_target) / L, 6)
            order = np.lexsort((coords_xy[on, 0], dist))
            out.append(int(on[order[0]]))
    out = np.asarray(out, dtype=np.int64)
    if np.unique(out).size != out.size:
        raise ValueError(f"n = {n}: two stations snapped to the same node")
    return out


def run_sample(npz_path: str, levels: tuple[int, ...], family: str, raw_dir: str) -> tuple:
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
    n = int(stored["graph_node_ids"].size)

    sim = resimulate(sampled, stored["matched_mode_indices"].astype(int))
    assert np.array_equal(sim["graph_node_ids"], stored["graph_node_ids"]), "node id drift"
    assert np.array_equal(sim["edge_index"], edge_index), "edge_index drift"
    a3, fs = sim["a3"], sim["fs"]
    x_resim = node_psd_features(a3, fs)

    diag = {"sample": sample, "Ng": ng, "Nx": int(sampled["Nx"]), "n_nodes": n,
            "balcony": bool(sampled["balcony"]["enable"]),
            "x_psd_resim_maxdiff": float(np.abs(x_resim - stored["x_psd"]).max()),
            "freq_resim_maxdiff_hz": float(np.abs(sim["freq_selected_hz"]
                                                  - stored["matched_freq_hz"]).max())}
    inputs = {"full": (stored["x_psd"], stored["edge_attr_cpsd_full"], 0),
              "zeros": (np.zeros_like(stored["x_psd"]),
                        np.zeros_like(stored["edge_attr_cpsd_full"]), 0)}
    sensors_by_level = {}
    for lv in levels:
        s = station_nodes(coords, sampled, lv, family)
        sensors_by_level[lv] = s
        x, ea, _, unreached = S2.propagated_inputs(a3, fs, s, edge_index, freq_bins)
        assert unreached == 0, f"{sample} n={lv}: unreached nodes"
        inputs[f"n{lv}"] = (x, ea, int(s.size))
        diag[f"x_over_L_n{lv}"] = ";".join(f"{v:.3f}" for v in np.unique(coords[s, 0])
                                           / float(sampled["L_mm"]))

    rows = []
    for ck, (model, freq_stats, norm_stats) in S2._MODELS.items():
        for v, (x, ea, ns) in inputs.items():
            shapes, freq = infer(model, freq_stats, norm_stats, edge_index, x, ea, n)
            r = {"sample": sample, "Ng": ng, "checkpoint": ck, "variant": v,
                 "n_sensors": ns if v.startswith("n") else (n if v == "full" else 0),
                 "n_nodes": n}
            for m, name in enumerate(MODE_NAMES):
                r[f"mac_{name}"] = mac(shapes[:, m], target[:, m])
            for m, name in enumerate(MODE_NAMES):
                r[f"ferr_{name}"] = float(abs(freq[m] - true_freq[m]))
            rows.append(r)
    diag["elapsed_s"] = round(time.time() - t0, 1)
    np.savez_compressed(Path(raw_dir) / f"{sample}.npz", coords=coords, edge_index=edge_index,
                        **{f"sensors_n{lv}": s for lv, s in sensors_by_level.items()})
    return rows, diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10])
    ap.add_argument("--family", choices=("outer", "all"), default="outer")
    ap.add_argument("--per-ng", type=int, default=None,
                    help="first K test structures of each Ng only (pilot)")
    ap.add_argument("--tag", default="stage3")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible - this repo runs CPU torch only (CLAUDE.md rule 2)")

    cfg = json.loads((RUN_DIR / "config.json").read_text(encoding="utf-8"))
    root = Path(cfg["dataset_root"])
    test_ids = [s.strip() for s in (RUN_DIR / "test_split.txt").read_text().splitlines() if s.strip()]
    if args.per_ng:
        test_ids = [s for g in (2, 3, 4, 5) for s in
                    [t for t in test_ids if f"_Ng{g}_" in t][: args.per_ng]]
    raw_dir = HERE / f"raw_{args.tag}"
    raw_dir.mkdir(exist_ok=True)
    print(f"run: {RUN_DIR.name}  family: {args.family}  levels: {args.levels}  "
          f"structures: {len(test_ids)}  workers: {args.workers}", flush=True)

    job = partial(run_sample, levels=tuple(args.levels), family=args.family, raw_dir=str(raw_dir))
    rows, diags, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=S2._init_worker) as pool:
        for i, (r, dg) in enumerate(pool.map(job, [str(root / s) for s in test_ids]), 1):
            rows.extend(r)
            diags.append(dg)
            if i % 25 == 0 or i == len(test_ids):
                el = time.time() - t0
                print(f"[{i:>4}/{len(test_ids)}] {el / 60:.1f} min, "
                      f"~{el / i * (len(test_ids) - i) / 60:.1f} min left", flush=True)

    write_csv(HERE / f"{args.tag}_per_sample.csv", rows)
    write_csv(HERE / f"{args.tag}_diagnostics.csv", diags)
    print(f"\nwrote {args.tag}_per_sample.csv, {args.tag}_diagnostics.csv")
    print("\n--- last_model.pt, mean MAC (pooled over Ng) ---")
    for v in ["full"] + [f"n{lv}" for lv in args.levels] + ["zeros"]:
        sel = [r for r in rows if r["checkpoint"] == "last" and r["variant"] == v]
        vals = [np.mean([r[f"mac_{m}"] for r in sel]) for m in MODE_NAMES]
        ns = np.mean([r["n_sensors"] for r in sel])
        pct = np.mean([r["n_sensors"] / r["n_nodes"] for r in sel]) * 100
        print(f"  {v:<6} sensors {ns:6.1f} ({pct:5.1f}% of nodes)  "
              + " ".join(f"{m}={x:.3f}" for m, x in zip(MODE_NAMES, vals))
              + f"  mean={np.mean(vals):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
