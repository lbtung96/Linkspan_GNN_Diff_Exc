"""Stage 4 - in-domain performance of every trained network on the fixed test set.

    python tools/run_stage.py Final_results_for_paper/pipeline/s4_indomain.py [--runs ...]

Every network is scored on the same 500 held-out structures (125 per girder
count, never trained or validated on), on CPU, from its best-validation
checkpoint - the checkpoint every number in this folder uses.  The last-epoch
checkpoint is scored too and written in separate columns, for the record only
(5000 epochs overfits the frequency head; E13 measured this).

MAC and frequency error use the library's own per-sample functions
(_compute_sample_metrics / compute_frequency_metrics), the same ones behind
every test_metrics.csv in this repository.  One training seed, so the spread
reported is over test structures: a bootstrap 95 % interval of the mean
(2000 resamples, fixed seed).

Writes results/s4_indomain_per_sample.csv, results/s4_indomain_summary.csv and
results/s4_indomain_by_ng.csv, and a predictions.npz per run and checkpoint under
runs/<run>/eval_<checkpoint>/ for figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

import fr_common as C
from fr_train import jobs

NAME_RE = re.compile(r"sample_(\d{4})_Ng(\d)_phase1_gnn\.npz$")


def bootstrap_ci(v: np.ndarray, n_boot: int = 2000, seed: int = 20260925) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(run: str, which: str, test_files: list[str]) -> list[dict]:
    import torch
    import EdgeGATConv_Modal_GNN as eg
    import GraphSAGE_Modal_GNN as gs
    import fr_edge_modes

    fr_edge_modes.install()   # the two ablation arms' edge-feature modes; others untouched
    torch.set_num_threads(2)
    job = jobs()[run]
    ck = C.RUNS / run / "checkpoints" / f"{which}.pt"
    dev = torch.device("cpu")
    if job.arch == "edgegat":
        model, ckpt = eg.load_trained_model_checkpoint(ck, device=dev)
        ds = eg.Phase2AEdgeDataset([C.POP_BROADBAND / f for f in test_files],
                                   edge_feature_mode=job.edge_feature_mode,
                                   shape_target_mode=job.shape_target_mode, expected_n_modes=4)
        metric = eg._compute_sample_metrics
    else:
        ckpt = torch.load(ck, map_location=dev, weights_only=False)
        model = gs.build_model_from_config(ckpt["config"]).to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
        ds = gs.Phase1NPZDataset([C.POP_BROADBAND / f for f in test_files], 400, 4,
                                 edge_feature_mode=job.edge_feature_mode,
                                 shape_target_mode=job.shape_target_mode)
        metric = gs._compute_sample_metrics
    model.eval()
    stats = gs.FrequencyNormalizationStats(freq_mean_hz=np.asarray(ckpt["freq_mean_hz"], np.float32),
                                           freq_std_hz=np.asarray(ckpt["freq_std_hz"], np.float32))
    rows, preds = [], {}
    with torch.no_grad():
        for i in range(len(ds)):
            g, f_true = ds[i]
            g = g.to(dev)
            out = model(g, g.ndata["x"], g.edata["edge_x"])
            pred = out.pred_mode_shape_abs.cpu().numpy()
            target = g.ndata["y"].cpu().numpy()
            f_pred = gs.denormalize_frequency_predictions(out.pred_freq, stats).cpu().numpy().reshape(-1)
            f_true = f_true.cpu().numpy().reshape(-1)
            m = metric(pred, target, C.MODE_NAMES)
            fm = gs.compute_frequency_metrics(f_pred[None, :], f_true[None, :], C.MODE_NAMES)
            name = test_files[i]
            r = {"run": run, "checkpoint": which, "sample": name,
                 "Ng": int(NAME_RE.search(name).group(2)), "epoch_of_checkpoint": int(ckpt.get("epoch", -1))}
            for k, mode in enumerate(C.MODE_NAMES):
                r[f"mac_{mode}"] = m[f"mac_{mode}"]
                r[f"freq_pred_{mode}_hz"] = float(f_pred[k])
                r[f"freq_true_{mode}_hz"] = float(f_true[k])
                r[f"freq_abs_err_{mode}_hz"] = float(abs(f_pred[k] - f_true[k]))
            r["mac_mean"] = float(np.mean([m[f"mac_{mode}"] for mode in C.MODE_NAMES]))
            r["freq_mae_hz"] = fm["freq_mae_hz"]
            rows.append(r)
            preds[name] = (pred.astype(np.float32), target.astype(np.float32),
                           f_pred.astype(np.float32), f_true.astype(np.float32))
    out_dir = C.RUNS / run / f"eval_{which}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "predictions.npz",
                        **{f"{k.removesuffix('.npz')}__{nm}": v[j] for k, v in preds.items()
                           for j, nm in enumerate(("pred", "target", "freq_pred", "freq_true"))})
    return rows


def summarise(rows: list[dict], keys: dict) -> dict:
    out = dict(keys)
    out["n_test"] = len(rows)
    for col in [f"mac_{m}" for m in C.MODE_NAMES] + ["mac_mean"] + \
               [f"freq_abs_err_{m}_hz" for m in C.MODE_NAMES] + ["freq_mae_hz"]:
        v = np.array([r[col] for r in rows], float)
        out[col] = float(v.mean())
        if col in ("mac_mean", "freq_mae_hz", "mac_T1"):
            lo, hi = bootstrap_ci(v)
            out[f"{col}_ci95_lo"], out[f"{col}_ci95_hi"] = lo, hi
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None, choices=sorted(jobs()))
    ap.add_argument("--which", nargs="+", default=["best_model", "last_model"],
                    choices=["best_model", "last_model"])
    args = ap.parse_args()
    test_files = [s.strip() for s in (C.SPLITS / "test.txt").read_text(encoding="utf-8").splitlines() if s.strip()]
    runs = [r for r in (args.runs or list(jobs())) if (C.RUNS / r / "run_summary.json").exists()]
    if not runs:
        print("[fail] no completed runs")
        return 1
    # keep rows of runs not re-scored this time, so partial invocations accumulate
    per_path = C.RESULTS / "s4_indomain_per_sample.csv"
    old = []
    if per_path.exists():
        old = [r for r in csv.DictReader(per_path.open(encoding="utf-8"))
               if not (r["run"] in runs and r["checkpoint"] in args.which)]
    new: list[dict] = []
    for run in runs:
        for which in args.which:
            print(f"[eval] {run} {which}", flush=True)
            new += evaluate(run, which, test_files)
    allrows = old + new
    for r in allrows:
        for k, v in list(r.items()):
            if k not in ("run", "checkpoint", "sample"):
                r[k] = float(v) if k not in ("Ng", "epoch_of_checkpoint") else int(float(v))
    with per_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(new[0].keys()))
        w.writeheader(); w.writerows(allrows)

    summ, by_ng = [], []
    for run in sorted({r["run"] for r in allrows}, key=lambda s: (jobs()[s].arch, jobs()[s].n, s)):
        job = jobs()[run]
        info = json.loads((C.RUNS / run / "run_summary.json").read_text(encoding="utf-8"))
        for which in ("best_model", "last_model"):
            rr = [r for r in allrows if r["run"] == run and r["checkpoint"] == which]
            if not rr:
                continue
            keys = {"run": run, "checkpoint": which, "arch": job.arch, "N": job.n,
                    "edge_feature_mode": job.edge_feature_mode, "shape_target_mode": job.shape_target_mode,
                    "epoch_of_checkpoint": rr[0]["epoch_of_checkpoint"], "best_epoch": info["best_epoch"],
                    "epochs": info["epochs"], "device": info["device"], "train_time_h": info["train_time_s"] / 3600}
            summ.append(summarise(rr, keys))
            for g in C.NG_VALUES:
                by_ng.append(summarise([r for r in rr if r["Ng"] == g], {**keys, "Ng": g}))
    for path, rows in ((C.RESULTS / "s4_indomain_summary.csv", summ), (C.RESULTS / "s4_indomain_by_ng.csv", by_ng)):
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    for s in summ:
        print(f"{s['run']:<32} {s['checkpoint']:<11} ep={s['epoch_of_checkpoint']:<5} "
              f"MAC {s['mac_mean']:.3f} [B1 {s['mac_B1']:.3f} T1 {s['mac_T1']:.3f} B2 {s['mac_B2']:.3f} "
              f"T2 {s['mac_T2']:.3f}]  freq MAE {s['freq_mae_hz']:.3f} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
