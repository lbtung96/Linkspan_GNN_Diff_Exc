"""T5 (EC-23) - shape metrics beyond MAC: sign-aligned error, success rates, cross-MAC.

    python tools/run_stage.py pipeline/c_ec23_metrics.py \
        [--runs ...] [--sets paper fresh_test]

Reads the predictions c_rescore.py stored (data/T5/predictions/<set>/<run>__best_model.npz;
best-validation checkpoint).  Default runs: the main network (edgegat_N2000) and the six
ablation-grid arms (fr_common.ABLATION_GRID).  Per structure and mode, over every graph node
(the nodes the paper's MAC is computed on):

  mac          the library's MAC (float64 here; checked against c_rescore's to 1e-5)
  rmse_signed  sign-aligned shape error: the prediction is scaled by its max-abs (the targets'
               own normalisation: y_mode_shape_signed has max |y| = 1 per mode), its global
               sign set by the sign of its dot product with the target, then
               RMSE = sqrt(mean_nodes (s p/max|p| - y)^2).  0 = identical; a shape orthogonal
               to the target gives ~ sqrt(2) x RMS(y).
  cross-MAC    MAC(predicted slot i, target mode j) for all i, j (4 x 4)
  swapped      the predicted slot matches another mode's target better than its own
               (max_j!=i crossMAC[i, j] > crossMAC[i, i])
Success rates: share of structures with MAC > 0.8 and > 0.9, per mode.

Writes ec23_per_sample.csv, ec23_summary.csv (per set x run x mode x Ng-group, with bootstrap
95 % intervals over structures), ec23_crossmac.csv (mean 4 x 4 per set x run).
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

import ec_common as E

C = E.C
from s4_indomain import bootstrap_ci  # noqa: E402

OUT = E.task_dir("T5")
PRED_ROOT = E.EC_DATA / "T5" / "predictions"
M = list(C.MODE_NAMES)
DEFAULT_RUNS = [C.sweep_run_name(C.MAIN_N)] + [C.ABLATION_GRID[(l, f)] for l in C.ABLATION_LAYERS for f in C.ABLATION_FEATURES]


def mac(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v) ** 2 / (np.dot(u, u) * np.dot(v, v) + 1e-30))


def rmse_signed(p: np.ndarray, y: np.ndarray) -> float:
    pn = p / (np.max(np.abs(p)) + 1e-30)
    s = 1.0 if np.dot(pn, y) >= 0 else -1.0
    return float(np.sqrt(np.mean((s * pn - y) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--sets", nargs="+", default=["paper", "fresh_test"])
    args = ap.parse_args()
    ref = {}
    for r in csv.DictReader((OUT / "rescore_per_sample.csv").open(encoding="utf-8")):
        if r["checkpoint"] == "best_model":
            ref[(r["test_set"], r["run"], r["sample"])] = r
    rows, cross_rows, summ = [], [], []
    worst = 0.0
    for ts in args.sets:
        for run in args.runs:
            f = PRED_ROOT / ts / f"{run.replace('/', '__')}__best_model.npz"
            if not f.exists():
                print(f"[skip] {f} missing (run c_rescore.py first)")
                continue
            d = np.load(f)
            cross = []
            for smp in [str(s) for s in d["samples"]]:
                k = smp.removesuffix(".npz")
                p, y = d[f"{k}__pred"].astype(np.float64), d[f"{k}__target"].astype(np.float64)
                X = np.array([[mac(p[:, i], y[:, j]) for j in range(4)] for i in range(4)])
                cross.append(X)
                rr = ref[(ts, run, smp)]
                for i, m in enumerate(M):
                    worst = max(worst, abs(X[i, i] - float(rr[f"mac_{m}"])))
                    off = [j for j in range(4) if j != i]
                    jbest = off[int(np.argmax(X[i, off]))]
                    rows.append({"test_set": ts, "run": run, "sample": smp, "Ng": int(rr["Ng"]), "mode": m,
                                 "mac": X[i, i], "rmse_signed": rmse_signed(p[:, i], y[:, i]),
                                 "best_other_mode": M[jbest], "best_other_mac": X[i, jbest],
                                 "swapped": bool(X[i, jbest] > X[i, i])})
            cm = np.mean(cross, axis=0)
            for i in range(4):
                cross_rows.append({"test_set": ts, "run": run, "pred_slot": M[i],
                                   **{f"target_{M[j]}": float(cm[i, j]) for j in range(4)}})
            for m in M:
                for g in ("all",) + C.NG_VALUES:
                    q = [r for r in rows if r["test_set"] == ts and r["run"] == run and r["mode"] == m
                         and (g == "all" or r["Ng"] == g)]
                    mv = np.array([r["mac"] for r in q])
                    ev = np.array([r["rmse_signed"] for r in q])
                    s8, s9 = (mv > 0.8).astype(float), (mv > 0.9).astype(float)
                    sw = np.array([r["swapped"] for r in q], float)
                    out = {"test_set": ts, "run": run, "mode": m, "Ng": g, "n": len(q), "mac_mean": mv.mean(),
                           "rmse_signed_mean": ev.mean(), "rmse_signed_median": float(np.median(ev)),
                           "rmse_signed_p90": float(np.percentile(ev, 90)),
                           "success_mac_gt_0.8": s8.mean(), "success_mac_gt_0.9": s9.mean(), "swap_rate": sw.mean()}
                    if g == "all":
                        for nm, v in (("rmse_signed_mean", ev), ("success_mac_gt_0.8", s8), ("success_mac_gt_0.9", s9),
                                      ("swap_rate", sw)):
                            out[f"{nm}_ci95_lo"], out[f"{nm}_ci95_hi"] = bootstrap_ci(v)
                    summ.append(out)
            print(f"{ts:<12} {run:<34} " + "  ".join(
                f"{m}: rmse {np.mean([r['rmse_signed'] for r in rows if (r['test_set'], r['run'], r['mode']) == (ts, run, m)]):.3f}"
                for m in M), flush=True)
    for path, rr in ((OUT / "ec23_per_sample.csv", rows), (OUT / "ec23_crossmac.csv", cross_rows),
                     (OUT / "ec23_summary.csv", summ)):
        keys: list[str] = []
        for r in rr:
            keys += [k for k in r if k not in keys]
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rr)
    print(f"[check] float64 MAC vs c_rescore's library MAC: max |diff| {worst:.2e} ({'PASS' if worst <= 1e-5 else 'FAIL'})")
    return 0 if worst <= 1e-5 else 1


if __name__ == "__main__":
    sys.exit(main())
