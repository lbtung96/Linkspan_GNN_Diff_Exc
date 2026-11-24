"""T17 step 3 - score the six scale-up runs (synthetic and real), seed by seed and across seeds.

    python tools/run_stage.py Final_results_for_paper/EC_comments/pipeline/q_eval.py [--workers 8]

T16's p_eval scoring (score / summarise / real_zeroshot), unchanged, with the T17 runs added to the
arm table in memory.  T16's two N600 counterparts are re-scored beside them; gate: their synthetic means
equal T16's eval_summary.csv (1e-9) and their real-bridge rows T16's real_zeroshot.csv exactly.
Writes results/T17_scale_up/: eval_per_sample.csv, eval_summary.csv (per run), eval_seeds.csv (per arm:
mean and SD over seeds 123-125), real_zeroshot.csv, real_summary.csv, real_seeds.csv, eval_gates.json.
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
import p_eval as PE

K, C = Q.K, Q.C
T16_REFS = ("ec_t16_cohorient_signinv", "ec_t16_orient_signinv")
P.ALL.update({k: {kk: v[kk] for kk in ("edge", "loss", "amp", "label")} for k, v in Q.RUNS.items()})
MODES = C.MODE_NAMES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    nets = list(T16_REFS) + list(Q.RUNS)
    miss = [n for n in nets if not (K.EC_RUNS / n / "checkpoints" / "best_model.pt").exists()]
    if miss:
        raise SystemExit(f"missing checkpoints: {miss}")
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, max_tasks_per_child=1) as ex:
        for rr in ex.map(PE.score, [(n, ts) for ts in PE.SETS for n in nets]):
            rows += rr
            print(f"[eval] {rr[0]['test_set']:<13} {rr[0]['run']:<40} MAC {np.mean([r['mac_mean'] for r in rr]):.4f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    Q.T17_RESULTS.mkdir(parents=True, exist_ok=True)
    K.write_csv(Q.T17_RESULTS / "eval_per_sample.csv", rows)
    summ = PE.summarise(rows)
    K.write_csv(Q.T17_RESULTS / "eval_summary.csv", summ)
    t16 = {(r["test_set"], r["run"]): float(r["mac_mean"]) for r in K.read_csv(P.T16_RESULTS / "eval_summary.csv")}
    g = {f"{s['test_set']}/{s['run']}": abs(s["mac_mean"] - t16[(s["test_set"], s["run"])])
         for s in summ if s["run"] in T16_REFS}
    gates = {"synthetic_t16_refs": {"max_abs_diff": max(g.values()), "pass": len(g) == 6 and max(g.values()) < 1e-9}}

    seeds = []
    for ts in PE.SETS:
        for arm in Q.BASE_ARMS:
            ss = [s for s in summ if s["test_set"] == ts and s["run"] in Q.RUNS and Q.RUNS[s["run"]]["base_arm"] == arm]
            r = {"test_set": ts, "arm": arm, "n_seeds": len(ss)}
            for col in ["mac_mean"] + [f"mac_{m}" for m in MODES] + ["freq_mae_hz"]:
                v = np.array([s[col] for s in ss])
                r[col], r[f"{col}_sd"], r[f"{col}_min"], r[f"{col}_max"] = float(v.mean()), float(v.std(ddof=1)), float(v.min()), float(v.max())
            seeds.append(r)
    K.write_csv(Q.T17_RESULTS / "eval_seeds.csv", seeds)

    rr, _ = PE.real_zeroshot(nets)
    K.write_csv(Q.T17_RESULTS / "real_zeroshot.csv", rr)
    ref = K.read_csv(P.T16_RESULTS / "real_zeroshot.csv")
    gr = {}
    for n in T16_REFS:
        x = [r for r in rr if r["arm"] == n]
        y = [r for r in ref if r["arm"] == n]
        gr[n] = max(abs(float(p["mac_ssi"]) - float(q["mac_ssi"])) for p, q in zip(x, y)) if len(x) == len(y) == 36 else None
    gates["real_t16_refs"] = {"max_abs_diff": gr, "pass": all(v == 0.0 for v in gr.values())}
    rs = []
    for n in nets:
        g_ = [r for r in rr if r["arm"] == n]
        rs.append({"run": n, "mac_ssi_mean": float(np.mean([r["mac_ssi"] for r in g_])),
                   "mac_fe_mean": float(np.mean([r["mac_fe_det"] for r in g_])),
                   **{f"mac_ssi_{m}": float(np.mean([r["mac_ssi"] for r in g_ if r["mode"] == m])) for m in MODES}})
    K.write_csv(Q.T17_RESULTS / "real_summary.csv", rs)
    rseeds = []
    for arm in Q.BASE_ARMS:
        ss = [r for r in rs if r["run"] in Q.RUNS and Q.RUNS[r["run"]]["base_arm"] == arm]
        out = {"arm": arm, "n_seeds": len(ss)}
        for col in ["mac_ssi_mean", "mac_fe_mean"] + [f"mac_ssi_{m}" for m in MODES]:
            v = np.array([s[col] for s in ss])
            out[col], out[f"{col}_sd"] = float(v.mean()), float(v.std(ddof=1))
        rseeds.append(out)
    K.write_csv(Q.T17_RESULTS / "real_seeds.csv", rseeds)
    gates["pass"] = all(v["pass"] for v in gates.values() if isinstance(v, dict))
    K.write_json(Q.T17_RESULTS / "eval_gates.json", gates)
    print(json.dumps(gates, indent=1))
    for s in summ:
        print(f"{s['test_set']:<13} {s['run']:<40} MAC {s['mac_mean']:.3f} B1 {s['mac_B1']:.3f} T1 {s['mac_T1']:.3f} "
              f"B2 {s['mac_B2']:.3f} T2 {s['mac_T2']:.3f} f {s['freq_mae_hz']:.3f}")
    for r in seeds:
        print(f"[seeds] {r['test_set']:<13} {r['arm']:<20} MAC {r['mac_mean']:.3f} +- {r['mac_mean_sd']:.3f} "
              f"(min {r['mac_mean_min']:.3f}, max {r['mac_mean_max']:.3f})  f {r['freq_mae_hz']:.3f}")
    for r in rseeds:
        print(f"[real]  {r['arm']:<20} vs SSI {r['mac_ssi_mean']:.3f} +- {r['mac_ssi_mean_sd']:.3f}   vs FE {r['mac_fe_mean']:.3f}")
    return 0 if gates["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
