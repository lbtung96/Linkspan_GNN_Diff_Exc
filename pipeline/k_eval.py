"""T13 step 3 - score the five arms and T11's two 5000-epoch arms on the new-excitation test sets.

    python tools/run_stage.py pipeline/k_eval.py --workers 6

Scoring code: T5's c_rescore.evaluate, imported and called unchanged (s4_indomain's path, CPU,
2 torch threads, best-validation checkpoint), as T11's h_eval.py.  Module globals re-pointed in
memory in every worker, so nothing is written into T5's or T11's folders:
  c_rescore.set_files  + 'paper_varexc'      (T11's data/T11/paper_varexc/samples, test.txt order)
                       + 'paper_varexc_T120' (data/T13/paper_varexc_T120/samples, test.txt order)
  c_rescore.PRED_ROOT  -> data/T13/predictions

Sets: paper_varexc (T11), fresh_test (T5), paper_varexc_T120 (same 500 structures and seeds as
paper_varexc, 120 s records).  Every net is scored on all three (cheap).

Gate (eval_gates.json): T11's two arms on paper_varexc and fresh_test reproduce T11's
eval_summary.csv exactly (every shared numeric column).

Writes results/T13_feasibility/: eval_per_sample.csv, eval_summary.csv (mean per mode, mean MAC,
frequency MAE, bootstrap 95 % over structures with s4's resampling), eval_by_ng.csv,
eval_contrasts.csv (paired by structure: each lever arm - the reference arm; coherency arms -
orientation K4 arm).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import k_common as K  # noqa: E402
import c_rescore as CR  # noqa: E402

C, H = K.C, K.H
_orig_set_files = CR.set_files
SET_DIRS = {"paper_varexc": K.T11_TEST, "paper_varexc_T120": K.sdir("paper_varexc_T120")}


def set_files(test_set: str) -> list[Path]:
    if test_set in SET_DIRS:
        files = [SET_DIRS[test_set] / n for n in K.read_list(K.TEST_LIST)]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise RuntimeError(f"{len(missing)} {test_set} samples missing, e.g. {missing[0]}")
        return files
    return _orig_set_files(test_set)


CR.set_files = set_files
CR.PRED_ROOT = K.T13_DATA / "predictions"
MODES = C.MODE_NAMES
COLS = [f"mac_{m}" for m in MODES] + ["mac_mean"] + [f"freq_abs_err_{m}_hz" for m in MODES] + ["freq_mae_hz"]
REF = "ec_t13_coh_K1_N600"
CONTRASTS = [("ec_t13_coh_K4_N600", REF, "augmentation: K 4 - K 1"),
             ("ec_t13_coh_K1_N600_T120", REF, "record length: 120 s - 60 s"),
             ("ec_t13_coh_K1_N2000", REF, "population: N 2000 - N 600"),
             ("ec_t13_coh_K4_N600", "ec_t13_orient_K4_N600", "coherency K4 - orientation K4"),
             ("ec_t13_coh_K1_N600_T120", "ec_t13_orient_K4_N600", "coherency 120 s - orientation K4"),
             ("ec_t13_coh_K1_N2000", "ec_t13_orient_K4_N600", "coherency N 2000 - orientation K4"),
             (REF, "ec_varexc_edgegat_N600", "400 ep. - T11 5000 ep., coherency K1"),
             ("ec_t13_orient_K4_N600", "ec_varexc_edgegat_geomorient_N600", "orientation K4 400 ep. - T11 K1 5000 ep.")]


def worker(task: tuple):
    return CR.evaluate(task)


def summarise(rows: list[dict], keys: dict) -> dict:
    out = dict(keys)
    out["n"] = len(rows)
    for col in COLS:
        v = np.array([float(r[col]) for r in rows])
        lo, hi = K.bootstrap_ci(v)
        out[col], out[f"{col}_lo"], out[f"{col}_hi"] = float(v.mean()), lo, hi
    return out


def paired(rows_a, rows_b, col):
    a = {r["sample"]: float(r[col]) for r in rows_a}
    b = {r["sample"]: float(r[col]) for r in rows_b}
    common = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in common])
    lo, hi = K.bootstrap_ci(d)
    return float(d.mean()), lo, hi, len(common)


def gate_t11(summ: list[dict]) -> dict:
    ref = {(r["test_set"], r["run"]): r for r in K.read_csv(H.T11_RESULTS / "eval_summary.csv")}
    out = {"reference": "results/T11_varied_excitation_pilot/eval_summary.csv",
           "rule": "every shared numeric column equal (==)", "rows": {}}
    for s in summ:
        k = (s["test_set"], s["run"])
        if s["run"] not in K.T11_ARMS or k not in ref:
            continue
        r = ref[k]
        bad = {}
        n = 0
        for c in r:
            if c in ("test_set", "run", "edge_feature_mode") or c not in s:
                continue
            try:
                a, b = float(r[c]), float(s[c])
            except (TypeError, ValueError):
                continue
            n += 1
            if a != b:
                bad[c] = (a, b)
        out["rows"][f"{k[0]}/{k[1]}"] = {"columns_compared": n, "mismatches": bad, "pass": not bad}
    out["pass"] = len(out["rows"]) == 4 and all(v["pass"] for v in out["rows"].values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=list(K.ALL_NETS))
    ap.add_argument("--sets", nargs="+", default=list(K.TEST_SETS), choices=list(K.TEST_SETS))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    K.T13_RESULTS.mkdir(parents=True, exist_ok=True)
    runs = [CR.resolve_run(r) for r in args.runs]
    tasks = [(k, root, str(d), "best_model", ts) for ts in args.sets for (k, root, d) in runs]
    t0 = time.time()
    new, metas = [], {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
        for rows, meta in ex.map(worker, tasks):
            new += rows
            metas[meta["run"]] = meta
            print(f"[eval] {rows[0]['test_set']:<18} {meta['run']:<36} MAC "
                  f"{np.mean([r['mac_mean'] for r in rows]):.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    per_path = K.T13_RESULTS / "eval_per_sample.csv"
    done = {(r["test_set"], r["run"]) for r in new}
    allrows = [r for r in K.read_csv(per_path) if (r["test_set"], r["run"]) not in done] + new
    K.write_csv(per_path, allrows)
    meta_path = K.T13_RESULTS / "eval_runs.json"
    allmeta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    allmeta.update(metas)
    K.write_json(meta_path, allmeta)

    order = list(K.ALL_NETS)
    groups = sorted({(r["test_set"], r["run"]) for r in allrows},
                    key=lambda t: (K.TEST_SETS.index(t[0]), order.index(t[1]) if t[1] in order else 99))
    by = {g: [r for r in allrows if (r["test_set"], r["run"]) == g] for g in groups}
    summ, by_ng = [], []
    for (ts, run), rr in by.items():
        m = allmeta.get(run, {})
        keys = {"test_set": ts, "run": run, "best_epoch": m.get("best_epoch"), "epochs": m.get("epochs"),
                "edge_feature_mode": m.get("edge_feature_mode"), "N": m.get("N")}
        summ.append(summarise(rr, keys))
        for g in C.NG_VALUES:
            by_ng.append(summarise([r for r in rr if int(float(r["Ng"])) == g], {**keys, "Ng": g}))
    K.write_csv(K.T13_RESULTS / "eval_summary.csv", summ)
    K.write_csv(K.T13_RESULTS / "eval_by_ng.csv", by_ng)

    con = []
    for ts in K.TEST_SETS:
        for a, b, lbl in CONTRASTS:
            if (ts, a) in by and (ts, b) in by:
                row = {"test_set": ts, "contrast": lbl, "a": a, "b": b}
                for col in ("mac_mean", "mac_T1", "mac_T2", "freq_mae_hz"):
                    d, lo, hi, n = paired(by[(ts, a)], by[(ts, b)], col)
                    row.update({f"{col}_diff": d, f"{col}_lo": lo, f"{col}_hi": hi})
                row["n"] = n
                con.append(row)
    # record-length effect within each net: paper_varexc_T120 - paper_varexc (same structures and seeds)
    for run in order:
        if ("paper_varexc_T120", run) in by and ("paper_varexc", run) in by:
            row = {"test_set": "paper_varexc_T120 - paper_varexc", "contrast": "test record 120 s - 60 s",
                   "a": run, "b": run}
            for col in ("mac_mean", "mac_T1", "mac_T2", "freq_mae_hz"):
                d, lo, hi, n = paired(by[("paper_varexc_T120", run)], by[("paper_varexc", run)], col)
                row.update({f"{col}_diff": d, f"{col}_lo": lo, f"{col}_hi": hi})
            row["n"] = n
            con.append(row)
    K.write_csv(K.T13_RESULTS / "eval_contrasts.csv", con)

    gates = {"t11_arms_vs_t11_eval_summary": gate_t11(summ)}
    gates["pass"] = gates["t11_arms_vs_t11_eval_summary"]["pass"]
    K.write_json(K.T13_RESULTS / "eval_gates.json", gates)
    print(f"[gate] T11 arms reproduce T11 eval_summary: {'PASS' if gates['pass'] else 'FAIL'}")
    if not gates["pass"]:
        print(json.dumps(gates, indent=1, default=float)[:3000])
    for s in summ:
        print(f"{s['test_set']:<18} {s['run']:<36} MAC {s['mac_mean']:.4f} [{s['mac_mean_lo']:.3f}, "
              f"{s['mac_mean_hi']:.3f}]  B1 {s['mac_B1']:.3f} T1 {s['mac_T1']:.3f} B2 {s['mac_B2']:.3f} "
              f"T2 {s['mac_T2']:.3f}  f MAE {s['freq_mae_hz']:.3f} Hz")
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    return 0 if gates["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
