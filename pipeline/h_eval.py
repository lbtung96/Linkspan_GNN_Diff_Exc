"""T11 step 3 - score the two new arms and three paper networks on four synthetic test sets.

    python tools/run_stage.py pipeline/h_eval.py [--runs ...] [--sets ...] --workers 4

Scoring code: T5's c_rescore.evaluate, imported and called unchanged (s4_indomain.evaluate's
path: CPU, 2 torch threads, the library datasets and per-sample metrics, best-validation
checkpoint, frequency de-normalised with the checkpoint's stats).  Two of its module globals
are re-pointed in memory, in every worker, so nothing is written into T5's folders:
  c_rescore.set_files   + 'paper_varexc' (data/T11/paper_varexc/samples, test.txt order)
  c_rescore.PRED_ROOT   -> data/T11/predictions

Sets: paper (fixed test set, exc. 12345), paper_varexc (same 500 test structures, a new
record per structure, base 20261126), fresh_test (T5: 500 new structures, one shared new
record 20261124), fresh_test_paperexc (T5: the same 500 new structures, exc. 12345).

Gates (eval_gates.json):
  paper set   the paper networks reproduce s4_indomain_summary.csv exactly (c_rescore.gate)
  fresh sets  the paper networks reproduce T5's rescore_summary.csv exactly (shared numeric columns)

Writes results/T11_varied_excitation_pilot/: eval_per_sample.csv, eval_summary.csv (mean per
mode, mean MAC, frequency MAE, bootstrap 95 % intervals over structures, s4's resampling),
eval_by_ng.csv, eval_contrasts.csv (paired by structure), eval_gates.json.
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

import h_common as H  # noqa: E402
import c_rescore as CR  # noqa: E402

C = H.C
_orig_set_files = CR.set_files


def set_files(test_set: str) -> list[Path]:
    if test_set == "paper_varexc":
        names = H.read_list(H.TEST_LIST)
        files = [H.TEST_DIR / n for n in names]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise RuntimeError(f"{len(missing)} paper_varexc samples missing, e.g. {missing[0]}")
        return files
    return _orig_set_files(test_set)


CR.set_files = set_files
CR.PRED_ROOT = H.T11_DATA / "predictions"
GATE_EXCLUDE = CR.GATE_COLS_EXCLUDE | {"test_set", "run_root", "N", "seed", "epoch_of_checkpoint", "best_epoch",
                                       "epochs", "train_time_h", "forward_ms_median", "n_test"}
MODES = C.MODE_NAMES
COLS = [f"mac_{m}" for m in MODES] + ["mac_mean"] + [f"freq_abs_err_{m}_hz" for m in MODES] + ["freq_mae_hz"]


def worker(task: tuple):
    return CR.evaluate(task)


def summarise(rows: list[dict], keys: dict) -> dict:
    out = dict(keys)
    out["n"] = len(rows)
    for col in COLS:
        v = np.array([float(r[col]) for r in rows])
        lo, hi = H.bootstrap_ci(v)
        out[col], out[f"{col}_lo"], out[f"{col}_hi"] = float(v.mean()), lo, hi
    return out


def paired(rows_a: list[dict], rows_b: list[dict], col: str) -> tuple[float, float, float, int]:
    a = {r["sample"]: float(r[col]) for r in rows_a}
    b = {r["sample"]: float(r[col]) for r in rows_b}
    common = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in common])
    lo, hi = H.bootstrap_ci(d)
    return float(d.mean()), lo, hi, len(common)


def gate_fresh(summ: list[dict]) -> dict:
    ref = {(r["test_set"], r["run"]): r for r in H.read_csv(H.EC.task_dir("T5") / "rescore_summary.csv")
           if r["checkpoint"] == "best_model"}
    out = {"reference": "results/T5_fresh_test_baselines/rescore_summary.csv",
           "rule": "every shared numeric column equal (==)", "rows": {}}
    for s in summ:
        k = (s["test_set"], s["run"])
        if s["test_set"] not in ("fresh_test", "fresh_test_paperexc") or k not in ref:
            continue
        r = ref[k]
        cols = [c for c in r if c in s and c not in GATE_EXCLUDE and r[c] not in ("", None)]
        bad = {}
        for c in cols:
            try:
                if float(r[c]) != float(s[c]):
                    bad[c] = (float(r[c]), float(s[c]))
            except (TypeError, ValueError):
                continue
        out["rows"][f"{k[0]}/{k[1]}"] = {"columns_compared": len(cols), "mismatches": bad, "pass": not bad}
    out["pass"] = bool(out["rows"]) and all(v["pass"] for v in out["rows"].values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=list(H.NET_ORDER))
    ap.add_argument("--sets", nargs="+", default=list(H.TEST_SETS), choices=list(H.TEST_SETS))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    runs = [CR.resolve_run(r) for r in args.runs]
    tasks = [(k, root, str(d), "best_model", ts) for ts in args.sets for (k, root, d) in runs]
    t0 = time.time()
    new, metas = [], {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
        for rows, meta in ex.map(worker, tasks):
            new += rows
            metas[meta["run"]] = meta
            print(f"[eval] {rows[0]['test_set']:<20} {meta['run']:<36} MAC "
                  f"{np.mean([r['mac_mean'] for r in rows]):.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    per_path = H.T11_RESULTS / "eval_per_sample.csv"
    done = {(r["test_set"], r["run"]) for r in new}
    old = [r for r in H.read_csv(per_path) if (r["test_set"], r["run"]) not in done]
    allrows = old + new
    H.write_csv(per_path, allrows)
    meta_path = H.T11_RESULTS / "eval_runs.json"
    allmeta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    allmeta.update(metas)
    H.write_json(meta_path, allmeta)

    groups = sorted({(r["test_set"], r["run"]) for r in allrows},
                    key=lambda t: (H.TEST_SETS.index(t[0]), H.NET_ORDER.index(t[1]) if t[1] in H.NET_ORDER else 99))
    by = {g: [r for r in allrows if (r["test_set"], r["run"]) == g] for g in groups}
    summ, by_ng, s4like = [], [], []
    for (ts, run), rr in by.items():
        m = allmeta.get(run, {})
        keys = {"test_set": ts, "run": run, "best_epoch": m.get("best_epoch"), "epochs": m.get("epochs"),
                "edge_feature_mode": m.get("edge_feature_mode"), "N": m.get("N")}
        summ.append(summarise(rr, keys))
        for g in C.NG_VALUES:
            by_ng.append(summarise([r for r in rr if int(float(r["Ng"])) == g], {**keys, "Ng": g}))
        if (ts, run) in done:
            rr_f = [{k: (float(v) if k.startswith(("mac_", "freq_")) else v) for k, v in r.items()} for r in rr]
            s4like.append({**CR.S4.summarise(rr_f, {"test_set": ts, "run": run, "checkpoint": "best_model"})})
    H.write_csv(H.T11_RESULTS / "eval_summary.csv", summ)
    H.write_csv(H.T11_RESULTS / "eval_by_ng.csv", by_ng)

    # paired contrasts by structure
    con = []
    pairs = [("ec_varexc_edgegat_N600", "edgegat_N600", "varied - fixed exc., coherency arm"),
             ("ec_varexc_edgegat_geomorient_N600", "edgegat_geomorient_N600", "varied - fixed exc., orientation arm"),
             ("ec_varexc_edgegat_N600", "ec_varexc_edgegat_geomorient_N600", "coherency - orientation, varied exc."),
             ("edgegat_N600", "edgegat_geomorient_N600", "coherency - orientation, fixed exc. (paper)"),
             ("ec_varexc_edgegat_N600", "edgegat_N2000", "varied-exc. coherency N=600 - paper main N=2000")]
    for ts in H.TEST_SETS:
        for a, b, lbl in pairs:
            if (ts, a) in by and (ts, b) in by:
                row = {"test_set": ts, "contrast": lbl, "a": a, "b": b}
                for col in ("mac_mean", "mac_T1", "freq_mae_hz"):
                    d, lo, hi, n = paired(by[(ts, a)], by[(ts, b)], col)
                    row.update({f"{col}_diff": d, f"{col}_lo": lo, f"{col}_hi": hi})
                row["n"] = n
                con.append(row)
    # excitation effect within each network: paper_varexc - paper (same structures), fresh_test - fresh_test_paperexc
    for run in H.NET_ORDER:
        for a_set, b_set, lbl in (("paper_varexc", "paper", "new record per structure - paper record (paper test structures)"),
                                  ("fresh_test", "fresh_test_paperexc", "shared new record - paper record (fresh structures)")):
            if (a_set, run) in by and (b_set, run) in by:
                row = {"test_set": f"{a_set} - {b_set}", "contrast": f"excitation effect: {lbl}", "a": run, "b": run}
                for col in ("mac_mean", "mac_T1", "freq_mae_hz"):
                    d, lo, hi, n = paired(by[(a_set, run)], by[(b_set, run)], col)
                    row.update({f"{col}_diff": d, f"{col}_lo": lo, f"{col}_hi": hi})
                row["n"] = n
                con.append(row)
    H.write_csv(H.T11_RESULTS / "eval_contrasts.csv", con)

    gpath = H.T11_RESULTS / "eval_gates.json"
    gates = json.loads(gpath.read_text(encoding="utf-8")) if gpath.exists() else {}
    if any(ts == "paper" and run in H.PAPER_NETS for ts, run in done):
        gates["paper_vs_s4_indomain"] = CR.gate([s for s in s4like if s["test_set"] == "paper"])
    fr = gate_fresh(s4like)
    if fr["rows"]:
        gates["fresh_vs_T5_rescore"] = fr
    gates["pass"] = all(v.get("pass", False) for k, v in gates.items() if isinstance(v, dict))
    H.write_json(gpath, gates)
    for k, v in gates.items():
        if isinstance(v, dict):
            print(f"[gate] {k}: {'PASS' if v['pass'] else 'FAIL'}")
            if not v["pass"]:
                print(json.dumps(v, indent=1, default=float)[:3000])
    for s in summ:
        if (s["test_set"], s["run"]) in done:
            print(f"{s['test_set']:<20} {s['run']:<36} MAC {s['mac_mean']:.4f} [{s['mac_mean_lo']:.3f}, "
                  f"{s['mac_mean_hi']:.3f}]  B1 {s['mac_B1']:.3f} T1 {s['mac_T1']:.3f} B2 {s['mac_B2']:.3f} "
                  f"T2 {s['mac_T2']:.3f}  f MAE {s['freq_mae_hz']:.3f} Hz")
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    return 0 if gates["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
