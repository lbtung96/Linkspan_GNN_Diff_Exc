"""Re-simulate every stored own-record sample in memory and compare it with the stored file.

    python tools/run_stage.py tools/verify_records.py [--workers 16] [--sets T11 T13 T17 T20] [--limit N]

The record sets under data/ were written by the archive's stages (h_prepare, k_prepare, q_prepare, v_prepare)
and are hard-linked here, read-only.  This stage proves that THIS repository's code - the simulator's explicit
`seed` / `t_total` arguments that replaced the archive's in-memory SIM.SEED / SIM.T_TOTAL patches - regenerates
each of them: for every record, the five excitation-dependent arrays (h_common.EXC_KEYS) are recomputed from
the stored structure (data/population) and its seed, and compared with the stored file.  Nothing is written
under data/.

    set                      structures                          seed                                  length
    T11/trainval_varexc      N600 train + val (600)              SeedSequence([20261125, idx])         60 s
    T11/paper_varexc         paper test (500)                    SeedSequence([20261126, idx])         60 s
    T13/rec1..rec3           N600 train (508 each)               SeedSequence([20261131..33, idx])     60 s
    T13/trainval_T120        N600 train + val (600)              SeedSequence([20261125, idx])         120 s
    T13/paper_varexc_T120    paper test (500)                    SeedSequence([20261126, idx])         120 s
    T13/trainval_N2000       N2000 train + val (2000)            SeedSequence([20261125, idx])         60 s
    T17/rec1..rec3_N2000     N2000 train (1700 each)             SeedSequence([20261131..33, idx])     60 s
    T20/{deck,walk}_p0..p3   N2000 train (+ val in p0)           SeedSequence([v_common.BASES[e][k], idx])  60 s
    T20/{deck,walk}_test     paper test (500)                    SeedSequence([v_common.TEST_BASE[e], idx]) 60 s
    (idx = the 4-digit structure index of the file name)

Tolerance 1e-5 absolute on every array (the FE model is re-solved and eigsh has no fixed start vector; the
archive measured up to 1.2e-7 on the float32 coherency).  Also checked byte-identical against the population
file: edge_index, y_mode_shape_signed, y_freq_hz (the labels and graph the record was written with).

Writes results/reproduction/records_per_sample.csv.gz (one row per record: max |diff| per key) and
results/reproduction/records_summary.csv / records_gates.json.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import gzip  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "pipeline")]

import numpy as np  # noqa: E402

import q_common as Q  # noqa: E402  (registers T17's record sets in k_common.DATASETS)
import v_common as V  # noqa: E402

K, H, C = Q.K, Q.H, Q.C
OUT = REPO / "results" / "reproduction"
TOL = 1e-5
LABEL_KEYS = ("edge_index", "y_mode_shape_signed", "y_freq_hz")


def hk_sets() -> dict:
    """key -> (folder, [(name, seed, t_total)])"""
    out = {}
    tv600 = K.read_list(K.TRAIN600) + K.read_list(K.VAL600)
    test = K.read_list(K.TEST_LIST)
    out["T11/trainval_varexc"] = (H.TRAINVAL_DIR, [(n, H.exc_seed(H.TRAIN_EXC_BASE, H.sample_index(n)), 60.0) for n in tv600])
    out["T11/paper_varexc"] = (H.TEST_DIR, [(n, H.exc_seed(H.TEST_EXC_BASE, H.sample_index(n)), 60.0) for n in test])
    for key, d in K.DATASETS.items():
        names = [n for lst in d["lists"] for n in K.read_list(lst)]
        task = "T17" if key in Q.REC_KEYS.values() else "T13"
        out[f"{task}/{key}"] = (K.sdir(key), [(n, K.exc_seed(d["base"], K.sample_index(n)), float(d["t_total"]))
                                                for n in names])
    return out


def t20_plan() -> dict:
    """structure name -> [(set key, excitation, base)] (v_prepare.plan, roles train / val / test)."""
    import v_prepare as VP
    tr, va, te = K.read_list(K.TRAIN2000), K.read_list(K.VAL2000), K.read_list(K.TEST_LIST)
    plan = {}
    for names, role in ((tr, "train"), (va, "val"), (te, "test")):
        for n in names:
            plan.setdefault(n, []).extend(VP.plan(n, role))
    return plan


def _load(p: Path) -> dict:
    with np.load(p, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def _cmp(feats: dict, stored: dict, pop: dict) -> dict:
    r = {f"d_{k}": float(np.abs(np.asarray(feats[k], np.float64) - np.asarray(stored[k], np.float64)).max())
         for k in H.EXC_KEYS}
    r["labels_identical"] = all(np.array_equal(stored[k], pop[k]) and stored[k].dtype == pop[k].dtype for k in LABEL_KEYS)
    r["pass"] = bool(max(r[f"d_{k}"] for k in H.EXC_KEYS) <= TOL and r["labels_identical"])
    return r


def job_hk(task: tuple) -> dict:
    key, folder, name, seed_entropy, t_total = task
    seed = np.random.SeedSequence(seed_entropy)
    res = K.resimulate(C.POP_BROADBAND / name, seed, t_total=t_total)
    r = _cmp(res["features"], _load(Path(folder) / name), _load(C.POP_BROADBAND / name))
    return {"set": key, "sample": name, **r}


def job_t20(task: tuple) -> list[dict]:
    name, recs = task
    al = V.alpha()
    S = V.solve(C.POP_BROADBAND / name)
    pop = _load(C.POP_BROADBAND / name)
    idx = K.sample_index(name)
    out = []
    for key, e, base in recs:
        a3, _ = V.simulate(S, e, np.random.SeedSequence([base, idx]), al[e])
        feats = V.features(S, a3)
        r = _cmp(feats, _load(V.sdir(key) / name), pop)
        r["labels_identical"] = bool(r["labels_identical"] and S["graph_equal"])
        r["pass"] = bool(r["pass"] and S["graph_equal"])
        out.append({"set": f"T20/{key}", "sample": name, **r})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sets", nargs="+", default=["T11", "T13", "T17", "T20"])
    ap.add_argument("--limit", type=int, default=None, help="first N structures per set (smoke test)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows: list[dict] = []
    hk_tasks = []
    for key, (folder, items) in hk_sets().items():
        if key.split("/")[0] not in a.sets:
            continue
        for name, ss, tt in items[: a.limit]:
            hk_tasks.append((key, str(folder), name, list(np.atleast_1d(ss.entropy)), tt))
    t20_tasks = list(t20_plan().items())[: a.limit] if "T20" in a.sets else []
    print(f"[verify] {len(hk_tasks)} T11/T13/T17 records, {sum(len(r) for _, r in t20_tasks)} T20 records "
          f"({len(t20_tasks)} structures), {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, max_tasks_per_child=50) as ex:
        for i, r in enumerate(ex.map(job_hk, hk_tasks, chunksize=4), 1):
            rows.append(r)
            if i % 500 == 0 or i == len(hk_tasks):
                print(f"[hk] {i}/{len(hk_tasks)}  fails so far {sum(not x['pass'] for x in rows)}  "
                      f"{time.time() - t0:.0f} s", flush=True)
        for i, rr in enumerate(ex.map(job_t20, t20_tasks, chunksize=2), 1):
            rows += rr
            if i % 200 == 0 or i == len(t20_tasks):
                print(f"[t20] {i}/{len(t20_tasks)} structures  fails so far {sum(not x['pass'] for x in rows)}  "
                      f"{time.time() - t0:.0f} s", flush=True)
    keys = list(rows[0]) if rows else []
    with gzip.open(OUT / "records_per_sample.csv.gz", "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summ = []
    for s in sorted({r["set"] for r in rows}):
        g = [r for r in rows if r["set"] == s]
        summ.append({"set": s, "records": len(g), **{f"max_d_{k}": max(r[f"d_{k}"] for r in g) for k in H.EXC_KEYS},
                     "labels_identical": all(r["labels_identical"] for r in g), "fails": sum(not r["pass"] for r in g),
                     "pass": all(r["pass"] for r in g)})
    K.write_csv(OUT / "records_summary.csv", summ)
    gates = {"tolerance": TOL, "records": len(rows), "fails": sum(not r["pass"] for r in rows),
             "limit_per_set": a.limit, "sets": {s["set"]: s["pass"] for s in summ},
             "pass": bool(rows) and all(r["pass"] for r in rows), "seconds": round(time.time() - t0, 1)}
    K.write_json(OUT / "records_gates.json", gates)
    for s in summ:
        print(f"{s['set']:<26} {s['records']:6d}  max x_psd {s['max_d_x_psd']:.1e}  max coh {s['max_d_edge_attr_cpsd_full']:.1e}"
              f"  labels {s['labels_identical']}  {'PASS' if s['pass'] else 'FAIL'}")
    print("RECORDS PASS" if gates["pass"] else "RECORDS FAIL")
    return 0 if gates["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
