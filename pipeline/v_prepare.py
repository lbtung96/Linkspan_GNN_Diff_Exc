"""T20 step 2 - simulate the N = 2000 population and the paper's test structures under deck forcing + noise.

    python tools/run_stage.py Final_results_for_paper/EC_comments/pipeline/v_prepare.py --check
    python tools/run_stage.py Final_results_for_paper/EC_comments/pipeline/v_prepare.py --run [--workers 16]

One job per structure: re-solve it once (v_common.solve: the generator's solver from the stored parameters),
then simulate every record it needs for both excitations (v_common.simulate, noise ALPHA from noise_fit.json):
  1700 training structures   pools 0-3 of deck and walk     (8 records)
   300 validation structures pool 0 of deck and walk        (2 records)
   500 paper test structures  deck_test and walk_test       (2 records)
Each written sample is the paper population's file with k_common's five excitation keys replaced (as T11, T13,
T17); a per-structure check file records the diagnostics.  Resumable: finished structures are skipped.
--check (results/T20_deck_forcing/prepare_check.json), on eight training structures (two per Ng):
  supports   v_common.simulate under the paper's support excitation with T13's pool-0 seed reproduces the
             five arrays of T13's trainval_N2000 file to 1e-5 (CLAUDE.md's tolerance for a re-run FE solve:
             eigsh has no fixed start vector; measured 5e-9 on x_psd, 1.2e-7 on the float32 edges)
  deck/walk  graph rows with no vertical mode content (the Ng hinge supports) at rest before the noise;
             achieved noise / in-band RMS = ALPHA (1e-2 rel.); x_psd rows peak at 1; every array finite;
             graph identical to the stored structure
--run: writes data/T20/<exc>_<pool|test>/samples and results/T20_deck_forcing/prepare_summary.json.
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

import v_common as V  # noqa: E402

K, H, C = V.K, V.H, V.C
CHECKS = V.T20_DATA / "checks"


def plan(name: str, role: str) -> list[tuple[str, str, int]]:
    if role == "train":
        return [(f"{e}_p{k}", e, V.BASES[e][k]) for e in V.EXCS for k in range(4)]
    if role == "val":
        return [(f"{e}_p0", e, V.BASES[e][0]) for e in V.EXCS]
    return [(f"{e}_test", e, V.TEST_BASE[e]) for e in V.EXCS]


def record_row(key: str, diag: dict, feats: dict, alpha: float) -> dict:
    x, e = feats["x_psd"], feats["edge_attr_cpsd_full"]
    return {"set": key, "exc": diag["exc"], "alpha": alpha, "noise_rel_achieved": diag.get("noise_rel_achieved"),
            "fixed_rows_max_before_noise": diag.get("fixed_rows_max_before_noise"),
            "n_fixed_rows": diag.get("n_fixed_rows"), "support_rows_rms_rel": diag.get("support_rows_rms_rel"),
            "ambient_scale": diag.get("ambient_scale"), "x_rowmax_min": float(np.abs(x).max(1).min()),
            "x_rowmax_max": float(np.abs(x).max(1).max()),
            "finite": bool(all(np.isfinite(v).all() for v in feats.values())), "edge_absmax": float(np.abs(e).max())}


def job(task: tuple) -> dict:
    name, role = task
    keys = plan(name, role)
    dsts = [V.sdir(k) / name for k, _, _ in keys]
    chk = CHECKS / (Path(name).stem + ".json")
    if chk.exists() and all(d.exists() for d in dsts):
        return json.loads(chk.read_text(encoding="utf-8"))
    t0 = time.time()
    al = V.alpha()
    S = V.solve(C.POP_BROADBAND / name)
    if not S["graph_equal"]:
        raise RuntimeError(f"{name}: re-solved graph differs from the stored one")
    st = S["st"]
    idx = K.sample_index(name)
    recs = []
    for (key, e, base), dst in zip(keys, dsts):
        a3, diag = V.simulate(S, e, np.random.SeedSequence([base, idx]), al[e])
        feats = V.features(S, a3)
        payload = dict(st)
        for k in H.EXC_KEYS:
            new = np.asarray(feats[k]).astype(st[k].dtype, copy=False)
            if new.shape != st[k].shape:
                raise RuntimeError(f"{name}: {k} shape {new.shape} != stored {st[k].shape}")
            payload[k] = new
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            np.savez(fh, **payload)
        os.replace(tmp, dst)
        recs.append(record_row(key, diag, feats, al[e]))
    out = {"sample": name, "role": role, "Ng": int(st["Ng"]), "sample_index": idx, "graph_equal": S["graph_equal"],
           "records": recs, "seconds": round(time.time() - t0, 1)}
    CHECKS.mkdir(parents=True, exist_ok=True)
    chk.write_text(json.dumps(out), encoding="utf-8")
    return out


def check_job(name: str) -> dict:
    import h_prepare as HP

    t0 = time.time()
    al = V.alpha()
    S = V.solve(C.POP_BROADBAND / name)
    idx = K.sample_index(name)
    ref = HP._load(K.sdir("trainval_N2000") / name)
    a3, _ = V.simulate(S, "supports", K.exc_seed(K.BASES[0], idx))
    f = V.features(S, a3)
    out = {"sample": name, "graph_equal": S["graph_equal"],
           "x_psd_maxdiff": float(np.abs(f["x_psd"] - ref["x_psd"]).max()),
           "x_psd_row_scale_maxrel": float(np.abs(f["x_psd_row_scale"] / ref["x_psd_row_scale"] - 1).max()),
           **{f"{k}_maxdiff": float(np.abs(f[k].astype(np.float64) - ref[k].astype(np.float64)).max())
              for k in ("edge_attr_cpsd", "edge_attr_cpsd_resonance", "edge_attr_cpsd_full")},
           "edge_attr_cpsd_full_p999": float(np.quantile(np.abs(f["edge_attr_cpsd_full"].astype(np.float64)
                                                                 - ref["edge_attr_cpsd_full"].astype(np.float64)), 0.999))}
    t1 = time.time()
    for e in V.EXCS:
        a3, diag = V.simulate(S, e, np.random.SeedSequence([V.BASES[e][0], idx]), al[e])
        r = record_row(e, diag, V.features(S, a3), al[e])
        out[e] = r
    out["seconds_supports"] = round(t1 - t0, 1)
    out["seconds_two_deck_records"] = round(time.time() - t1, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    V.T20_RESULTS.mkdir(parents=True, exist_ok=True)
    tr, va, te = K.read_list(K.TRAIN2000), K.read_list(K.VAL2000), K.read_list(K.TEST_LIST)
    if a.check:
        al = V.alpha()
        picks = []
        for ng in (2, 3, 4, 5):
            picks += [n for n in tr if f"_Ng{ng}_" in n and (K.sdir("trainval_N2000") / n).exists()][:2]
        with ProcessPoolExecutor(max_workers=8, max_tasks_per_child=1) as ex:
            rows = list(ex.map(check_job, picks))
        dk = [r[e] for r in rows for e in V.EXCS]
        gates = {"supports_reproduce_t13": {
                     "x_psd_max": max(r["x_psd_maxdiff"] for r in rows),
                     "x_psd_row_scale_maxrel": max(r["x_psd_row_scale_maxrel"] for r in rows),
                     "edge_full_max": max(r["edge_attr_cpsd_full_maxdiff"] for r in rows),
                     "edge_full_p999": max(r["edge_attr_cpsd_full_p999"] for r in rows),
                     "pass": max(r["x_psd_maxdiff"] for r in rows) < 1e-5
                     and max(r["x_psd_row_scale_maxrel"] for r in rows) < 1e-5
                     and max(r["edge_attr_cpsd_full_maxdiff"] for r in rows) < 1e-5},
                 "graph_equal": {"pass": all(r["graph_equal"] for r in rows)},
                 "hinges_at_rest": {"max": max(r["fixed_rows_max_before_noise"] for r in dk),
                                    "n_fixed_rows_eq_ng": all(r[e]["n_fixed_rows"] == int(r["sample"].split("_Ng")[1][0])
                                                              for r in rows for e in V.EXCS),
                                    "front_support_rms_rel": [round(r["support_rows_rms_rel"], 3) for r in dk],
                                    "pass": max(r["fixed_rows_max_before_noise"] for r in dk) == 0.0
                                    and all(r[e]["n_fixed_rows"] == int(r["sample"].split("_Ng")[1][0])
                                            for r in rows for e in V.EXCS)},
                 "noise_level": {"max_rel_err": max(abs(r["noise_rel_achieved"] / r["alpha"] - 1) for r in dk),
                                 "pass": max(abs(r["noise_rel_achieved"] / r["alpha"] - 1) for r in dk) < 1e-2},
                 "rows_peak_at_1": {"pass": all(abs(r["x_rowmax_min"] - 1) < 1e-12 and abs(r["x_rowmax_max"] - 1) < 1e-12
                                                for r in dk)},
                 "finite": {"pass": all(r["finite"] for r in dk)}}
        gates["pass"] = all(v["pass"] for v in gates.values())
        out = {"alpha": al, "gates": gates, "rows": rows,
               "est_hours_16_workers": round((len(tr) * (sum(r["seconds_supports"] for r in rows) / len(rows)
                                                         + 4 * sum(r["seconds_two_deck_records"] for r in rows) / len(rows))
                                              + (len(va) + len(te)) * (sum(r["seconds_supports"] for r in rows) / len(rows)
                                                                       + sum(r["seconds_two_deck_records"] for r in rows) / len(rows)))
                                             / 16 / 3600, 2)}
        K.write_json(V.T20_RESULTS / "prepare_check.json", out)
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
        return 0 if gates["pass"] else 1
    if not json.loads((V.T20_RESULTS / "prepare_check.json").read_text(encoding="utf-8"))["gates"]["pass"]:
        raise SystemExit("prepare_check.json does not pass")
    tasks = [(n, "train") for n in tr] + [(n, "val") for n in va] + [(n, "test") for n in te]
    t0 = time.time()
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=a.workers, max_tasks_per_child=50) as ex:
        for r in ex.map(job, tasks, chunksize=4):
            rows.append(r)
            done += 1
            if done % 100 == 0 or done == len(tasks):
                print(f"[prepare] {done}/{len(tasks)}  {time.time() - t0:.0f} s", flush=True)
    recs = [rec for r in rows for rec in r["records"]]
    counts = {}
    for r in rows:
        for rec in r["records"]:
            counts[rec["set"]] = counts.get(rec["set"], 0) + 1
    expect = {**{f"{e}_p{k}": len(tr) + (len(va) if k == 0 else 0) for e in V.EXCS for k in range(4)},
              **{f"{e}_test": len(te) for e in V.EXCS}}
    summ = {"alpha": V.alpha(), "counts": counts, "expected": expect, "wall_s": round(time.time() - t0, 1),
            "graph_equal_all": all(r["graph_equal"] for r in rows),
            "finite_all": all(rec["finite"] for rec in recs),
            "hinges_at_rest_max": max(rec["fixed_rows_max_before_noise"] for rec in recs),
            "noise_rel_err_max": max(abs(rec["noise_rel_achieved"] / rec["alpha"] - 1) for rec in recs),
            "files_present": all((V.sdir(k) / r["sample"]).exists() for r in rows for k in [x["set"] for x in r["records"]])}
    summ["pass"] = (counts == expect and summ["graph_equal_all"] and summ["finite_all"]
                    and summ["hinges_at_rest_max"] == 0.0 and summ["noise_rel_err_max"] < 1e-2 and summ["files_present"])
    K.write_json(V.T20_RESULTS / "prepare_summary.json", summ)
    print(json.dumps(summ, indent=1))
    return 0 if summ["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
