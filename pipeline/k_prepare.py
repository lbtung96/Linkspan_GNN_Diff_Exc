"""T13 step 1 - gates (a)-(c), then the new excitation records.

    python tools/run_stage.py Final_results_for_paper/EC_comments/pipeline/k_prepare.py --part gate --workers 10
    python tools/run_stage.py Final_results_for_paper/EC_comments/pipeline/k_prepare.py --part data --workers 10 [--sets ...]
    python tools/run_stage.py Final_results_for_paper/EC_comments/pipeline/k_prepare.py --part crosscheck

--part gate (results/T13_feasibility/prepare_gates.json)
  (a) T11's gate (h_prepare --part gate: 40 structures, SEED 12345, T_TOTAL 60 reproduce the
      stored paper features to <= 1e-5) is reused and re-read from its json; in addition 8 new
      structures (2 per Ng, from the N2000 structures T11 never touched) are re-simulated here
      through k_common.resimulate (T_TOTAL patch, set to 60) and must reproduce the stored
      paper features (<= 1e-5); and 4 N600 structures with base 20261125 must reproduce T11's
      stored samples (so record 0 / the N2000 arm's N600 records ARE arm 1's).
  (b) records: for two structures, the four records k = 0..3 (60 s) and the 120 s record have
      pairwise max |corr| of the support-joint records < 0.2; the same (structure, k)
      re-simulated twice gives the same record and features (<= 1e-5); the 120 s record has
      24000 samples (22 Welch segments vs 10).
  (c) labels and graphs: re-solved at 60 s and 120 s, identical within 1e-5 to the stored
      sample (the response length cannot touch them) - and per written sample in --part data,
      and across all variants in --part crosscheck.

--part data  every set in k_common.DATASETS: the stored paper sample with h_common.EXC_KEYS
  replaced (h_prepare's write-and-read-back checks, plus the record length).  Resumable.
  trainval_N2000: the 600 structures T11 already simulated with the same seed are copied
  byte for byte (checked), the other 1400 simulated.

--part crosscheck  (c) over everything written: for every structure, every non-excitation
  key byte-identical across the paper sample, T11's sample and every T13 variant.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import filecmp  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402

import k_common as K  # noqa: E402
import h_prepare as HP  # noqa: E402  (T11: _load, compare_to_stored, max_abs_corr)

H, C = K.H, K.C
TOL = 1e-5


# ---------------------------------------------------------------- gates
def gate_a_job(job: tuple) -> dict:
    name, kind = job
    src = C.POP_BROADBAND / name
    t0 = time.time()
    if kind == "paper":
        res = K.resimulate(src, H.PAPER_EXC_SEED, K.PAPER_T_TOTAL)
        ref = HP._load(src)
    else:                                       # base 20261125 vs T11's stored sample
        res = K.resimulate(src, K.exc_seed(K.BASES[0], K.sample_index(name)), K.PAPER_T_TOTAL)
        ref = HP._load(K.T11_TRAINVAL / name)
    out = {"sample": name, "kind": kind, "Ng": int(ref["Ng"]), **HP.compare_to_stored(res, ref),
           "seconds": round(time.time() - t0, 2)}
    num = [k for k in out if k.endswith(("maxdiff", "relmax", "maxdiff_hz"))]
    out["pass"] = bool(out["graph_ids_equal"] and out["edge_index_equal"] and all(out[k] <= TOL for k in num))
    return out


def gate_b_job(job: tuple) -> dict:
    name, k, t_total = job
    src = C.POP_BROADBAND / name
    res = K.resimulate(src, K.exc_seed(K.BASES[k], K.sample_index(name)), t_total)
    sup = H.support_rows(res["edge_index"], res["graph_node_ids"].size)
    st = HP._load(src)
    cmp = HP.compare_to_stored(res, st)
    return {"name": name, "k": k, "t_total": t_total, "rec": res["a3"][sup], "nt": int(res["a3"].shape[1]),
            "features": res["features"], "label_freq_maxdiff_hz": cmp["label_freq_maxdiff_hz"],
            "label_shape_abs_maxdiff": cmp["label_shape_abs_maxdiff"],
            "graph_equal": cmp["graph_ids_equal"] and cmp["edge_index_equal"]}


def run_gates(workers: int) -> dict:
    t0 = time.time()
    n600 = set(K.read_list(K.TRAIN600) + K.read_list(K.VAL600))
    new2000 = [n for n in K.read_list(K.TRAIN2000) + K.read_list(K.VAL2000) if n not in n600]
    a_jobs = []
    for ng in C.NG_VALUES:
        a_jobs += [(n, "paper") for n in [n for n in new2000 if f"_Ng{ng}_" in n][:2]]
    tr = K.read_list(K.TRAIN600)
    a_jobs += [(n, "t11") for n in tr[:2] + [n for n in tr if "_Ng5_" in n][:2]]
    A = tr[0]
    B = [n for n in tr if "_Ng4_" in n][0]
    b_jobs = [(A, k, 60.0) for k in range(K.K)] + [(A, 1, 60.0)] + [(B, k, 60.0) for k in range(K.K)] \
        + [(A, 0, 120.0), (A, 0, 120.0), (B, 0, 120.0)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fa = ex.map(gate_a_job, a_jobs)
        fb = ex.map(gate_b_job, b_jobs)
        rows_a, rb = list(fa), list(fb)
    HP_num = [k for k in rows_a[0] if k.endswith(("maxdiff", "relmax", "maxdiff_hz"))]
    t11 = json.loads((H.T11_RESULTS / "prepare_gates.json").read_text(encoding="utf-8"))
    ga = {"t11_gate_reused": {"pass": t11["a_reproduction_paper_seed"]["pass"],
                              "n_structures": t11["a_reproduction_paper_seed"]["n_structures"],
                              "worst": t11["a_reproduction_paper_seed"]["worst"],
                              "source": "results/T11_varied_excitation_pilot/prepare_gates.json"},
          "paper_seed_new_structures": {
              "n": sum(r["kind"] == "paper" for r in rows_a),
              "pass": all(r["pass"] for r in rows_a if r["kind"] == "paper"),
              "worst": {k: max(r[k] for r in rows_a if r["kind"] == "paper") for k in HP_num}},
          "base20261125_vs_T11_samples": {
              "n": sum(r["kind"] == "t11" for r in rows_a),
              "pass": all(r["pass"] for r in rows_a if r["kind"] == "t11"),
              "worst": {k: max(r[k] for r in rows_a if r["kind"] == "t11") for k in HP_num}},
          "tol": TOL}
    ga["pass"] = all(v["pass"] for v in ga.values() if isinstance(v, dict))
    K.write_csv(K.T13_RESULTS / "prepare_gate_a.csv", rows_a)

    a60 = {r["k"]: r for r in rb[:4]}
    a1_again = rb[4]
    b60 = {r["k"]: r for r in rb[5:9]}
    a120, a120_again, b120 = rb[9], rb[10], rb[11]
    recs = {f"A_k{k}": a60[k]["rec"] for k in range(K.K)}
    recs.update({f"B_k{k}": b60[k]["rec"] for k in range(K.K)})
    # 120 s vs 60 s: compare the overlapping first 60 s
    n60 = a60[0]["rec"].shape[1]
    corr = {}
    keys = list(recs)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            corr[f"{a}~{b}"] = HP.max_abs_corr(recs[a], recs[b])
    corr["A_k0_120s[:60s]~A_k0_60s"] = HP.max_abs_corr(a120["rec"][:, :n60], a60[0]["rec"])
    corr["A_k0_120s[:60s]~B_k0_120s[:60s]"] = HP.max_abs_corr(a120["rec"][:, :n60], b120["rec"][:, :n60])
    for k in range(1, K.K):
        corr[f"A_k0_120s[:60s]~A_k{k}"] = HP.max_abs_corr(a120["rec"][:, :n60], a60[k]["rec"])
    rep_rec = float(np.abs(a60[1]["rec"] - a1_again["rec"]).max() / np.sqrt(np.mean(a60[1]["rec"] ** 2)))
    rep_feat = max(HP._maxdiff(a60[1]["features"][k], a1_again["features"][k]) for k in ("x_psd", "edge_attr_cpsd_full"))
    rep120_rec = float(np.abs(a120["rec"] - a120_again["rec"]).max() / np.sqrt(np.mean(a120["rec"] ** 2)))
    rep120_feat = max(HP._maxdiff(a120["features"][k], a120_again["features"][k]) for k in ("x_psd", "edge_attr_cpsd_full"))
    nseg = lambda nt: (nt - 2048) // 1024 + 1  # noqa: E731
    gb = {"structures": {"A": A, "B": B},
          "max_abs_corr_pairs": corr, "worst_pair_corr": max(corr.values()),
          "same_seed_twice_record_maxdiff_rel_rms_60s": rep_rec, "same_seed_twice_feature_maxdiff_60s": rep_feat,
          "same_seed_twice_record_maxdiff_rel_rms_120s": rep120_rec, "same_seed_twice_feature_maxdiff_120s": rep120_feat,
          "record_samples_60s": int(a60[0]["nt"]), "record_samples_120s": int(a120["nt"]),
          "welch_segments_60s": nseg(a60[0]["nt"]), "welch_segments_120s": nseg(a120["nt"]),
          "features_differ_between_records_x_psd_min": min(
              HP._maxdiff(a60[i]["features"]["x_psd"], a60[j]["features"]["x_psd"])
              for i in range(K.K) for j in range(i + 1, K.K)),
          "threshold_independent_corr_lt": 0.2, "tol": TOL}
    gb["pass"] = bool(gb["worst_pair_corr"] < 0.2 and rep_rec <= TOL and rep_feat <= TOL
                      and rep120_rec <= TOL and rep120_feat <= TOL
                      and gb["record_samples_60s"] == 12000 and gb["record_samples_120s"] == 24000
                      and gb["features_differ_between_records_x_psd_min"] > 1e-2)
    lab = [{"name": r["name"], "k": r["k"], "t_total": r["t_total"], "graph_equal": r["graph_equal"],
            "label_freq_maxdiff_hz": r["label_freq_maxdiff_hz"],
            "label_shape_abs_maxdiff": r["label_shape_abs_maxdiff"]} for r in rb]
    gc = {"rule": "labels re-solved in every gate (b) simulation (60 s and 120 s, k = 0..3) equal the stored "
                  "sample's within 1e-5; graphs identical",
          "rows": lab,
          "worst_label_freq_maxdiff_hz": max(r["label_freq_maxdiff_hz"] for r in lab),
          "worst_label_shape_abs_maxdiff": max(r["label_shape_abs_maxdiff"] for r in lab)}
    gc["pass"] = all(r["graph_equal"] and r["label_freq_maxdiff_hz"] <= TOL and r["label_shape_abs_maxdiff"] <= TOL
                     for r in lab)
    out = {"a_reproduction": ga, "b_records": gb, "c_labels_sim": gc,
           "pass": ga["pass"] and gb["pass"] and gc["pass"], "wall_s": round(time.time() - t0, 1)}
    K.write_json(K.T13_RESULTS / "prepare_gates.json", out)
    return out


# ---------------------------------------------------------------- data
def data_job(job: tuple) -> dict:
    name, key = job
    spec = K.DATASETS[key]
    src = C.POP_BROADBAND / name
    dst_dir = K.sdir(key)
    dst = dst_dir / name
    chk = dst_dir / (Path(name).stem + ".check.json")
    if dst.exists() and chk.exists():
        return json.loads(chk.read_text(encoding="utf-8"))
    t0 = time.time()
    idx = K.sample_index(name)
    seed = K.exc_seed(spec["base"], idx)
    st = HP._load(src)
    t11_copy = key == "trainval_N2000" and (K.T11_TRAINVAL / name).exists()
    if t11_copy:
        shutil.copyfile(K.T11_TRAINVAL / name, dst.with_suffix(".tmp"))
        os.replace(dst.with_suffix(".tmp"), dst)
        res = None
    else:
        res = K.resimulate(src, seed, spec["t_total"])
        payload = dict(st)
        for k in H.EXC_KEYS:
            new = np.asarray(res["features"][k]).astype(st[k].dtype, copy=False)
            if new.shape != st[k].shape:
                raise RuntimeError(f"{name}: {k} shape {new.shape} != stored {st[k].shape}")
            payload[k] = new
        tmp = dst.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            np.savez(fh, **payload)
        os.replace(tmp, dst)

    back = HP._load(dst)
    copied = [k for k in st if k not in H.EXC_KEYS]
    not_identical = [k for k in copied if not (back[k].dtype == st[k].dtype and back[k].shape == st[k].shape
                                               and np.array_equal(back[k], st[k]))]
    x, e = back["x_psd"], back["edge_attr_cpsd_full"]
    row = {"sample": name, "set": key, "Ng": int(st["Ng"]), "sample_index": idx, "exc_seed": H.seed_repr(seed),
           "t_total_s": spec["t_total"], "source": "T11 copy" if t11_copy else "simulated",
           "keys_total": len(st), "keys_copied_identical": len(copied) - len(not_identical),
           "keys_not_identical": ";".join(not_identical), "same_key_set": sorted(back) == sorted(st),
           "x_rowmax_min": float(np.abs(x).max(axis=1).min()), "x_rowmax_max": float(np.abs(x).max(axis=1).max()),
           "edge_absmax": float(np.abs(e).max()),
           "x_psd_diff_vs_paper": HP._maxdiff(x, st["x_psd"])}
    if t11_copy:
        row.update({"t11_byte_identical": filecmp.cmp(dst, K.T11_TRAINVAL / name, shallow=False),
                    "record_samples": None, "graph_ids_equal": True, "edge_index_equal": True,
                    "label_freq_maxdiff_hz": 0.0, "label_shape_abs_maxdiff": 0.0})
        ok_src = row["t11_byte_identical"]
    else:
        cmp = HP.compare_to_stored(res, st)
        row.update({"t11_byte_identical": None, "record_samples": int(res["a3"].shape[1]),
                    "graph_ids_equal": cmp["graph_ids_equal"], "edge_index_equal": cmp["edge_index_equal"],
                    "label_freq_maxdiff_hz": cmp["label_freq_maxdiff_hz"],
                    "label_shape_abs_maxdiff": cmp["label_shape_abs_maxdiff"]})
        ok_src = row["record_samples"] == int(round(spec["t_total"] / 0.005))
    row["seconds"] = round(time.time() - t0, 2)
    row["pass"] = bool(ok_src and not not_identical and row["same_key_set"] and row["graph_ids_equal"]
                       and row["edge_index_equal"] and row["label_freq_maxdiff_hz"] <= TOL
                       and row["label_shape_abs_maxdiff"] <= TOL
                       and abs(row["x_rowmax_min"] - 1) <= 1e-9 and abs(row["x_rowmax_max"] - 1) <= 1e-9
                       and row["edge_absmax"] <= 1 + 1e-5 and row["x_psd_diff_vs_paper"] > 1e-3)
    chk.write_text(json.dumps(row, indent=1), encoding="utf-8")
    return row


def set_names(key: str) -> list[str]:
    out = []
    for lst in K.DATASETS[key]["lists"]:
        out += K.read_list(lst)
    return out


def run_data(keys: list[str], workers: int, limit: int | None) -> dict:
    t0 = time.time()
    jobs = []
    for key in keys:
        K.sdir(key).mkdir(parents=True, exist_ok=True)
        names = set_names(key)
        jobs += [(n, key) for n in (names[:limit] if limit else names)]
    # longest first (120 s), so the pool's tail is short
    jobs.sort(key=lambda j: -K.DATASETS[j[1]]["t_total"])
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(data_job, jobs, chunksize=2)):
            rows.append(r)
            if (i + 1) % 100 == 0 or i + 1 == len(jobs):
                print(f"[data] {i + 1}/{len(jobs)}  fails {sum(not x['pass'] for x in rows)}  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    path = K.T13_RESULTS / "prepare_checks.csv"
    done = {(r["set"], r["sample"]) for r in rows}
    old = [r for r in K.read_csv(path) if (r["set"], r["sample"]) not in done]
    K.write_csv(path, old + rows)
    summ_path = K.T13_RESULTS / "prepare_summary.json"
    summ = json.loads(summ_path.read_text(encoding="utf-8")) if summ_path.exists() else {}
    for key in keys:
        rr = [r for r in rows if r["set"] == key]
        sim = [r for r in rr if r["source"] == "simulated"]
        summ[key] = {"what": K.DATASETS[key]["what"], "n": len(rr), "expected": len(set_names(key)),
                     "n_fail": sum(not r["pass"] for r in rr),
                     "n_simulated": len(sim), "n_t11_copy": len(rr) - len(sim),
                     "base": K.DATASETS[key]["base"], "t_total_s": K.DATASETS[key]["t_total"],
                     "worst_label_freq_maxdiff_hz": max(float(r["label_freq_maxdiff_hz"]) for r in rr),
                     "worst_label_shape_abs_maxdiff": max(float(r["label_shape_abs_maxdiff"]) for r in rr),
                     "all_copied_keys_identical": all(not r["keys_not_identical"] for r in rr),
                     "x_psd_diff_vs_paper_min": min(float(r["x_psd_diff_vs_paper"]) for r in rr),
                     "seconds_per_structure_median": float(np.median([float(r["seconds"]) for r in sim])) if sim else None}
        summ[key]["pass"] = bool(summ[key]["n"] == summ[key]["expected"] and summ[key]["n_fail"] == 0)
    summ["wall_min_last"] = round((time.time() - t0) / 60, 2)
    summ["workers_last"] = workers
    K.write_json(summ_path, summ)
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in summ.items()}, indent=1))
    return summ


# ---------------------------------------------------------------- crosscheck (c)
def cross_job(name: str) -> dict:
    st = HP._load(C.POP_BROADBAND / name)
    copied = [k for k in st if k not in H.EXC_KEYS]
    files = {"paper": C.POP_BROADBAND / name}
    for tag, d in (("T11_trainval", K.T11_TRAINVAL), ("T11_test", K.T11_TEST)):
        if (d / name).exists():
            files[tag] = d / name
    for key in K.DATASETS:
        if (K.sdir(key) / name).exists():
            files[key] = K.sdir(key) / name
    bad = []
    for tag, f in files.items():
        if tag == "paper":
            continue
        d = HP._load(f)
        bad += [f"{tag}:{k}" for k in copied if not (d[k].dtype == st[k].dtype and d[k].shape == st[k].shape
                                                    and np.array_equal(d[k], st[k]))]
    return {"sample": name, "n_files": len(files), "files": ";".join(files), "not_identical": ";".join(bad),
            "pass": not bad}


def run_crosscheck(workers: int) -> dict:
    names = sorted(set(K.read_list(K.TRAIN2000) + K.read_list(K.VAL2000) + K.read_list(K.TEST_LIST)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(cross_job, names, chunksize=8))
    K.write_csv(K.T13_RESULTS / "prepare_crosscheck.csv", rows)
    hist = {}
    for r in rows:
        hist[r["n_files"]] = hist.get(r["n_files"], 0) + 1
    out = {"rule": "every key outside h_common.EXC_KEYS (graph, labels, matcher record, parameters) "
                   "byte-identical (dtype, shape, values) between the paper sample, T11's samples and every "
                   "T13 variant (records k = 1..3, 120 s, N2000)",
           "structures": len(rows), "files_per_structure_histogram": hist,
           "files_compared": sum(r["n_files"] - 1 for r in rows), "n_fail": sum(not r["pass"] for r in rows),
           "pass": all(r["pass"] for r in rows)}
    g = json.loads((K.T13_RESULTS / "prepare_gates.json").read_text(encoding="utf-8"))
    g["c_labels_crosscheck"] = out
    g["pass"] = all(v["pass"] for k, v in g.items() if isinstance(v, dict) and "pass" in v)
    K.write_json(K.T13_RESULTS / "prepare_gates.json", g)
    print(json.dumps(out, indent=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["gate", "data", "crosscheck"], required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--sets", nargs="+", default=list(K.DATASETS), choices=list(K.DATASETS))
    ap.add_argument("--limit", type=int, default=None, help="first N per set (smoke only)")
    args = ap.parse_args()
    K.T13_RESULTS.mkdir(parents=True, exist_ok=True)
    if args.part == "gate":
        out = run_gates(args.workers)
        print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in out.items()}, indent=1))
        print(json.dumps(out["b_records"], indent=1, default=float)[:3000])
        print("GATES PASS" if out["pass"] else "GATES FAIL")
        return 0 if out["pass"] else 1
    if args.part == "data":
        g = json.loads((K.T13_RESULTS / "prepare_gates.json").read_text(encoding="utf-8"))
        if not g.get("pass"):
            raise SystemExit("prepare_gates.json missing or failing: run --part gate first")
        summ = run_data(args.sets, args.workers, args.limit)
        return 0 if all(summ[k]["pass"] for k in args.sets) else 1
    out = run_crosscheck(args.workers)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
