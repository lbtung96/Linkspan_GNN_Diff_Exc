"""T11 step 1 - gates, then the varied-excitation training/validation and test samples.

    python tools/run_stage.py Final_results_for_paper/EC_comments/pipeline/h_prepare.py --part gate --workers 10
    python tools/run_stage.py Final_results_for_paper/EC_comments/pipeline/h_prepare.py --part data --workers 10

--part gate (results/T11_varied_excitation_pilot/prepare_gates.json)
  (a) reproduction: with the patch disabled (SEED 12345) the re-simulation reproduces the
      stored paper features of 40 structures (10 per Ng, from the N = 600 training and
      validation lists and the test list) to <= 1e-5 (the FE re-solve tolerance of CLAUDE.md,
      T4's resim gate): x_psd, edge_attr_cpsd(_resonance), edge_attr_cpsd_full; x_psd_row_scale
      relative; graph (node ids, edge index) identical; labels re-solved (selected-mode
      frequencies, y_mode_shape_abs) within 1e-5.
  (b) records: the support joints' (graph degree-1 nodes) absolute acceleration IS the injected
      base record (band-passed).  Under 12345 two different structures share it (the T5 finding,
      shown directly); with the patch they do not (max |correlation| over all support-row
      pairs); the RNG draw is a pure function of the seed (bit for bit); the same structure
      re-simulated twice with its seed gets the same record and features (<= 1e-5: the support
      rows also carry a tiny modal part, so ARPACK's start vector reaches them); its
      training-base and test-base records differ.

--part data
  N600_train (508) + N600_val (92) with base 20261125 -> data/T11/trainval_varexc/samples/
  test (500)                       with base 20261126 -> data/T11/paper_varexc/samples/
  Each written sample = the stored sample with the five excitation-dependent arrays replaced
  (h_common.EXC_KEYS); read back and checked: every other key byte-identical (dtype, shape,
  values); graph identical to the re-solve; labels re-solved within 1e-5; node rows peak at 1;
  |coherency| <= 1; features differ from the stored (paper-record) ones.  One row per
  structure in results/.../prepare_checks.csv.  Resumable (a sample whose .npz and
  .check.json exist is skipped).
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402

import h_common as H  # noqa: E402

C = H.C
TOL = 1e-5


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def _maxdiff(a, b) -> float:
    return float(np.abs(np.asarray(a, float) - np.asarray(b, float)).max())


def _relmax(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((np.abs(a - b) / np.maximum(np.abs(b), 1e-300)).max())


def compare_to_stored(res: dict, st: dict) -> dict:
    f = res["features"]
    return {
        "graph_ids_equal": bool(np.array_equal(res["graph_node_ids"], st["graph_node_ids"])),
        "edge_index_equal": bool(np.array_equal(res["edge_index"], st["edge_index"])),
        "freq_bins_maxdiff": _maxdiff(res["freq_bins_hz"], st["freq_bins_hz"]),
        "label_freq_maxdiff_hz": _maxdiff(res["freq_selected_hz"], st["matched_freq_hz"]),
        "label_shape_abs_maxdiff": _maxdiff(res["y_mode_shape_abs"], st["y_mode_shape_abs"]),
        "x_psd_maxdiff": _maxdiff(f["x_psd"], st["x_psd"]),
        "x_psd_row_scale_relmax": _relmax(f["x_psd_row_scale"], st["x_psd_row_scale"]),
        "edge_cpsd8_maxdiff": _maxdiff(f["edge_attr_cpsd"], st["edge_attr_cpsd"]),
        "edge_cpsd_res_maxdiff": _maxdiff(f["edge_attr_cpsd_resonance"], st["edge_attr_cpsd_resonance"]),
        "edge_full_maxdiff": _maxdiff(f["edge_attr_cpsd_full"], st["edge_attr_cpsd_full"]),
    }


# ---------------------------------------------------------------- gate (a)
def gate_a_job(job: tuple) -> dict:
    name, split = job
    path = C.POP_BROADBAND / name
    t0 = time.time()
    res = H.resimulate_payload(path, H.PAPER_EXC_SEED)
    st = _load(path)
    out = {"sample": name, "split": split, "Ng": int(st["Ng"]), **compare_to_stored(res, st),
           "seconds": round(time.time() - t0, 2)}
    num = [k for k in out if k.endswith(("maxdiff", "relmax", "maxdiff_hz"))]
    out["pass"] = bool(out["graph_ids_equal"] and out["edge_index_equal"] and all(out[k] <= TOL for k in num))
    return out


def gate_a_sample() -> list[tuple[str, str]]:
    lists = {"train": H.read_list(H.TRAIN_LIST), "val": H.read_list(H.VAL_LIST), "test": H.read_list(H.TEST_LIST)}
    take = {"train": 4, "val": 2, "test": 4}
    out = []
    for ng in C.NG_VALUES:
        for split, k in take.items():
            out += [(n, split) for n in [n for n in lists[split] if f"_Ng{ng}_" in n][:k]]
    return out


# ---------------------------------------------------------------- gate (b)
def gate_b_job(job: tuple) -> dict:
    name, kind = job
    path = C.POP_BROADBAND / name
    idx = H.sample_index(name)
    if kind == "paper":
        res = H.resimulate_payload(path, H.PAPER_EXC_SEED)
    elif kind == "train":
        res = H.resimulate_payload(path, H.exc_seed(H.TRAIN_EXC_BASE, idx))
    elif kind == "test":
        res = H.resimulate_payload(path, H.exc_seed(H.TEST_EXC_BASE, idx))
    else:
        raise ValueError(kind)
    sup = H.support_rows(res["edge_index"], res["graph_node_ids"].size)
    return {"name": name, "kind": kind, "support": sup, "rec": res["a3"][sup], "rms": np.sqrt(np.mean(res["a3"] ** 2, axis=1)),
            "features": res["features"]}


def max_abs_corr(A: np.ndarray, B: np.ndarray) -> float:
    A = (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True)
    B = (B - B.mean(1, keepdims=True)) / B.std(1, keepdims=True)
    return float(np.abs(A @ B.T / A.shape[1]).max())


def run_gate_b(ex) -> dict:
    tr = H.read_list(H.TRAIN_LIST)
    a, b = tr[0], [n for n in tr if "_Ng2_" in n][1]          # two Ng = 2 decks
    c = [n for n in tr if "_Ng4_" in n][0]                      # a four-girder deck
    jobs = [(a, "paper"), (b, "paper"), (c, "paper"), (a, "train"), (a, "train"), (b, "train"), (c, "train"),
            (a, "test")]
    r = list(ex.map(gate_b_job, jobs))
    pa, pb, pc, ta1, ta2, tb, tc, sa = r
    support_is_record = {}
    # support rows hold the injected record: louder than every deck node by far
    for x in (pa, pc):
        rms = x["rms"]
        deck = np.setdiff1d(np.arange(rms.size), x["support"])
        ratio = rms[x["support"]] / rms[deck].max()
        support_is_record[x["name"]] = {"n_degree1": int(x["support"].size),
                                        "n_loud_degree1_rms_gt_10x_deck": int(np.sum(ratio > 10)),
                                        "degree1_rms_over_max_deck_rms": [round(float(v), 3) for v in ratio]}
    # the draw itself (simulator lines 344-345): default_rng(seed).standard_normal is a pure function of the seed
    sq = H.exc_seed(H.TRAIN_EXC_BASE, H.sample_index(a))
    d1 = np.random.default_rng(sq).standard_normal((6, 4000))
    d2 = np.random.default_rng(H.exc_seed(H.TRAIN_EXC_BASE, H.sample_index(a))).standard_normal((6, 4000))
    d3 = np.random.default_rng(sq).standard_normal((6, 4000))          # the same SeedSequence object reused
    rec_rel = float(np.abs(ta1["rec"] - ta2["rec"]).max() / np.sqrt(np.mean(ta1["rec"] ** 2)))
    out = {
        "structures": {"A": a, "B": b, "C": c},
        "support_rows": support_is_record,
        "paper_seed_A_vs_B_max_abs_corr": max_abs_corr(pa["rec"], pb["rec"]),
        "paper_seed_A_vs_C_max_abs_corr": max_abs_corr(pa["rec"], pc["rec"]),
        "varied_A_vs_B_max_abs_corr": max_abs_corr(ta1["rec"], tb["rec"]),
        "varied_A_vs_C_max_abs_corr": max_abs_corr(ta1["rec"], tc["rec"]),
        "varied_A_vs_paper_A_max_abs_corr": max_abs_corr(ta1["rec"], pa["rec"]),
        "train_base_vs_test_base_A_max_abs_corr": max_abs_corr(ta1["rec"], sa["rec"]),
        "rng_draw_reproducible_bit_identical": bool(np.array_equal(d1, d2) and np.array_equal(d1, d3)),
        "same_structure_same_seed_record_maxdiff_rel_rms": rec_rel,
        "same_structure_same_seed_feature_maxdiff": max(
            _maxdiff(ta1["features"][k], ta2["features"][k]) for k in ("x_psd", "edge_attr_cpsd_full")),
        "varied_vs_paper_A_x_psd_maxdiff": _maxdiff(ta1["features"]["x_psd"], pa["features"]["x_psd"]),
        "varied_vs_paper_A_edge_full_maxdiff": _maxdiff(ta1["features"]["edge_attr_cpsd_full"],
                                                        pa["features"]["edge_attr_cpsd_full"]),
    }
    out["pass"] = bool(out["paper_seed_A_vs_B_max_abs_corr"] > 0.999
                       and out["varied_A_vs_B_max_abs_corr"] < 0.2 and out["varied_A_vs_C_max_abs_corr"] < 0.2
                       and out["varied_A_vs_paper_A_max_abs_corr"] < 0.2
                       and out["train_base_vs_test_base_A_max_abs_corr"] < 0.2
                       and out["rng_draw_reproducible_bit_identical"]
                       and out["same_structure_same_seed_record_maxdiff_rel_rms"] <= TOL
                       and out["same_structure_same_seed_feature_maxdiff"] <= TOL
                       and out["varied_vs_paper_A_x_psd_maxdiff"] > 1e-2
                       and all(v["n_loud_degree1_rms_gt_10x_deck"] >= 2 for v in support_is_record.values()))
    out["thresholds"] = {"shared_record_corr_gt": 0.999, "independent_record_corr_lt": 0.2,
                         "feature_tol": TOL, "changed_features_x_psd_gt": 1e-2}
    return out


# ---------------------------------------------------------------- data
def out_dir_for(split: str) -> Path:
    return H.TEST_DIR if split == "test" else H.TRAINVAL_DIR


def base_for(split: str) -> int:
    return H.TEST_EXC_BASE if split == "test" else H.TRAIN_EXC_BASE


def data_job(job: tuple) -> dict:
    name, split = job
    src = C.POP_BROADBAND / name
    dst_dir = out_dir_for(split)
    dst = dst_dir / name
    chk = dst_dir / (Path(name).stem + ".check.json")
    if dst.exists() and chk.exists():
        return json.loads(chk.read_text(encoding="utf-8"))
    t0 = time.time()
    idx = H.sample_index(name)
    seed = H.exc_seed(base_for(split), idx)
    res = H.resimulate_payload(src, seed)
    st = _load(src)
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

    back = _load(dst)
    copied = [k for k in st if k not in H.EXC_KEYS]
    not_identical = [k for k in copied if not (back[k].dtype == st[k].dtype and back[k].shape == st[k].shape
                                               and np.array_equal(back[k], st[k]))]
    cmp = compare_to_stored(res, st)          # features here differ by design (new record)
    x, e = back["x_psd"], back["edge_attr_cpsd_full"]
    row = {"sample": name, "split": split, "Ng": int(st["Ng"]), "sample_index": idx,
           "exc_seed": H.seed_repr(seed),
           "keys_total": len(st), "keys_replaced": len(H.EXC_KEYS), "keys_copied_identical": len(copied) - len(not_identical),
           "keys_not_identical": ";".join(not_identical),
           "same_key_set": sorted(back) == sorted(st),
           "graph_ids_equal": cmp["graph_ids_equal"], "edge_index_equal": cmp["edge_index_equal"],
           "label_freq_maxdiff_hz": cmp["label_freq_maxdiff_hz"],
           "label_shape_abs_maxdiff": cmp["label_shape_abs_maxdiff"],
           "x_rowmax_min": float(np.abs(x).max(axis=1).min()), "x_rowmax_max": float(np.abs(x).max(axis=1).max()),
           "edge_absmax": float(np.abs(e).max()),
           "x_psd_diff_vs_paper": cmp["x_psd_maxdiff"], "edge_full_diff_vs_paper": cmp["edge_full_maxdiff"],
           "seconds": round(time.time() - t0, 2)}
    row["pass"] = bool(not not_identical and row["same_key_set"] and row["graph_ids_equal"]
                       and row["edge_index_equal"] and row["label_freq_maxdiff_hz"] <= TOL
                       and row["label_shape_abs_maxdiff"] <= TOL
                       and abs(row["x_rowmax_min"] - 1) <= 1e-9 and abs(row["x_rowmax_max"] - 1) <= 1e-9
                       and row["edge_absmax"] <= 1 + 1e-5 and row["x_psd_diff_vs_paper"] > 1e-3)
    chk.write_text(json.dumps(row, indent=1), encoding="utf-8")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["gate", "data"], required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="first N structures per split (smoke only)")
    args = ap.parse_args()
    H.T11_RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.part == "gate":
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows_a = list(ex.map(gate_a_job, gate_a_sample()))
            gb = run_gate_b(ex)
        H.write_csv(H.T11_RESULTS / "prepare_gate_a.csv", rows_a)
        num = [k for k in rows_a[0] if k.endswith(("maxdiff", "relmax", "maxdiff_hz"))]
        ga = {"n_structures": len(rows_a), "tol": TOL, "pass": all(r["pass"] for r in rows_a),
              "worst": {k: max(r[k] for r in rows_a) for k in num},
              "graph_identical_all": all(r["graph_ids_equal"] and r["edge_index_equal"] for r in rows_a),
              "per_ng": {int(g): sum(r["Ng"] == g for r in rows_a) for g in C.NG_VALUES},
              "per_split": {s: sum(r["split"] == s for r in rows_a) for s in ("train", "val", "test")},
              "seconds_median": float(np.median([r["seconds"] for r in rows_a]))}
        out = {"a_reproduction_paper_seed": ga, "b_records": gb, "pass": ga["pass"] and gb["pass"],
               "wall_s": round(time.time() - t0, 1)}
        H.write_json(H.T11_RESULTS / "prepare_gates.json", out)
        print(json.dumps({"a": {k: ga[k] for k in ("pass", "worst")}, "b": gb}, indent=1, default=float))
        print("GATES PASS" if out["pass"] else "GATES FAIL")
        return 0 if out["pass"] else 1

    jobs = []
    for split, lst in (("train", H.TRAIN_LIST), ("val", H.VAL_LIST), ("test", H.TEST_LIST)):
        names = H.read_list(lst)[: args.limit] if args.limit else H.read_list(lst)
        jobs += [(n, split) for n in names]
    for d in (H.TRAINVAL_DIR, H.TEST_DIR):
        d.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(data_job, jobs, chunksize=2)):
            rows.append(r)
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                print(f"[data] {i + 1}/{len(jobs)}  fails so far {sum(not x['pass'] for x in rows)}  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    tag = "" if not args.limit else f"_limit{args.limit}"
    H.write_csv(H.T11_RESULTS / f"prepare_checks{tag}.csv", rows)
    summ = {"n": len(rows), "pass": all(r["pass"] for r in rows),
            "n_fail": sum(not r["pass"] for r in rows),
            "per_split": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
            "worst_label_freq_maxdiff_hz": max(r["label_freq_maxdiff_hz"] for r in rows),
            "worst_label_shape_abs_maxdiff": max(r["label_shape_abs_maxdiff"] for r in rows),
            "all_copied_keys_identical": all(not r["keys_not_identical"] for r in rows),
            "graph_identical_all": all(r["graph_ids_equal"] and r["edge_index_equal"] for r in rows),
            "x_psd_diff_vs_paper_min": min(r["x_psd_diff_vs_paper"] for r in rows),
            "x_psd_diff_vs_paper_median": float(np.median([r["x_psd_diff_vs_paper"] for r in rows])),
            "edge_full_diff_vs_paper_median": float(np.median([r["edge_full_diff_vs_paper"] for r in rows])),
            "seconds_per_structure_median": float(np.median([r["seconds"] for r in rows])),
            "workers": args.workers, "wall_min": round((time.time() - t0) / 60, 2)}
    H.write_json(H.T11_RESULTS / f"prepare_summary{tag}.json", summ)
    print(json.dumps(summ, indent=1))
    return 0 if summ["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
