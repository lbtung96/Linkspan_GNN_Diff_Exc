"""T5 (EC-15) - checks on the fresh test sets: novelty, distribution, rejections, cost.

    python tools/run_stage.py pipeline/c_fresh_checks.py

  1. count and balance: 500 structures, 125 per Ng, in each fresh set;
  2. the two fresh sets hold the same structures (identical params_json, FE targets and
     graphs) and differ only in the excitation (node PSDs differ);
  3. novelty: no fresh params_json equals any of the 5500 paper structures; the nearest paper
     structure (same Ng, standardised numeric parameters) is no closer than the paper's own
     test structures are to the paper's training pool;
  4. distribution: fresh vs the paper's fixed test set, per parameter and per target
     frequency, two-sample Kolmogorov-Smirnov (pooled over Ng and per Ng), and categorical
     shares;
  5. rejections by reason (as s1 parses failures.log) per Ng, against the paper's;
  6. generation time.

Writes fresh_checks.json, fresh_param_distribution.csv, fresh_rejections.csv in
results/T5_fresh_test_baselines/.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

import ec_common as E

C = E.C
from generate_population_dataset_modes4 import _parse_params_json  # noqa: E402

OUT = E.task_dir("T5")
T5_DATA = E.EC_DATA / "T5"
FRESH = ("fresh_test", "fresh_test_paperexc")
NUMERIC = ["L_mm", "Nx", "dy_mm", "E", "rho", "bottom_beam.x_frac", "bottom_beam.overhang_mm",
           "spring.kx_kN_per_m", "spring.kz_kN_per_m", "nsm.total_mass_tonne",
           "balcony.length_mm", "balcony.dx_mm", "balcony.nBeams", "balcony.x_center_frac"]
CATEG = ["diag_pattern", "balcony.enable", "balcony.side"]


def flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def load(path: Path, arrays: bool = False) -> dict:
    with np.load(path, allow_pickle=True) as d:
        p = _parse_params_json(d)
        r = {"name": path.name, "params_raw": str(d["params_json"]), "p": flat(p), "Ng": int(p["Ng"]),
             "freq": d["y_freq_hz"].astype(float).reshape(-1), "n_nodes": int(d["n_graph_nodes"])}
        if arrays:
            r.update(x_psd=d["x_psd"], y=d["y_mode_shape_signed"])
        return r


def vec(rec: dict) -> np.ndarray:
    p = rec["p"]
    bal = bool(p.get("balcony.enable", False))
    return np.array([float(p.get(k, 0.0)) if (not k.startswith("balcony.") or bal) else 0.0 for k in NUMERIC])


def main() -> int:
    paper_all = sorted(C.POP_BROADBAND.glob("sample_*_Ng*_phase1_gnn.npz"))
    test_names = set((C.SPLITS / "test.txt").read_text(encoding="utf-8").split())
    pool_names = set()
    for g in C.NG_VALUES:
        pool_names |= {f.name for f in paper_all if f"_Ng{g}_" in f.name and f.name not in test_names}
    print(f"paper population: {len(paper_all)} files", flush=True)
    paper = [load(f) for f in paper_all]
    fresh = {s: [load(f, arrays=True) for f in sorted((T5_DATA / s / "samples_phase2_bandpass").glob("sample_*.npz"))] for s in FRESH
             if (T5_DATA / s / "samples_phase2_bandpass").exists()}
    chk: dict = {}

    # 1. count and balance
    chk["counts"] = {s: {"n": len(v), "per_Ng": dict(Counter(r["Ng"] for r in v))} for s, v in fresh.items()}
    # 2. the two fresh sets: same structures, different excitation
    if len(fresh) == 2:
        a, b = fresh[FRESH[0]], fresh[FRESH[1]]
        same_par = all(x["params_raw"] == y["params_raw"] for x, y in zip(a, b))
        same_y = max(float(np.abs(x["y"] - y["y"]).max()) for x, y in zip(a, b))
        same_f = max(float(np.abs(x["freq"] - y["freq"]).max()) for x, y in zip(a, b))
        dpsd = [float(np.abs(x["x_psd"] - y["x_psd"]).max()) for x, y in zip(a, b)]
        chk["fresh_sets_paired"] = {"same_params_json_all": same_par, "max_abs_diff_targets": same_y,
                                    "max_abs_diff_freq_hz": same_f, "node_psd_max_abs_diff_min": min(dpsd),
                                    "node_psd_max_abs_diff_median": float(np.median(dpsd))}
    # 3. novelty
    paper_raw = {r["params_raw"] for r in paper}
    Xp = np.array([vec(r) for r in paper])
    mu, sd = Xp.mean(0), Xp.std(0) + 1e-12
    ng_p = np.array([r["Ng"] for r in paper])
    is_pool = np.array([r["name"] in pool_names for r in paper])
    is_test = np.array([r["name"] in test_names for r in paper])

    def nn_dist(x: np.ndarray, ng: int, mask: np.ndarray) -> float:
        idx = np.where(mask & (ng_p == ng))[0]
        return float(np.min(np.linalg.norm((Xp[idx] - x) / sd, axis=1)))

    nov = {}
    for s, v in fresh.items():
        dup = sum(r["params_raw"] in paper_raw for r in v)
        d_all = np.array([nn_dist(vec(r), r["Ng"], np.ones_like(is_pool)) for r in v])
        d_pool = np.array([nn_dist(vec(r), r["Ng"], is_pool) for r in v])
        nov[s] = {"identical_params_json_in_paper_population": int(dup),
                  "nn_dist_to_any_paper_structure_min": float(d_all.min()),
                  "nn_dist_to_paper_pool_median": float(np.median(d_pool)),
                  "nn_dist_to_paper_pool_min": float(d_pool.min())}
    d_ref = np.array([nn_dist(Xp[i], ng_p[i], is_pool) for i in np.where(is_test)[0]])
    nov["paper_test_to_paper_pool"] = {"nn_dist_median": float(np.median(d_ref)), "nn_dist_min": float(d_ref.min())}
    nov["space"] = f"standardised over the 5500 paper structures: {NUMERIC} (balcony fields 0 when no balcony), same Ng"
    chk["novelty"] = nov

    # 4. distribution vs the paper's fixed test set
    ptest = [r for r in paper if r["name"] in test_names]
    rows = []
    if FRESH[0] in fresh:
        ft = fresh[FRESH[0]]

        def col(recs, key):
            if key.startswith("freq_"):
                return np.array([r["freq"][C.MODE_NAMES.index(key[5:])] for r in recs])
            if key == "n_nodes":
                return np.array([r["n_nodes"] for r in recs], float)
            bal_only = key.startswith("balcony.")
            return np.array([float(r["p"][key]) for r in recs if not bal_only or r["p"]["balcony.enable"]])

        for key in NUMERIC + [f"freq_{m}" for m in C.MODE_NAMES] + ["n_nodes"]:
            for g in (None,) + C.NG_VALUES:
                a = [r for r in ft if g is None or r["Ng"] == g]
                b = [r for r in ptest if g is None or r["Ng"] == g]
                xa, xb = col(a, key), col(b, key)
                ks = ks_2samp(xa, xb)
                rows.append({"quantity": key, "Ng": "all" if g is None else g, "n_fresh": len(xa), "n_paper_test": len(xb),
                             "mean_fresh": float(xa.mean()), "mean_paper_test": float(xb.mean()),
                             "sd_fresh": float(xa.std()), "sd_paper_test": float(xb.std()),
                             "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue)})
        cat = {}
        for key in CATEG:
            for nm, recs in (("fresh", ft), ("paper_test", ptest)):
                cat.setdefault(key, {})[nm] = {str(k): v / len(recs) for k, v in
                                               sorted(Counter(str(r["p"].get(key)) for r in recs).items())}
        chk["categorical_shares"] = cat
        ps = np.array([r["ks_p"] for r in rows])
        chk["ks_summary"] = {"n_tests": len(rows), "n_p_below_0.05": int((ps < 0.05).sum()),
                             "n_p_below_0.05_over_n_tests_expected_under_null": 0.05,
                             "min_p": float(ps.min()),
                             "min_p_quantity": rows[int(ps.argmin())]["quantity"] + f" (Ng {rows[int(ps.argmin())]['Ng']})",
                             "bonferroni_0.05_pass": bool(ps.min() > 0.05 / len(rows))}
        with (OUT / "fresh_param_distribution.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # 5. rejections, parsed as s1 does
    rej = []
    for s in fresh:
        c = Counter()
        for line in (T5_DATA / s / "failures.log").read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"Ng=(\d) attempt=\d+ error=(\w+): (.*)", line)
            if m:
                msg = re.sub(r"selected-mode MAC below .*", "selected-mode MAC below gate", m.group(3))
                msg = re.sub(r"(No feasible candidate survives MMI scoring for target \w+).*", r"\1", msg)
                c[(int(m.group(1)), msg[:90])] += 1
        for (g, msg), n in sorted(c.items()):
            rej.append({"set": s, "Ng": g, "reason": msg, "count": n, "accepted": 125})
    for r in csv.DictReader((C.RESULTS / "s1_rejections.csv").open(encoding="utf-8")):
        if r["arm"] == "broadband":
            rej.append({"set": "paper_population", "Ng": int(r["Ng"]), "reason": r["reason"], "count": int(r["count"]),
                        "accepted": C.N_TOTAL // 4})
    for r in rej:
        r["per_100_accepted"] = 100.0 * r["count"] / r["accepted"]
    with (OUT / "fresh_rejections.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rej[0]))
        w.writeheader()
        w.writerows(rej)
    tot = {}
    for r in rej:
        t = tot.setdefault(r["set"], {})
        t[r["Ng"]] = t.get(r["Ng"], 0) + r["per_100_accepted"]
    chk["logged_rejections_per_100_accepted"] = tot

    # 6. time
    chk["generation"] = {}
    for s in fresh:
        cfg = json.loads((T5_DATA / s / "generation_config.json").read_text(encoding="utf-8"))
        chk["generation"][s] = {k: cfg.get(k) for k in ("param_seed", "excitation_seed", "seconds", "started_utc",
                                                        "finished_utc", "threads")}
    pc = json.loads((C.POP_BROADBAND_ROOT / "generation_config.json").read_text(encoding="utf-8"))
    chk["generation"]["paper_broadband_5500"] = {"seconds": pc.get("seconds"), "per_structure_s": pc.get("seconds", 0) / C.N_TOTAL}
    (OUT / "fresh_checks.json").write_text(json.dumps(chk, indent=1, default=str), encoding="utf-8")
    print(json.dumps(chk, indent=1, default=str))
    ok = all(v["n"] == 500 and all(n == 125 for n in v["per_Ng"].values()) for v in chk["counts"].values()) \
        and all(nov[s]["identical_params_json_in_paper_population"] == 0 for s in fresh)
    if "fresh_sets_paired" in chk:
        ok = ok and chk["fresh_sets_paired"]["same_params_json_all"] and chk["fresh_sets_paired"]["max_abs_diff_targets"] <= 1e-5
    print("CHECKS PASS" if ok else "CHECK FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
