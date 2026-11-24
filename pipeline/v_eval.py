"""T20 step 4 - score the deck-forcing networks beside T17's, across excitations and on the real bridges.

    python tools/run_stage.py pipeline/v_eval.py [--workers 8]

Networks: the six T20 runs (deck, walk x seeds 123-125) and T17's three coherency + orientation runs (support
excitation), all coherency + orientation, sign-invariant, N 2000, 1000 epochs.
Synthetic test sets, the paper's 500 test structures, each with its own record:
  paper_varexc   support excitation (T11; T17's main set)
  deck_test      deck forcing + noise floor (v_prepare)
  walk_test      walkers + ambient + noise floor (v_prepare)
so every training excitation is scored on every test excitation.  Real bridges: s5's zero-shot, MAC vs SSI and
vs the FE target (T16's p_eval.real_zeroshot, unchanged).
Scoring is p_eval.score / summarise / real_zeroshot unchanged, with the two new test sets added in memory.
Gates (eval_gates.json): T17's three networks reproduce T17's paper_varexc means (1e-9) and real rows exactly.
Writes results/T20_deck_forcing/: eval_per_sample.csv, eval_summary.csv, eval_cross.csv (train x test
excitation, mean and SD over seeds), real_zeroshot.csv, real_summary.csv, real_arms.csv, real_bridges.csv,
eval_gates.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import v_common as V
import q_common as Q
import p_common as P
import p_eval as PE

K, C = V.K, V.C
MODES = list(C.MODE_NAMES)
T17_NETS = [n for n in Q.RUNS if Q.RUNS[n]["base_arm"] == "cohorient_signinv"]
for _r in (Q.RUNS, V.RUNS):
    P.ALL.update({k: {kk: v[kk] for kk in ("edge", "loss", "amp", "label")} for k, v in _r.items()})
_orig_set_files = PE.set_files


def set_files(ts: str):
    if ts in ("deck_test", "walk_test"):
        return [V.sdir(ts) / n for n in K.read_list(K.TEST_LIST)]
    return _orig_set_files(ts)


PE.set_files = set_files
SETS = ("paper_varexc", "deck_test", "walk_test")
PE.SETS = SETS
TRAIN_EXC = {**{n: "supports" for n in T17_NETS}, **{n: V.RUNS[n]["exc"] for n in V.RUNS}}
SEED_OF = {**{n: Q.RUNS[n]["seed"] for n in T17_NETS}, **{n: V.RUNS[n]["seed"] for n in V.RUNS}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    nets = T17_NETS + list(V.RUNS)
    miss = [n for n in nets if not (K.EC_RUNS / n / "checkpoints" / "best_model.pt").exists()]
    if miss:
        raise SystemExit(f"missing checkpoints: {miss}")
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, max_tasks_per_child=1) as ex:
        for rr in ex.map(PE.score, [(n, ts) for ts in SETS for n in nets]):
            rows += rr
            print(f"[eval] {rr[0]['test_set']:<13} {rr[0]['run']:<40} MAC {np.mean([r['mac_mean'] for r in rr]):.4f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    K.write_csv(V.T20_RESULTS / "eval_per_sample.csv", rows)
    summ = PE.summarise(rows)
    K.write_csv(V.T20_RESULTS / "eval_summary.csv", summ)
    t17 = {(r["test_set"], r["run"]): float(r["mac_mean"]) for r in K.read_csv(Q.T17_RESULTS / "eval_summary.csv")}
    g = {s["run"]: abs(s["mac_mean"] - t17[("paper_varexc", s["run"])]) for s in summ
         if s["run"] in T17_NETS and s["test_set"] == "paper_varexc"}
    gates = {"synthetic_t17": {"max_abs_diff": max(g.values()), "pass": len(g) == 3 and max(g.values()) < 1e-9}}

    cross = []
    for tr in ("supports",) + V.EXCS:
        for ts in SETS:
            ss = [s for s in summ if s["test_set"] == ts and TRAIN_EXC[s["run"]] == tr]
            r = {"train_excitation": tr, "test_set": ts, "n_seeds": len(ss)}
            for col in ["mac_mean"] + [f"mac_{m}" for m in MODES] + ["freq_mae_hz"]:
                v = np.array([s[col] for s in ss])
                r[col], r[f"{col}_sd"] = float(v.mean()), float(v.std(ddof=1))
            cross.append(r)
    K.write_csv(V.T20_RESULTS / "eval_cross.csv", cross)

    rr, _ = PE.real_zeroshot(nets)
    K.write_csv(V.T20_RESULTS / "real_zeroshot.csv", rr)
    ref = {(r["arm"], r["structure"], r["mode"]): float(r["mac_ssi"]) for r in K.read_csv(Q.T17_RESULTS / "real_zeroshot.csv")}
    x = [r for r in rr if r["arm"] in T17_NETS]
    dr = max(abs(float(r["mac_ssi"]) - ref[(r["arm"], r["structure"], r["mode"])]) for r in x)
    gates["real_t17"] = {"rows": len(x), "max_abs_diff": dr, "pass": len(x) == 3 * 36 and dr == 0.0}
    rs = []
    for n in nets:
        g_ = [r for r in rr if r["arm"] == n]
        rs.append({"run": n, "train_excitation": TRAIN_EXC[n], "seed": SEED_OF[n],
                   "mac_ssi_mean": float(np.mean([r["mac_ssi"] for r in g_])),
                   "mac_fe_mean": float(np.mean([r["mac_fe_det"] for r in g_])),
                   "freq_mae_vs_ssi_hz": float(np.mean([abs(r["pred_freq_hz"] - r["ssi_freq_hz"]) for r in g_])),
                   **{f"mac_ssi_{m}": float(np.mean([r["mac_ssi"] for r in g_ if r["mode"] == m])) for m in MODES}})
    K.write_csv(V.T20_RESULTS / "real_summary.csv", rs)
    arms = []
    for tr in ("supports",) + V.EXCS:
        ss = [r for r in rs if r["train_excitation"] == tr]
        out = {"train_excitation": tr, "n_seeds": len(ss), "per_seed_mac_ssi": [round(s["mac_ssi_mean"], 4) for s in ss]}
        for col in ["mac_ssi_mean", "mac_fe_mean", "freq_mae_vs_ssi_hz"] + [f"mac_ssi_{m}" for m in MODES]:
            v = np.array([s[col] for s in ss])
            out[col], out[f"{col}_sd"] = float(v.mean()), float(v.std(ddof=1))
        arms.append(out)
    K.write_csv(V.T20_RESULTS / "real_arms.csv", arms)
    # per bridge, seed means, and the paired difference against T17 (bridge = unit)
    br = []
    for s in C.LOOCV:
        row = {"structure": s}
        for tr in ("supports",) + V.EXCS:
            row[tr] = float(np.mean([r["mac_ssi"] for r in rr if r["structure"] == s and TRAIN_EXC[r["arm"]] == tr]))
        br.append(row)
    K.write_csv(V.T20_RESULTS / "real_bridges.csv", br)
    for tr in V.EXCS:
        d = np.array([b[tr] - b["supports"] for b in br])
        lo, hi = K.bootstrap_ci(d)
        gates.setdefault("_info", {})[f"real_{tr}_minus_supports"] = {"mean": float(d.mean()), "lo": lo, "hi": hi,
                                                                     "bridges_better": int((d > 0).sum())}
    gates["pass"] = all(v["pass"] for k, v in gates.items() if not k.startswith("_"))
    K.write_json(V.T20_RESULTS / "eval_gates.json", gates)
    print(json.dumps(gates, indent=1))
    for r in cross:
        print(f"[cross] train {r['train_excitation']:<9} test {r['test_set']:<13} MAC {r['mac_mean']:.3f} +- {r['mac_mean_sd']:.3f}  "
              + " ".join(f"{m} {r[f'mac_{m}']:.3f}" for m in MODES) + f"  f {r['freq_mae_hz']:.3f}")
    for r in arms:
        print(f"[real]  train {r['train_excitation']:<9} vs SSI {r['mac_ssi_mean']:.3f} +- {r['mac_ssi_mean_sd']:.3f} "
              f"{r['per_seed_mac_ssi']}  " + " ".join(f"{m} {r[f'mac_ssi_{m}']:.3f}" for m in MODES)
              + f"  vs FE {r['mac_fe_mean']:.3f}  f vs SSI {r['freq_mae_vs_ssi_hz']:.2f} Hz")
    return 0 if gates["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
