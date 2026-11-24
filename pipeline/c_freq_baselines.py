"""T5 (EC-25a) - frequency baselines that never look at a response record.

    python tools/run_stage.py Final_results_for_paper/EC_comments/pipeline/c_freq_baselines.py

Fitted on the main network's training split only (data/splits/N2000_train.txt: 1700 decks;
never its validation or a test set), scored on the paper's fixed test set and the fresh one:

  mean        the training-set mean of each mode's frequency
  ng_mean     the same, per girder count
  geom_knn    k = 10 nearest training decks of the same girder count, in standardised
              sampled parameters from params_json (z-scored on the training split):
              L, Nx, dy, E, rho, spring kx and kz, bottom-beam x_frac and overhang, non-
              structural mass, balcony on/off, side (one-hot), beams, centre, spacing and
              length (0 when there is no balcony), diagonal pattern (one-hot).  E and rho
              are material properties the drawings may not give; the kNN is therefore an
              optimistic geometry-only predictor.
  geom_knn_noEr  the same without E and rho

The network's numbers (edgegat_N2000, best_model) are read from c_rescore's per-sample
table, on the same structures.  Writes freq_baselines_per_sample.csv and freq_baselines.csv
(MAE per mode and mean, bootstrap 95 % intervals over structures).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

import ec_common as E

C = E.C
from generate_population_dataset_modes4 import _parse_params_json  # noqa: E402
from s4_indomain import bootstrap_ci  # noqa: E402

OUT = E.task_dir("T5")
T5_DATA = E.EC_DATA / "T5"
M = list(C.MODE_NAMES)
K = 10
NUM = ["L_mm", "Nx", "dy_mm", "E", "rho"]
SUB = [("spring", "kx_kN_per_m"), ("spring", "kz_kN_per_m"), ("bottom_beam", "x_frac"),
       ("bottom_beam", "overhang_mm"), ("nsm", "total_mass_tonne")]
BAL = ["nBeams", "x_center_frac", "dx_mm", "length_mm"]
SIDES = ["left", "right", "both"]
DIAG = ["global_x", "staggered_x", "single_partial_x", "None"]


def features(p: dict) -> tuple[np.ndarray, list[str]]:
    b = p["balcony"]
    on = bool(b["enable"])
    v = [float(p[k]) for k in NUM] + [float(p[a][k]) for a, k in SUB] + [float(on)]
    v += [float(on and b["side"] == s) for s in SIDES] + [float(b[k]) if on else 0.0 for k in BAL]
    v += [float(p["diag_pattern"] == d) for d in DIAG]
    names = NUM + [f"{a}.{k}" for a, k in SUB] + ["balcony.enable"] + [f"side_{s}" for s in SIDES] \
        + [f"balcony.{k}" for k in BAL] + [f"diag_{d}" for d in DIAG]
    return np.array(v), names


def load(files: list[Path]) -> dict:
    X, F, ng, names = [], [], [], []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            p = _parse_params_json(d)
            F.append(d["y_freq_hz"].astype(np.float64).reshape(-1))
        x, fn = features(p)
        X.append(x)
        ng.append(int(p["Ng"]))
        names.append(f.name)
    return {"X": np.array(X), "F": np.array(F), "ng": np.array(ng), "names": names, "feat_names": fn}


def predict(tr: dict, te: dict, drop: list[str]) -> np.ndarray:
    keep = [i for i, n in enumerate(tr["feat_names"]) if n not in drop]
    Xtr, Xte = tr["X"][:, keep], te["X"][:, keep]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    out = np.zeros_like(te["F"])
    for i in range(len(Zte)):
        idx = np.where(tr["ng"] == te["ng"][i])[0]
        d = np.linalg.norm(Ztr[idx] - Zte[i], axis=1)
        out[i] = tr["F"][idx[np.argsort(d, kind="stable")[:K]]].mean(0)
    return out


def main() -> int:
    tr = load([C.POP_BROADBAND / n for n in (C.SPLITS / f"N{C.MAIN_N}_train.txt").read_text(encoding="utf-8").split()])
    print(f"training split: {len(tr['names'])} decks", flush=True)
    sets = {"paper": [C.POP_BROADBAND / n for n in (C.SPLITS / "test.txt").read_text(encoding="utf-8").split()],
            "fresh_test": sorted((T5_DATA / "fresh_test" / "samples_phase2_bandpass").glob("sample_*.npz"))}
    net = {}
    for r in csv.DictReader((OUT / "rescore_per_sample.csv").open(encoding="utf-8")):
        if r["run"] == C.sweep_run_name(C.MAIN_N) and r["checkpoint"] == "best_model":
            net[(r["test_set"], r["sample"])] = np.array([float(r[f"freq_pred_{m}_hz"]) for m in M])
    per, summ = [], []
    for ts, files in sets.items():
        if not files:
            continue
        te = load(files)
        preds = {"mean": np.tile(tr["F"].mean(0), (len(te["F"]), 1)),
                 "ng_mean": np.array([tr["F"][tr["ng"] == g].mean(0) for g in te["ng"]]),
                 "geom_knn": predict(tr, te, []),
                 "geom_knn_noEr": predict(tr, te, ["E", "rho"])}
        if all((ts, n) in net for n in te["names"]):
            preds["network_N2000"] = np.array([net[(ts, n)] for n in te["names"]])
        for meth, P in preds.items():
            err = np.abs(P - te["F"])
            for i, n in enumerate(te["names"]):
                per.append({"test_set": ts, "method": meth, "sample": n, "Ng": int(te["ng"][i]),
                            **{f"abs_err_{m}_hz": float(err[i, k]) for k, m in enumerate(M)},
                            "mae_hz": float(err[i].mean())})
            for g in ("all",) + C.NG_VALUES:
                sel = np.ones(len(err), bool) if g == "all" else te["ng"] == g
                row = {"test_set": ts, "method": meth, "Ng": g, "n": int(sel.sum())}
                for k, m in enumerate(M):
                    row[f"mae_{m}_hz"] = float(err[sel, k].mean())
                row["mae_hz"] = float(err[sel].mean())
                if g == "all":
                    row["mae_hz_ci95_lo"], row["mae_hz_ci95_hi"] = bootstrap_ci(err[sel].mean(1))
                    row["rel_err_pct_mean"] = float((err[sel] / te["F"][sel]).mean() * 100)
                summ.append(row)
        print(f"{ts}: " + "  ".join(f"{s['method']} {s['mae_hz']:.3f}" for s in summ
                                    if s["test_set"] == ts and s["Ng"] == "all"), flush=True)
    for path, rr in ((OUT / "freq_baselines_per_sample.csv", per), (OUT / "freq_baselines.csv", summ)):
        keys: list[str] = []
        for r in rr:
            keys += [k for k in r if k not in keys]
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rr)
    ok = any(s["test_set"] == "paper" and s["method"] == "network_N2000" and s["Ng"] == "all"
             and abs(s["mae_hz"] - 0.2186514123082161) < 1e-5 for s in summ)
    print(f"[check] network freq MAE on the paper test set equals s4's 0.21865 Hz (1e-5; truth read in float64 here): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
