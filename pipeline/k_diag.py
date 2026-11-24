"""T13 step 4b - what do the nets predict for the torsion modes?  (stored predictions only)

    python tools/run_stage.py pipeline/k_diag.py

Reads the predictions k_eval.py stored (data/T13/predictions/<set>/<run>__best_model.npz: per
structure the predicted and target signed shapes on every graph node).  Per net, set and mode:
  norm_ratio_to_B1   median over structures of ||pred mode|| / ||pred B1||  (a collapsed output
                     is ~0: the MSE-optimal prediction for a target whose sign cannot be inferred)
  mac_pred_vs_<m>    mean MAC of the predicted mode against each target mode (what it resembles)
  mac_to_pred_B1     mean MAC of the predicted mode against the predicted B1
Writes results/T13_feasibility/diag_torsion_output.csv.
"""

from __future__ import annotations

import sys

import numpy as np

import k_common as K

C = K.C
MODES = C.MODE_NAMES


def mac(a, b) -> float:
    return float((a @ b) ** 2 / ((a @ a) * (b @ b) + 1e-30))


def main() -> int:
    rows = []
    for ts in K.TEST_SETS:
        for run in K.ALL_NETS:
            p = K.T13_DATA / "predictions" / ts / f"{run}__best_model.npz"
            if not p.exists():
                continue
            d = np.load(p)
            names = [str(s).removesuffix(".npz") for s in d["samples"]]
            pr = [d[f"{n}__pred"] for n in names]
            tg = [d[f"{n}__target"] for n in names]
            for k, m in enumerate(MODES):
                r = {"test_set": ts, "run": run, "mode": m, "n": len(names),
                     "norm_ratio_to_B1": float(np.median([np.linalg.norm(a[:, k]) / np.linalg.norm(a[:, 0]) for a in pr])),
                     "mac_to_pred_B1": float(np.mean([mac(a[:, k], a[:, 0]) for a in pr]))}
                for j, mt in enumerate(MODES):
                    r[f"mac_pred_vs_{mt}"] = float(np.mean([mac(a[:, k], t[:, j]) for a, t in zip(pr, tg)]))
                rows.append(r)
    K.write_csv(K.T13_RESULTS / "diag_torsion_output.csv", rows)
    for r in rows:
        if r["test_set"] == "paper_varexc" and r["mode"] in ("T1", "T2"):
            print(f"{r['run']:<36} {r['mode']} norm/B1 {r['norm_ratio_to_B1']:.3f}  MAC vs targets "
                  + " ".join(f"{mt} {r[f'mac_pred_vs_{mt}']:.3f}" for mt in MODES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
