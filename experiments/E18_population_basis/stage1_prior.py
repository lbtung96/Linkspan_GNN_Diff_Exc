"""E18 stage 1 - the population prior over mode shapes.

    python tools/run_stage.py experiments/E18_population_basis/stage1_prior.py

Reads the main network's training split (Final_results_for_paper, N = 2000: the
training part of it, never validation or test), expands every target mode shape in
popbasis's tensor Legendre basis, and writes per girder count:

  prior.npz        Sigma and tau^2, pooled over the four target modes and per class
                   (bend1, tors1, bend2, tors2), with the basis degrees
  prior_fit.csv    per training shape: class, matcher label, projection MAC (how well
                   the basis represents it), residual
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "Final_results_for_paper" / "pipeline"))
sys.path.insert(0, str(HERE))

import fr_common as C  # noqa: E402
import popbasis as PB  # noqa: E402
from generate_population_dataset_modes4 import _parse_params_json  # noqa: E402

X_TOL_FRAC = 0.02          # nodes on the two outer girders are paired within 2 % of L


def load(name: str):
    with np.load(C.POP_BROADBAND / name, allow_pickle=True) as d:
        p = _parse_params_json(d)
        return (p, d["coords_xy_mm"].astype(np.float64), d["y_mode_shape_signed"].astype(np.float64),
                d["y_freq_hz"].astype(np.float64).reshape(-1))


def main() -> int:
    t0 = time.time()
    train = (C.SPLITS / f"N{C.MAIN_N}_train.txt").read_text(encoding="utf-8").split()
    acc: dict = {}
    rows = []
    nb = (PB.DEG_X + 1) * (PB.DEG_Y + 1)
    for name in train:
        p, c, y, f = load(name)
        ng, L, dy = int(p["Ng"]), float(p["L_mm"]), float(p["dy_mm"])
        B = PB.design(*PB.norm_coords(c, L, ng, dy))
        cl = PB.classes(y, f, c, 0.0, (ng - 1) * dy, X_TOL_FRAC * L)
        for m in range(4):
            a, r = PB.fit(B, y[:, m])
            v = y[:, m] / np.sqrt(np.mean(y[:, m] ** 2))
            fitted = B @ a
            rows.append({"sample": name, "Ng": ng, "label": C.MODE_NAMES[m], "class": cl[m],
                         "freq_hz": float(f[m]),
                         "projection_mac": float((fitted @ v) ** 2 / ((fitted @ fitted) * (v @ v))),
                         "residual_ms": r})
            for key in ("pooled", cl[m]):
                s = acc.setdefault((ng, key), [np.zeros((nb, nb)), 0.0, 0])
                s[0] += np.outer(a, a)
                s[1] += r
                s[2] += 1
    out = {"deg_x": PB.DEG_X, "deg_y": PB.DEG_Y, "eta_scale": PB.ETA_SCALE, "n_train": len(train)}
    for (ng, key), (S, r, k) in acc.items():
        out[f"Sigma_Ng{ng}_{key}"] = S / k
        out[f"tau2_Ng{ng}_{key}"] = np.float64(r / k)
        out[f"count_Ng{ng}_{key}"] = np.int64(k)
    np.savez(HERE / "prior.npz", **out)
    with (HERE / "prior_fit.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    pm = np.array([r["projection_mac"] for r in rows])
    print(f"{len(train)} training structures, {len(rows)} shapes, basis {nb}, {time.time() - t0:.0f}s")
    print(f"projection MAC: median {np.median(pm):.5f}, 1st percentile {np.percentile(pm, 1):.5f}, min {pm.min():.5f}")
    for ng in C.NG_VALUES:
        print(f"  Ng {ng}: " + ", ".join(f"{k} {int(out[f'count_Ng{ng}_{k}'])}" for k in ("pooled",) + PB.CLASSES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
