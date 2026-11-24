"""Gate 1 - feature-distribution check.

Compares node/edge feature statistics of the corrected real datasets against
(a) the synthetic training set and (b) the deterministic FE features, plus the
old (broken) production real features for reference.
"""

from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from structures import LOOCV  # single source of truth
OUT = Path(__file__).resolve().parent
STRUCTURES = list(LOOCV)


def stats(x: np.ndarray, e: np.ndarray, label: str, group: str) -> dict:
    rm = np.max(np.abs(x), axis=1)
    ae = np.abs(e)
    return {
        "group": group,
        "label": label,
        "n_nodes": int(x.shape[0]),
        "n_edges": int(e.shape[0]),
        "node_rowmax_min": float(rm.min()),
        "node_rowmax_median": float(np.median(rm)),
        "node_rowmax_max": float(rm.max()),
        "edge_absv_median": float(np.median(ae)),
        "edge_absv_max": float(ae.max()),
    }


def main() -> int:
    rows: list[dict] = []

    # (a) synthetic training set - 40 samples is plenty for a distribution check
    paths = sorted(glob.glob(str(REPO / "population_dataset_modes4" /
                                 "samples_phase2_bandpass" / "*.npz")))[:40]
    rmin, rmed, rmax, emed, emax, nn, ne = [], [], [], [], [], [], []
    for p in paths:
        d = np.load(p)
        x, e = d["x_psd"], d["edge_attr_cpsd_full"]
        rm = np.max(np.abs(x), axis=1)
        ae = np.abs(e)
        rmin.append(rm.min()); rmed.append(np.median(rm)); rmax.append(rm.max())
        emed.append(np.median(ae)); emax.append(ae.max())
        nn.append(x.shape[0]); ne.append(e.shape[0])
    rows.append({
        "group": "synthetic_training", "label": f"samples_phase2_bandpass (n={len(paths)})",
        "n_nodes": int(np.median(nn)), "n_edges": int(np.median(ne)),
        "node_rowmax_min": float(np.min(rmin)), "node_rowmax_median": float(np.median(rmed)),
        "node_rowmax_max": float(np.max(rmax)),
        "edge_absv_median": float(np.median(emed)), "edge_absv_max": float(np.max(emax)),
    })

    # (b) deterministic FE features (production, per structure).
    #     These come from a separate deterministic-inference run, not from this
    #     pipeline, so a newly added structure has none until that is run:
    #         python infer_phase2a_deterministic_edgegat.py \
    #             --run-dir runs_phase2/EdgeGATConv/800_Samples_Ng2Split \
    #             --case-name <S>_det --structure-name <S> --ssi-dir SSI_modal_infor
    #     Report the gap rather than dying on it - this is a diagnostic, and the
    #     other three groups are still worth having.
    missing_det = []
    for s in STRUCTURES:
        p = (REPO / "experiments" / "E1_architecture_ablation" / "inference" / "800_Samples_Ng2Split"
             / f"{s}_det" / "assembled_phase2a_sample.npz")
        if not p.exists():
            missing_det.append(s)
            continue
        d = np.load(p, allow_pickle=True)
        rows.append(stats(np.asarray(d["x_psd"]), np.asarray(d["edge_attr_cpsd_full"]),
                          s, "deterministic_FE"))
    if missing_det:
        print(f"[warn] no deterministic FE features for {missing_det}; "
              f"group 'deterministic_FE' covers "
              f"{len(STRUCTURES) - len(missing_det)}/{len(STRUCTURES)} structures")

    # (c) old production real features (broken)
    for s in STRUCTURES:
        d = np.load(REPO / "phase3_real_graphs" / s / "phase3_dataset.npz")
        rows.append(stats(d["x_psd"], d["edge_attr_cpsd_full"], s, "real_production_OLD"))

    # (d) corrected real features
    for s in STRUCTURES:
        d = np.load(OUT / "phase3_corrected" / s / "phase3_dataset_corrected.npz")
        rows.append(stats(d["x_psd"], d["edge_attr_cpsd_full"], s, "real_corrected"))
        rb = np.max(np.abs(d["x_psd_order_B"]), axis=1)
        rows.append({
            "group": "real_corrected_orderB", "label": s,
            "n_nodes": int(d["x_psd_order_B"].shape[0]), "n_edges": int(d["edge_attr_cpsd_full"].shape[0]),
            "node_rowmax_min": float(rb.min()), "node_rowmax_median": float(np.median(rb)),
            "node_rowmax_max": float(rb.max()),
            "edge_absv_median": float(np.median(np.abs(d["edge_attr_cpsd_full"]))),
            "edge_absv_max": float(np.abs(d["edge_attr_cpsd_full"]).max()),
        })

    path = OUT / "stage1_feature_stats.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {path}\n")

    # gate evaluation
    ok = True
    for r in rows:
        if r["group"] == "real_corrected":
            if abs(r["node_rowmax_min"] - 1.0) > 1e-5 or abs(r["node_rowmax_max"] - 1.0) > 1e-5:
                ok = False; print(f"GATE FAIL node rowmax: {r['label']} {r}")
            if r["edge_absv_max"] > 1.0 + 1e-5:
                ok = False; print(f"GATE FAIL edge |v|: {r['label']} {r}")

    hdr = f"{'group':<24}{'label':<38}{'node rowmax min/med/max':<34}{'edge |v| med/max'}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['group']:<24}{r['label']:<38}"
              f"{r['node_rowmax_min']:.3e} {r['node_rowmax_median']:.3e} {r['node_rowmax_max']:.3e}   "
              f"{r['edge_absv_median']:.3e} {r['edge_absv_max']:.3e}")
    print()
    print("GATE 1: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
