"""Stage 2 - round-trip validation of the corrected builders.

Feeds the corrected builders the FULL FE white-noise response for each real
geometry (same seed, excitation, damping as the deterministic path), with no
sensor subsampling and no propagation, and compares the resulting feature
matrices against the deterministic features produced by the production pipeline.

"No propagation" is achieved by clamping every node (mask = all True), which
makes ``feature_propagation`` an identity - i.e. the same code path is exercised,
not a bypass.

Gate 2: differences must be at float32 round-off level.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
AUDIT = REPO / "experiments" / "E6_controls"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(AUDIT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from task2_propagation_only_control import resimulate_deterministic  # noqa: E402
from infer_phase2a_deterministic_edgegat import (  # noqa: E402
    _build_params,
    assemble_deterministic_phase2a_payload,
)
from build_phase3_features_corrected import (  # noqa: E402
    corrected_edge_features,
    corrected_node_features,
    training_freq_grid,
)
from structures import LOOCV  # single source of truth

OUT = Path(__file__).resolve().parent
STRUCTURES = list(LOOCV)
MAC_THRESHOLD_OVERRIDES = {"Eidsdal": 0.70}


def reldiff(a: np.ndarray, b: np.ndarray) -> float:
    """max |a-b| / max(|b|), a scale-free difference measure."""
    denom = float(np.max(np.abs(b)))
    return float(np.max(np.abs(a - b)) / denom) if denom > 0 else float("nan")


def main() -> int:
    grid = training_freq_grid()
    rows: list[dict] = []
    for s in STRUCTURES:
        params = _build_params(s)
        payload, _ = assemble_deterministic_phase2a_payload(
            params, structure_name=s,
            min_mac_threshold=MAC_THRESHOLD_OVERRIDES.get(s))
        det_x = np.asarray(payload["x_psd"], dtype=np.float32)
        det_e = np.asarray(payload["edge_attr_cpsd_full"], dtype=np.float32)
        ei = np.asarray(payload["edge_index"], dtype=np.int64)
        n_fe = det_x.shape[0]

        sim = resimulate_deterministic(s, params)
        a3 = sim["a3_graph"]                       # (n_fe, nt), fs = 200 Hz, no bandpass
        fs = sim["fs"]
        all_idx = np.arange(n_fe, dtype=int)       # clamp every node -> no propagation

        x_full = corrected_node_features(a3, fs, grid, n_nodes=n_fe,
                                         sensor_idx=all_idx, edge_index=ei,
                                         order="A", norm_range="full")
        x_grid = corrected_node_features(a3, fs, grid, n_nodes=n_fe,
                                         sensor_idx=all_idx, edge_index=ei,
                                         order="A", norm_range="grid")
        e_cor = corrected_edge_features(a3, fs, grid, ei)

        rows.append({
            "structure": s, "n_nodes": n_fe, "n_edges": int(ei.shape[1]), "fs_hz": fs,
            "node_maxabsdiff_normfull": float(np.max(np.abs(x_full - det_x))),
            "node_reldiff_normfull": reldiff(x_full, det_x),
            "node_maxabsdiff_normgrid": float(np.max(np.abs(x_grid - det_x))),
            "node_reldiff_normgrid": reldiff(x_grid, det_x),
            "edge_maxabsdiff": float(np.max(np.abs(e_cor - det_e))),
            "edge_reldiff": reldiff(e_cor, det_e),
        })
        r = rows[-1]
        print(f"  [{s}] node(norm=full) maxabs={r['node_maxabsdiff_normfull']:.3e} "
              f"rel={r['node_reldiff_normfull']:.3e} | "
              f"node(norm=grid) maxabs={r['node_maxabsdiff_normgrid']:.3e} | "
              f"edge maxabs={r['edge_maxabsdiff']:.3e} rel={r['edge_reldiff']:.3e}")

    path = OUT / "stage2_roundtrip_validation.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {path}")

    tol = 1e-5
    ok = all(r["node_maxabsdiff_normfull"] < tol and r["edge_maxabsdiff"] < tol
             for r in rows)
    print(f"\nGATE 2 (tolerance {tol:g} on max abs diff): " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
