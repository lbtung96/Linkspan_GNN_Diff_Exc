"""Stage 0 - resolve the three outstanding discrepancies from the audit.

Reads only; computes nothing new beyond diffs and table joins.
Writes Audit/Rerun/stage0_tables.md (raw tables consumed by stage0_discrepancies.md).
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from build_phase3_graphs import STRUCTURES as GRAPH_PARAMS  # noqa: E402
from infer_phase2a_deterministic_edgegat import REAL_STRUCTURE_PARAMS as DET_PARAMS  # noqa: E402
from grid_modal_Kinematic_MPC import default_params_grid_3d_offset_upgraded_v22  # noqa: E402

OUT = Path(__file__).resolve().parent
S8 = ["Hellesylt", "Hundeidvik", "Leknes", "Linge", "Saebo",
      "Eidsdal", "Liabygda", "Stranda"]
MODES = ["B1", "T1", "B2", "T2"]
# Mode-matching corrections declared only in REAL_STRUCTURE_PARAMS; they are not
# geometry, so they are reported as declared rows rather than diffed.
MATCHING_KEYS = ("mode_col_swap", "mode_candidate_override")

HANDOVER_DET = {  # docs/JOURNAL_PAPER_HANDOVER.md §6.5(a); identical in Progress.md and CLAUDE.md
    "Hellesylt": [0.997, 0.874, 0.939, 0.787],
    "Hundeidvik": [0.968, 0.922, 0.892, 0.830],
    "Leknes": [0.982, 0.859, 0.932, 0.860],
    "Linge": [0.967, 0.987, 0.964, 0.913],
    "Saebo": [0.990, 0.972, 0.906, 0.989],
    "Eidsdal": [0.940, 0.882, 0.498, 0.604],
    "Liabygda": [0.924, 0.855, 0.876, 0.790],
    "Stranda": [0.989, 0.832, 0.783, 0.708],
}


def merged(base_overrides: dict) -> dict:
    """Shallow-merge overrides onto the FE defaults, as both call sites do."""
    p = default_params_grid_3d_offset_upgraded_v22()
    for k, v in deepcopy(base_overrides).items():
        if isinstance(v, dict) and isinstance(p.get(k), dict):
            p[k] = {**p[k], **v}
        else:
            p[k] = v
    return p


def mtime(p: str) -> str:
    return datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")


def main() -> int:
    lines: list[str] = []
    w = lines.append

    # ---------------- 0.1 ----------------
    w("## 0.1 raw data\n")
    w("### Every stored artefact containing a deterministic Hellesylt MAC\n")
    w("| Path | mtime | run dir | checkpoint | GNN-vs-FE (B1/T1/B2/T2) | SSI-vs-FE (B1/T1/B2/T2) |")
    w("|---|---|---|---|---|---|")
    pats = [
        "experiments/E1_architecture_ablation/inference/*/Hellesylt_det/summary.json",
        "experiments/E1_architecture_ablation/inference/*/Hellesylt_ssi/summary.json",
        "Progress_update/07_Deterministic_Baseline/Hellesylt/summary.json",
        "Progress_update/08_Deterministic_Bandpass/Hellesylt/summary.json",
    ]
    for pat in pats:
        for p in sorted(glob.glob(pat)):
            d = json.load(open(p))
            m = d["mode_shape_metrics"]
            gnn = [round(m[f"mac_{k}"], 3) for k in MODES]
            s = d.get("ssi_summary")
            ssi = [round(v, 3) for v in s["ssi_mac_vs_fe_target"]] if s else None
            w(f"| `{p}` | {mtime(p)} | `{d.get('run_dir','?')}` | `{d.get('checkpoint_path','?')}` | {gnn} | {ssi} |")
    w("")

    w("### Handover value vs stored value, all eight structures (baseline checkpoint)\n")
    w("| Structure | Mode | Handover / CLAUDE.md / Progress.md | Stored `mode_shape_metrics.mac_*` (GNN vs FE) | Stored `ssi_summary.ssi_mac_vs_fe_target` (SSI vs FE) | Handover == which? |")
    w("|---|---|---|---|---|---|")
    for s in S8:
        p = f"experiments/E1_architecture_ablation/inference/800_Samples_Ng2Split/{s}_det/summary.json"
        d = json.load(open(p))
        m = d["mode_shape_metrics"]
        ssi = d["ssi_summary"]["ssi_mac_vs_fe_target"]
        for i, k in enumerate(MODES):
            h = HANDOVER_DET[s][i]
            g = m[f"mac_{k}"]
            v = ssi[i]
            which = "SSI-vs-FE" if abs(h - v) < 5e-4 else ("GNN-vs-FE" if abs(h - g) < 5e-4 else "neither")
            w(f"| {s} | {k} | {h:.3f} | {g:.3f} | {v:.3f} | {which} |")
    w("")

    # ---------------- 0.2 ----------------
    w("## 0.2 raw data\n")
    w("### Field-by-field diff of the two parameter dictionaries (post-merge onto FE defaults)\n")
    w("| Structure | Field | `build_phase3_graphs.STRUCTURES` | `REAL_STRUCTURE_PARAMS` | differs? |")
    w("|---|---|---|---|---|")
    n_diff = 0
    for s in S8:
        a = merged(GRAPH_PARAMS[s])
        b = merged({k: v for k, v in DET_PARAMS[s].items() if k not in MATCHING_KEYS})
        keys = sorted(set(a) | set(b))
        for k in keys:
            va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
            if isinstance(va, dict) and isinstance(vb, dict):
                for kk in sorted(set(va) | set(vb)):
                    x, y = va.get(kk, "<absent>"), vb.get(kk, "<absent>")
                    if x != y:
                        n_diff += 1
                        w(f"| {s} | `{k}.{kk}` | `{x}` | `{y}` | **yes** |")
            elif va != vb:
                n_diff += 1
                w(f"| {s} | `{k}` | `{va}` | `{vb}` | **yes** |")
        for key in MATCHING_KEYS:
            if key in DET_PARAMS[s]:
                n_diff += 1
                w(f"| {s} | `{key}` | `<absent>` | `{DET_PARAMS[s][key]}` | **yes** |")
    w("")
    w(f"Total differing effective fields across the eight structures: **{n_diff}**.")
    w("")

    # ---------------- 0.3 ----------------
    w("## 0.3 raw data\n")
    w("### Real field records\n")
    w("| Structure | n_samples | fs (Hz) | duration (s) | Welch df at nfft=2048 (Hz) | span of first 400 bins (Hz) |")
    w("|---|---|---|---|---|---|")
    for s in S8:
        d = np.load(Path("field_modal_filtering_outputs") / s / "data" /
                    "field_modal_filtering_outputs.npz", allow_pickle=False)
        u = d["raw_u3_timeseries_clean"]
        fs = float(2.0 * d["freq_hz"][-1])
        w(f"| {s} | {u.shape[0]} | {fs:.1f} | {u.shape[0]/fs:.1f} | {fs/2048:.5f} | 0 – {399*fs/2048:.2f} |")
    w("")
    g = np.load("population_dataset_modes4/samples_phase2_bandpass/"
                "sample_0000_Ng2_phase1_gnn.npz")["freq_bins_hz"]
    w(f"Training/deterministic grid: 400 bins, df = {g[1]-g[0]:.8f} Hz, span 0 – {g[-1]:.5f} Hz "
      f"(fs = 200 Hz, nfft = 2048).")
    w("")

    path = OUT / "stage0_tables.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")
    print("\n".join(lines[:4]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
