"""T10 - sensor channel order: SSI labels (sensors 7 and 9 swapped by the MATLAB SSI scripts) vs GNN field inputs (raw order).

    python tools/run_stage.py pipeline/g_t10_channel_order.py

Read-only.  Nothing is trained; the only network call is s5's inference over the stored corrected features.
Sensor index convention below: 0-based rows of the sensor arrays.  Rows 6 and 8 = sensors 7 and 9 (1-based) =
logger serials x40131 / x40133 = grid_<S>.txt nodes 9 / 11.

Two hypotheses about which order is the grid (physical) order:
  H_ssi  the SSI order (raw records with rows 6/8 swapped) is the grid order -> the GNN field inputs are mis-ordered
  H_raw  the raw record order is the grid order                              -> the SSI labels are mis-ordered

Tests (per bridge x mode):
  1. FE target at the sensors (label-corrected, experiments/E6_controls/task2_raw/<S>.npz fe_target, read at
     nearest_fe_node_idx) against SSI as stored / SSI with rows 6, 8 swapped.        stored better -> H_ssi
  2. FE target against the FDD first singular vector of the raw-order records (complex MAC), raw / rows swapped.
                                                                                    swapped better -> H_ssi
  3. model-free: roughness (second divided differences along the y = 0 girder line, which carries both sensors)
     of SSI stored / swapped and of FDD raw / swapped.                              smoother order wins
  4. zero-shot predictions of the main network (s5 path, stored features) scored against SSI stored / swapped.
     Gate: the stored scoring reproduces s5_zeroshot.csv exactly (all numeric columns).

Writes results/T10_channel_order/: t10_geometry_tests.csv, t10_zeroshot_rescore.csv, t10_sensor_positions.csv,
t10_checks.json.
"""

from __future__ import annotations

import ec_common as E  # noqa: I001  (sets sys.path)

import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

import s5_zeroshot as S5  # noqa: E402

C = E.C
REPO = E.REPO
OUT = E.EC_RESULTS / "T10_channel_order"
OUT.mkdir(parents=True, exist_ok=True)
MODES = S5.MODES
I6, I8 = 6, 8


def swap(a: np.ndarray) -> np.ndarray:
    b = np.array(a, copy=True)
    b[[I6, I8]] = b[[I8, I6]]
    return b


def cmac(u, v) -> float:
    """MAC that accepts a complex vector (|u^H v|^2 / (|u|^2 |v|^2))."""
    u = np.asarray(u).reshape(-1); v = np.asarray(v).reshape(-1)
    return float(np.abs(np.vdot(u, v)) ** 2 / (np.vdot(u, u).real * np.vdot(v, v).real + 1e-30))


def to_real(u: np.ndarray) -> np.ndarray:
    """Rotate a complex mode so that it is as real as possible, return the real part."""
    u = np.asarray(u, dtype=complex)
    th = 0.5 * np.angle(np.sum(u * u))
    return np.real(u * np.exp(-1j * th))


def roughness(vals: np.ndarray, x: np.ndarray) -> float:
    """Sum of squared second divided differences along a line, normalised by the mean square value."""
    o = np.argsort(x); v = vals[o] / (np.max(np.abs(vals)) + 1e-30); xx = x[o] / 1000.0
    d1 = np.diff(v) / np.diff(xx)
    d2 = np.diff(d1) / (0.5 * (xx[2:] - xx[:-2]))
    return float(np.sum(d2 ** 2))


def main() -> int:
    torch.set_num_threads(int(json.loads((C.RESULTS / "s5_gates.json").read_text())["torch_threads"]))
    checks: dict = {"torch_threads": int(torch.get_num_threads())}
    data = S5.load_structures()
    geo_rows, pos_rows = [], []
    for s, d in data.items():
        g = np.load(S5.GRAPH_ROOT / s / "graph_package.npz", allow_pickle=False)
        xy = g["node_coords_xy_mm"].astype(float); si = g["sensor_node_indices"].astype(int)
        near = g["nearest_fe_node_idx"].astype(int); pid = g["sensor_physical_ids"].astype(int)
        ds = np.load(REPO / "phase3_real_graphs" / s / "phase3_dataset.npz", allow_pickle=False)
        field = np.load(C.LEGACY_FIELD_OUTPUTS / s / "data" / "field_modal_filtering_outputs.npz",
                        allow_pickle=False)
        # --- provenance: GNN raw order = raw .mat column order (ch3 of each triple, no reorder) ---------------
        # (the summary names the archive's external copy; field_measurements_raw/ holds the same bytes,
        # checked at migration; the recorded string is kept as it was)
        raw_rec = json.loads((C.LEGACY_FIELD_OUTPUTS / s / "logs" / "structure_summary.json")
                             .read_text(encoding="utf-8"))["raw_file"]
        raw_path = str(C.FIELD_RAW_MAT_DIR / Path(raw_rec).name)
        raw = sio.loadmat(raw_path)
        names = [str(n[0]) for n in raw["Names"].ravel()]
        full = field["raw_u3_timeseries_full"]
        raw_u3 = raw["Data"][:, 2::3]
        same = bool(full.shape == raw_u3.shape and np.array_equal(np.isnan(full), np.isnan(raw_u3))
                    and np.array_equal(np.nan_to_num(full), np.nan_to_num(raw_u3)))
        ssi_mat = sio.loadmat(REPO / "SSI_modal_infor" / f"{s}_Data_picked.mat")
        phi = ssi_mat["phi_sorted"][2::3].astype(float)
        y = d["y_ssi"]
        checks[s] = {
            "raw_file": raw_rec,
            "raw_u3_full_equals_raw_mat_Data_ch3_in_column_order": same,
            "u3_channel_indices_zero_based": field["u3_channel_indices_zero_based"].tolist(),
            "raw_names_rows_6_8": [names[3 * I6 + 2], names[3 * I8 + 2]],
            "sensor_physical_ids": pid.tolist(),
            "y_ssi_equals_phi_sorted_uz_up_to_sign": bool(all(np.allclose(np.abs(phi[:, k]), np.abs(y[:, k]),
                                                                           rtol=0, atol=1e-6 * np.abs(phi).max())
                                                              for k in range(4))),
        }
        for r in (I6, I8):
            pos_rows.append({"structure": s, "row0": r, "sensor_1based": r + 1, "physical_id": int(pid[r]),
                             "raw_channel_name": names[3 * r + 2],
                             "x_mm": xy[si[r], 0], "y_mm": xy[si[r], 1],
                             "deck_width_max_y_mm": float(xy[:, 1].max()),
                             "nearest_fe_node": int(near[r])})
        fe_s = d["fe_t_det"][near]                       # (n_sensors, 4) FE target at the sensors (nearest node)
        line = np.abs(xy[si, 1] - xy[si[I6], 1]) < 1.0  # sensors on the girder line of rows 6 and 8
        assert line[I8], s
        f = field["freq_hz"]; U = field["first_singular_vectors"]; Sv = field["singular_values"][0]
        mac_gate = []
        for m, name in enumerate(MODES):
            ssi = y[:, m].astype(float); ssw = swap(ssi); fe = fe_s[:, m]
            i0 = int(np.argmin(np.abs(f - d["ssi_freq"][m])))
            cand = [j for j in (i0 - 1, i0, i0 + 1) if 0 <= j < f.size]
            j = max(cand, key=lambda k: Sv[k])
            u = U[:, j]
            ur = to_real(u)
            st = cmac(fe, ssi); sw = cmac(fe, ssw)
            mac_gate.append(abs(st - float(ds["ssi_mac_vs_fe"][m])))
            geo_rows.append({
                "structure": s, "mode": name, "ssi_freq_hz": float(d["ssi_freq"][m]), "fdd_freq_hz": float(f[j]),
                "rel_diff_ssi_rows_6_8": float(abs(ssi[I6] - ssi[I8]) / (np.max(np.abs(ssi)) + 1e-30)),
                "rel_diff_fe_rows_6_8": float(abs(fe[I6] - fe[I8]) / (np.max(np.abs(fe)) + 1e-30)),
                "mac_fe_vs_ssi_stored": st, "mac_fe_vs_ssi_swapped": sw,
                "mac_fe_vs_fdd_raw": cmac(fe, u), "mac_fe_vs_fdd_swapped": cmac(fe, swap(u)),
                "mac_ssi_stored_vs_fdd_raw": cmac(ssi, u), "mac_ssi_stored_vs_fdd_swapped": cmac(ssi, swap(u)),
                "rough_fe": roughness(fe[line], xy[si[line], 0]),
                "rough_ssi_stored": roughness(ssi[line], xy[si[line], 0]),
                "rough_ssi_swapped": roughness(ssw[line], xy[si[line], 0]),
                "rough_fdd_raw": roughness(ur[line], xy[si[line], 0]),
                "rough_fdd_swapped": roughness(swap(ur)[line], xy[si[line], 0]),
                "n_sensors_on_line": int(line.sum()),
            })
        checks[s]["ssi_mac_vs_fe_reproduced_maxabs"] = float(max(mac_gate))

    # --- zero-shot rescoring (s5 path) ------------------------------------------------------------------------
    run = C.RUNS / C.sweep_run_name(C.MAIN_N)
    rows5 = S5.evaluate(run, "best_model", data, "broadband", C.SEED)
    ref = list(csv.DictReader((C.RESULTS / "s5_zeroshot.csv").open(encoding="utf-8")))
    num = [k for k in ref[0] if k not in ("arm", "seed", "checkpoint", "structure", "mode", "N")]
    md = max(abs(float(a[k]) - float(b[k])) for a, b in zip(rows5, ref) for k in num)
    same = [(a["structure"], a["mode"]) for a in rows5] == [(b["structure"], b["mode"]) for b in ref]
    checks["gate_stored_reproduces_s5"] = {"pass": bool(same and md == 0.0), "max_abs_diff": md}

    from EdgeGATConv_Modal_GNN import DEVICE, load_trained_model_checkpoint
    from infer_phase2a_unseen_structure import load_frequency_normalization
    from infer_phase2a_unseen_structure_edgegat import load_input_normalization
    ck = run / "checkpoints" / "best_model.pt"
    model, ckpt = load_trained_model_checkpoint(ck, device=DEVICE); model.eval()
    fstats = load_frequency_normalization(checkpoint_path=ck, checkpoint=ckpt, run_dir=run)
    nstats = load_input_normalization(run)
    zs_rows = []
    k = 0
    for s, d in data.items():
        p, _ = S5.infer(model, fstats, nstats, d["ei"], d["x"], d["ea"], d["n"])
        at = p[d["sensor_idx"]].astype(np.float64)
        for m, name in enumerate(MODES):
            st = S5.mac(at[:, m], d["y_ssi"][:, m]); sw = S5.mac(at[:, m], swap(d["y_ssi"][:, m]))
            assert abs(st - rows5[k]["mac_ssi"]) < 1e-12, (s, name, st, rows5[k]["mac_ssi"]); k += 1
            zs_rows.append({"structure": s, "mode": name, "mac_ssi_stored": st, "mac_ssi_swapped": sw,
                            "delta": sw - st, "mac_pred_swapped_vs_ssi_stored": S5.mac(swap(at[:, m]), d["y_ssi"][:, m])})
    checks["gate_prediction_rescoring_matches_evaluate"] = "abs diff < 1e-12 (last-bit BLAS alignment)"

    E_ = lambda rows, key: float(np.mean([r[key] for r in rows]))  # noqa: E731
    summary = {"geometry_mean": {k_: E_(geo_rows, k_) for k_ in geo_rows[0] if k_ not in ("structure", "mode")},
               "zeroshot_mean": {k_: E_(zs_rows, k_) for k_ in zs_rows[0] if k_ not in ("structure", "mode")},
               "zeroshot_per_mode": {m: {k_: float(np.mean([r[k_] for r in zs_rows if r["mode"] == m]))
                                         for k_ in ("mac_ssi_stored", "mac_ssi_swapped")} for m in MODES},
               "votes": {
                   "fe_vs_ssi_stored_better": sum(r["mac_fe_vs_ssi_stored"] > r["mac_fe_vs_ssi_swapped"] for r in geo_rows),
                   "fe_vs_fdd_swapped_better": sum(r["mac_fe_vs_fdd_swapped"] > r["mac_fe_vs_fdd_raw"] for r in geo_rows),
                   "rough_ssi_stored_smoother": sum(r["rough_ssi_stored"] < r["rough_ssi_swapped"] for r in geo_rows),
                   "rough_fdd_swapped_smoother": sum(r["rough_fdd_swapped"] < r["rough_fdd_raw"] for r in geo_rows),
                   "n_cells": len(geo_rows)}}
    checks["summary"] = summary
    for name, rows in (("t10_geometry_tests.csv", geo_rows), ("t10_zeroshot_rescore.csv", zs_rows),
                       ("t10_sensor_positions.csv", pos_rows)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / "t10_checks.json").write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    ok = checks["gate_stored_reproduces_s5"]["pass"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
