"""Stage 5a - the paper's real-bridge inputs, in the sensor order of the SSI labels.

    python tools/run_stage.py pipeline/s5a_field_inputs.py

The loggers x40131 and x40133 (0-based sensor rows 6 and 8) are interchanged in the
raw field records of all nine bridges; every SSI script swaps them back before
identification ("% Swap 131 and 133"), the GNN field path never did (fr_common,
FIELD_SENSOR_SWAP; PROVENANCE Phase P; EC_comments T10).  This stage regenerates the
field outputs with that reorder and rebuilds the corrected real-bridge features from
them, into data/field_inputs_v2/:

  field_modal_filtering_outputs/<S>/   process_field_modal_filtering.process_structure,
                                       imported and called with explicit file paths and
                                       settings.raw_channel_reorder = the SSI scripts'
                                       column order (no file dialog, no popup: the saved
                                       mode assignment is reused, as it was originally)
  phase3_corrected/<S>/phase3_dataset_corrected.npz
                                       experiments/E5_real_transfer/
                                       build_phase3_features_corrected.build_structure,
                                       imported, with its FIELD_ROOT pointed at the above

Both scripts are imported unmodified; only their output roots and the reorder are set
in memory.  The legacy outputs (field_modal_filtering_outputs/,
experiments/E5_real_transfer/phase3_corrected/) are read, never written: experiments/
keeps them as history.

Gates (results/s5a_field_inputs_checks.json; any failure exits 1):
  serials           raw .mat Names: the columns of rows 6 and 8 are x40131 / x40133
  settings          every data-affecting setting equals the legacy run's settings_used.json
  identity          with no reorder, the producer reproduces the legacy field npz and the
                    builder the legacy corrected features, every array bit for bit
  swap              with the reorder, the raw u3 records equal the legacy ones with rows
                    6 and 8 exchanged, bit for bit; the arrays read from the SSI file are
                    unchanged; the non-feature arrays of the corrected dataset (labels,
                    geometry, FE targets) are unchanged
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402

import fr_common as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "experiments" / "E5_real_transfer"))

import process_field_modal_filtering as P  # noqa: E402
import build_phase3_features_corrected as B  # noqa: E402

FIELD_NPZ = Path("data") / "field_modal_filtering_outputs.npz"
FEATURE_NPZ = "phase3_dataset_corrected.npz"
FEATURE_KEYS = ("x_psd", "x_psd_order_B", "x_psd_norm_full", "edge_attr_cpsd_full")
# settings that cannot change any array: where things are written and whether plots show
NON_DATA_SETTINGS = ("run_mode", "output_root", "mode_assignment_json", "show_plots", "save_png",
                     "save_svg", "save_mat", "raw_channel_reorder")
# arrays the producer reads from the SSI-picked file alone, or from the channel layout
SSI_ONLY_ARRAYS = ("ssi_picked_frequencies_hz", "ssi_selected_indices", "u3_channel_indices_zero_based",
                   "freq_hz")


def _no_dialog(*_a, **_k):
    raise RuntimeError("s5a: the producer asked for a mode assignment it should have reused")


# The producer reuses a saved mode assignment only after a yes/no popup; answer it here.
P.popup_yes_no = lambda root, title, message: True
P.popup_ask_string = _no_dialog


def equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def maxdiff(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape != b.shape or a.dtype.kind not in "fciu":
        return None
    return float(np.nanmax(np.abs(np.asarray(a) - np.asarray(b)))) if a.size else 0.0   # raw_*_full has NaN gaps


def produce(s: str, out_root: Path, reorder: list[int] | None, save_png: bool) -> dict:
    """One bridge through process_field_modal_filtering.process_structure."""
    legacy_logs = C.LEGACY_FIELD_OUTPUTS / s / "logs"
    raw_name, ssi_name = C.FIELD_RAW_FILES[s]
    raw_file, ssi_file = C.FIELD_RAW_MAT_DIR / raw_name, C.FIELD_RAW_MAT_DIR / ssi_name
    leg_summary = json.loads((legacy_logs / "structure_summary.json").read_text(encoding="utf-8"))
    # the legacy summary names the archive's external copy ("For GNN"); this repository's
    # field_measurements_raw/ holds the same files (byte-identical, checked at migration)
    assert (Path(leg_summary["raw_file"]).name == raw_file.name
            and Path(leg_summary["ssi_file"]).name == ssi_file.name), s

    settings = P.build_default_settings()
    settings.output_root = out_root
    settings.mode_assignment_json = out_root / "manual_mode_assignments.json"
    settings.show_plots = False
    settings.save_png = save_png
    settings.raw_channel_reorder = reorder
    leg_settings = json.loads((legacy_logs / "settings_used.json").read_text(encoding="utf-8"))
    mine = json.loads(json.dumps(asdict(settings), default=P._json_default))
    settings_diff = sorted(k for k in set(mine) | set(leg_settings)
                           if k not in NON_DATA_SETTINGS and mine.get(k) != leg_settings.get(k))

    # the mode assignment the legacy run used (its store keys Stranda as "Stranda_Walk")
    mapping = json.loads((legacy_logs / "mode_assignment_used.json").read_text(encoding="utf-8"))
    pair = P.PairCandidate(structure_key=s.lower(), structure_name=s, raw_file=raw_file, ssi_file=ssi_file,
                           notes=["explicit pairing, pipeline/s5a_field_inputs.py"])
    P.process_structure(pair, settings, None, {s: mapping})
    return {"settings_diff_vs_legacy": settings_diff, "settings": mine}


def build_features(s: str, field_root: Path, grid: np.ndarray) -> dict:
    B.FIELD_ROOT = field_root               # read by build_structure at call time
    return B.build_structure(s, grid)


def serial_check(s: str) -> dict:
    m = sio.loadmat(C.FIELD_RAW_MAT_DIR / C.FIELD_RAW_FILES[s][0], variable_names=["Names"])
    names = [str(n[0]) for n in m["Names"].ravel()]
    k = C.FIELD_CHANNELS_PER_SENSOR
    got = {int(r): names[k * r:k * r + k] for r in C.FIELD_SENSOR_SWAP}
    ok = all(all(n.startswith(ser + "_") for n in got[r]) for r, ser in zip(C.FIELD_SENSOR_SWAP, C.FIELD_SENSOR_SERIALS))
    return {"pass": bool(ok), "n_channels": len(names), "names": got}


def main() -> int:
    grid = B.training_freq_grid()
    C.FIELD_OUTPUTS.mkdir(parents=True, exist_ok=True)
    C.REAL_FEATURES.mkdir(parents=True, exist_ok=True)
    checks: dict = {"field_sensor_swap": list(C.FIELD_SENSOR_SWAP), "serials": list(C.FIELD_SENSOR_SERIALS),
                    "field_raw_mat_dir": str(C.FIELD_RAW_MAT_DIR), "per_structure": {}}
    summaries = []
    for s in C.LOOCV:
        t0 = time.time()
        c: dict = {"serials": serial_check(s)}
        n_ch = c["serials"]["n_channels"]
        reorder = C.field_channel_reorder(n_ch)
        order = C.field_sensor_order(n_ch // C.FIELD_CHANNELS_PER_SENSOR)
        c["channel_reorder"] = reorder
        leg_field = np.load(C.LEGACY_FIELD_OUTPUTS / s / FIELD_NPZ, allow_pickle=False)
        leg_feat = np.load(C.LEGACY_REAL_FEATURES / s / FEATURE_NPZ, allow_pickle=False)

        # ---- identity: the imported producer and builder reproduce the legacy inputs ----------
        with tempfile.TemporaryDirectory(prefix="_identity_", dir=C.FIELD_INPUTS, ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            r_id = produce(s, tmp, None, save_png=False)
            new = np.load(tmp / s / FIELD_NPZ, allow_pickle=False)
            diff_field = sorted(k for k in set(new.files) | set(leg_field.files)
                                if k not in new.files or k not in leg_field.files or not equal(new[k], leg_field[k]))
            feat = build_features(s, tmp, grid)
            new.close()
        diff_feat = sorted(k for k in set(feat) | set(leg_feat.files)
                           if k not in feat or k not in leg_feat.files or not equal(np.asarray(feat[k]), leg_feat[k]))
        c["identity"] = {"pass": not diff_field and not diff_feat and not r_id["settings_diff_vs_legacy"],
                         "settings_diff_vs_legacy": r_id["settings_diff_vs_legacy"],
                         "field_arrays": len(leg_field.files), "field_arrays_differing": diff_field,
                         "feature_arrays": len(leg_feat.files), "feature_arrays_differing": diff_feat}

        # ---- the swap: the paper's inputs ------------------------------------------------------
        r_sw = produce(s, C.FIELD_OUTPUTS, reorder, save_png=True)
        sw = np.load(C.FIELD_OUTPUTS / s / FIELD_NPZ, allow_pickle=False)
        per_array = {}
        for k in leg_field.files:
            a, b = sw[k], leg_field[k]
            if k in SSI_ONLY_ARRAYS or k.startswith("mask_") or a.ndim == 0:
                ref, how = b, "unpermuted"
            elif k in ("raw_u3_timeseries_full", "raw_u3_timeseries_clean", "filtered_u3_timeseries_visualization_only"):
                ref, how = b[:, order], "sensor columns permuted"
            elif k in ("raw_psd", "filtered_psd", "first_singular_vectors"):
                ref, how = b[order], "sensor rows permuted"
            elif k.endswith("spectral_density_matrix") or k.startswith("filtered_sdm_"):
                ref, how = b[np.ix_(order, order)], "sensor rows and columns permuted"
            else:
                ref, how = b, "unpermuted"
            per_array[k] = {"compared_to_legacy": how, "equal": equal(a, ref), "max_abs_diff": maxdiff(a, ref)}
        raw_ok = all(per_array[k]["equal"] for k in ("raw_u3_timeseries_full", "raw_u3_timeseries_clean"))
        ssi_ok = all(per_array[k]["equal"] for k in SSI_ONLY_ARRAYS)
        sw.close()
        feat = build_features(s, C.FIELD_OUTPUTS, grid)
        d = C.REAL_FEATURES / s
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / FEATURE_NPZ, **feat)
        other_changed = sorted(k for k in leg_feat.files if k not in FEATURE_KEYS and not equal(np.asarray(feat[k]), leg_feat[k]))
        c["swap"] = {"pass": raw_ok and ssi_ok and not other_changed and not r_sw["settings_diff_vs_legacy"],
                     "raw_u3_equals_legacy_rows_exchanged": raw_ok, "ssi_file_arrays_unchanged": ssi_ok,
                     "non_feature_arrays_changed": other_changed,
                     "feature_rel_change": {k: float(np.linalg.norm(feat[k].astype(np.float64) - leg_feat[k])
                                                     / np.linalg.norm(leg_feat[k].astype(np.float64)))
                                            for k in FEATURE_KEYS},
                     "field_arrays": per_array}
        c["pass"] = c["serials"]["pass"] and c["identity"]["pass"] and c["swap"]["pass"]
        checks["per_structure"][s] = c
        summaries.append(json.loads((C.FIELD_OUTPUTS / s / "logs" / "structure_summary.json").read_text(encoding="utf-8")))
        leg_field.close(); leg_feat.close()
        print(f"[{s}] {time.time() - t0:.0f}s  serials {c['serials']['pass']}  identity {c['identity']['pass']}"
              f"  swap {c['swap']['pass']}  x_psd rel change {c['swap']['feature_rel_change']['x_psd']:.3f}"
              f"  edge {c['swap']['feature_rel_change']['edge_attr_cpsd_full']:.3f}", flush=True)
        if not c["identity"]["pass"]:
            print(f"   identity differs: settings {c['identity']['settings_diff_vs_legacy']} field "
                  f"{diff_field} features {diff_feat}", flush=True)

    P.save_json(C.FIELD_OUTPUTS / "run_summary.json",
                {"mode": "pipeline/s5a_field_inputs.py", "field_sensor_swap": list(C.FIELD_SENSOR_SWAP),
                 "settings": r_sw["settings"], "structures": summaries})
    checks["pass"] = all(c["pass"] for c in checks["per_structure"].values())
    (C.RESULTS / "s5a_field_inputs_checks.json").write_text(json.dumps(checks, indent=2, default=P._json_default),
                                                            encoding="utf-8")
    print("s5a GATES PASS" if checks["pass"] else "s5a GATES FAILED")
    return 0 if checks["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
