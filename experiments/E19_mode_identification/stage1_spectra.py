"""E19 stage 1 - the sensors' cross-spectra on the synthetic validation and test decks.

    python tools/run_stage.py experiments/E19_mode_identification/stage1_spectra.py --workers 12

Re-simulates every deck of the main network's validation split (300; used to tune
E19's two settings) and of the fixed test set (500), exactly as s8 / E16 / E18 do,
and caches per deck and layout the Welch cross-spectral matrix of the kept sensors
(0-20 Hz), with what scoring needs: every FE mode's frequency (to tell whether a
picked peak is the target mode), the target modes and their classes, coordinates,
true shapes and the support mask.

Layouts: s8's outer girders at n = 1 (the two front corners) and n = 2 (front and
mid-span pairs: four sensors).

Writes cache/<split>/<sample>.npz (generated, not tracked) and stage1_checks.json.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "experiments" / "E14_corner_sensors"))
sys.path.insert(0, str(REPO / "experiments" / "E18_population_basis"))

import fr_common as C  # noqa: E402
import popbasis as PB  # noqa: E402
import stage1_corner_propagation as E14  # noqa: E402
import stage3_sensor_sweep as S3  # noqa: E402
from generate_population_dataset_modes4 import _parse_params_json  # noqa: E402
from stage2_synthetic import segment_spectra  # noqa: E402

LAYOUTS = {"outer1": ("outer", 1), "outer2": ("outer", 2)}
F_MAX_HZ = 20.0
X_TOL_FRAC = 0.02


def resimulate(sampled: dict, selected_idx: np.ndarray) -> dict:
    """E14's resimulate, also returning every solved mode's frequency and the support rows."""
    p = E14.build_solver_params(sampled, n_candidate_modes=E14.N_CANDIDATE_MODES)
    solve_res = E14.run_solver(p, ref_x_fracs=E14.REF_X_FRACS, tol_x=E14.REF_SENSOR_TOL_X)
    graph_payload = E14.build_projected_graph_from_model(solve_res.model, include_sensor_map=False)
    sim_model = dict(solve_res.model)
    sim_model["M_full"] = solve_res.M_full
    sim = E14.simulate_white_noise_from_modal_model(
        sim_model, solve_res.phi_full, solve_res.freq_all_hz, selected_idx,
        zeta=E14.PHASE1_ZETA, plot=False, return_time_series=True,
        use_all_modes=True, bandpass_hz=E14.BANDPASS_HZ)
    ids = np.asarray(graph_payload["graph_node_ids"], dtype=np.int64)
    return {"a3": np.asarray(sim["a3_abs"], dtype=np.float64)[ids - 1, :],
            "fs": float(np.asarray(sim["fs"]).reshape(())), "graph_node_ids": ids,
            "edge_index": np.asarray(graph_payload["edge_index"], dtype=np.int64),
            "support": np.isin(ids, np.asarray(sim["support_nodes"]).ravel()),
            "freq_all_hz": np.asarray(solve_res.freq_all_hz, dtype=np.float64).reshape(-1)}


def features(p: dict) -> np.ndarray:
    """Geometry a new deck's drawings give: log L, log dy, balcony, bottom-beam position."""
    return np.array([np.log(float(p["L_mm"])), np.log(float(p["dy_mm"])),
                     float(bool(p["balcony"]["enable"])), float(p["bottom_beam"]["x_frac"])])


def run_sample(args: tuple[str, str]) -> dict:
    split, path = args
    t0 = time.time()
    with np.load(path, allow_pickle=True) as d:
        p = _parse_params_json(d)
        st = {k: d[k] for k in ("graph_node_ids", "edge_index", "coords_xy_mm", "y_mode_shape_signed",
                                "y_freq_hz", "matched_mode_indices", "matched_freq_hz")}
    sample = Path(path).stem
    ng, L, dy = int(p["Ng"]), float(p["L_mm"]), float(p["dy_mm"])
    c = st["coords_xy_mm"].astype(np.float64)
    y = st["y_mode_shape_signed"].astype(np.float64)
    f_true = st["y_freq_hz"].astype(np.float64).reshape(-1)
    matched = st["matched_mode_indices"].astype(int)
    cls = PB.classes(y, f_true, c, 0.0, (ng - 1) * dy, X_TOL_FRAC * L)
    sim = resimulate(p, matched)
    assert np.array_equal(sim["graph_node_ids"], st["graph_node_ids"])
    assert np.allclose(sim["freq_all_hz"][matched], st["matched_freq_hz"], rtol=0, atol=1e-5)
    fs = sim["fs"]
    freq_axis = np.fft.rfftfreq(E14.WELCH_NFFT, 1.0 / fs)
    nb = int(np.searchsorted(freq_axis, F_MAX_HZ, side="right"))
    out = {"sample": sample, "Ng": ng, "L_mm": L, "dy_mm": dy, "feat": features(p), "coords": c, "y": y,
           "f_true": f_true, "classes": np.array(cls), "matched": matched, "freq_all": sim["freq_all_hz"],
           "support": sim["support"], "f_axis": freq_axis[:nb], "fs": fs}
    for key, (fam, lv) in LAYOUTS.items():
        s = S3.station_nodes(c, p, lv, fam)
        X = segment_spectra(sim["a3"][s], fs)[:, :, :nb]                   # (s, nseg, F)
        out[f"{key}_nodes"] = s
        out[f"{key}_G"] = np.einsum("igf,jgf->ijf", np.conj(X), X) / X.shape[1]
    dst = HERE / "cache" / split
    dst.mkdir(parents=True, exist_ok=True)
    np.savez(dst / f"{sample}.npz", **out)
    return {"split": split, "sample": sample, "Ng": ng, "classes": cls, "seconds": round(time.time() - t0, 1),
            "freq_check_maxdiff": float(np.abs(sim["freq_all_hz"][matched] - st["matched_freq_hz"]).max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--per-ng", type=int, default=None, help="first K per Ng and split (pilot)")
    args = ap.parse_args()
    splits = {"val": (C.SPLITS / f"N{C.MAIN_N}_val.txt").read_text(encoding="utf-8").split(),
              "test": (C.SPLITS / "test.txt").read_text(encoding="utf-8").split()}
    jobs = []
    for split, names in splits.items():
        if args.per_ng:
            names = [t for g in C.NG_VALUES for t in [x for x in names if f"_Ng{g}_" in x][: args.per_ng]]
        jobs += [(split, str(C.POP_BROADBAND / t)) for t in names]
    t0, res = time.time(), []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(run_sample, jobs), 1):
            res.append(r)
            if i % 100 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] {(time.time() - t0) / 60:.1f} min", flush=True)
    chk = {"n": {s: sum(r["split"] == s for r in res) for s in splits},
           "matched_freq_maxdiff_hz": max(r["freq_check_maxdiff"] for r in res),
           "class_sets_complete": sum(sorted(r["classes"]) == sorted(PB.CLASSES) for r in res)}
    (HERE / "stage1_checks.json").write_text(json.dumps(chk, indent=1), encoding="utf-8")
    print(json.dumps(chk, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
