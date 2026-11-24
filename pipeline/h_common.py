"""T11 - varied-excitation pilot: shared paths, seeds and the per-structure re-simulation.

Every h_*.py script imports this.  Run from the repository root through tools/run_stage.py.

Why T11 exists (T5, 2026-11-23): `simulate_uniform_inertial_u3_white_noise.py` sets
`SEED = 12345` (line 62) and calls `np.random.default_rng(SEED)` inside
`simulate_white_noise_from_modal_model` (line 344), afresh for every structure, so every one
of the 5500 paper structures was excited by the same white-noise record.  On a new record the
networks collapse (main network 0.934 -> 0.377).  T11 asks whether training with a DIFFERENT
record per structure fixes that.

Excitation seeding.  Each structure's simulation gets its own record,

    seed = np.random.SeedSequence([base, sample_index])

passed to `simulate_white_noise_from_modal_model(..., seed=seed)` (in the archive: the module
global SIM.SEED was set to it just before the call and 12345 restored after - the same
default_rng(SeedSequence) either way).  `sample_index`
is the paper's global structure index (the 4 digits of `sample_XXXX_...npz`).  Bases:

    20261125   N = 600 training and validation structures   (TRAIN_EXC_BASE)
    20261126   the paper's 500 test structures -> paper_varexc (TEST_EXC_BASE)
    20261127   the nine real bridges' FE-simulated ceiling      (REAL_EXC_BASE, index = LOOCV position)

Different entropy, so no test record equals a training record.  Nothing on disk is edited.

The re-simulation is the generator's own broadband path (`assemble_phase1_sample_payload`
with bandpass=True: FE solve from params_json, all 15 modes, 1.5-20 Hz zero-phase bandpass,
Welch, row-normalised PSD; edges by the generator's `build_cpsd_edge_payload` and
`build_cpsd_full_edge_payload`), the same path E14's `resimulate` / T4's `sp_prepare` follow.
Only the excitation-dependent arrays are replaced in the written sample:

    x_psd, x_psd_row_scale, edge_attr_cpsd, edge_attr_cpsd_resonance, edge_attr_cpsd_full

every other key (graph, labels, matcher record, parameters) is copied from the stored sample
and checked byte-identical on read-back.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ec_common as EC  # noqa: E402  (puts REPO and the paper pipeline on sys.path)

C = EC.C
REPO = EC.REPO

T11_DATA = EC.EC_DATA / "T11"
T11_RESULTS = EC.EC_RESULTS / "T11_varied_excitation_pilot"
T11_LOGS = EC.EC / "logs"
EC_RUNS = EC.EC_RUNS

PAPER_EXC_SEED = 12345
TRAIN_EXC_BASE = 20261125
TEST_EXC_BASE = 20261126
REAL_EXC_BASE = 20261127

TRAINVAL_DIR = T11_DATA / "trainval_varexc" / "samples"      # 508 + 92 re-simulated samples
TEST_DIR = T11_DATA / "paper_varexc" / "samples"             # 500 re-simulated test samples
TRAIN_LIST = C.SPLITS / "N600_train.txt"
VAL_LIST = C.SPLITS / "N600_val.txt"
TEST_LIST = C.SPLITS / "test.txt"

# keys of a sample that depend on the excitation record (everything else is copied)
EXC_KEYS = ("x_psd", "x_psd_row_scale", "edge_attr_cpsd", "edge_attr_cpsd_resonance", "edge_attr_cpsd_full")

# the two new arms (fr_train.Job fields) and the paper networks they are compared with
ARMS = {
    "ec_varexc_edgegat_N600": {"edge_feature_mode": "cpsd_full", "paper_twin": "edgegat_N600",
                               "label": "EdgeGAT + coherency, varied excitation"},
    "ec_varexc_edgegat_geomorient_N600": {"edge_feature_mode": "geom_orient",
                                          "paper_twin": "edgegat_geomorient_N600",
                                          "label": "EdgeGAT + orientation, varied excitation"},
}
PAPER_NETS = ("edgegat_N600", "edgegat_geomorient_N600", "edgegat_N2000")
NET_LABEL = {
    "ec_varexc_edgegat_N600": "EdgeGAT + coherency, N = 600, varied exc. (new)",
    "ec_varexc_edgegat_geomorient_N600": "EdgeGAT + orientation, N = 600, varied exc. (new)",
    "edgegat_N600": "EdgeGAT + coherency, N = 600, paper (fixed exc.)",
    "edgegat_geomorient_N600": "EdgeGAT + orientation, N = 600, paper (fixed exc.)",
    "edgegat_N2000": "EdgeGAT + coherency, N = 2000, paper main (fixed exc.)",
}
NET_ORDER = ("ec_varexc_edgegat_N600", "edgegat_N600", "ec_varexc_edgegat_geomorient_N600",
             "edgegat_geomorient_N600", "edgegat_N2000")
TEST_SETS = ("paper", "paper_varexc", "fresh_test", "fresh_test_paperexc")
SET_LABEL = {
    "paper": "paper test set (paper structures, paper exc. 12345)",
    "paper_varexc": "paper_varexc (paper test structures, a new record per structure)",
    "fresh_test": "fresh_test (new structures, one shared new record 20261124)",
    "fresh_test_paperexc": "fresh_test_paperexc (new structures, paper exc. 12345)",
}


def read_list(p: Path) -> list[str]:
    return [s.strip() for s in Path(p).read_text(encoding="utf-8").splitlines() if s.strip()]


def sample_index(name: str) -> int:
    return int(Path(name).stem.split("_")[1])


def exc_seed(base: int, index: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([int(base), int(index)])


def seed_repr(seed) -> str:
    if isinstance(seed, np.random.SeedSequence):
        return f"SeedSequence({list(np.atleast_1d(seed.entropy))})"
    return str(seed)


def resimulate_payload(sample_path: Path, seed, t_total: float | None = None) -> dict:
    """Re-simulate one stored sample with excitation `seed` (12345 = the paper's record).

    The generator's broadband assemble path, line for line for everything the excitation
    touches.  Returns the five excitation-dependent arrays, the re-solved label quantities
    (for the label check) and the graph rows' absolute acceleration (for the record checks).
    seed and t_total are passed to the simulator (archive: the module globals SIM.SEED and
    SIM.T_TOTAL were set for the call and restored after); t_total None = the simulator's 60 s."""
    import generate_population_dataset_modes4 as gen
    import simulate_uniform_inertial_u3_white_noise as SIM
    from export_projected_graph import build_projected_graph_from_model

    if gen.simulate_white_noise_from_modal_model is not SIM.simulate_white_noise_from_modal_model:
        raise RuntimeError("the generator's simulate function is not the simulator module's")
    if seed is None:
        raise ValueError("resimulate_payload needs an explicit excitation seed")
    with np.load(sample_path, allow_pickle=True) as d:
        sampled = gen._parse_params_json(d)
        st = {k: d[k] for k in ("graph_node_ids", "edge_index", "edge_attr_cpsd_matched_freq_hz",
                                "freq_bins_hz", "matched_mode_indices")}
    sel = np.asarray(st["matched_mode_indices"], dtype=int).reshape(-1)
    p = gen.build_solver_params(sampled, n_candidate_modes=gen.N_CANDIDATE_MODES)
    solve_res = gen.run_solver(p, ref_x_fracs=gen.REF_X_FRACS, tol_x=gen.REF_SENSOR_TOL_X)
    graph_payload = build_projected_graph_from_model(solve_res.model, include_sensor_map=False)
    sim_model = dict(solve_res.model)
    sim_model["M_full"] = solve_res.M_full
    sim = gen.simulate_white_noise_from_modal_model(
        sim_model, solve_res.phi_full, solve_res.freq_all_hz, sel,
        zeta=gen.PHASE1_ZETA, plot=False, return_time_series=True,
        use_all_modes=True, bandpass_hz=(1.5, 20.0), seed=seed, t_total=t_total)
    gids = np.asarray(graph_payload["graph_node_ids"], dtype=np.int64)
    rows = gids - 1
    ei = np.asarray(graph_payload["edge_index"], dtype=np.int64)
    freq_bins = np.asarray(sim["freq_bins_hz"], dtype=float).reshape(-1)[: gen.PHASE1_FREQ_BINS]
    a3 = np.asarray(sim["a3_abs"], dtype=float)[rows, :]
    fs = float(np.asarray(sim["fs"]).reshape(()))
    kw = dict(edge_index=ei, freq_bins_hz=freq_bins, a3_abs=a3, fs=fs, nperseg=SIM.WELCH_NPERSEG,
              noverlap=SIM.WELCH_NOVERLAP, nfft=SIM.WELCH_NFFT)
    cpsd = gen.build_cpsd_edge_payload(matched_freq_hz=np.asarray(st["edge_attr_cpsd_matched_freq_hz"], float), **kw)
    full = gen.build_cpsd_full_edge_payload(**kw)
    return {
        "features": {
            "x_psd": np.asarray(sim["Paa_norm"], dtype=float)[rows, : gen.PHASE1_FREQ_BINS],
            "x_psd_row_scale": np.asarray(sim["Paa_row_scale"], dtype=float)[rows],
            "edge_attr_cpsd": np.asarray(cpsd["edge_attr_cpsd"], dtype=np.float32),
            "edge_attr_cpsd_resonance": np.asarray(cpsd["edge_attr_cpsd_resonance"], dtype=np.float32),
            "edge_attr_cpsd_full": np.asarray(full["edge_attr_cpsd_full"], dtype=np.float32),
        },
        "graph_node_ids": gids, "edge_index": ei, "freq_bins_hz": freq_bins,
        "freq_selected_hz": np.asarray(solve_res.freq_all_hz, dtype=float)[sel],
        "y_mode_shape_abs": np.asarray(sim["phi_u3_abs_norm"], dtype=float)[rows, :],
        "a3": a3,
    }


def support_rows(edge_index: np.ndarray, n: int) -> np.ndarray:
    """The graph's degree-1 nodes: the support joints that carry the injected base noise."""
    deg = np.bincount(np.asarray(edge_index[0], dtype=np.int64), minlength=n)
    return np.where(deg == 1)[0]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(Path(path).open(encoding="utf-8"))) if Path(path).exists() else []


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def bootstrap_ci(v, n_boot: int = 2000, seed: int = 20260925) -> tuple[float, float]:
    """s4_indomain.bootstrap_ci (same resampling and seed)."""
    v = np.asarray(v, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
