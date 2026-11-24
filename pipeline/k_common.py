"""T13 - feasibility round: can the coherency-edge network learn torsion once the excitation varies?

Every k_*.py script imports this.  Run from the repository root through tools/run_stage.py.

T11 (h_*.py) re-simulated the N = 600 training/validation structures with one excitation record
per structure and trained 5000 epochs: the coherency arm scored 0.41 on every test set with
T1 ~ 0.01 (best epoch 83, then validation rose while training fell), the orientation arm
0.71-0.73.  T13 screens three levers at 400 epochs, one seed:

  arm                        edge input   N     records per training structure   record length
  ec_t13_coh_K1_N600         coherency    600   1 (base 20261125 = T11's)        60 s
  ec_t13_coh_K4_N600         coherency    600   4 (bases below), 1 drawn/epoch    60 s
  ec_t13_coh_K1_N600_T120    coherency    600   1 (base 20261125)                120 s
  ec_t13_coh_K1_N2000        coherency    2000  1 (base 20261125)                60 s
  ec_t13_orient_K4_N600      orientation  600   4 (as arm 2)                      60 s

Excitation seeding is T11's (h_common.resimulate_payload): SIM.SEED = SeedSequence([base, index])
for the one call, 12345 restored after.  Record k of structure i is SeedSequence([BASES[k], i]);
BASES[0] = 20261125 = T11's training base, so record 0 IS T11's sample (reused, not re-simulated).
Record length: SIM.T_TOTAL patched in memory for the one call (60.0 restored after); the
generator's simulate function reads it at call time (same __globals__ as SEED - checked).
Welch settings unchanged, so a 120 s record gives 22 Welch segments instead of 10.

Validation (best-epoch selection) is always the 92 N600 validation structures with their single
base-20261125 record (60 s), except the 120 s arm, whose validation records are 120 s (same seed).
Augmentation: each epoch every training structure contributes ONE of its K records, drawn by
numpy default_rng([20261130, epoch]) - independent of torch's RNG stream, as T4's sp_train.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import h_common as H  # noqa: E402  (T11: seeding patch, re-simulation, csv/json/bootstrap helpers)

EC = H.EC
C = H.C
REPO = H.REPO

T13_DATA = EC.EC_DATA / "T13"
T13_RESULTS = EC.EC_RESULTS / "T13_feasibility"
T13_LOGS = EC.EC / "logs"
EC_RUNS = EC.EC_RUNS
GATE_ROOT = EC_RUNS / "ec_t13_gates"

PAPER_T_TOTAL = 60.0
LONG_T_TOTAL = 120.0
EPOCHS = 400
EPOCH_SEED = 20261130
# record k of structure i: SeedSequence([BASES[k], i]); BASES[0] is T11's training base
BASES = (H.TRAIN_EXC_BASE, 20261131, 20261132, 20261133)
K = len(BASES)

TRAIN600 = C.SPLITS / "N600_train.txt"
VAL600 = C.SPLITS / "N600_val.txt"
TRAIN2000 = C.SPLITS / "N2000_train.txt"
VAL2000 = C.SPLITS / "N2000_val.txt"
TEST_LIST = C.SPLITS / "test.txt"

T11_TRAINVAL = H.TRAINVAL_DIR                         # base 20261125, 60 s, N600 train + val
T11_TEST = H.TEST_DIR                                 # paper_varexc: base 20261126, 60 s


def sdir(key: str) -> Path:
    return T13_DATA / key / "samples"


# data sets written by k_prepare.py: key -> (manifests, base, t_total, folder)
DATASETS = {
    "rec1": {"lists": (TRAIN600,), "base": BASES[1], "t_total": PAPER_T_TOTAL,
             "what": "N600 training structures, record k = 1"},
    "rec2": {"lists": (TRAIN600,), "base": BASES[2], "t_total": PAPER_T_TOTAL,
             "what": "N600 training structures, record k = 2"},
    "rec3": {"lists": (TRAIN600,), "base": BASES[3], "t_total": PAPER_T_TOTAL,
             "what": "N600 training structures, record k = 3"},
    "trainval_T120": {"lists": (TRAIN600, VAL600), "base": BASES[0], "t_total": LONG_T_TOTAL,
                      "what": "N600 training + validation structures, base 20261125, 120 s"},
    "paper_varexc_T120": {"lists": (TEST_LIST,), "base": H.TEST_EXC_BASE, "t_total": LONG_T_TOTAL,
                          "what": "paper test structures, base 20261126 (as T11's paper_varexc), 120 s"},
    "trainval_N2000": {"lists": (TRAIN2000, VAL2000), "base": BASES[0], "t_total": PAPER_T_TOTAL,
                       "what": "N2000 training + validation structures, base 20261125, 60 s "
                               "(the 600 N600 structures copied from T11, the rest simulated)"},
}

# arms: pools[k] = folder holding training record k; pool[0] also holds the validation set
ARMS = {
    "ec_t13_coh_K1_N600": {"edge": "cpsd_full", "n": 600, "pools": (T11_TRAINVAL,), "t_total": 60,
                           "train": TRAIN600, "val": VAL600, "lever": "reference (T11 condition, 400 ep.)",
                           "label": "coherency, N 600, K 1, 60 s"},
    "ec_t13_coh_K4_N600": {"edge": "cpsd_full", "n": 600,
                           "pools": (T11_TRAINVAL, sdir("rec1"), sdir("rec2"), sdir("rec3")), "t_total": 60,
                           "train": TRAIN600, "val": VAL600, "lever": "augmentation (4 records)",
                           "label": "coherency, N 600, K 4, 60 s"},
    "ec_t13_coh_K1_N600_T120": {"edge": "cpsd_full", "n": 600, "pools": (sdir("trainval_T120"),), "t_total": 120,
                                "train": TRAIN600, "val": VAL600, "lever": "record length (120 s)",
                                "label": "coherency, N 600, K 1, 120 s"},
    "ec_t13_coh_K1_N2000": {"edge": "cpsd_full", "n": 2000, "pools": (sdir("trainval_N2000"),), "t_total": 60,
                            "train": TRAIN2000, "val": VAL2000, "lever": "population size (N 2000)",
                            "label": "coherency, N 2000, K 1, 60 s"},
    "ec_t13_orient_K4_N600": {"edge": "geom_orient", "n": 600,
                              "pools": (T11_TRAINVAL, sdir("rec1"), sdir("rec2"), sdir("rec3")), "t_total": 60,
                              "train": TRAIN600, "val": VAL600, "lever": "orientation, augmented",
                              "label": "orientation, N 600, K 4, 60 s"},
}
T11_ARMS = ("ec_varexc_edgegat_N600", "ec_varexc_edgegat_geomorient_N600")
ALL_NETS = tuple(ARMS) + T11_ARMS
NET_LABEL = {**{k: v["label"] + " (400 ep.)" for k, v in ARMS.items()},
             "ec_varexc_edgegat_N600": "T11 coherency, N 600, K 1, 60 s (5000 ep.)",
             "ec_varexc_edgegat_geomorient_N600": "T11 orientation, N 600, K 1, 60 s (5000 ep.)"}

TEST_SETS = ("paper_varexc", "fresh_test", "paper_varexc_T120")
SET_LABEL = {
    "paper_varexc": "paper_varexc (500 paper test structures, own record each, base 20261126, 60 s; T11)",
    "fresh_test": "fresh_test (500 new structures, one shared new record 20261124, 60 s; T5)",
    "paper_varexc_T120": "paper_varexc_T120 (the same 500 structures and seeds, 120 s records; new)",
}

read_list = H.read_list
sample_index = H.sample_index
write_csv, read_csv, write_json, bootstrap_ci = H.write_csv, H.read_csv, H.write_json, H.bootstrap_ci


def exc_seed(base: int, index: int) -> np.random.SeedSequence:
    return H.exc_seed(base, index)


def resimulate(sample_path: Path, seed, t_total: float = PAPER_T_TOTAL) -> dict:
    """h_common.resimulate_payload with SIM.T_TOTAL = t_total for the one call (60.0 restored)."""
    import generate_population_dataset_modes4 as gen
    import simulate_uniform_inertial_u3_white_noise as SIM

    if gen.simulate_white_noise_from_modal_model.__globals__ is not SIM.__dict__:
        raise RuntimeError("the generator's simulate function does not read the simulator module's T_TOTAL")
    if SIM.T_TOTAL != PAPER_T_TOTAL:
        raise RuntimeError(f"simulator T_TOTAL is {SIM.T_TOTAL!r} on entry, expected 60.0")
    SIM.T_TOTAL = float(t_total)
    try:
        res = H.resimulate_payload(sample_path, seed)
    finally:
        SIM.T_TOTAL = PAPER_T_TOTAL
    nt_expected = int(round(float(t_total) / SIM.DT))
    if res["a3"].shape[1] != nt_expected:
        raise RuntimeError(f"record has {res['a3'].shape[1]} samples, expected {nt_expected} for {t_total} s")
    return res


def run_dir(name: str) -> Path:
    for root in (C.RUNS, EC_RUNS):
        if (root / name / "run_summary.json").exists():
            return root / name
    raise FileNotFoundError(name)
