"""T17 - scale-up of T16's best set-ups: N = 2000, 1000 epochs, seeds 123-125, own records.

Every q_*.py script imports this.  Run from the repository root through tools/run_stage.py.

Arms (sign-invariant loss, T16's in-memory additions from p_common):
  ec_t17_cohorient_signinv_N2000_s{seed}   coherency + orientation edges (802-d)
  ec_t17_orient_signinv_N2000_s{seed}      orientation edges (2-d)
Data: N2000 training list (1700) and validation list (300); four own records per training structure,
one drawn per epoch (T13's augmentation wrapper):
  pool 0  T13 trainval_N2000           SeedSequence([20261125, i])  (train + val)
  pool k  T17 rec{k}_N2000, k = 1..3   SeedSequence([20261131..33, i])  - T13's record bases, so the 426
          N2000 training structures that are also N600 training structures are T13's rec{k} files
          (hard-linked, not re-simulated); the other 1274 are simulated here with k_prepare.data_job.
Seeds: 123 (the paper's), 124, 125 - the torch/numpy seed of training (split and data fixed).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import p_common as P  # noqa: E402

K, H, EC, C = P.K, P.H, P.EC, P.C
T17_DATA = EC.EC_DATA / "T17"
T17_RESULTS = EC.EC_RESULTS / "T17_scale_up"
T17_LOGS = EC.EC / "logs"
EPOCHS = 1000
SEEDS = (123, 124, 125)
REC_KEYS = {1: "rec1_N2000", 2: "rec2_N2000", 3: "rec3_N2000"}

# register the three record sets with k_prepare's data job (in memory; T13's own entries untouched)
for _k, _key in REC_KEYS.items():
    K.DATASETS[_key] = {"lists": (K.TRAIN2000,), "base": K.BASES[_k], "t_total": K.PAPER_T_TOTAL,
                        "what": f"N2000 training structures, record k = {_k} (T17)"}
_orig_sdir = K.sdir


def sdir(key: str) -> Path:
    return T17_DATA / key / "samples" if key in REC_KEYS.values() else _orig_sdir(key)


K.sdir = sdir
POOLS = (K.sdir("trainval_N2000"), sdir(REC_KEYS[1]), sdir(REC_KEYS[2]), sdir(REC_KEYS[3]))

BASE_ARMS = {"cohorient_signinv": P.ARMS["ec_t16_cohorient_signinv"],
             "orient_signinv": P.ARMS["ec_t16_orient_signinv"]}


def run_name(arm: str, seed: int) -> str:
    return f"ec_t17_{arm}_N2000_s{seed}"


RUNS = {run_name(a, s): {**BASE_ARMS[a], "base_arm": a, "seed": s} for a in BASE_ARMS for s in SEEDS}
