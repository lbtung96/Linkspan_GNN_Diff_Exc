"""T16 - sign test: does coherency carry usable information once torsion's global sign is no longer asked for?

Every p_*.py script imports this.  Run from the repository root through tools/run_stage.py.

T14/T15 (2026-11-23): the paper's coherency networks took torsion's sign from the shared excitation
record (mirroring the record assignment flips 100 % of their torsion predictions); with a record per
structure (T11, T13) that cue is gone and coherency networks output T1 ~ 0 - the squared-error answer
to an undetermined sign.  T16 trains six arms on T13's data (N = 600, four own records per training
structure, one drawn per epoch; validation the 92 N600 validation structures with record r0), 400
epochs, seed 123, EdgeGAT, through k_train.run_one, with three additions (in-memory patches in the
archive; library and fr_train.Job options since the 2026-11-24 restructure, see the end of this file):

  loss "signinv"  per structure and per mode, min(||p - y||^2, ||p + y||^2) summed over its nodes,
                  divided by the element count, i.e. MSE when every sign matches.  MAC (the metric) is
                  sign-blind, so the evaluation is unchanged.  Used for training AND validation loss.
                  (EC-10 suggested this loss.)
  edge "cpsd_full_orient"  the 800-d coherency and the 2-d orientation (cos, sin) side by side (802-d).
  node "amp"      amplitude-keeping node input: sqrt(PSD), deck nodes divided by the deck-wide maximum
                  (so at a modal peak the values across the deck follow |phi|), support (degree-1) nodes
                  by their own maximum (they carry the injected record, 40-540x louder).  Built from
                  x_psd * x_psd_row_scale.  Synthetic evaluation only (the field inputs carry no scale).

  arm                         edges              loss     node input  question
  ec_t16_coh_signinv          coherency          signinv  row-norm    does coherency give torsion once the sign is free?
  ec_t16_none_signinv         none (zeros)       signinv  row-norm    control: the same without any edge input
  ec_t16_cohorient            coh. + orient.     MSE      row-norm    does phase add to orientation?
  ec_t16_cohorient_signinv    coh. + orient.     signinv  row-norm    both
  ec_t16_orient_signinv       orientation        signinv  row-norm    does the loss help orientation?
  ec_t16_coh_signinv_amp      coherency          signinv  amp         does keeping amplitude help?

References (already trained, T13, standard MSE, same data and budget): ec_t13_coh_K4_N600 (0.423),
ec_t13_orient_K4_N600 (0.758).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import k_common as K  # noqa: E402

H, EC, C = K.H, K.EC, K.C
T16_RESULTS = EC.EC_RESULTS / "T16_sign_test"
T16_LOGS = EC.EC / "logs"
POOLS_K4 = (K.T11_TRAINVAL, K.sdir("rec1"), K.sdir("rec2"), K.sdir("rec3"))
COH_ORIENT = "cpsd_full_orient"

ARMS = {
    "ec_t16_coh_signinv": {"edge": "cpsd_full", "loss": "signinv", "amp": False,
                           "label": "coherency, sign-invariant loss"},
    "ec_t16_none_signinv": {"edge": "zero", "loss": "signinv", "amp": False,
                            "label": "no edge input, sign-invariant loss (control)"},
    "ec_t16_cohorient": {"edge": COH_ORIENT, "loss": "mse", "amp": False,
                         "label": "coherency + orientation, MSE"},
    "ec_t16_cohorient_signinv": {"edge": COH_ORIENT, "loss": "signinv", "amp": False,
                                 "label": "coherency + orientation, sign-invariant loss"},
    "ec_t16_orient_signinv": {"edge": "geom_orient", "loss": "signinv", "amp": False,
                              "label": "orientation, sign-invariant loss"},
    "ec_t16_coh_signinv_amp": {"edge": "cpsd_full", "loss": "signinv", "amp": True,
                               "label": "coherency, sign-invariant loss, amplitude-keeping nodes"},
}
REFS = {"ec_t13_coh_K4_N600": {"edge": "cpsd_full", "loss": "mse", "amp": False, "label": "coherency, MSE (T13)"},
        "ec_t13_orient_K4_N600": {"edge": "geom_orient", "loss": "mse", "amp": False, "label": "orientation, MSE (T13)"}}
ALL = {**ARMS, **REFS}

# ------------------------------------------------------------------ the three T16 options: library code now
# The archive installed these in memory (install_signinv / install_cohorient / install_amp).  Since the
# 2026-11-24 restructure they are options of the library and of fr_train.Job:
#   loss "signinv"           GraphSAGE_Modal_GNN.signinv_shape_loss, via Job(shape_loss="signinv")
#   edge "cpsd_full_orient"  an EdgeGATConv_Modal_GNN edge mode
#   node "amp"               EdgeGATConv_Modal_GNN.amp_node_features, via Job(node_feature_mode="amp") or
#                            EdgeGATConv_Modal_GNN.set_node_feature_mode("amp") for inference
# The names below stay for the callers (checks, reports).


def signinv_shape_loss(pred, target, bnn):
    import GraphSAGE_Modal_GNN as gs
    return gs.signinv_shape_loss(pred, target, bnn)


def amp_features(x_psd: np.ndarray, row_scale: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    import EdgeGATConv_Modal_GNN as eg
    return eg.amp_node_features(x_psd, row_scale, edge_index)


def shape_loss(arm: dict) -> str:
    return "signinv" if arm["loss"] == "signinv" else "mse"


def node_feature_mode(arm: dict) -> str:
    return "amp" if arm["amp"] else "psd"


def install_cohorient() -> None:
    """No patching: the 802-d mode is a library mode (kept as a check for the callers)."""
    import fr_edge_modes
    fr_edge_modes.install()


def install_for(arm: dict) -> None:
    """Select the arm's node input for every graph this process builds (training or inference).
    Loss and edge mode need no installing: the loss is a Job option, the edge mode a library mode."""
    import EdgeGATConv_Modal_GNN as eg
    install_cohorient()
    eg.set_node_feature_mode(node_feature_mode(arm))
