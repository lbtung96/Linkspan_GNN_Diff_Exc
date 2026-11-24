"""T16 - sign test: does coherency carry usable information once torsion's global sign is no longer asked for?

Every p_*.py script imports this.  Run from the repository root through tools/run_stage.py.

T14/T15 (2026-11-23): the paper's coherency networks took torsion's sign from the shared excitation
record (mirroring the record assignment flips 100 % of their torsion predictions); with a record per
structure (T11, T13) that cue is gone and coherency networks output T1 ~ 0 - the squared-error answer
to an undetermined sign.  T16 trains six arms on T13's data (N = 600, four own records per training
structure, one drawn per epoch; validation the 92 N600 validation structures with record r0), 400
epochs, seed 123, EdgeGAT, fr_train unchanged (through k_train.run_one), with three in-memory
additions (library modules unedited):

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

_STATE = {"bnn": None, "installed": set()}


# ------------------------------------------------------------------ sign-invariant loss
def signinv_shape_loss(pred, target, bnn):
    import torch
    bnn = torch.as_tensor(bnn, dtype=torch.int64, device=pred.device)
    if int(bnn.sum()) != pred.shape[0]:
        raise RuntimeError(f"batch has {pred.shape[0]} nodes, batch_num_nodes sums to {int(bnn.sum())}")
    seg = torch.repeat_interleave(torch.arange(bnn.numel(), device=pred.device), bnn)
    sp = torch.zeros(bnn.numel(), pred.shape[1], dtype=pred.dtype, device=pred.device).index_add_(0, seg, (pred - target) ** 2)
    sm = torch.zeros(bnn.numel(), pred.shape[1], dtype=pred.dtype, device=pred.device).index_add_(0, seg, (pred + target) ** 2)
    return torch.minimum(sp, sm).sum() / pred.numel()


def install_signinv() -> None:
    """Make gs._compute_multitask_loss use the sign-invariant shape loss; the batch's node counts come
    from the model's forward (patched to record graph.batch_num_nodes())."""
    import EdgeGATConv_Modal_GNN as eg
    import GraphSAGE_Modal_GNN as gs

    if "signinv" in _STATE["installed"]:
        return
    orig_fwd = eg.EdgeGATModalGNN.forward

    def fwd(self, graph, node_x, edge_x):
        _STATE["bnn"] = graph.batch_num_nodes()
        return orig_fwd(self, graph, node_x, edge_x)

    def multitask(outputs, shape_target, freq_target_hz, *, loss_fn, enable_freq_multitask, freq_loss_weight, freq_stats):
        loss_shape = signinv_shape_loss(outputs.pred_mode_shape_abs, shape_target, _STATE["bnn"])
        freq_target_z = gs.normalize_frequency_targets(freq_target_hz, freq_stats)
        loss_freq = loss_fn(outputs.pred_freq, freq_target_z)
        loss_total = loss_shape + float(freq_loss_weight) * loss_freq
        return loss_total, loss_shape, loss_freq, gs.denormalize_frequency_predictions(outputs.pred_freq, freq_stats)

    eg.EdgeGATModalGNN.forward = fwd
    gs._compute_multitask_loss = multitask
    _STATE["installed"].add("signinv")


# ------------------------------------------------------------------ coherency + orientation edges
def install_cohorient() -> None:
    import EdgeGATConv_Modal_GNN as eg
    import fr_edge_modes

    fr_edge_modes.install()
    if "cohorient" in _STATE["installed"]:
        return
    prev = eg._resolve_edge_attr_array

    def resolve(data, edge_feature_mode: str):
        if edge_feature_mode == COH_ORIENT:
            return np.hstack([prev(data, "cpsd_full"), fr_edge_modes.orientation(data)]).astype(np.float32)
        return prev(data, edge_feature_mode)

    eg._resolve_edge_attr_array = resolve
    eg.EDGE_FEATURE_MODES = (*eg.EDGE_FEATURE_MODES, COH_ORIENT)
    _STATE["installed"].add("cohorient")


# ------------------------------------------------------------------ amplitude-keeping node input
def amp_features(x_psd: np.ndarray, row_scale: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    P = np.asarray(x_psd, np.float64) * np.asarray(row_scale, np.float64).reshape(-1, 1)
    A = np.sqrt(np.maximum(P, 0.0))
    deg = np.bincount(np.asarray(edge_index)[0], minlength=A.shape[0])
    sup = deg == 1
    out = np.empty_like(A)
    out[~sup] = A[~sup] / A[~sup].max()
    out[sup] = A[sup] / np.maximum(A[sup].max(axis=1, keepdims=True), 1e-300)
    return out.astype(np.float32)


def install_amp() -> None:
    import EdgeGATConv_Modal_GNN as eg

    if "amp" in _STATE["installed"]:
        return
    prev = eg.build_phase2a_graph_from_payload

    def build(payload, *, edge_feature_mode, shape_target_mode):
        p = dict(payload)
        p["x_psd"] = amp_features(p["x_psd"], p["x_psd_row_scale"], p["edge_index"])
        return prev(p, edge_feature_mode=edge_feature_mode, shape_target_mode=shape_target_mode)

    eg.build_phase2a_graph_from_payload = build
    _STATE["installed"].add("amp")


def install_for(arm: dict) -> None:
    import fr_edge_modes
    fr_edge_modes.install()
    install_cohorient()            # harmless for other modes
    if arm["loss"] == "signinv":
        install_signinv()
    if arm["amp"]:
        install_amp()
