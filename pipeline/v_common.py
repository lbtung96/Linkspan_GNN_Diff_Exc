"""T20 - deck forcing plus a measurement-noise floor: does a more field-like excitation transfer better?

Every v_*.py script imports this.  Run from the repository root through tools/run_stage.py.

T18 found the paper's excitation (independent white base acceleration at every support DOF) the least
field-like of six, and deck forcing closer.  T20 re-simulates the N = 2000 population under two deck
excitations, each with a white measurement-noise floor, and retrains T17's best arm on them:

  deck   independent white vertical forces at every deck graph node (every graph node except the Ng + 2
         supports).  The Ng hinge supports are fixed vertically and stay at rest; the two front supports
         (bottom-beam ends) are springs in the FE model and respond to the deck load.
  walk   three 70 kg walkers (r_excitation_realism.walker_force: random straight paths at 1.1-1.5 m/s, pacing
         1.6-2.4 Hz with harmonics) plus deck noise at 30 % of the walkers' in-band RMS (RMS over deck nodes)
  noise  white noise added to the absolute acceleration at every graph node, before the training bandpass,
         sigma = ALPHA[exc] x the in-band (1.5-20 Hz) RMS of that record's deck-node response.  ALPHA is
         fitted by v_fit_noise.py on the nine FE twins: the median over bridges of T18's peak_to_bg equals
         the field's.  The hinge supports carry only this noise.

Simulation (simulate()): the library simulator's arithmetic for everything after the load - all 15 modes,
Newmark per mode, absolute vertical acceleration at the graph nodes, phase3_utils bandpass, the simulator's
Welch and row normalisation, the generator's coherency builders - with only the load and the base term
swapped.  Gate (v_prepare --check): under the paper's support excitation and a T13 seed it reproduces
k_common.resimulate's five excitation-dependent arrays.

Records (one numpy Generator per record, drawn in the order forces -> ambient -> noise):
  pool k of training structure i     SeedSequence([BASES[exc][k], i]),  k = 0..3  (val: pool 0)
  test structure i                   SeedSequence([TEST_BASE[exc], i])
Data: data/T20/<exc>_p<k>/samples, data/T20/<exc>_test/samples; labels and graph copied from the paper
population, only k_common's five excitation keys replaced (as T11/T13/T17).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import p_common as P  # noqa: E402

K, H, EC, C = P.K, P.H, P.EC, P.C
EXCS = ("deck", "walk")
BASES = {"deck": (20261160, 20261161, 20261162, 20261163), "walk": (20261170, 20261171, 20261172, 20261173)}
TEST_BASE = {"deck": 20261169, "walk": 20261179}
FIT_NOISE_BASE = 20261189
AMBIENT_SHARE = 0.3
T_TOTAL = K.PAPER_T_TOTAL                   # 60 s, as training
T20_DATA = EC.EC_DATA / "T20"
T20_RESULTS = EC.EC_RESULTS / "T20_deck_forcing"
T20_LOGS = EC.EC / "logs"
NOISE_FIT = T20_RESULTS / "noise_fit.json"
EPOCHS = 1000
SEEDS = (123, 124, 125)
ARM = P.ARMS["ec_t16_cohorient_signinv"]


def sdir(key: str) -> Path:
    return T20_DATA / key / "samples"


def pools(exc: str) -> list[Path]:
    return [sdir(f"{exc}_p{k}") for k in range(4)]


def run_name(exc: str, seed: int) -> str:
    return f"ec_t20_{exc}_cohorient_signinv_N2000_s{seed}"


RUNS = {run_name(e, s): {**ARM, "label": f"coherency + orientation, sign-invariant, {e} forcing + noise floor",
                         "exc": e, "seed": s} for e in EXCS for s in SEEDS}


def alpha() -> dict:
    d = json.loads(NOISE_FIT.read_text(encoding="utf-8"))
    if not d.get("pass"):
        raise RuntimeError("noise_fit.json does not pass")
    return {e: float(d["alpha"][e]) for e in EXCS}


# ---------------------------------------------------------------------------------------------- structure
def solve(sample_path: Path) -> dict:
    """Re-solve one stored population structure exactly as h_common.resimulate_payload does."""
    import generate_population_dataset_modes4 as gen
    import simulate_uniform_inertial_u3_white_noise as SIM
    from export_projected_graph import build_projected_graph_from_model
    from grid_modal_Kinematic_MPC import gdof0
    import h_prepare as HP

    st = HP._load(sample_path)
    with np.load(sample_path, allow_pickle=True) as d:
        sampled = gen._parse_params_json(d)
    p = gen.build_solver_params(sampled, n_candidate_modes=gen.N_CANDIDATE_MODES)
    sr = gen.run_solver(p, ref_x_fracs=gen.REF_X_FRACS, tol_x=gen.REF_SENSOR_TOL_X)
    gp = build_projected_graph_from_model(sr.model, include_sensor_map=False)
    gids = np.asarray(gp["graph_node_ids"], dtype=np.int64)
    ei = np.asarray(gp["edge_index"], dtype=np.int64)
    sup = set(int(v) for v in SIM._resolve_support_nodes(sr.model))
    is_sup = np.array([int(g) in sup for g in gids])
    model = dict(sr.model)
    model["M_full"] = sr.M_full
    return {"name": sample_path.name, "st": st, "model": model, "Phi": np.asarray(sr.phi_full, float),
            "M": sr.M_full, "freqs": np.asarray(sr.freq_all_hz, float), "gids": gids, "ei": ei,
            "zrow": np.array([gdof0(int(g), 3) for g in gids]), "is_sup": is_sup,
            "deck_xy": np.asarray(gp["coords_xy_mm"], float)[~is_sup],
            "graph_equal": bool(np.array_equal(ei, st["edge_index"]) and np.array_equal(gids, st["graph_node_ids"]))}


def _modal(S: dict, Pm: np.ndarray) -> np.ndarray:
    """Modal loads (n_modes, nt) -> vertical relative acceleration at the graph nodes (n_graph, nt)."""
    import generate_population_dataset_modes4 as gen
    import simulate_uniform_inertial_u3_white_noise as SIM

    dt = float(SIM.DT)
    qdd = np.zeros_like(Pm)
    for k, om in enumerate(2.0 * np.pi * S["freqs"]):
        _, _, qdd[k] = SIM.newmark_sdof(om, float(gen.PHASE1_ZETA), Pm[k], dt, beta=0.25, gamma=0.5)
    return S["Phi"][S["zrow"]] @ qdd


def _bp(a: np.ndarray) -> np.ndarray:
    import simulate_uniform_inertial_u3_white_noise as SIM
    from phase3_utils import bandpass_filter

    return bandpass_filter(a, float(SIM.FS), f_low=1.5, f_high=20.0)


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a ** 2)))


def simulate(S: dict, exc: str, seed, alpha_: float = 0.0, t_total: float = T_TOTAL) -> tuple[np.ndarray, dict]:
    """One record -> bandpassed absolute vertical acceleration at the graph nodes (n_graph, nt), diagnostics.

    exc 'supports' is the library simulator's load (gate only; the Generator is default_rng(seed) as there)."""
    import simulate_uniform_inertial_u3_white_noise as SIM
    from grid_modal_Kinematic_MPC import build_influence_matrix
    from r_excitation_realism import walker_force

    dt = float(SIM.DT)
    nt = int(round(float(t_total) / dt))
    rng = np.random.default_rng(seed)
    Phi, zrow, deck = S["Phi"], S["zrow"], ~S["is_sup"]
    diag = {"exc": exc, "nt": nt}
    if exc == "supports":
        sup_nodes = SIM._resolve_support_nodes(S["model"])
        R, _ = build_influence_matrix(S["model"], sup_nodes, [1, 2, 3])
        ag = float(SIM.SIGMA_AG_MMPS2) * rng.standard_normal((int(R.shape[1]), nt))
        Pm = -np.asarray(Phi.T @ (S["M"] @ R), dtype=float) @ ag
        a = _modal(S, Pm) + np.asarray((R @ ag)[zrow], dtype=float)
        if alpha_:
            raise ValueError("the supports case is the gate: no noise")
        return _bp(a), diag
    dz = zrow[deck]
    Fd = rng.standard_normal((int(deck.sum()), nt))
    a = _modal(S, Phi[dz].T @ Fd)
    if exc == "walk":
        Fw = walker_force(nt, dt, S["deck_xy"], rng)
        aw = _modal(S, Phi[dz].T @ Fw)
        c = AMBIENT_SHARE * _rms(_bp(aw)[deck]) / _rms(_bp(a)[deck])
        a = aw + c * a
        diag["ambient_scale"] = c
    elif exc != "deck":
        raise ValueError(exc)
    fixed = ~np.any(Phi[zrow] != 0.0, axis=1)               # graph rows with no vertical mode content
    diag["n_fixed_rows"] = int(fixed.sum())
    diag["fixed_rows_max_before_noise"] = float(np.abs(a[fixed]).max()) if fixed.any() else 0.0
    ref = _rms(_bp(a)[deck])
    diag["support_rows_rms_rel"] = _rms(_bp(a)[~deck]) / ref if ref else float("nan")
    noise = alpha_ * ref * rng.standard_normal(a.shape)
    a = a + noise
    diag["inband_rms_deck"] = ref
    diag["noise_rel_achieved"] = _rms(noise) / ref if ref else float("nan")
    return _bp(a), diag


def features(S: dict, a3: np.ndarray) -> dict:
    """k_common's five excitation-dependent arrays from the bandpassed graph-row record (resimulate_payload's path)."""
    import generate_population_dataset_modes4 as gen
    import simulate_uniform_inertial_u3_white_noise as SIM
    from scipy.signal import get_window

    fs = float(SIM.FS)
    window = get_window("hamming", SIM.WELCH_NPERSEG, fftbins=True)
    f, Paa = SIM.welch_all_nodes(a3, fs, window, nperseg=SIM.WELCH_NPERSEG, noverlap=SIM.WELCH_NOVERLAP, nfft=SIM.WELCH_NFFT)
    Pn, scale = SIM.normalize_rows_by_max_abs(Paa)
    freq_bins = np.asarray(f, dtype=float).reshape(-1)[: gen.PHASE1_FREQ_BINS]
    kw = dict(edge_index=S["ei"], freq_bins_hz=freq_bins, a3_abs=a3, fs=fs, nperseg=SIM.WELCH_NPERSEG,
              noverlap=SIM.WELCH_NOVERLAP, nfft=SIM.WELCH_NFFT)
    cpsd = gen.build_cpsd_edge_payload(matched_freq_hz=np.asarray(S["st"]["edge_attr_cpsd_matched_freq_hz"], float), **kw)
    full = gen.build_cpsd_full_edge_payload(**kw)
    return {"x_psd": np.asarray(Pn, dtype=float)[:, : gen.PHASE1_FREQ_BINS],
            "x_psd_row_scale": np.asarray(scale, dtype=float),
            "edge_attr_cpsd": np.asarray(cpsd["edge_attr_cpsd"], dtype=np.float32),
            "edge_attr_cpsd_resonance": np.asarray(cpsd["edge_attr_cpsd_resonance"], dtype=np.float32),
            "edge_attr_cpsd_full": np.asarray(full["edge_attr_cpsd_full"], dtype=np.float32)}
