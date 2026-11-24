"""Plot the synthetic support excitation exactly as the simulator draws it (EC_comments, T12).

    python Final_results_for_paper/EC_comments/pipeline/x_plot_excitation.py

simulate_uniform_inertial_u3_white_noise.simulate_white_noise_from_modal_model draws
    rng = np.random.default_rng(SEED)                              # line 344
    ag_inputs = SIGMA_AG_MMPS2 * rng.standard_normal((n_inputs, nt))
with one row per (support node, DOF), node-major, DOF order x, y, z
(grid_modal_Kinematic_MPC.build_influence_matrix).  The support nodes are the girder ends on the
hinge line x = 0 plus the two bottom-beam ends (_resolve_support_nodes).  This script rebuilds
the FE model of two paper test decks only to find those nodes, then redraws the records with
  * the paper's SEED = 12345 (every structure), and
  * T11's per-structure seed SeedSequence([20261126, sample_index]) (paper_varexc test set).
Nothing is simulated or written outside pipeline/results/T12_excitation_plots/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import csd, welch  # noqa: E402

import ec_common as E  # noqa: E402,F401  (puts the repo and the paper pipeline on sys.path)
import generate_population_dataset_modes4 as gen  # noqa: E402
import simulate_uniform_inertial_u3_white_noise as SIM  # noqa: E402
from grid_modal_Kinematic_MPC import build_influence_matrix  # noqa: E402

OUT = E.EC_RESULTS / "T12_excitation_plots"
POP = E.C.POP_BROADBAND
DECKS = ("sample_1250_Ng2_phase1_gnn.npz", "sample_4000_Ng4_phase1_gnn.npz")   # paper test set
VAREXC_BASE = 20261126                                                       # T11 paper_varexc
DOF = {1: "x", 2: "y", 3: "z"}
NT = int(round(SIM.T_TOTAL / SIM.DT))
FS = SIM.FS


def deck(name: str) -> dict:
    with np.load(POP / name, allow_pickle=True) as d:
        sampled = gen._parse_params_json(d)
        idx = int(d["sample_index"])
    p = gen.build_solver_params(sampled, n_candidate_modes=gen.N_CANDIDATE_MODES)
    model = gen.run_solver(p, ref_x_fracs=gen.REF_X_FRACS, tol_x=gen.REF_SENSOR_TOL_X).model
    sup = SIM._resolve_support_nodes(model)
    _, imap = build_influence_matrix(model, sup, [1, 2, 3])
    nodes = np.asarray(model["nodes"], float)
    xy = {int(r[0]): (r[1] / 1000.0, r[2] / 1000.0) for r in nodes}
    labels = []
    for k, n in enumerate(sup, 1):
        x, y = xy[int(n)]
        labels.append(f"S{k} ({'hinge' if abs(x) < 1e-9 else 'bottom-beam end'}, x={x:.1f} m, y={y:.1f} m)")
    return dict(name=name, idx=idx, Ng=int(model["Ng"]), sup=sup, imap=np.asarray(imap), xy=xy,
                nodes=nodes, labels=labels)


def records(n_inputs: int, seed) -> np.ndarray:
    rng = np.random.default_rng(seed)                      # exactly the simulator's draw
    return SIM.SIGMA_AG_MMPS2 * rng.standard_normal((n_inputs, NT))


def fig_paper_record(d: dict, path: Path) -> None:
    ag = records(d["imap"].shape[0], SIM.SEED)
    t = np.arange(NT) * SIM.DT
    ns = len(d["sup"])
    fig = plt.figure(figsize=(15, 2.0 * ns + 3.2))
    gs = fig.add_gridspec(ns + 1, 4, width_ratios=[1.25, 1, 1, 1], height_ratios=[1] * ns + [1.6])
    ax = fig.add_subplot(gs[:ns, 0])
    ax.scatter(d["nodes"][:, 1] / 1000, d["nodes"][:, 2] / 1000, s=6, c="0.75")
    for k, n in enumerate(d["sup"], 1):
        x, y = d["xy"][int(n)]
        ax.scatter([x], [y], s=60, c="C3", zorder=3)
        ax.annotate(f"S{k}", (x, y), xytext=(6, 4), textcoords="offset points", color="C3", fontsize=10)
    ax.set_title(f"{d['name'][:-4]} (Ng = {d['Ng']}): support nodes = excitation points")
    ax.set_xlabel("x (m) - hinge line at x = 0")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    sel = t <= 2.0
    for row in range(d["imap"].shape[0]):
        k, dof = row // 3, int(d["imap"][row, 1])
        a = fig.add_subplot(gs[k, dof])
        a.plot(t[sel], ag[row, sel], lw=0.6, color=f"C{dof - 1}")
        a.set_ylim(-400, 400)
        a.tick_params(labelsize=7)
        if k == 0:
            a.set_title(f"base acceleration, {DOF[dof]} (mm/s$^2$)", fontsize=9)
        if dof == 1:
            a.set_ylabel(f"S{k + 1}", rotation=0, labelpad=14, fontsize=10)
        if k == ns - 1:
            a.set_xlabel("t (s), first 2 of 60 s", fontsize=8)
    a = fig.add_subplot(gs[ns, :])
    for row in range(ag.shape[0]):
        f, p = welch(ag[row], fs=FS, window="hamming", nperseg=SIM.WELCH_NPERSEG,
                     noverlap=SIM.WELCH_NOVERLAP, nfft=SIM.WELCH_NFFT, detrend=False)
        a.plot(f, p, lw=0.5, alpha=0.6)
    a.axhline(2 * SIM.SIGMA_AG_MMPS2 ** 2 / FS, color="k", ls="--", lw=1,
              label=r"ideal white-noise level $2\sigma^2/f_s$ = 100 (mm/s$^2$)$^2$/Hz")
    a.axvspan(1.5, 20, color="0.9", zorder=0, label="band-pass 1.5-20 Hz (applied to the response)")
    a.set_xlim(0, 40)
    a.set_yscale("log")
    a.set_xlabel("frequency (Hz)")
    a.set_ylabel("Welch PSD")
    a.set_title(f"all {ag.shape[0]} input records (paper SEED = 12345, sigma = {SIM.SIGMA_AG_MMPS2:.0f} mm/s$^2$): "
                "same level, but each record's spectrum has its own fixed random ripple", fontsize=10)
    a.legend(fontsize=8, loc="lower right")
    fig.suptitle("Paper excitation: independent white-noise base acceleration in x, y, z at every support node "
                 "(re-drawn from the simulator's own RNG call)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
    b = (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)
    return a @ b.T / a.shape[1]


def fig_same_vs_new(d2: dict, d4: dict, path: Path) -> dict:
    n2, n4 = d2["imap"].shape[0], d4["imap"].shape[0]
    cases = {
        "paper: SEED = 12345 for every deck": (records(n2, SIM.SEED), records(n4, SIM.SEED)),
        "T11: SeedSequence([20261126, sample_index]) per deck": (
            records(n2, np.random.SeedSequence([VAREXC_BASE, d2["idx"]])),
            records(n4, np.random.SeedSequence([VAREXC_BASE, d4["idx"]]))),
    }
    t = np.arange(NT) * SIM.DT
    sel = t <= 1.0
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1.6, 1]})
    out = {}
    for r, (title, (a2, a4)) in enumerate(cases.items()):
        ax = axes[r, 0]
        ax.plot(t[sel], a2[0, sel], lw=1.4, label=f"{d2['name'][:-4]} (Ng = 2), S1 x")
        ax.plot(t[sel], a4[0, sel], lw=0.8, ls="--", label=f"{d4['name'][:-4]} (Ng = 4), S1 x")
        ax.set_title(title)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("base acceleration (mm/s$^2$)")
        ax.legend(fontsize=8)
        c = corr(a2, a4)
        out[title] = float(np.abs(np.diag(c[:, :n2])).mean())
        ax = axes[r, 1]
        im = ax.imshow(c, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(f"correlation, Ng = 2 inputs (rows) x Ng = 4 inputs (cols)\nmean |diagonal| = {out[title]:.3f}",
                     fontsize=9)
        ax.set_xlabel("Ng = 4 input (S1x, S1y, S1z, S2x, ...)")
        ax.set_ylabel("Ng = 2 input")
    fig.colorbar(im, ax=axes[:, 1], shrink=0.8)
    fig.suptitle("Same record or not: with SEED = 12345 the Ng = 2 deck's 12 inputs are exactly the Ng = 4 deck's "
                 "first 12 inputs", fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_fingerprint(d4: dict, path: Path) -> None:
    """Sample coherency between two INDEPENDENT inputs (S1 z, S2 z): ideal 0, realised a fixed random pattern."""
    seeds = {"paper SEED 12345": SIM.SEED,
             "record B (T11 seed, sample 4000)": np.random.SeedSequence([VAREXC_BASE, d4["idx"]]),
             "record C (T11 seed, sample 4001)": np.random.SeedSequence([VAREXC_BASE, d4["idx"] + 1])}
    iz1, iz2 = 2, 5                                      # S1 z and S2 z (node-major, x y z)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for k, (lab, s) in enumerate(seeds.items()):
        ag = records(d4["imap"].shape[0], s)
        kw = dict(fs=FS, window="hamming", nperseg=SIM.WELCH_NPERSEG, noverlap=SIM.WELCH_NOVERLAP,
                  nfft=SIM.WELCH_NFFT, detrend=False)
        f, pxy = csd(ag[iz1], ag[iz2], **kw)
        _, pxx = welch(ag[iz1], **kw)
        _, pyy = welch(ag[iz2], **kw)
        g = pxy / np.sqrt(pxx * pyy)
        m = f <= 39.0
        axes[0].plot(f[m], np.abs(g[m]), lw=0.8 if k else 1.4, label=lab, color=f"C{k}")
        axes[1].plot(f[m], np.angle(g[m]), lw=0.6 if k else 1.0, label=lab, color=f"C{k}", alpha=0.9 if k == 0 else 0.6)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("|coherency| between S1 z and S2 z")
    axes[0].set_title("Two independent support inputs should have zero coherency. Over a 60 s record with ~10 Welch "
                      "segments they do not: each record leaves its own random pattern, bin by bin.", fontsize=10)
    axes[1].set_ylabel("phase (rad)")
    axes[1].set_xlabel("frequency (Hz) - the paper's 400-bin grid spans 0-39 Hz")
    axes[0].legend(fontsize=9)
    fig.suptitle("Why 'white noise of the same level' is not the same input: the per-record fingerprint "
                 "(identical in all 5500 paper structures)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d2, d4 = deck(DECKS[0]), deck(DECKS[1])
    fig_paper_record(d4, OUT / "fig1_paper_excitation_Ng4.png")
    fig_paper_record(d2, OUT / "fig1b_paper_excitation_Ng2.png")
    c = fig_same_vs_new(d2, d4, OUT / "fig2_same_record_vs_per_structure.png")
    fig_fingerprint(d4, OUT / "fig3_record_fingerprint_coherency.png")
    for d in (d2, d4):
        print(d["name"], "inputs:", d["imap"].shape[0], "|", "; ".join(d["labels"]))
    for k, v in c.items():
        print(f"{k}: mean |corr| of matching inputs = {v:.3f}")
    print("written to", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
