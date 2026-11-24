"""Assemble REPORT.md, summary.csv and the figures from the E14 stage-1 outputs.

Generated, never hand-edited - the same rule `results/RESULTS.md` lives under.
Every number in REPORT.md is computed here from `stage1_per_sample.csv`,
`stage1_diagnostics.csv` and `raw/*.npz`; the prose around it only names which
number to read.

The unit of replication is the structure (90 held-out structures, one seed). No
statement here is about retraining.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

RUN_DIR = REPO / "runs_phase2" / "EdgeGATConv" / "sensitivity_N600_ep500"
MODES = ("B1", "T1", "B2", "T2")
GOOD_MAC = 0.9

# figure colours - validated reference palette, slots 1-3 (all-pairs safe), plus a
# neutral for the floor, which is a reference level rather than a series
C_FULL, C_CORNERS, C_OPERATOR, C_FLOOR = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def mac(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return float(abs(np.dot(u, v)) ** 2 / (np.dot(u, u) * np.dot(v, v) + 1e-30))


def matrix(rows: list[dict], key: str, samples: list[str], **where) -> np.ndarray:
    """(n_samples, 4) array of `key`, ordered by `samples` and MODES."""
    idx = {(r["sample"], r["mode"]): r[key] for r in rows
           if all(r[k] == v for k, v in where.items())}
    return np.array([[idx[(s, m)] for m in MODES] for s in samples], dtype=float)


def fmt_row(label: str, vals, spec: str = ".3f") -> str:
    return f"| {label} | " + " | ".join(format(v, spec) for v in vals) + " |"


def main() -> int:
    rows = read_csv(HERE / "stage1_per_sample.csv")
    diags = read_csv(HERE / "stage1_diagnostics.csv")
    samples = [d["sample"] for d in diags]
    n = len(samples)
    bal = np.array([str(d["balcony"]) == "True" for d in diags])

    M = {(ck, v): matrix(rows, "mac", samples, checkpoint=ck, variant=v)
         for ck in ("last", "best") for v in ("full", "corners_prop", "corner_copy", "zeros")}
    FE = {(ck, v): matrix(rows, "abs_freq_err_hz", samples, checkpoint=ck, variant=v)
          for ck in ("last", "best") for v in ("full", "corners_prop", "corner_copy", "zeros")}
    op = matrix(rows, "prop_shape_mac", samples, checkpoint="last", variant="full")
    true_f = matrix(rows, "true_freq_hz", samples, checkpoint="last", variant="full")
    fnorm = json.loads((RUN_DIR / "frequency_normalization.json").read_text(encoding="utf-8"))
    const_err = np.abs(true_f - np.asarray(fnorm["freq_mean_hz"], dtype=float))

    # ---- the two controls collapse to one flat floor -------------------------
    flat_gap = float(np.abs(M[("last", "corner_copy")] - M[("last", "zeros")]).max())
    raw = {s: np.load(HERE / "raw" / f"{s}.npz") for s in samples}
    flat_std = max(float((np.std(raw[s][f"last_{v}"], axis=0)
                          / (np.abs(raw[s][f"last_{v}"]).max(axis=0) + 1e-30)).max())
                   for s in samples for v in ("zeros", "corner_copy"))
    ones_mac = np.array([[mac(np.ones(len(raw[s]["target"])), raw[s]["target"][:, m])
                          for m in range(4)] for s in samples])
    floor = M[("last", "zeros")]

    # ---- observability of each mode at the two corners -----------------------
    obs, phase, echo = [], [], []
    for s in samples:
        t, c = raw[s]["target"], raw[s]["corners"]
        obs.append(np.abs(t[c]).max(axis=0) / np.abs(t).max(axis=0))
        phase.append(np.sign(t[c[0]] * t[c[1]]))
        echo.append([mac(raw[s]["last_corners_prop"][:, m], raw[s]["prop_shape"][:, m])
                     for m in range(4)])
    obs, phase, echo = np.array(obs), np.array(phase), np.array(echo)

    # ---- reproduction against the run's own test metrics ---------------------
    tm = {r["sample_identifier"]: r for r in read_csv(RUN_DIR / "evaluation" / "test_metrics.csv")}
    rep_mac = max(abs(M[("last", "full")][i, j] - tm[s][f"mac_{m}"])
                  for i, s in enumerate(samples) for j, m in enumerate(MODES))
    rep_freq = max(abs(FE[("last", "full")][i, j] - tm[s][f"freq_mae_{m}_hz"])
                   for i, s in enumerate(samples) for j, m in enumerate(MODES))
    resim_x = max(d["x_psd_resim_maxdiff"] for d in diags)
    resim_e = max(d["edge_resim_maxdiff"] for d in diags)
    resim_f = max(d["freq_resim_maxdiff_hz"] for d in diags)
    reach = min(d["prop_min_rms_ratio"] for d in diags)
    unreached = int(sum(d["prop_n_unreached"] for d in diags))

    # ---- paired test: corners vs the flat floor, unit = structure ------------
    cp = M[("last", "corners_prop")]
    tests = []
    for j in range(4):
        diff = cp[:, j] - floor[:, j]
        p = float(wilcoxon(cp[:, j], floor[:, j]).pvalue) if np.any(diff != 0) else float("nan")
        tests.append((float(diff.mean()), int((diff > 0).sum()), p))

    # ---- summary.csv -----------------------------------------------------------
    out_rows = []
    for ck in ("last", "best"):
        for v in ("full", "corners_prop", "corner_copy", "zeros"):
            for j, m in enumerate(MODES):
                x = M[(ck, v)][:, j]
                out_rows.append({
                    "checkpoint": ck, "variant": v, "mode": m, "n_structures": n,
                    "mac_mean": x.mean(), "mac_sd": x.std(ddof=1), "mac_median": np.median(x),
                    "mac_q25": np.percentile(x, 25), "mac_q75": np.percentile(x, 75),
                    f"frac_mac_ge_{GOOD_MAC}": float((x >= GOOD_MAC).mean()),
                    "freq_mae_hz": FE[(ck, v)][:, j].mean(),
                    "freq_mape_pct": 100 * (FE[(ck, v)][:, j] / true_f[:, j]).mean(),
                })
    with open(HERE / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    rep_i = make_figures(M, op, raw, samples)

    # ---- REPORT.md -------------------------------------------------------------
    L = M[("last", "full")].mean(0), cp.mean(0), floor.mean(0)
    cfg = json.loads((RUN_DIR / "run_summary.json").read_text(encoding="utf-8"))
    ng = sorted({int(d["Ng"]) for d in diags})
    md: list[str] = []
    add = md.append
    add("# E14 - two front-corner sensors, the rest propagated\n")
    add("**Generated by `make_report.py`. Do not hand-edit.** Design and assumptions: "
        "[`README.md`](README.md).\n")
    add(f"Network `sensitivity_N600_ep500` (broadband, seed 123, 500 epochs, "
        f"`last_model.pt`), evaluated on its own {n} held-out synthetic structures "
        f"(Ng = {', '.join(map(str, ng))}). Synthetic domain only. The unit of "
        f"replication is the structure; there is one seed, so nothing here says "
        f"whether retraining would reproduce it.\n")

    add("## The answer\n")
    add(f"**No.** Measured at the two front corners and propagated, the input takes "
        f"mean MAC from {L[0].min():.2f}-{L[0].max():.2f} down to "
        f"{cp.mean(0).min():.2f}-{cp.mean(0).max():.2f}, and "
        f"{int((cp >= GOOD_MAC).sum())} of {cp.size} structure-mode cells reach "
        f"MAC {GOOD_MAC} (full graph: {int((M[('last', 'full')] >= GOOD_MAC).sum())}). "
        f"Frequency error grows from {FE[('last', 'full')].mean():.2f} Hz to "
        f"{FE[('last', 'corners_prop')].mean():.2f} Hz on average - against "
        f"{FE[('last', 'corner_copy')].mean():.2f} Hz with no spatial information at "
        f"all, and {const_err.mean():.2f} Hz for guessing the training-set mean.\n")

    add("| | " + " | ".join(MODES) + " |")
    add("|---|" + "---:|" * 4)
    add(fmt_row("full graph (every node measured)", L[0]))
    add(fmt_row("**two corners, propagated**", L[1]))
    add(fmt_row("flat floor (no spatial information)", L[2]))
    add(fmt_row("median, two corners", np.median(cp, 0)))
    add(fmt_row(f"share of structures at MAC >= {GOOD_MAC}, two corners", (cp >= GOOD_MAC).mean(0)))
    add(fmt_row("operator ceiling (true shape, propagated)", op.mean(0)))
    add("")
    add("Mean MAC over all graph nodes against the stored signed FE target. "
        "**Operator ceiling** is not a network output: the true mode shape, known only "
        "at the two corners, pushed through the same propagation operator.\n")

    add("## Two different failures\n")
    add("How much of each mode the two corners can see at all - corner amplitude over "
        "the mode's peak amplitude anywhere on the deck, and whether the two corners move "
        "together (+1) or against each other (-1):\n")
    add("| | " + " | ".join(MODES) + " |")
    add("|---|" + "---:|" * 4)
    add(fmt_row("corner amplitude / peak, mean", obs.mean(0)))
    add(fmt_row("corner amplitude / peak, worst structure", obs.min(0)))
    add(fmt_row("corner phase relation, mean", phase.mean(0), "+.2f"))
    add("")
    add(f"**B1 and T1 are not short of information.** The front corners are where both "
        f"modes peak ({obs[:, 0].min():.2f} and {obs[:, 1].min():.2f} of peak amplitude "
        f"in the worst structure), and the phase relation separates them cleanly "
        f"(in phase {phase[:, 0].mean():+.2f}, anti-phase {phase[:, 1].mean():+.2f}). "
        f"The network still reaches only {cp[:, 0].mean():.3f} and {cp[:, 1].mean():.3f}. "
        f"What it lacks is the field: propagation turns two point records into a smooth "
        f"fall-off from the front, and this network reads mode shapes off the spatial "
        f"pattern of its input rather than off the geometry. Fed any spatially uniform "
        f"input it returns a spatially uniform shape (largest spatial std / max-abs over "
        f"both controls and all structures: {flat_std:.1e}), so it has no geometric "
        f"prior to fall back on. Nor does it simply echo the diffusion pattern: MAC "
        f"between the two-corner prediction and the propagated true shape is only "
        + ", ".join(f"{m} {e:.2f}" for m, e in zip(MODES, echo.mean(0)))
        + ". It is off the distribution it was trained on, where every node carried a "
          "genuine response, and its output there is neither the field nor the mode.\n")
    add(f"**T2 is barely observable from the front, and so is B2 on decks without a "
        f"balcony.** On average the corners carry {obs[:, 3].mean():.2f} of T2's peak "
        f"amplitude ({obs[:, 3].min():.3f} in the worst structure) and "
        f"{obs[~bal, 2].mean():.2f} of B2's on decks without a balcony. Both modes have "
        f"their structure along the span, which two sensors on one cross-section cannot "
        f"place. The operator ceiling ({op[:, 2].mean():.3f}, {op[:, 3].mean():.3f}) "
        f"says the propagated field carries little of either shape before any network "
        f"is involved.\n")

    add("### B2 and T2 mean different things on balcony decks\n")
    add(f"{int(bal.sum())} of the {n} test decks have a balcony. At the front corners "
        f"the two labels swap character between the two groups, which is what the "
        f"pooled B2/T2 floors average over:\n")
    add("| | B2, no balcony | B2, balcony | T2, no balcony | T2, balcony |")
    add("|---|---:|---:|---:|---:|")
    for label, arr, spec in (("corner phase relation", phase, "+.2f"),
                             ("corner amplitude / peak", obs, ".2f"),
                             ("flat floor MAC", floor, ".3f"),
                             ("two corners MAC", cp, ".3f"),
                             ("full graph MAC", M[("last", "full")], ".3f")):
        vals = [arr[~bal, 2].mean(), arr[bal, 2].mean(), arr[~bal, 3].mean(), arr[bal, 3].mean()]
        add(f"| {label} | " + " | ".join(format(v, spec) for v in vals) + " |")
    add("")
    add("Without a balcony B2 is in phase across the deck and T2 anti-phase, as the "
        "names say. With one, at the front, 'B2' moves anti-phase and 'T2' in phase. "
        "Which physical mode each label holds on a balcony deck is a question about "
        "the synthetic population's mode matching, not about this experiment, and is "
        "not pursued here. B1 and T1 behave the same in both groups.\n")
    add("## Against the no-information floor\n")
    add(f"Both controls - every node carrying one corner's record, and all-zero input - "
        f"produce a spatially flat mode shape, so both score exactly MAC(flat vector, "
        f"target) (largest difference between them: {flat_gap:.1e}; largest difference "
        f"from an analytic all-ones vector: "
        f"{float(np.abs(floor - ones_mac).max()):.1e}). That floor is not zero wherever "
        f"the target has a large one-signed component - B2 without a balcony "
        f"({floor[~bal, 2].mean():.3f}), T2 with one ({floor[bal, 3].mean():.3f}). "
        f"Paired over the {n} structures (Wilcoxon signed-rank):\n")
    add("| | " + " | ".join(MODES) + " |")
    add("|---|" + "---:|" * 4)
    add(fmt_row("two corners minus floor, mean", [t[0] for t in tests], "+.3f"))
    add("| structures where two corners beat the floor | "
        + " | ".join(f"{t[1]}/{n}" for t in tests) + " |")
    add("| p | " + " | ".join(f"{t[2]:.1e}" for t in tests) + " |")
    add("")
    verdict = []
    for m, (dmean, wins, p) in zip(MODES, tests):
        if p < 0.01 and dmean > 0 and wins > n / 2:
            verdict.append(f"{m} is clearly above the floor ({dmean:+.3f}, {wins}/{n})")
        elif p >= 0.01:
            verdict.append(f"{m} is not distinguishable from it ({dmean:+.3f}, p = {p:.2f})")
        else:
            verdict.append(f"{m} is mixed - {dmean:+.3f} on average but ahead on "
                           f"{wins}/{n} structures")
    sentence = "; ".join(verdict)
    add(sentence[0].upper() + sentence[1:] + ".")
    t2_nb, t2_b = (cp[~bal, 3] - floor[~bal, 3]).mean(), (cp[bal, 3] - floor[bal, 3]).mean()
    t2_line = (f"The T2 split is the balcony split above: without a balcony the floor is "
               f"{floor[~bal, 3].mean():.3f} and two corners change it by {t2_nb:+.3f}; "
               f"with one the floor is {floor[bal, 3].mean():.3f} and they change it by "
               f"{t2_b:+.3f}.")
    if max(t2_nb, t2_b) < 0.05:
        t2_line += " In neither group do two corners add usable T2 information."
    add(t2_line + "\n")

    add("## Frequencies\n")
    add("Mean absolute error, Hz:\n")
    add("| | " + " | ".join(MODES) + " |")
    add("|---|" + "---:|" * 4)
    add(fmt_row("full graph", FE[("last", "full")].mean(0), ".2f"))
    add(fmt_row("**two corners, propagated**", FE[("last", "corners_prop")].mean(0), ".2f"))
    add(fmt_row("one corner copied everywhere", FE[("last", "corner_copy")].mean(0), ".2f"))
    add(fmt_row("all-zero input", FE[("last", "zeros")].mean(0), ".2f"))
    add(fmt_row("constant guess: training-set mean", const_err.mean(0), ".2f"))
    add(fmt_row("true frequency, mean", true_f.mean(0), ".2f"))
    add("")
    add("The constant guess needs no network: it predicts the training-set mean "
        "frequency (`frequency_normalization.json`) for every structure.\n")
    f_cp = FE[("last", "corners_prop")].mean()
    f_copy = FE[("last", "corner_copy")].mean()
    f_full = FE[("last", "full")].mean()
    f_const = const_err.mean()
    add(f"Every modal peak is in the corner spectra - B1 and T1 at full amplitude - yet "
        f"averaged over the four modes the two-corner error is {f_cp:.2f} Hz, against "
        f"{f_copy:.2f} Hz with one corner's record copied to every node, "
        f"{f_const:.2f} Hz for a constant guess and {f_full:.2f} Hz on the full graph. "
        f"Given two propagated records the frequency head does "
        f"{'worse than' if f_cp > f_const else 'no better than' if f_cp >= 0.9 * f_const else 'only a little better than'} "
        f"guessing the population mean. The copy control suggests why: identical "
        f"spectra at every node do no better, so the head appears to need the spatial "
        f"pattern of the spectra, not the spectra alone.\n")

    add("## Robustness: `best_model.pt` (epoch "
        f"{int(cfg.get('best_epoch', -1))})\n")
    add("| | " + " | ".join(MODES) + " |")
    add("|---|" + "---:|" * 4)
    add(fmt_row("full graph", M[("best", "full")].mean(0)))
    add(fmt_row("two corners, propagated", M[("best", "corners_prop")].mean(0)))
    add(fmt_row("flat floor", M[("best", "zeros")].mean(0)))
    add("")
    gap_ck = float(np.abs(M[("best", "corners_prop")].mean(0) - cp.mean(0)).max())
    worst_best = float(M[("best", "corners_prop")].mean(0).max())
    add(f"The two-corner row moves by at most {gap_ck:.3f} MAC between checkpoints and "
        f"stays at or below {worst_best:.3f} in every mode under `best_model.pt`, "
        f"against {M[('best', 'full')].mean(0).min():.3f} or more on the full graph.\n")

    add("## Figures\n")
    add("- `fig_mac_by_mode.png` - every structure's MAC, per mode, for the three inputs.")
    add(f"- `fig_modeshapes_{samples[rep_i]}.png` - the structure whose two-corner mean MAC "
        f"is the median of the {n}: FE target, full-graph prediction, two-corner "
        f"prediction, and the operator ceiling. Black rings mark the two corners.\n")

    add("## Checks\n")
    add(f"- **Re-simulation is faithful.** Across all {n} structures the rebuilt inputs "
        f"match the stored ones to {resim_x:.1e} (node PSD) and {resim_e:.1e} (edge "
        f"coherency); the re-solved target frequencies to {resim_f:.1e} Hz.")
    add(f"- **The full-graph row reproduces the run's own test metrics** "
        f"(`evaluation/test_metrics.csv`, trained and tested on GPU in the source "
        f"repository): largest per-structure MAC difference {rep_mac:.1e}, frequency "
        f"error {rep_freq:.1e} Hz - CPU against GPU.")
    add(f"- **Propagation reaches every node** ({unreached} unreached). The farthest "
        f"node receives as little as {reach:.1e} of the corner amplitude after 50 "
        f"iterations; the features are scale-invariant (row-normalised PSD, coherency), "
        f"so what reaches the network is the mixing ratio, not the amplitude.")
    add("- **The corners are exact FE nodes** - (L, 0) and (L, (Ng-1)·dy), located "
        "within 1 mm, per structure (`stage1_diagnostics.csv`).\n")

    add("## What this does not establish\n")
    add("- **Whether two corners carry enough information in principle.** This network "
        "never saw a propagated input. A network trained on inputs built the same way - "
        "two corners, propagated - is the experiment that answers it; B1/T1, fully "
        "observable at the corners, are where it could plausibly work, T2 where it "
        "plausibly cannot.")
    add("- **Anything about Ng = 3-5.** The N = 600 network has only seen two-girder "
        "decks, so every test structure here is Ng = 2 and the two corners sit on the "
        "only two girders there are. Stage 2, [`REPORT_N10000.md`](REPORT_N10000.md), "
        "repeats the test on the N = 10 000 network across all four girder counts and "
        "adds a four-sensor layout.")
    add("- **Real sensors.** The corner records are noise-free simulation.")
    add("- **Other operators.** Only the production diffusion operator was run. E9 "
        "ranks alternatives, but with two known values none of them can place a "
        "nodal line along the span either.")

    (HERE / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {HERE / 'REPORT.md'}, summary.csv and figures")
    return 0


def make_figures(M: dict, op: np.ndarray, raw: dict, samples: list[str]) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from infer_phase2a_deterministic_edgegat import _MODE_SHAPE_CMAP

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2, "font.family": "sans-serif"})

    # ---- figure 1: MAC per structure, per mode --------------------------------
    rng = np.random.default_rng(0)
    series = [("Full graph", M[("last", "full")], C_FULL),
              ("Two corners, propagated", M[("last", "corners_prop")], C_CORNERS),
              ("Flat floor", M[("last", "zeros")], C_FLOOR)]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    offs = (-0.26, 0.0, 0.26)
    for (label, x, col), dx in zip(series, offs):
        for j in range(4):
            xs = j + dx + rng.uniform(-0.08, 0.08, size=x.shape[0])
            ax.scatter(xs, x[:, j], s=10, color=col, alpha=0.45, linewidths=0,
                       label=label if j == 0 else None, zorder=2)
            ax.plot([j + dx - 0.11, j + dx + 0.11], [x[:, j].mean()] * 2, color=INK,
                    lw=2, solid_capstyle="round", zorder=3)
    for j in range(4):
        ax.scatter([j], [op[:, j].mean()], marker="D", s=46, color=C_OPERATOR,
                   edgecolors="white", linewidths=1.5, zorder=4,
                   label="Operator ceiling (mean)" if j == 0 else None)
    ax.set_xticks(range(4), MODES)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("MAC vs FE target")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"Two front-corner sensors vs the full graph - {len(samples)} held-out "
                 "structures (black tick = mean)", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              fontsize=9, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(HERE / "fig_mac_by_mode.png", dpi=170)
    plt.close(fig)

    # ---- figure 2: the median structure's mode shapes -------------------------
    cp_mean = M[("last", "corners_prop")].mean(axis=1)
    rep_i = int(np.argsort(cp_mean)[len(cp_mean) // 2])
    s = samples[rep_i]
    d = raw[s]
    xy, c = d["coords"], d["corners"]
    cols = [("FE target", d["target"]), ("Full graph", d["last_full"]),
            ("Two corners, propagated", d["last_corners_prop"]),
            ("Operator ceiling", d["prop_shape"])]
    fig, axes = plt.subplots(4, 4, figsize=(15, 8.6), squeeze=False)
    ei = d["edge_index"]
    for m, name in enumerate(MODES):
        for k, (title, arr) in enumerate(cols):
            ax = axes[m, k]
            v = np.asarray(arr[:, m], dtype=float)
            v = v / (np.abs(v).max() or 1.0)
            for a, b in ei.T[ei[0] < ei[1]]:
                ax.plot(xy[[a, b], 0], xy[[a, b], 1], color=GRID, lw=0.8, zorder=1)
            ax.scatter(xy[:, 0], xy[:, 1], c=v, cmap=_MODE_SHAPE_CMAP, vmin=-1, vmax=1,
                       s=34, edgecolors="#555555", linewidths=0.4, zorder=2)
            ax.scatter(xy[c, 0], xy[c, 1], s=150, facecolors="none", edgecolors=INK,
                       linewidths=1.6, zorder=3)
            score = "" if k == 0 else f"  MAC {mac(arr[:, m], d['target'][:, m]):.2f}"
            ax.set_title(f"{name} - {title}{score}", fontsize=9, color=INK)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
    fig.suptitle(f"{s}: the structure with the median two-corner MAC "
                 "(each panel max-abs normalised; black rings = the two corners, x = L)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / f"fig_modeshapes_{s}.png", dpi=150)
    plt.close(fig)
    return rep_i


if __name__ == "__main__":
    raise SystemExit(main())
