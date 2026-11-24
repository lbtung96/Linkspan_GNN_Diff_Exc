"""E18 - the two explainer pages: the methodology chapter and the shape explorer.

    python tools/run_stage.py experiments/E18_population_basis/make_pages.py      # ~1 min

  e18_population_prior.html   the method as a chapter (Bayes, the population prior, kriging),
                              with a worked example on an Ng = 3 deck
  e18_shape_explorer.html     the method step by step, interactive: the basis on any deck, the
                              fit, the prior, the reading at a Welch bin, the posterior

Both are self-contained (only MathJax and fonts load from a CDN) and are built from the
templates and scripts in pages/.  Nothing here changes E18's results; the pages read them.

The chapter.  Tables 1 and 2 and Eq. (28) are computed here from the Ng = 3 training decks
and prior.npz.  Every other number its text quotes is gated: against the same computation,
against E18's synthetic.csv / real.csv / prior_fit.csv, or against E19's test.csv.  The build
fails if one has drifted, so the prose cannot silently disagree with the results.

The explorer.  Embeds E18's prior, eight synthetic test decks with E19's cached cross-spectra
(per girder count, the decks at the median and the 10th percentile of E18's two-sensor PB-class
score) and the nine real bridges' recorded cross-spectra.  Gate: E18's own Python code, run on
the stored (32-bit) data, reproduces E18's recorded MAC on every deck, layout, prior and mode,
and E19's template MAC.  pages/explorer_test.js (Node) checks the page's JavaScript port
against the same numbers.

Needs E18 stages 1-3, E19 stages 1-2 (its cache/test is read) and the field recordings.
"""

from __future__ import annotations

import base64
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
E19 = REPO / "experiments" / "E19_mode_identification"
PAGES = HERE / "pages"
for p in (REPO / "Final_results_for_paper" / "pipeline", REPO / "experiments" / "E17_real_holdout", HERE, E19):
    sys.path.insert(0, str(p))

import stage1_holdout as E17  # noqa: E402  (puts the pipeline on sys.path)
import popbasis as PB  # noqa: E402
import ident as ID  # noqa: E402
import build_phase3_graphs as BPG  # noqa: E402
from phase3_utils import bandpass_filter, BANDPASS_F_HIGH, BANDPASS_F_LOW, load_graph_package  # noqa: E402
from generate_population_dataset_modes4 import _parse_params_json  # noqa: E402

C = E17.C
MODES = ["B1", "T1", "B2", "T2"]
KEYS = ["pooled", *PB.CLASSES]
F_MAX = 20.0
TOL = 1e-5
SUPD = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def rows_of(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def uni(x: float, d: int = 2, plus: bool = False) -> str:
    """Typographic minus; '+' if asked; zero unsigned."""
    s = f"{x:.{d}f}"
    if float(s) == 0:
        return f"{0:.{d}f}"
    return s.replace("-", "−") if x < 0 else ("+" + s if plus else s)


def tex(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


def times10(x: float, d: int = 1) -> str:
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if abs(round(m, d)) >= 10:
        e, m = e + 1, m / 10
    return f"{m:.{d}f} × 10{str(e).translate(SUPD)}"


def mac_c(u, v) -> float:
    return float(np.abs(np.vdot(u, v)) ** 2 / (np.vdot(u, u).real * np.vdot(v, v).real + 1e-300))


PRIOR = np.load(HERE / "prior.npz")


def sigma(ng: int, key: str) -> tuple[np.ndarray, float]:
    return PRIOR[f"Sigma_Ng{ng}_{key}"], float(PRIOR[f"tau2_Ng{ng}_{key}"])


# ============================================================================ the chapter
def chapter() -> str:
    ng = 3
    train = [t for t in (C.SPLITS / f"N{C.MAIN_N}_train.txt").read_text(encoding="utf-8").split() if f"_Ng{ng}_" in t]
    names = {"cornerA": (1.0, -1.0), "cornerB": (1.0, 1.0), "midA": (0.5, -1.0)}
    Bev = PB.design(np.array([v[0] for v in names.values()]), np.array([v[1] for v in names.values()]))
    acc = {k: [] for k in KEYS}
    vals = {k: [] for k in KEYS}
    geo = []
    for name in train:
        with np.load(C.POP_BROADBAND / name, allow_pickle=True) as d:
            p = _parse_params_json(d)
            c = d["coords_xy_mm"].astype(np.float64)
            y = d["y_mode_shape_signed"].astype(np.float64)
            f = d["y_freq_hz"].astype(np.float64).reshape(-1)
        L, dy = float(p["L_mm"]), float(p["dy_mm"])
        geo.append((L, dy))
        B = PB.design(*PB.norm_coords(c, L, ng, dy))
        cl = PB.classes(y, f, c, 0.0, (ng - 1) * dy, 0.02 * L)
        for m in range(4):
            a, _ = PB.fit(B, y[:, m])
            for k in ("pooled", cl[m]):
                acc[k].append(a)
                vals[k].append(Bev @ a)
    st, sig_diff = {}, 0.0
    for k in KEYS:
        A = np.array(acc[k])
        S, tau2 = sigma(ng, k)
        sig_diff = max(sig_diff, float(np.abs(A.T @ A / len(A) - S).max()))
        Cv = Bev @ S @ Bev.T
        sd = np.sqrt(np.diag(Cv))
        st[k] = {"n": len(A), "tau2": tau2, "sd": dict(zip(names, sd)),
                 "rho_cc": Cv[0, 1] / np.sqrt(Cv[0, 0] * Cv[1, 1]), "rho_cm": Cv[0, 2] / np.sqrt(Cv[0, 0] * Cv[2, 2]),
                 "points": np.round(np.vstack([np.array(vals[k]), -np.array(vals[k])]), 4)}

    # Table 1
    t1 = "\n".join(
        f'      <tr><td class="lbl">{k}</td><td>{st[k]["n"]}</td><td>{st[k]["sd"]["cornerA"]:.2f}</td>'
        f'<td>{st[k]["sd"]["midA"]:.2f}</td><td>{uni(st[k]["rho_cc"], plus=True)}</td>'
        f'<td>{uni(st[k]["rho_cm"], plus=True)}</td><td>{times10(st[k]["tau2"])}</td></tr>' for k in KEYS)

    # worked example: L = 18 m, dy = 2.5 m, the two front corners measured
    L, dy = 18000.0, 2500.0
    W_ = (ng - 1) * dy
    pts = np.array([[L, 0], [L, W_], [0.25 * L, 0], [0.5 * L, 0], [0.75 * L, 0], [0.5 * L, W_ / 2], [0.5 * L, W_]])
    B = PB.design(*PB.norm_coords(pts, L, ng, dy))
    kept = np.array([0, 1])
    ex, Css = {}, {}
    for k, dvec in (("pooled", (1, 1)), ("bend1", (1, 1)), ("bend2", (1, 1)),
                    ("pooled", (1, -1)), ("tors1", (1, -1)), ("tors2", (1, -1))):
        S, t = sigma(ng, k)
        Cf = B @ S @ B.T
        Css[k] = Cf[np.ix_(kept, kept)] + t * np.eye(2)
        W = np.linalg.solve(Css[k], Cf[:, kept].T).T
        assert np.allclose(W[2:], PB.operator(B, kept, S, t)[2:])
        post = np.diag(Cf - Cf[:, kept] @ np.linalg.solve(Css[k], Cf[:, kept].T))
        pred = W @ np.array(dvec, float)
        ex[(k, dvec)] = [(pred[i], 1 - max(post[i], 0) / Cf[i, i]) for i in range(2, 7)]
    t2 = []
    for title, dvec, ks in (("Readings d = (1, 1): corners in phase", (1, 1), ("pooled", "bend1", "bend2")),
                            ("Readings d = (1, −1): corners in anti-phase", (1, -1), ("pooled", "tors1", "tors2"))):
        t2.append(f'      <tr class="grp"><td colspan="6">{title}</td></tr>')
        for k in ks:
            t2.append(f'      <tr><td class="lbl">{k}</td>' + "".join(
                f'<td>{uni(v, plus=True)}<span class="r">{round(100 * r)} %</span></td>' for v, r in ex[(k, dvec)]) + "</tr>")
    mat = lambda M, d: (rf"\begin{{pmatrix}} {tex(M[0, 0], d)} &amp; {tex(M[0, 1], d)} \\ "  # noqa: E731
                        rf"{tex(M[1, 0], d)} &amp; {tex(M[1, 1], d)} \end{{pmatrix}}")

    # the numbers the prose quotes, gated
    b2, po = st["bend2"], st["pooled"]
    sv, su, rho = b2["sd"]["cornerA"], b2["sd"]["midA"], b2["rho_cm"]
    psv, psu, prho = po["sd"]["cornerA"], po["sd"]["midA"], po["rho_cm"]
    syn = rows_of(HERE / "synthetic.csv")
    mm = lambda q: np.mean([float(q[f"mac_{m}"]) for m in MODES])  # noqa: E731
    two = {meth: np.mean([mm(q) for q in syn if q["route"] == "ods" and q["family"] == "outer" and q["level"] == "1"
                          and q["method"] == meth]) for meth in ("PB", "PB-class")}
    dense = np.mean([mm(q) for q in syn if q["method"] == "dense"])
    real = rows_of(HERE / "real.csv")
    rdense = np.mean([float(q["pred_mac_label"]) for q in real if q["source"] == "real" and q["layout"] == "full"])
    nseg = [int(q["nseg"]) for q in real if q["source"] == "real"]
    tmpl = [q for q in rows_of(E19 / "test.csv") if q["method"] == "template" and q["layout"] == "outer1"]
    tng = {g: np.mean([float(q["mac"]) for q in tmpl if int(q["Ng"]) == g]) for g in C.NG_VALUES}
    proj = np.median([float(q["projection_mac"]) for q in rows_of(HERE / "prior_fit.csv")])
    n_train = len({q["sample"] for q in rows_of(HERE / "prior_fit.csv")})
    Lr = (min(g[0] for g in geo) / 1000, max(g[0] for g in geo) / 1000)
    dyr = (min(g[1] for g in geo) / 1000, max(g[1] for g in geo) / 1000)
    t2v = {k: v for k, v in ex.items()}
    pool_m = t2v[("pooled", (1, -1))]
    tors2 = t2v[("tors2", (1, -1))]
    tors2_r = [round(100 * tors2[i][1]) for i in (0, 1, 2, 4)]
    gates = [
        f"holds {n_train} linkspans, {len(train)} for each girder count",
        f"\\(K_c = {po['n']}\\) shapes ({len(train)} decks × 4 modes)",
        f"\\(K_c = {b2['n']}\\) each",
        f"\\(L\\) from {Lr[0]:.1f} to {Lr[1]:.1f} m and \\(\\dyy\\) from {dyr[0]:.2f} to {dyr[1]:.2f} m",
        f"L ({Lr[0]:.1f}–{Lr[1]:.1f} m on the Ng = 3 training decks)",
        f"a median of {proj:.5f}",
        f"{200 / 2048:.4f} Hz", f"{128 / 2048:.4f} Hz",
        f"the two front corners have \\(\\rho = {tex(st['tors1']['rho_cc'])}\\)",
        f"front corner A and midspan A have \\(\\rho = {tex(rho)}\\)",
        f"the same two points have \\(\\rho = {tex(prho)}\\)",
        f"\\(\\sigma_v = {sv:.2f}\\), the midspan of the same girder has \\(\\sigma_u = {su:.2f}\\), and \\(\\rho = {tex(rho)}\\)",
        f"\\({tex(rho)} \\times {su:.2f} / {sv:.2f} = {tex(rho * su / sv)}\\)",
        f"\\({su:.2f} \\times \\sqrt{{1 - {abs(rho):.2f}^2}} = {su * np.sqrt(1 - rho ** 2):.2f}\\), down from {su:.2f}",
        f"predicts {uni(prho * psu / psv)} with standard deviation {psu * np.sqrt(1 - prho ** 2):.2f} (prior {psu:.2f})",
        f"midspan at about {uni(t2v[('bend2', (1, 1))][1][0])} on both girders",
        f"give {uni(t2v[('bend1', (1, 1))][2][0], plus=True)} at three-quarter span",
        f"predicts {uni(pool_m[1][0])} at midspan A and {uni(pool_m[4][0], plus=True)} at midspan B",
        f"removes at most {max(round(100 * pool_m[i][1]) for i in (0, 1, 3, 4))} % of the prior variance",
        f"Under tors1 they remove {round(100 * t2v[('tors1', (1, -1))][2][1])} % of the prior variance",
        f"only {min(tors2_r)} to {max(tors2_r)} % along girder A and at midspan B",
        f"the predicted ±{abs(tors2[1][0]):.2f} at midspan",
        f"a mean MAC of only {two['PB']:.3f}, against {two['PB-class']:.3f} with the class prior",
        f"scores a mean MAC of {np.mean([float(q['mac']) for q in tmpl]):.3f} on the synthetic test decks, "
        f"above the two-corner fill ({two['PB-class']:.3f})",
        f"a mean MAC of {tng[2]:.3f} there, against {min(tng[g] for g in (3, 4, 5)):.3f}–{max(tng[g] for g in (3, 4, 5)):.3f}",
        f"scores a mean MAC of {dense:.3f} against the true shape",
        f"({min(nseg)} to {max(nseg)} Welch segments)",
        f"the deflection read at every sensor scores {rdense:.3f} against SSI",
        f"to {times10(sig_diff, 0)}.",
    ]
    html = (PAGES / "method_template.html").read_text(encoding="utf-8")
    missing = [g for g in gates if g not in html]
    if missing:
        raise SystemExit("chapter text disagrees with the computation; not found:\n  " + "\n  ".join(missing))
    print(f"chapter: {len(gates)} quoted numbers agree; Sigma recomputed vs prior.npz {sig_diff:.1e}")

    def panel(title, key, xk, yk):
        s = st[key]
        col = list(names)
        return {"title": title, "xName": xk.replace("cornerA", "corner A").replace("cornerB", "corner B"),
                "yName": {"cornerB": "corner B", "midA": "midspan A"}[yk],
                "pts": [[round(float(p[col.index(xk)]), 3), round(float(p[col.index(yk)]), 3)] for p in s["points"]],
                "sx": float(s["sd"][xk]), "sy": float(s["sd"][yk]), "rho": float(s["rho_cc"] if yk == "cornerB" else s["rho_cm"])}

    panels = [panel("(a) bend2 · corner A → midspan A", "bend2", "cornerA", "midA"),
              panel("(b) pooled · corner A → midspan A", "pooled", "cornerA", "midA"),
              panel("(c) tors1 · corner A → corner B", "tors1", "cornerA", "cornerB")]
    for k, v in (("<!--__TABLE1__-->", t1), ("<!--__TABLE2__-->", "\n".join(t2)),
                 ("__CSS_BEND2__", mat(Css["bend2"], 3)), ("__CSS_TORS1__", mat(Css["tors1"], 2)),
                 ("/*__DATA__*/null", json.dumps(panels, separators=(",", ":")))):
        assert html.count(k) == 1, k
        html = html.replace(k, v)
    return html


# ============================================================================ the explorer
def b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def dec(s: str, shape) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f4").astype(np.float64).reshape(shape)


def r6(x) -> float:
    return float(f"{float(x):.6g}")


def edges_of(ei: np.ndarray) -> list:
    return sorted({(int(min(a, b)), int(max(a, b))) for a, b in ei.T if a != b})


def s1_curve(G: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(np.moveaxis(G, -1, 0))[:, -1]


def e19_block(rows, sample, lay_key):
    out = {"template": {}, "labellers": {}}
    for q in rows:
        if q["sample"] != sample or q["layout"] != lay_key:
            continue
        if q["method"] == "template":
            out["template"][q["mode"]] = float(q["mac"])
        elif q["method"] in ("rank", "population"):
            b = out["labellers"].setdefault(q["method"], {"theta": float(q["theta"]), "lam": q["lam"], "modes": {}})
            b["modes"][q["mode"]] = {"f_hat": float(q["f_hat"]) if q["f_hat"] not in ("", "nan") else None,
                                     "correct": q["correct"] == "True", "mac": float(q["mac"])}
        elif q["method"] == "given":
            out.setdefault("given", {})[q["mode"]] = float(q["mac"])
    return out


def peaks_block(G, f, sxy, ya, yb, xtol, theta):
    return [{"f": float(f"{p['f']:.7g}"), "bin": p["bin"], "ch": ID.character(p["u"], sxy, ya, yb, xtol)}
            for p in ID.peaks(G, f, theta)]


def explorer_data() -> dict:
    prior = {}
    for ng in C.NG_VALUES:
        prior[str(ng)] = {}
        for k in KEYS:
            S, t = sigma(ng, k)
            w, v = np.linalg.eigh(S)
            o = np.argsort(-w)
            prior[str(ng)][k] = {"Sigma": b64(S), "tau2": t, "evals": [float(x) for x in np.maximum(w[o], 0)],
                                 "evecs": b64(v[:, o].T)}
    fit_rows = rows_of(HERE / "prior_fit.csv")
    fit_stats = {}
    for ng in C.NG_VALUES:
        for k in PB.CLASSES:
            m = np.array([float(q["projection_mac"]) for q in fit_rows if int(q["Ng"]) == ng and q["class"] == k])
            fit_stats[f"{ng}_{k}"] = {"n": int(m.size), "median": float(np.median(m)),
                                      "p05": float(np.percentile(m, 5)), "min": float(m.min())}
    train = sorted({q["sample"] for q in fit_rows})
    rng_geo = {str(ng): {"L": [1e9, 0], "dy": [1e9, 0], "Nx": [1e9, 0], "n": 0} for ng in C.NG_VALUES}
    for s in train:
        with np.load(C.POP_BROADBAND / s, allow_pickle=True) as d:
            p = _parse_params_json(d)
        g = rng_geo[str(int(p["Ng"]))]
        g["n"] += 1
        for key, val in (("L", float(p["L_mm"])), ("dy", float(p["dy_mm"])), ("Nx", int(p["Nx"]))):
            g[key] = [min(g[key][0], val), max(g[key][1], val)]

    # synthetic test decks: per girder count, the median and the 10th percentile of E18's two-sensor score
    syn = rows_of(HERE / "synthetic.csv")
    score, e18_syn = {}, {}
    for q in syn:
        if q["route"] == "ods" and q["family"] == "outer" and q["level"] == "1" and q["method"] == "PB-class":
            score[q["sample"]] = np.mean([float(q[f"mac_{m}"]) for m in MODES])
        if q["family"] == "outer" and q["level"] in ("1", "2") and q["method"] in ("PB", "PB-class", "PB-class-data"):
            lay = "outer1" if q["level"] == "1" else "outer2"
            e18_syn.setdefault(q["sample"], {}).setdefault(q["route"], {}).setdefault(lay, {})[q["method"]] = \
                [float(q[f"mac_{m}"]) for m in MODES]
    e19_test, e19_real = rows_of(E19 / "test.csv"), rows_of(E19 / "real.csv")
    picked = []
    for ng in C.NG_VALUES:
        ss = sorted([s for s in score if f"_Ng{ng}_" in s], key=lambda s: score[s])
        picked += [(ss[len(ss) // 2], "typical"), (ss[int(0.10 * len(ss))], "hard")]
    synth = []
    for sample, tag in picked:
        d = np.load(E19 / "cache" / "test" / f"{sample}.npz")
        with np.load(C.POP_BROADBAND / f"{sample}.npz", allow_pickle=True) as z:
            p = _parse_params_json(z)
            ei = z["edge_index"].astype(np.int64)
        ng, L, dy = int(d["Ng"]), float(d["L_mm"]), float(d["dy_mm"])
        c = d["coords"].astype(np.float64)
        o1, o2, G2 = d["outer1_nodes"].astype(int), d["outer2_nodes"].astype(int), d["outer2_G"]
        pos = [int(np.where(o2 == k)[0][0]) for k in o1]
        assert np.array_equal(d["outer1_G"], G2[np.ix_(pos, pos)]), sample
        f = d["f_axis"].astype(np.float64)
        ya, yb, xtol = 0.0, (ng - 1) * dy, 0.02 * L
        ent = {"id": sample, "tag": tag, "Ng": ng, "L": L, "dy": dy, "Nx": int(p["Nx"]),
               "balcony": bool(p["balcony"]["enable"]), "diag": str(p.get("diag_pattern")),
               "coords": [[round(x, 2), round(y, 2)] for x, y in c], "edges": edges_of(ei),
               "support": np.flatnonzero(d["support"]).tolist(), "y": [[r6(v) for v in row] for row in d["y"]],
               "f": [float(x) for x in d["f_true"]], "classes": [str(x) for x in d["classes"]], "fs": float(d["fs"]),
               "nf": int(f.size), "df": float(f[1] - f[0]), "kept": {"outer1": o1.tolist(), "outer2": o2.tolist()},
               "keptPos": {"outer1": pos, "outer2": list(range(len(o2)))}, "G_re": b64(G2.real), "G_im": b64(G2.imag),
               "lines": [ya, yb], "xtol": xtol, "s1": {"outer1": b64(s1_curve(d["outer1_G"])), "outer2": b64(s1_curve(G2))},
               "e18": e18_syn[sample], "score_pbclass_2": score[sample], "e19": {}}
        for lay, G in (("outer1", d["outer1_G"]), ("outer2", G2)):
            blk = e19_block(e19_test, sample, lay)
            for b in blk["labellers"].values():
                b["peaks"] = peaks_block(G, f, c[d[f"{lay}_nodes"]], ya, yb, xtol, b["theta"])
            ent["e19"][lay] = blk
        synth.append(ent)

    # real bridges: recorded cross-spectra, rows of the four front + midspan sensors
    e18_real = rows_of(HERE / "real.csv")
    real = []
    for s in C.LOOCV:
        pkg = load_graph_package(s, graph_root=str(E17.S6.GRAPH_ROOT))
        coords = pkg["node_coords_xy_mm"].astype(np.float64)
        n_fe = int(pkg["n_fe_nodes"])
        sens = pkg["sensor_node_indices"].astype(int)
        sxy = coords[sens]
        lay, (ya, yb) = E17.layouts(sxy, coords[:n_fe, 0].max())
        assert set(lay["front_pair"]) <= set(lay["front_mid"])
        geo = BPG.STRUCTURES[s]
        L, ng, dy = float(geo["L_mm"]), int(geo["Ng"]), float(geo["dy_mm"])
        raw6 = np.load(E17.RUN / "decomposition_raw" / f"{s}.npz")
        y_ssi, f_ssi = raw6["y_ssi"].astype(np.float64), raw6["ssi_freq_hz"].astype(np.float64)
        fe_t = raw6["fe_target"].astype(np.float64)
        assert fe_t.shape[0] == n_fe, s
        field = np.load(E17.FIELD_ROOT / s / "data" / "field_modal_filtering_outputs.npz", allow_pickle=False)
        u3 = field["raw_u3_timeseries_clean"].astype(np.float64).T
        fs = float(2.0 * field["freq_hz"].astype(np.float64)[-1])
        x = bandpass_filter(u3, fs, f_low=BANDPASS_F_LOW, f_high=BANDPASS_F_HIGH)
        Gall, fax, nseg = E17.csd_matrix(x, fs)
        nb = int(np.searchsorted(fax, F_MAX, side="right"))
        Gall, fax = Gall[:, :, :nb], fax[:nb]
        fm = lay["front_mid"]
        ent = {"id": s, "Ng": ng, "L": L, "dy": dy, "balcony": bool(geo["balcony"]["enable"]),
               "coords": [[round(a, 2), round(b, 2)] for a, b in coords], "n_fe": n_fe,
               "edges": edges_of(pkg["edge_index"].astype(np.int64)), "sens": sens.tolist(),
               "layouts": {"front_pair": lay["front_pair"].tolist(), "front_mid": fm.tolist()},
               "lines": [ya, yb], "xtol": 300.0, "y": [[r6(v) for v in row] for row in y_ssi],
               "f": [float(v) for v in f_ssi], "classes": PB.classes(y_ssi, f_ssi, sxy, ya, yb, 300.0), "fs": fs,
               "nseg": int(nseg), "nf": int(nb), "df": float(fax[1] - fax[0]),
               "rows_re": b64(Gall[fm].real), "rows_im": b64(Gall[fm].imag),
               "fe_target": [[r6(v) for v in row] for row in fe_t], "fe_freq": [float(v) for v in raw6["fe_freq_hz"]],
               "s1": {k: b64(s1_curve(Gall[np.ix_(lay[k], lay[k])])) for k in ("front_pair", "front_mid")},
               "e18": {}, "e19": {}}
        for q in e18_real:
            if q["source"] != "real" or q["structure"] != s:
                continue
            if q["layout"] in ("front_pair", "front_mid") and q["imputer"] in ("PB", "PB-class", "PB-class-data"):
                ent["e18"].setdefault(q["layout"], {}).setdefault(q["imputer"], {})[q["mode"]] = {
                    "mac": float(q["pred_mac_label"]), "class_used": q["class_used"], "ref": int(q["ref_sensor"])}
            if q["layout"] == "full":
                ent["e18"].setdefault("dense", {})[q["mode"]] = float(q["pred_mac_label"])
        for key, lname in (("outer1", "front_pair"), ("outer2", "front_mid")):
            blk = e19_block(e19_real, s, key)
            G = Gall[np.ix_(lay[lname], lay[lname])]
            for b in blk["labellers"].values():
                b["peaks"] = peaks_block(G, fax, sxy[lay[lname]], ya, yb, 300.0, b["theta"])
            ent["e19"][lname] = blk
        real.append(ent)
    data = {"meta": {"deg_x": PB.DEG_X, "deg_y": PB.DEG_Y, "eta_scale": PB.ETA_SCALE, "ridge": PB.RIDGE,
                     "classes": list(PB.CLASSES), "modes": MODES, "fit_stats": fit_stats, "geo_ranges": rng_geo,
                     "n_train": len(train)},
            "prior": prior, "synth": synth, "real": real}
    data["meta"]["check_vs_e18"] = check_stored(data)
    return data


def check_stored(data: dict) -> dict:
    """E18's own code on the stored (32-bit) data must give E18's and E19's recorded MAC."""
    Sig = lambda ng, k: (dec(data["prior"][str(ng)][k]["Sigma"], (36, 36)), data["prior"][str(ng)][k]["tau2"])  # noqa: E731
    dmax = {"synthetic_ods": 0.0, "synthetic_shape": 0.0, "real": 0.0, "template": 0.0}
    for e in data["synth"]:
        c, y = np.array(e["coords"]), np.array(e["y"])
        B = PB.design(*PB.norm_coords(c, e["L"], e["Ng"], e["dy"]))
        deck = np.ones(len(c), bool)
        deck[e["support"]] = False
        G = dec(e["G_re"], (4, 4, e["nf"])) + 1j * dec(e["G_im"], (4, 4, e["nf"]))
        f = np.arange(e["nf"]) * e["df"]
        bins = [int(np.argmin(np.abs(f - fm))) for fm in e["f"]]
        for m in range(4):
            lead = np.linalg.eigh(Sig(e["Ng"], e["classes"][m])[0])[1][:, -1]
            dmax["template"] = max(dmax["template"], abs(mac_c((B @ lead)[deck], y[deck, m])
                                                         - e["e19"]["outer1"]["template"][MODES[m]]))
        for lay in ("outer1", "outer2"):
            kp, s = e["keptPos"][lay], np.array(e["kept"][lay])
            D = []
            for m in range(4):
                Gb = G[np.ix_(kp, kp)][:, :, bins[m]]
                ref = int(np.argmax(np.real(np.diag(Gb))))
                D.append(Gb[ref, :] / np.sqrt(Gb[ref, ref].real))
            D = np.stack(D, 1)
            spans = (np.abs(c[s, 1] - e["lines"][0]) < 1).any() and (np.abs(c[s, 1] - e["lines"][1]) < 1).any()
            cl_ods = PB.classes(np.real(D), np.array(e["f"]), c[s], *e["lines"], e["xtol"]) if spans else None
            cl_shp = PB.classes(y[s], np.array(e["f"]), c[s], *e["lines"], e["xtol"]) if spans else None
            for meth in ("PB", "PB-class", "PB-class-data"):
                for m in range(4):
                    for route, cl, vec, keep in (("ods", cl_ods, D[:, m], deck), ("shape", cl_shp, y[s, m], slice(None))):
                        key = {"PB": "pooled", "PB-class": e["classes"][m], "PB-class-data": cl[m] if cl else "pooled"}[meth]
                        v = PB.operator(B, s, *Sig(e["Ng"], key)) @ vec
                        k = f"synthetic_{route}"
                        dmax[k] = max(dmax[k], abs(mac_c(v[keep], y[keep, m]) - e["e18"][route][lay][meth][m]))
    for e in data["real"]:
        c, sens, y = np.array(e["coords"]), np.array(e["sens"]), np.array(e["y"])
        ns = len(sens)
        B = PB.design(*PB.norm_coords(c, e["L"], e["Ng"], e["dy"]))
        R = dec(e["rows_re"], (4, ns, e["nf"])) + 1j * dec(e["rows_im"], (4, ns, e["nf"]))
        fm = e["layouts"]["front_mid"]
        f = np.arange(e["nf"]) * e["df"]
        bins = [int(np.argmin(np.abs(f - q))) for q in e["f"]]
        for m in range(4):
            lead = np.linalg.eigh(Sig(e["Ng"], e["classes"][m])[0])[1][:, -1]
            dmax["template"] = max(dmax["template"], abs(mac_c((B @ lead)[sens], y[:, m])
                                                         - e["e19"]["front_pair"]["template"][MODES[m]]))
        for lname in ("front_pair", "front_mid"):
            kept = np.array(e["layouts"][lname])
            hidden = np.setdiff1d(np.arange(ns), kept)
            refs = [int(kept[np.argmax([R[fm.index(k), k, bins[m]].real for k in kept])]) for m in range(4)]
            ks = c[sens[kept]]
            ya, yb = e["lines"]
            cl = None
            if (np.abs(ks[:, 1] - ya) < 1).any() and (np.abs(ks[:, 1] - yb) < 1).any():
                vals = np.stack([np.real(R[fm.index(refs[m]), kept, bins[m]]) for m in range(4)], 1)
                cl = PB.classes(vals, np.array(e["f"]), ks, ya, yb, 300.0)
            for meth in ("PB", "PB-class", "PB-class-data"):
                for m in range(4):
                    key = {"PB": "pooled", "PB-class": e["classes"][m], "PB-class-data": cl[m] if cl else "pooled"}[meth]
                    want = e["e18"][lname][meth][MODES[m]]
                    assert want["ref"] == refs[m] and (meth != "PB-class-data" or want["class_used"] == key), (e["id"], lname, meth, m)
                    W = PB.operator(B, sens[kept], *Sig(e["Ng"], key))
                    rr = R[fm.index(refs[m]), :, bins[m]]
                    v = rr / np.sqrt(rr[refs[m]].real)
                    v[hidden] = W[sens[hidden]] @ v[kept]
                    dmax["real"] = max(dmax["real"], abs(mac_c(v, y[:, m]) - want["mac"]))
    bad = {k: v for k, v in dmax.items() if v > TOL}
    if bad:
        raise SystemExit(f"stored explorer data do not reproduce E18/E19 (tol {TOL}): {bad}")
    print("explorer: stored data reproduce E18 / E19, max |dMAC|", {k: f"{v:.1e}" for k, v in dmax.items()})
    return dmax


def explorer() -> str:
    data = explorer_data()
    d = data["meta"]["check_vs_e18"]
    check = (f"Check, at build time: E18's own Python code, run on the data stored in this page, reproduces E18's recorded"
             f" MAC to within {times10(d['synthetic_ods'])} on the synthetic resonance route,"
             f" {times10(d['synthetic_shape'])} on the exact-shape route and {times10(d['real'])} on the real bridges,"
             f" for every deck, layout, prior and mode here, and E19's template MAC to within {times10(d['template'])}."
             f" Step 5 makes the same comparison live, with the maths running in your browser.")
    html = (PAGES / "explorer_template.html").read_text(encoding="utf-8")
    js = {k: (PAGES / f"explorer_{k.lower()}.js").read_text(encoding="utf-8") for k in ("CORE", "MODEL", "UI")}
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    for k, v in (("__CHECK__", check), ("/*__CORE__*/", js["CORE"]), ("/*__MODEL__*/", js["MODEL"]),
                 ("/*__DATA__*/null", blob), ("/*__UI__*/", js["UI"])):
        assert html.count(k) == 1, k
        html = html.replace(k, v)
    return html


def main() -> int:
    for name, build in (("e18_population_prior.html", chapter), ("e18_shape_explorer.html", explorer)):
        html = build()
        (HERE / name).write_text(html, encoding="utf-8", newline="\n")
        print(f"wrote {name} ({len(html.encode('utf-8')) // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
