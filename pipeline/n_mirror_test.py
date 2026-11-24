"""T15 - mirror test: is the excitation record the only thing that tells the network left from right?

    python tools/run_stage.py Final_results_for_paper/EC_comments/pipeline/n_mirror_test.py [--workers 6] [--per-ng 25]

Hypothesis (T13): a GNN without coordinates cannot tell a deck's left half from its right, and the
signed targets follow a template convention (`compute_reference_sign_alignment`: dot product with a
fixed per-Ng sensor template), so torsion's global sign is undetermined from a mirror-symmetric
input.  With the paper's ONE excitation record, support j always receives the same record rows, so
the record could act as a left/right label.

Test.  The first `--per-ng` paper test structures of each Ng are re-simulated with the paper record
(seed 12345) under four support-record assignments, patched in memory into the simulator's
`build_influence_matrix` (the library module is not edited):

  paper        the paper's assignment (gate: must reproduce the stored features)
  mirror       every support's three records move to its mirror-image support, y-record negated:
               the excitation field becomes the exact mirror image of the paper's
  swap_hinge   only the x = 0 hinge supports are mirrored
  swap_ends    only the two bottom-beam-end supports are mirrored

For a mirror-symmetric deck, `mirror` must give exactly the stored features with the nodes permuted
left <-> right (gate below).  Then a permutation-equivariant network with no coordinates (the
coherency networks) must output the mirror image of its paper prediction - its torsion sign flips
against the unchanged target while MAC is unchanged.  That part follows from symmetry; what the
test establishes is (i) that the record assignment is the ONLY left/right cue in those inputs
(the gate), (ii) what the orientation networks do, whose edge features (cos, sin) are NOT mirrored
by the swap, and (iii) which supports carry the label (partial swaps).

Metric: per mode, MAC and the signed cosine between prediction and target (MAC = cos^2; the sign of
the cosine is what MAC cannot see).  Writes to EC_comments/pipeline/results/T15_mirror_test/.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import h_common as H  # noqa: E402

EC, C = H.EC, H.C
T15_DATA = EC.EC_DATA / "T15"
OUT = EC.EC_RESULTS / "T15_mirror_test"
VARIANTS = ("paper", "mirror", "swap_hinge", "swap_ends")
NETS = ("edgegat_N2000", "edgegat_N600", "edgegat_geomorient_N600",
        "ec_varexc_edgegat_N600", "ec_varexc_edgegat_geomorient_N600")
NET_LABEL = {
    "edgegat_N2000": "paper main, coherency, N 2000",
    "edgegat_N600": "paper, coherency, N 600",
    "edgegat_geomorient_N600": "paper, orientation, N 600",
    "ec_varexc_edgegat_N600": "T11 varied exc., coherency, N 600",
    "ec_varexc_edgegat_geomorient_N600": "T11 varied exc., orientation, N 600",
}


class Asymmetric(RuntimeError):
    pass


# ------------------------------------------------------------------ geometry helpers
def mirror_partner(xyz: np.ndarray, i: int, yc: float, tol: float) -> int:
    p = xyz[i]
    d = np.abs(xyz[:, 0] - p[0]) + np.abs(xyz[:, 1] - (2.0 * yc - p[1]))
    if xyz.shape[1] > 2:
        d = d + np.abs(xyz[:, 2] - p[2])
    j = int(np.argmin(d))
    if d[j] > tol:
        raise Asymmetric(f"no mirror partner within {tol:g} for point {p}")
    return j


def graph_mirror_maps(coords: np.ndarray, edge_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Node permutation (i -> mirror i) and directed-edge permutation for a graph; Asymmetric if none."""
    y = coords[:, 1]
    yc = 0.5 * (y.min() + y.max())
    tol = 1e-6 * max(1.0, float(np.abs(coords).max()))
    perm = np.array([mirror_partner(coords, i, yc, tol) for i in range(coords.shape[0])], dtype=np.int64)
    if not np.array_equal(perm[perm], np.arange(perm.size)):
        raise Asymmetric("mirror map is not an involution")
    lookup = {(int(a), int(b)): k for k, (a, b) in enumerate(edge_index.T)}
    try:
        eperm = np.array([lookup[(int(perm[a]), int(perm[b]))] for a, b in edge_index.T], dtype=np.int64)
    except KeyError as exc:
        raise Asymmetric(f"mirror of edge {exc} is not an edge") from exc
    return perm, eperm


def make_patched_bim(orig, which: str | None, calls: list):
    """build_influence_matrix with support records re-assigned to mirror-image supports."""
    from scipy.sparse import coo_matrix
    from grid_modal_Kinematic_MPC import find_nodes_by_coordinate, gdof0

    def bim(model, node_ids, dofs):
        R, imap = orig(model, node_ids, dofs)
        calls.append(which)
        if which is None:
            return R, imap
        nodes = np.asarray(model["nodes"], dtype=float)
        ids = nodes[:, 0].astype(int)
        xyz = nodes[:, 1:4]
        yc = 0.5 * (xyz[:, 1].min() + xyz[:, 1].max())
        tol = 1e-6 * max(1.0, float(np.abs(xyz).max()))
        hinge = set(int(v) for v in find_nodes_by_coordinate(model, x=0.0, tol=1e-6))
        supports = set(int(v) for v in np.asarray(node_ids).reshape(-1))
        rows, cols, vals, new_map = [], [], [], []
        for k, (nid, dof) in enumerate(np.asarray(imap, dtype=int)):
            nid, dof = int(nid), int(dof)
            is_hinge = nid in hinge
            swap = which == "all" or (which == "hinge" and is_hinge) or (which == "ends" and not is_hinge)
            if swap:
                m = int(ids[mirror_partner(xyz, int(np.where(ids == nid)[0][0]), yc, tol)])
                if m not in supports:
                    raise Asymmetric(f"mirror of support {nid} is node {m}, not a support")
                s = -1.0 if dof == 2 else 1.0
            else:
                m, s = nid, 1.0
            rows.append(gdof0(m, dof))
            cols.append(k)
            vals.append(s)
            new_map.append([m, dof])
        R2 = coo_matrix((vals, (rows, cols)), shape=R.shape).tocsc()
        return R2, np.asarray(new_map, dtype=int)

    return bim


WHICH = {"paper": None, "mirror": "all", "swap_hinge": "hinge", "swap_ends": "ends"}


def _load(p: Path) -> dict:
    with np.load(p, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


# ------------------------------------------------------------------ simulation (worker)
def sim_job(job: tuple) -> dict:
    name, variant = job
    import simulate_uniform_inertial_u3_white_noise as SIM

    src = C.POP_BROADBAND / name
    dst = T15_DATA / variant / "samples" / name
    chk = dst.with_name(Path(name).stem + ".check.json")
    if dst.exists() and chk.exists():
        return json.loads(chk.read_text(encoding="utf-8"))
    t0 = time.time()
    st = _load(src)
    row = {"sample": name, "variant": variant, "Ng": int(st["Ng"])}
    calls: list = []
    orig = SIM.build_influence_matrix
    SIM.build_influence_matrix = make_patched_bim(orig, WHICH[variant], calls)
    try:
        res = H.resimulate_payload(src, H.PAPER_EXC_SEED)
    except Asymmetric as exc:
        row.update({"status": "asymmetric_supports", "detail": str(exc)})
        dst.parent.mkdir(parents=True, exist_ok=True)
        chk.write_text(json.dumps(row), encoding="utf-8")
        return row
    finally:
        SIM.build_influence_matrix = orig
    if calls != [WHICH[variant]]:
        raise RuntimeError(f"{name}/{variant}: patched influence matrix called {calls}")
    payload = dict(st)
    for k in H.EXC_KEYS:
        new = np.asarray(res["features"][k]).astype(st[k].dtype, copy=False)
        if new.shape != st[k].shape:
            raise RuntimeError(f"{name}: {k} shape {new.shape} != stored {st[k].shape}")
        payload[k] = new
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, **payload)
    os.replace(tmp, dst)

    x_new, e_new = payload["x_psd"], payload["edge_attr_cpsd_full"]
    x_old, e_old = st["x_psd"], st["edge_attr_cpsd_full"]
    row.update({"status": "ok", "graph_ids_equal": bool(np.array_equal(res["graph_node_ids"], st["graph_node_ids"])),
                "x_maxdiff_vs_stored": float(np.abs(x_new - x_old).max()),
                "e_maxdiff_vs_stored": float(np.abs(e_new - e_old).max())})
    try:
        perm, eperm = graph_mirror_maps(np.asarray(st["coords_xy_mm"], float), np.asarray(st["edge_index"]))
        row["deck_symmetric"] = True
        row["x_maxdiff_vs_stored_mirrored"] = float(np.abs(x_new - x_old[perm]).max())
        row["e_maxdiff_vs_stored_mirrored"] = float(np.abs(e_new - e_old[eperm]).max())
        y = np.asarray(st["y_mode_shape_signed"], float)
        row.update({f"target_mirror_parity_{m}": float(np.sum(y[:, j] * y[perm, j]) / np.sum(y[:, j] ** 2))
                    for j, m in enumerate(C.MODE_NAMES)})
    except Asymmetric as exc:
        row["deck_symmetric"] = False
        row["detail"] = str(exc)
    row["seconds"] = round(time.time() - t0, 2)
    chk.write_text(json.dumps(row), encoding="utf-8")
    return row


# ------------------------------------------------------------------ evaluation
def evaluate(names: list[str], rows_sim: list[dict]) -> list[dict]:
    import torch
    import EdgeGATConv_Modal_GNN as eg
    import GraphSAGE_Modal_GNN as gs
    import c_rescore as R

    R.install_modes()
    torch.set_num_threads(2)
    ok = {(r["sample"], r["variant"]) for r in rows_sim if r.get("status") == "ok"}
    out = []
    for net in NETS:
        key, _, rd = R.resolve_run(net)
        info = json.loads((rd / "run_summary.json").read_text(encoding="utf-8"))
        model, ckpt = eg.load_trained_model_checkpoint(rd / "checkpoints" / "best_model.pt", device=torch.device("cpu"))
        model.eval()
        stats = gs.FrequencyNormalizationStats(freq_mean_hz=np.asarray(ckpt["freq_mean_hz"], np.float32),
                                               freq_std_hz=np.asarray(ckpt["freq_std_hz"], np.float32))
        for variant in ("stored",) + VARIANTS:
            if variant == "stored":
                files = [C.POP_BROADBAND / n for n in names]
                use = names
            else:
                use = [n for n in names if (n, variant) in ok]
                files = [T15_DATA / variant / "samples" / n for n in use]
            ds = eg.Phase2AEdgeDataset(files, edge_feature_mode=info["edge_feature_mode"],
                                       shape_target_mode="signed", expected_n_modes=4)
            with torch.no_grad():
                for i, n in enumerate(use):
                    g, ft = ds[i]
                    o = model(g, g.ndata["x"], g.edata["edge_x"])
                    p = o.pred_mode_shape_abs.numpy()
                    y = g.ndata["y"].numpy()
                    fp = gs.denormalize_frequency_predictions(o.pred_freq, stats).numpy().reshape(-1)
                    r = {"run": net, "variant": variant, "sample": n, "Ng": int(n.split("_Ng")[1][0]),
                         "freq_mae_hz": float(np.abs(fp - ft.numpy().reshape(-1)).mean())}
                    for j, m in enumerate(C.MODE_NAMES):
                        c = float(np.dot(p[:, j], y[:, j]) / (np.linalg.norm(p[:, j]) * np.linalg.norm(y[:, j]) + 1e-12))
                        r[f"cos_{m}"] = c
                        r[f"mac_{m}"] = c * c
                        r[f"pnorm_{m}"] = float(np.linalg.norm(p[:, j]) / np.sqrt(p.shape[0]))
                    out.append(r)
        print(f"[eval] {net} done", flush=True)
    return out


def summarise(rows: list[dict], rows_sim: list[dict]) -> tuple[list[dict], list[dict]]:
    """Per (run, variant, mode) means, and - by target parity, not mode name (the Ng = 2 balcony
    decks label a torsion mode 'B2', see project memory) - how often the sign of the prediction
    against the target flipped relative to the paper assignment."""
    import pandas as pd
    d = pd.DataFrame(rows)
    par = {r["sample"]: r for r in rows_sim if r["variant"] == "paper" and r.get("deck_symmetric")}
    res = []
    for (run, variant), g in d.groupby(["run", "variant"], sort=False):
        r = {"run": run, "label": NET_LABEL[run], "variant": variant, "n": len(g)}
        for m in C.MODE_NAMES:
            r[f"mac_{m}"] = float(g[f"mac_{m}"].mean())
            r[f"cos_{m}"] = float(g[f"cos_{m}"].mean())
            r[f"frac_cos_pos_{m}"] = float((g[f"cos_{m}"] > 0).mean())
        r["mac_mean"] = float(np.mean([r[f"mac_{m}"] for m in C.MODE_NAMES]))
        r["freq_mae_hz"] = float(g["freq_mae_hz"].mean())
        res.append(r)
    flips = []
    base = d[d.variant == "paper"].set_index(["run", "sample"])
    for (run, variant), g in d[d.variant.isin(VARIANTS[1:])].groupby(["run", "variant"], sort=False):
        acc = {"anti": [], "sym": []}
        for _, row in g.iterrows():
            if row["sample"] not in par or (run, row["sample"]) not in base.index:
                continue
            b = base.loc[(run, row["sample"])]
            for m in C.MODE_NAMES:
                p = par[row["sample"]][f"target_mirror_parity_{m}"]
                if abs(abs(p) - 1.0) > 0.05 or abs(b[f"cos_{m}"]) < 0.3:
                    continue            # target not (anti)symmetric, or paper prediction too weak to have a sign
                acc["anti" if p < 0 else "sym"].append((np.sign(row[f"cos_{m}"]) != np.sign(b[f"cos_{m}"]),
                                                        row[f"cos_{m}"] * np.sign(b[f"cos_{m}"])))
        for cls, v in acc.items():
            if v:
                a = np.array(v, dtype=float)
                flips.append({"run": run, "label": NET_LABEL[run], "variant": variant,
                              "target_class": "antisymmetric (torsion-like)" if cls == "anti" else "symmetric (bending-like)",
                              "n_mode_cells": len(v), "frac_sign_flipped": float(a[:, 0].mean()),
                              "mean_cos_relative_to_paper_sign": float(a[:, 1].mean())})
    return res, flips


def figure(summ: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(NETS), figsize=(3.3 * len(NETS), 3.4), sharey=True)
    cols = {"stored": "#9aa0a6", "paper": "#4c72b0", "mirror": "#c44e52", "swap_hinge": "#dd8452", "swap_ends": "#55a868"}
    vs = ("paper", "mirror", "swap_hinge", "swap_ends")
    for ax, net in zip(axes, NETS):
        rs = {r["variant"]: r for r in summ if r["run"] == net}
        w = 0.2
        for k, v in enumerate(vs):
            if v not in rs:
                continue
            ax.bar(np.arange(4) + (k - 1.5) * w, [rs[v][f"cos_{m}"] for m in C.MODE_NAMES], w, color=cols[v], label=v)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(4), C.MODE_NAMES)
        ax.set_title(NET_LABEL[net], fontsize=8)
        ax.set_ylim(-1, 1)
    axes[0].set_ylabel("mean signed cosine (prediction, target)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Paper record, support records re-assigned: does the torsion sign follow the record?", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mirror_signed_cosine.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--per-ng", type=int, default=25)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    test = H.read_list(C.SPLITS / "test.txt")
    names = []
    for ng in C.NG_VALUES:
        names += [n for n in test if f"_Ng{ng}_" in n][: a.per_ng]
    jobs = [(n, v) for v in VARIANTS for n in names]
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        rows_sim = []
        for k, r in enumerate(pool.imap_unordered(sim_job, jobs), 1):
            rows_sim.append(r)
            if k % 40 == 0:
                print(f"[sim] {k}/{len(jobs)}  {time.time() - t0:.0f} s", flush=True)
    H.write_csv(OUT / "sim_checks.csv", sorted(rows_sim, key=lambda r: (r["variant"], r["sample"])))

    # gates
    import pandas as pd
    s = pd.DataFrame(rows_sim)
    okp = s[(s.variant == "paper") & (s.status == "ok")]
    okm = s[(s.variant == "mirror") & (s.status == "ok") & (s.deck_symmetric == True)]  # noqa: E712
    gates = {
        "n_structures": len(names),
        "paper_reproduces_stored_x_max": float(okp.x_maxdiff_vs_stored.max()),
        "paper_reproduces_stored_e_max": float(okp.e_maxdiff_vs_stored.max()),
        "n_symmetric_decks": int((s[s.variant == "paper"].deck_symmetric == True).sum()),  # noqa: E712
        "n_asymmetric_supports": {v: int((s[s.variant == v].status != "ok").sum()) for v in VARIANTS},
        "mirror_equals_permuted_stored_x_max": float(okm.x_maxdiff_vs_stored_mirrored.max()) if len(okm) else None,
        "mirror_equals_permuted_stored_e_max": float(okm.e_maxdiff_vs_stored_mirrored.max()) if len(okm) else None,
        "target_parity_by_mode_median": {m: float(okp[f"target_mirror_parity_{m}"].median()) for m in C.MODE_NAMES
                                         if f"target_mirror_parity_{m}" in okp},
    }
    tol = 1e-3
    gates["gate_paper_reproduces"] = gates["paper_reproduces_stored_x_max"] < tol and gates["paper_reproduces_stored_e_max"] < tol
    gates["gate_mirror_is_permutation"] = (gates["mirror_equals_permuted_stored_x_max"] is not None
                                           and gates["mirror_equals_permuted_stored_x_max"] < tol
                                           and gates["mirror_equals_permuted_stored_e_max"] < tol)
    H.write_json(OUT / "gates.json", gates)
    print(json.dumps(gates, indent=1), flush=True)

    rows = evaluate(names, rows_sim)
    H.write_csv(OUT / "per_sample.csv", rows)
    summ, flips = summarise(rows, rows_sim)
    H.write_csv(OUT / "summary.csv", summ)
    H.write_csv(OUT / "sign_flips.csv", flips)
    for r in flips:
        print(f"{r['run']:<36} {r['variant']:<11} {r['target_class']:<30} n {r['n_mode_cells']:>3}  "
              f"flipped {r['frac_sign_flipped']:.2f}  cos rel. {r['mean_cos_relative_to_paper_sign']:+.3f}", flush=True)
    # stored vs re-simulated paper predictions must agree (same inputs to ~1e-6)
    d = pd.DataFrame(rows)
    worst = 0.0
    for net in NETS:
        a = d[(d.run == net) & (d.variant == "stored")].set_index("sample")
        b = d[(d.run == net) & (d.variant == "paper")].set_index("sample")
        common = a.index.intersection(b.index)
        worst = max(worst, float(np.abs(a.loc[common, [f"mac_{m}" for m in C.MODE_NAMES]].values
                                        - b.loc[common, [f"mac_{m}" for m in C.MODE_NAMES]].values).max()))
    gates["stored_vs_resimulated_paper_mac_maxdiff"] = worst
    H.write_json(OUT / "gates.json", gates)
    for r in summ:
        print(f"{r['run']:<36} {r['variant']:<11} MAC {r['mac_mean']:.3f}  cos B1 {r['cos_B1']:+.3f} T1 {r['cos_T1']:+.3f} "
              f"B2 {r['cos_B2']:+.3f} T2 {r['cos_T2']:+.3f}", flush=True)
    figure(summ)
    print(f"[done] {time.time() - t0:.0f} s; gates paper {gates['gate_paper_reproduces']} mirror {gates['gate_mirror_is_permutation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
