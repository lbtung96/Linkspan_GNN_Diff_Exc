"""T14 (part e) - which frequency bins of the edge coherency does the paper network depend on?

    python tools/run_stage.py pipeline/m_occlusion.py

The 500 paper test structures exist under the paper record (data/pop_broadband) and under a record
of their own (T11 paper_varexc, same graphs and targets).  T5 showed that swapping the EDGE inputs
alone to the new record collapses the main network (0.934 -> 0.410), the node inputs barely matter.
Here the edge coherency is swapped bin by bin: one class of bins from the paper record, the
complement from the new record, node PSDs always from the paper record.  Bin classes per structure:

  modal      within +-3 bins (+-0.29 Hz) of one of the four target-mode frequencies
  peaks      the 15 % of in-band bins with the highest mean node PSD (all resonances, target or not)
  outband    outside the 1.5-20 Hz bandpass (0-1.5 Hz and 20-39 Hz), where the filtered response is
             tiny but coherency, being normalised, is not

If the network read the physics, the modal/peak bins would carry its performance; if it read the
record, the bins between and outside the resonances would.  Same library dataset and best-validation
checkpoint as c_excitation_diag.  Writes occlusion.csv to results/T14_input_anatomy/.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import h_common as H  # noqa: E402

EC, C = H.EC, H.C
OUT = EC.EC_RESULTS / "T14_input_anatomy"
NETS = ("edgegat_N2000", "edgegat_N600")
VARIANTS = (
    ("paper", "all paper"),
    ("new", "all new (nodes and edges)"),
    ("edges_new", "edges all new"),
    ("edges_modal_paper", "edges: modal bins paper, rest new"),
    ("edges_modal_new", "edges: modal bins new, rest paper"),
    ("edges_peaks_paper", "edges: peak bins paper, rest new"),
    ("edges_peaks_new", "edges: peak bins new, rest paper"),
    ("edges_outband_new", "edges: out-of-band bins new, in-band paper"),
    ("edges_inband_new", "edges: in-band bins new, out-of-band paper"),
)
RES_W = 3
PEAK_FRAC = 0.15


def main() -> int:
    import torch
    import EdgeGATConv_Modal_GNN as eg
    import GraphSAGE_Modal_GNN as gs
    import c_rescore as R

    R.install_modes()
    torch.set_num_threads(2)
    names = H.read_list(C.SPLITS / "test.txt")
    fb_files = [C.POP_BROADBAND / n for n in names]
    fa_files = [H.TEST_DIR / n for n in names]
    df = 200.0 / 2048
    fbins = np.arange(400) * df
    inband = (fbins >= 1.5) & (fbins <= 20.0)
    rows, per = [], []
    for net in NETS:
        key, _, rd = R.resolve_run(net)
        info = json.loads((rd / "run_summary.json").read_text(encoding="utf-8"))
        model, ckpt = eg.load_trained_model_checkpoint(rd / "checkpoints" / "best_model.pt", device=torch.device("cpu"))
        model.eval()
        stats = gs.FrequencyNormalizationStats(freq_mean_hz=np.asarray(ckpt["freq_mean_hz"], np.float32),
                                               freq_std_hz=np.asarray(ckpt["freq_std_hz"], np.float32))
        mk = lambda fs: eg.Phase2AEdgeDataset(fs, edge_feature_mode=info["edge_feature_mode"],  # noqa: E731
                                              shape_target_mode="signed", expected_n_modes=4)
        da, db = mk(fa_files), mk(fb_files)
        acc = {v: {"mac": [], "ferr": []} for v, _ in VARIANTS}
        frac = {"modal": [], "peaks": []}
        with torch.no_grad():
            for i in range(len(names)):
                ga, ft = da[i]
                gb, ftb = db[i]
                assert torch.equal(ga.edges()[0], gb.edges()[0]) and torch.equal(ga.edges()[1], gb.edges()[1])
                assert torch.allclose(ga.ndata["y"], gb.ndata["y"], rtol=0, atol=1e-5)
                if i < 3:     # the tensors are the stored arrays, bins in order (Re, Im interleaved)
                    with np.load(fb_files[i]) as z:
                        assert np.allclose(gb.ndata["x"].numpy(), z["x_psd"], atol=1e-6)
                        assert np.allclose(gb.edata["edge_x"].numpy(), z["edge_attr_cpsd_full"], atol=1e-6)
                xa, xb, ea, eb = ga.ndata["x"], gb.ndata["x"], ga.edata["edge_x"], gb.edata["edge_x"]
                tb = np.rint(ft.numpy().reshape(-1) / df).astype(int)
                modal = inband & (np.min(np.abs(np.arange(400)[:, None] - tb[None, :]), axis=1) <= RES_W)
                lvl = xb.numpy().mean(0)
                thr = np.quantile(lvl[inband], 1 - PEAK_FRAC)
                peaks = inband & (lvl >= thr)
                frac["modal"].append(modal.mean())
                frac["peaks"].append(peaks.mean())

                def mix(paper_bins: np.ndarray) -> torch.Tensor:
                    cols = np.repeat(paper_bins, 2)          # (Re, Im) per bin
                    e = ea.clone()
                    e[:, torch.from_numpy(cols)] = eb[:, torch.from_numpy(cols)]
                    return e

                inp = {"paper": (xb, eb), "new": (xa, ea), "edges_new": (xb, ea),
                       "edges_modal_paper": (xb, mix(modal)), "edges_modal_new": (xb, mix(~modal)),
                       "edges_peaks_paper": (xb, mix(peaks)), "edges_peaks_new": (xb, mix(~peaks)),
                       "edges_outband_new": (xb, mix(inband)), "edges_inband_new": (xb, mix(~inband))}
                for v, (x, e) in inp.items():
                    o = model(gb, x, e)
                    p, y = o.pred_mode_shape_abs.numpy(), gb.ndata["y"].numpy()
                    cos = [float(np.dot(p[:, j], y[:, j]) / (np.linalg.norm(p[:, j]) * np.linalg.norm(y[:, j]) + 1e-12))
                           for j in range(4)]
                    fp = gs.denormalize_frequency_predictions(o.pred_freq, stats).numpy().reshape(-1)
                    acc[v]["mac"].append([c * c for c in cos])
                    acc[v]["ferr"].append(float(np.abs(fp - ftb.numpy().reshape(-1)).mean()))
        for v, lab in VARIANTS:
            mm = np.array(acc[v]["mac"])
            rows.append({"run": net, "variant": v, "what": lab,
                         **{f"mac_{m}": float(mm[:, j].mean()) for j, m in enumerate(C.MODE_NAMES)},
                         "mac_mean": float(mm.mean()), "freq_mae_hz": float(np.mean(acc[v]["ferr"])),
                         "frac_bins_modal": float(np.mean(frac["modal"])), "frac_bins_peaks": float(np.mean(frac["peaks"])),
                         "frac_bins_outband": float(1 - inband.mean())})
            print(f"{net:<15} {lab:<46} MAC {rows[-1]['mac_mean']:.3f}  T1 {rows[-1]['mac_T1']:.3f}  "
                  f"f MAE {rows[-1]['freq_mae_hz']:.3f}", flush=True)
    H.write_csv(OUT / "e_occlusion.csv", rows)
    # gate: unmixed variants equal the T11 evaluation of the same networks and sets
    ref = {(r["test_set"], r["run"]): r for r in H.read_csv(H.T11_RESULTS / "eval_summary.csv")}
    worst = 0.0
    for r in rows:
        ts = {"paper": "paper", "new": "paper_varexc"}.get(r["variant"])
        if ts and (ts, r["run"]) in ref:
            worst = max(worst, abs(r["mac_mean"] - float(ref[(ts, r["run"])]["mac_mean"])))
    print(f"[check] unmixed variants vs T11 eval_summary: max |diff| {worst:.1e} ({'PASS' if worst < 1e-6 else 'FAIL'})")
    H.write_json(OUT / "e_occlusion_gate.json", {"max_abs_diff_vs_T11": worst, "pass": worst < 1e-6})
    return 0 if worst < 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main())
