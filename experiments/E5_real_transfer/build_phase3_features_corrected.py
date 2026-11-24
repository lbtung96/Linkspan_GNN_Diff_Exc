"""Stage 1 - corrected Phase-3 feature builders.

Fixes the three convention mismatches identified in Audit/AUDIT_REPORT.md:

  1. node PSD is row-normalised by max-abs, using the production
     ``normalize_rows_by_max_abs`` imported from the training pipeline;
  2. edge features are complex coherency, using the production
     ``_compute_edge_complex_coherency_spectra`` imported from
     ``generate_population_dataset_modes4`` (no reimplementation);
  3. Welch estimator settings match the synthetic pipeline exactly:
     Hamming(2048), nperseg=2048, noverlap=1024, nfft=2048, detrend=False,
     one-sided, scaling='density'.

Deliberate deviation from the task brief, justified in stage0_discrepancies.md
§0.3: the real recordings are sampled at **128 Hz**, not the 200 Hz of the
synthetic pipeline.  Taking "the first 400 bins directly" would place bin k at
0.0625*k Hz where the network expects 0.09766*k Hz - a 1.5625x compression of the
whole frequency axis.  The spectra are therefore still **interpolated onto the
400-bin training frequency grid**, exactly as production does.  Everything else
about the estimator is matched.

Node-feature ordering: the training pipeline contains no propagation stage at all
(the synthetic response is full-field), so there is no ordering to copy.  Both
orderings are therefore produced:
    order A (primary) : PSD -> propagate -> interpolate -> row-normalise
    order B           : PSD -> row-normalise -> propagate -> interpolate

Usage
-----
  python Audit/Rerun/build_phase3_features_corrected.py            # all 8 structures
  python Audit/Rerun/build_phase3_features_corrected.py --structures Linge
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import get_window, welch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

# --- production imports (nothing below is reimplemented) --------------------
from generate_population_dataset_modes4 import (  # noqa: E402
    PHASE1_FREQ_BINS,
    _compute_edge_complex_coherency_spectra,
)
from simulate_uniform_inertial_u3_white_noise import (  # noqa: E402
    WELCH_NFFT,
    WELCH_NOVERLAP,
    WELCH_NPERSEG,
    normalize_rows_by_max_abs,
)
from phase3_utils import (  # noqa: E402
    BANDPASS_F_HIGH,
    BANDPASS_F_LOW,
    bandpass_filter,
    feature_propagation,
    load_graph_package,
)
from structures import LOOCV  # single source of truth

STRUCTURES = list(LOOCV)
N_PROP_ITER = 50                      # same constant as the production builders
FIELD_ROOT = REPO / "field_modal_filtering_outputs"
GRAPH_ROOT = REPO / "phase3_real_graphs"
OUT_ROOT = Path(__file__).resolve().parent / "phase3_corrected"
# samples_phase2/ is empty in the current working tree, so the training grid is
# read from an equally valid source; both datasets carry the identical grid
# (verified: 400 bins, df = 0.09765625 Hz, span 0 - 38.96484 Hz).
TRAIN_GRID_SAMPLE = (REPO / "population_dataset_modes4" / "samples_phase2_bandpass"
                     / "sample_0000_Ng2_phase1_gnn.npz")


def training_freq_grid() -> np.ndarray:
    g = np.load(TRAIN_GRID_SAMPLE)["freq_bins_hz"].astype(np.float64)
    assert g.shape == (PHASE1_FREQ_BINS,), g.shape
    return g


# ---------------------------------------------------------------------------
# Welch, matched to the synthetic pipeline
# ---------------------------------------------------------------------------
def welch_psd_rows(sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise Welch PSD with the synthetic pipeline's estimator settings.

    Mirrors simulate_uniform_inertial_u3_white_noise.welch_all_nodes:
    Hamming(nperseg), noverlap, nfft, detrend=False, one-sided, density.
    """
    nperseg = int(min(WELCH_NPERSEG, sig.shape[1]))
    noverlap = int(min(WELCH_NOVERLAP, nperseg // 2))
    nfft = int(WELCH_NFFT)
    window = get_window("hamming", nperseg, fftbins=True)
    f, p0 = welch(sig[0], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
                  nfft=nfft, detrend=False, return_onesided=True, scaling="density")
    out = np.zeros((sig.shape[0], p0.size), dtype=np.float64)
    out[0] = p0
    for i in range(1, sig.shape[0]):
        _, out[i] = welch(sig[i], fs=fs, window=window, nperseg=nperseg,
                          noverlap=noverlap, nfft=nfft, detrend=False,
                          return_onesided=True, scaling="density")
    return f, out


def to_training_grid(f_native: np.ndarray, rows: np.ndarray,
                     grid: np.ndarray) -> np.ndarray:
    """Map a native-grid spectrum onto the 400-bin training grid.

    Direct truncation when the native grid already equals the training grid
    (synthetic path, fs = 200 Hz); linear interpolation otherwise (real path,
    fs = 128 Hz).  Real and imaginary parts are handled by the caller.
    """
    n = grid.size
    if f_native.size >= n and np.allclose(f_native[:n], grid, rtol=0, atol=1e-9):
        return np.ascontiguousarray(rows[:, :n])
    return np.stack([np.interp(grid, f_native, r) for r in rows], axis=0)


def propagate(x: np.ndarray, sensor_idx: np.ndarray, edge_index: np.ndarray,
              n_nodes: int) -> np.ndarray:
    """Production clamped harmonic extension (imported operator, 50 iterations)."""
    mask = torch.zeros(n_nodes, dtype=torch.bool)
    mask[sensor_idx] = True
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    return feature_propagation(xt, mask, torch.from_numpy(edge_index),
                               n_nodes, n_iter=N_PROP_ITER).numpy()


# ---------------------------------------------------------------------------
# Corrected feature builders
# ---------------------------------------------------------------------------
def corrected_node_features(
    sensor_sig: np.ndarray,
    fs: float,
    grid: np.ndarray,
    *,
    n_nodes: int,
    sensor_idx: np.ndarray,
    edge_index: np.ndarray,
    order: str = "A",
    norm_range: str = "grid",
) -> np.ndarray:
    """(n_nodes, 400) float32 node features, row-normalised by max-abs.

    order 'A': PSD -> propagate -> interpolate -> row-normalise
    order 'B': PSD -> row-normalise -> propagate -> interpolate
    norm_range 'grid': normalise over the 400 retained bins
    norm_range 'full': normalise over the full native one-sided spectrum
                       (this is what the synthetic pipeline does before it
                       truncates); only meaningful for order 'A'.
    """
    f_native, psd = welch_psd_rows(sensor_sig, fs)          # (n_sensors, n_native)

    if order == "B":
        psd = normalize_rows_by_max_abs(psd)[0]

    x = np.zeros((n_nodes, psd.shape[1]), dtype=np.float32)
    x[sensor_idx] = psd.astype(np.float32)
    prop = propagate(x, sensor_idx, edge_index, n_nodes).astype(np.float64)

    if order == "A" and norm_range == "full":
        prop = normalize_rows_by_max_abs(prop)[0]
        return to_training_grid(f_native, prop, grid).astype(np.float32)

    out = to_training_grid(f_native, prop, grid)
    if order == "A":
        out = normalize_rows_by_max_abs(out)[0]
    return out.astype(np.float32)


def corrected_edge_features(
    node_sig: np.ndarray,
    fs: float,
    grid: np.ndarray,
    edge_index: np.ndarray,
) -> np.ndarray:
    """(E_dir, 800) float32 complex-coherency edge features, interleaved [Re, Im].

    The coherency spectra come straight from the production function
    ``_compute_edge_complex_coherency_spectra``; only the mapping onto the
    training frequency grid and the interleaving are done here, matching
    ``build_cpsd_full_edge_payload``.
    """
    nperseg = int(min(WELCH_NPERSEG, node_sig.shape[1]))
    noverlap = int(min(WELCH_NOVERLAP, nperseg // 2))
    coh = _compute_edge_complex_coherency_spectra(
        edge_index=edge_index, a3_abs=node_sig, fs=fs,
        nperseg=nperseg, noverlap=noverlap, nfft=int(WELCH_NFFT),
    )                                                        # (E, nfft//2+1) complex64
    f_native = np.fft.rfftfreq(int(WELCH_NFFT), d=1.0 / fs)
    re = to_training_grid(f_native, np.real(coh).astype(np.float64), grid)
    im = to_training_grid(f_native, np.imag(coh).astype(np.float64), grid)
    out = np.zeros((coh.shape[0], 2 * grid.size), dtype=np.float32)
    out[:, 0::2] = re.astype(np.float32)
    out[:, 1::2] = im.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
def build_structure(structure: str, grid: np.ndarray) -> dict:
    """Corrected real-data features for one structure."""
    field = np.load(FIELD_ROOT / structure / "data" /
                    "field_modal_filtering_outputs.npz", allow_pickle=False)
    u3 = field["raw_u3_timeseries_clean"].astype(np.float64).T     # (n_sensors, n_samples)
    fs = float(2.0 * field["freq_hz"].astype(np.float64)[-1])
    u3_bp = bandpass_filter(u3, fs, f_low=BANDPASS_F_LOW, f_high=BANDPASS_F_HIGH)

    pkg = load_graph_package(structure, graph_root=str(GRAPH_ROOT))
    sensor_idx = pkg["sensor_node_indices"].astype(int)
    edge_index = pkg["edge_index"].astype(np.int64)
    n_nodes = int(pkg["node_coords_xy_mm"].shape[0])
    assert sensor_idx.size == u3.shape[0], (
        f"{structure}: {sensor_idx.size} sensor nodes vs {u3.shape[0]} channels")

    # node features, both orderings
    x_a = corrected_node_features(u3_bp, fs, grid, n_nodes=n_nodes,
                                  sensor_idx=sensor_idx, edge_index=edge_index,
                                  order="A", norm_range="grid")
    x_b = corrected_node_features(u3_bp, fs, grid, n_nodes=n_nodes,
                                  sensor_idx=sensor_idx, edge_index=edge_index,
                                  order="B")
    # third variant: order A but normalised over the full native spectrum, which
    # is literally what the synthetic pipeline does before truncating.  On the
    # real path the subsequent resample onto the coarser training grid means the
    # retained row max lands slightly below 1.0.
    x_a_full = corrected_node_features(u3_bp, fs, grid, n_nodes=n_nodes,
                                       sensor_idx=sensor_idx, edge_index=edge_index,
                                       order="A", norm_range="full")

    # edge features: propagate the bandpassed time series, then coherency
    ts = np.zeros((n_nodes, u3_bp.shape[1]), dtype=np.float32)
    ts[sensor_idx] = u3_bp.astype(np.float32)
    ts_prop = propagate(ts, sensor_idx, edge_index, n_nodes).astype(np.float64)
    ea = corrected_edge_features(ts_prop, fs, grid, edge_index)

    # labels and geometry are carried over unchanged from the production dataset
    ds = np.load(GRAPH_ROOT / structure / "phase3_dataset.npz", allow_pickle=False)
    out = {k: ds[k] for k in ds.files}
    out["x_psd"] = x_a
    out["x_psd_order_B"] = x_b
    out["x_psd_norm_full"] = x_a_full
    out["edge_attr_cpsd_full"] = ea
    out["freq_hz_grid"] = grid.astype(np.float32)
    out["fs_hz"] = np.float64(fs)
    out["n_samples"] = np.int64(u3.shape[1])
    assert out["edge_attr_cpsd_full"].shape[0] == edge_index.shape[1]
    assert np.array_equal(out["edge_index"], edge_index)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures", nargs="+", default=STRUCTURES, choices=STRUCTURES)
    args = ap.parse_args(argv)

    grid = training_freq_grid()
    print(f"training grid: {grid.size} bins, df = {grid[1]-grid[0]:.8f} Hz, "
          f"span 0 - {grid[-1]:.5f} Hz")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for s in args.structures:
        res = build_structure(s, grid)
        d = OUT_ROOT / s
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "phase3_dataset_corrected.npz", **res)
        x, e = res["x_psd"], res["edge_attr_cpsd_full"]
        rm = x.max(axis=1)
        print(f"  {s:<12} fs={float(res['fs_hz']):.1f}Hz n={int(res['n_samples'])} "
              f"x_psd{x.shape} row-max [{rm.min():.6f}, {rm.max():.6f}]  "
              f"edge{e.shape} |v|max={np.abs(e).max():.4f}")
    print(f"\nWrote -> {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
