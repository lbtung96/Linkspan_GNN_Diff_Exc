"""Standalone Phase 3 field-data preprocessing for SSI-guided modal filtering.

This script is intentionally decoupled from the existing PBSHM / GNN training and
inference pipeline. Its purpose is to preprocess raw field acceleration data,
retain only manually selected SSI-identified target modes (`B1`, `T1`, `B2`,
`T2`), and save filtered spectral products for later GNN fine-tuning
experiments.

Primary scope
-------------
- Load paired raw field `.mat` files and SSI-picked `.mat` files.
- Extract only `U3` / vertical acceleration channels from raw field data.
- Compute raw diagnostics: time series, PSD, selected CPSD, coherence, and FDD
  singular values.
- Use manually assigned SSI-picked modes as anchors for a Version 1 pragmatic
  SSI-guided band + FDD filtering workflow.
- Save filtered PSD / CPSD / spectral-density outputs in a clean run folder.

Out of scope
------------
- Graph construction
- FE-to-field projection
- Changes to any current training or inference scripts

Interactive workflow
--------------------
This script is designed first for editor-driven use. Set parameters in the
`USER SETTINGS` block near the top of the file, press Run in VS Code, and the
script will:

1. Open a popup file selector for one or more `.mat` files.
2. Pair raw and SSI files using filename logic around `_Data_picked.mat`.
3. Show a popup summary of matched pairs and unmatched files.
4. Prompt for SSI-picked mode indices corresponding to `B1`, `T1`, `B2`, `T2`
   for each structure, with JSON-backed reuse of prior assignments.
5. Process all selected structures automatically and save outputs.

CLI workflow
------------
CLI is supported for reproducibility and batch runs, but it only takes over when
explicit command-line arguments are passed. Running the script normally with no
arguments uses the settings defined in this file.

README-style summary
--------------------
Inputs:
- Raw field acceleration `.mat` files
- Paired SSI-picked mode `.mat` files ending in `_Data_picked.mat`

Outputs:
- Raw and filtered spectral density products
- Diagnostic plots
- Pairing reports, variable-detection logs, mode-mapping JSON, and metadata

Current limitations:
- Version 1 filtering uses SSI frequencies and mode shapes as anchors together
  with FDD first-singular-vector similarity. It is intentionally pragmatic
  rather than a full parametric operational modal analysis refinement.
- Optional filtered time-series synthesis is only for visualization / sanity
  checking. The primary scientific output is the filtered spectral content.

Usage examples
--------------
Interactive VS Code run:
- Edit the `USER SETTINGS` block below.
- Set `RUN_MODE = "interactive"`.
- Press Run.

CLI batch example:
- `python process_field_modal_filtering.py --mat-files Eidsdal2.mat Eisdal_Data_picked.mat`

Reuse saved JSON assignments:
- Keep `MODE_ASSIGNMENT_JSON = None` to use the default JSON under `OUTPUT_ROOT`.

Run with sensor reorder:
- Set `RAW_CHANNEL_REORDER = [0, 1, 2, 3, ...]` in the settings block
  or pass `--sensor-order 0,1,2,3,...` in CLI mode.

Run with selected CPSD pairs:
- Set `SELECTED_CPSD_PAIRS = [(0, 1), (0, 7), (3, 10)]`

Run with filtered time-series visualization:
- Set `MAKE_FILTERED_TIMESERIES = True`
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import io as sio
from scipy.signal import coherence, csd

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency
    h5py = None

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog
except ImportError:  # pragma: no cover - tkinter may be unavailable in some envs
    tk = None
    filedialog = None
    messagebox = None
    simpledialog = None


# =========================
# USER SETTINGS
# =========================
RUN_MODE = "interactive"  # "interactive" or "cli"
OUTPUT_ROOT = Path("field_modal_filtering_outputs")
MODE_ASSIGNMENT_JSON = None  # None -> OUTPUT_ROOT / "manual_mode_assignments.json"

FS_OVERRIDE = 128.0
RAW_CHANNEL_REORDER = None
U3_START_INDEX = 2
U3_STRIDE = 3
AUTO_FIND_CLEAN_WINDOW = True
MIN_WINDOW_SECONDS = 60.0
USE_MANUAL_WINDOW_OVERRIDES = True
MANUAL_WINDOW_OVERRIDES = {}

BAND_HALF_WIDTH = 0.5
CLOSE_MODE_HALF_WIDTH = 0.25
MODE_SIMILARITY_THRESHOLD = 0.70
MASK_SMOOTHING_HZ = 0.15

WELCH_NPERSEG = 1024
WELCH_NOVERLAP = 512
WELCH_WINDOW = "hann"
WELCH_NFFT = None
DETREND = False

FMIN = None
FMAX = None
FDD_N_SINGULAR_VALUES = 4

MAKE_FILTERED_TIMESERIES = True
SHOW_PLOTS = True
SAVE_NPZ = True
SAVE_MAT = False
SAVE_PNG = True
SAVE_SVG = False

PLOT_MAX_SENSORS_TIMESERIES = 12
PLOT_MAX_PAIRS = 6
SELECTED_CPSD_PAIRS = [(0, 1), (0, 7), (3, 10)]
PLOT_COHERENCE = True


TARGET_MODE_LABELS = ("B1", "T1", "B2", "T2")
SSI_SUFFIX = "_Data_picked.mat"
DEFAULT_FIELD_FS_HZ = 128.0


@dataclass
class Settings:
    run_mode: str = RUN_MODE
    output_root: Path = OUTPUT_ROOT
    mode_assignment_json: Path | None = MODE_ASSIGNMENT_JSON
    fs_override: float | None = FS_OVERRIDE
    fs_override_source: str = "settings"
    raw_channel_reorder: list[int] | None = RAW_CHANNEL_REORDER
    u3_start_index: int = U3_START_INDEX
    u3_stride: int = U3_STRIDE
    auto_find_clean_window: bool = AUTO_FIND_CLEAN_WINDOW
    min_window_seconds: float = MIN_WINDOW_SECONDS
    use_manual_window_overrides: bool = USE_MANUAL_WINDOW_OVERRIDES
    manual_window_overrides: dict[str, tuple[int, int]] = field(
        default_factory=lambda: dict(MANUAL_WINDOW_OVERRIDES)
    )
    band_half_width: float = BAND_HALF_WIDTH
    close_mode_half_width: float = CLOSE_MODE_HALF_WIDTH
    mode_similarity_threshold: float = MODE_SIMILARITY_THRESHOLD
    mask_smoothing_hz: float = MASK_SMOOTHING_HZ
    welch_nperseg: int = WELCH_NPERSEG
    welch_noverlap: int = WELCH_NOVERLAP
    welch_window: str = WELCH_WINDOW
    welch_nfft: int | None = WELCH_NFFT
    detrend: bool | str = DETREND
    fmin: float | None = FMIN
    fmax: float | None = FMAX
    fdd_n_singular_values: int = FDD_N_SINGULAR_VALUES
    make_filtered_timeseries: bool = MAKE_FILTERED_TIMESERIES
    show_plots: bool = SHOW_PLOTS
    save_npz: bool = SAVE_NPZ
    save_mat: bool = SAVE_MAT
    save_png: bool = SAVE_PNG
    save_svg: bool = SAVE_SVG
    plot_max_sensors_timeseries: int = PLOT_MAX_SENSORS_TIMESERIES
    plot_max_pairs: int = PLOT_MAX_PAIRS
    selected_cpsd_pairs: list[tuple[int, int]] = field(
        default_factory=lambda: list(SELECTED_CPSD_PAIRS)
    )


@dataclass
class PairCandidate:
    structure_key: str
    structure_name: str
    raw_file: Path | None
    ssi_file: Path | None
    score: float = 1.0
    notes: list[str] = field(default_factory=list)


@dataclass
class LoadedMat:
    path: Path
    variables: dict[str, Any]
    loader: str
    summary: dict[str, Any]


@dataclass
class SpectralResult:
    freq_hz: np.ndarray
    spectral_matrix: np.ndarray
    psd: np.ndarray
    selected_cpsd: dict[str, np.ndarray]
    selected_coherence: dict[str, np.ndarray]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, complex):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def build_default_settings() -> Settings:
    settings = Settings()
    if settings.mode_assignment_json is None:
        settings.mode_assignment_json = settings.output_root / "manual_mode_assignments.json"
    return settings


def parse_cli_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Phase 3 field-data preprocessing with SSI-guided modal filtering."
    )
    parser.add_argument("--mat-files", nargs="+", help="Explicit .mat files to process")
    parser.add_argument("--output-root", type=Path, help="Output root directory")
    parser.add_argument("--mode-assignment-json", type=Path, help="Manual mode assignment JSON path")
    parser.add_argument("--fs-override", type=float, help="Override sampling frequency")
    parser.add_argument("--band-half-width", type=float, help="Target mode half-width in Hz")
    parser.add_argument(
        "--close-mode-half-width",
        type=float,
        help="Reduced half-width in Hz for closely spaced target modes",
    )
    parser.add_argument("--fmin", type=float, help="Minimum frequency to keep")
    parser.add_argument("--fmax", type=float, help="Maximum frequency to keep")
    parser.add_argument("--welch-nperseg", type=int, help="Welch segment length")
    parser.add_argument("--welch-noverlap", type=int, help="Welch overlap")
    parser.add_argument("--welch-window", type=str, help="Welch window")
    parser.add_argument("--welch-nfft", type=int, help="Welch nfft")
    parser.add_argument(
        "--selected-cpsd-pairs",
        type=str,
        help='Comma-separated pairs like "0-1,0-7,3-10"',
    )
    parser.add_argument(
        "--sensor-order",
        type=str,
        help='Comma-separated raw channel reorder like "0,1,2,3"',
    )
    parser.add_argument("--make-filtered-timeseries", action="store_true", help="Enable time-series synthesis")
    parser.add_argument("--no-show-plots", action="store_true", help="Disable plot display")
    parser.add_argument("--save-mat", action="store_true", help="Also save .mat output")
    parser.add_argument("--no-save-npz", action="store_true", help="Disable .npz output")
    return parser.parse_args(argv)


def cli_has_explicit_arguments(argv: list[str]) -> bool:
    return len(argv) > 0


def apply_cli_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    if args.output_root is not None:
        settings.output_root = args.output_root
    if args.mode_assignment_json is not None:
        settings.mode_assignment_json = args.mode_assignment_json
    if args.fs_override is not None:
        settings.fs_override = args.fs_override
        settings.fs_override_source = "cli"
    if args.band_half_width is not None:
        settings.band_half_width = args.band_half_width
    if args.close_mode_half_width is not None:
        settings.close_mode_half_width = args.close_mode_half_width
    if args.fmin is not None:
        settings.fmin = args.fmin
    if args.fmax is not None:
        settings.fmax = args.fmax
    if args.welch_nperseg is not None:
        settings.welch_nperseg = args.welch_nperseg
    if args.welch_noverlap is not None:
        settings.welch_noverlap = args.welch_noverlap
    if args.welch_window is not None:
        settings.welch_window = args.welch_window
    if args.welch_nfft is not None:
        settings.welch_nfft = args.welch_nfft
    if args.selected_cpsd_pairs:
        settings.selected_cpsd_pairs = parse_pairs_string(args.selected_cpsd_pairs)
    if args.sensor_order:
        settings.raw_channel_reorder = [int(x.strip()) for x in args.sensor_order.split(",") if x.strip()]
    if args.make_filtered_timeseries:
        settings.make_filtered_timeseries = True
    if args.no_show_plots:
        settings.show_plots = False
    if args.save_mat:
        settings.save_mat = True
    if args.no_save_npz:
        settings.save_npz = False
    return settings


def parse_pairs_string(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        left, right = token.split("-")
        pairs.append((int(left), int(right)))
    return pairs


def init_tk_root() -> tk.Tk | None:
    if tk is None:
        return None
    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    return root


def select_mat_files_interactive(root: tk.Tk | None) -> list[Path]:
    if filedialog is None:
        raise RuntimeError("tkinter filedialog is unavailable; interactive mode cannot open file dialogs.")
    paths = filedialog.askopenfilenames(
        title="Select raw field and SSI .mat files",
        filetypes=[("MATLAB files", "*.mat"), ("All files", "*.*")],
    )
    return [Path(p) for p in paths]


def popup_yes_no(root: tk.Tk | None, title: str, message: str) -> bool:
    if messagebox is None:
        print(message)
        return True
    return bool(messagebox.askyesno(title, message, parent=root))


def popup_show_info(root: tk.Tk | None, title: str, message: str) -> None:
    if messagebox is None:
        print(f"[{title}] {message}")
        return
    messagebox.showinfo(title, message, parent=root)


def popup_ask_string(root: tk.Tk | None, title: str, prompt: str, initial: str = "") -> str | None:
    if simpledialog is None:
        return input(f"{title}: {prompt}\n[{initial}] ") or initial
    return simpledialog.askstring(title, prompt, initialvalue=initial, parent=root)


def normalize_structure_key(name: str, *, strip_trailing_digits: bool) -> str:
    stem = Path(name).stem.lower()
    stem = stem.replace("_data_picked", "")
    filtered = "".join(ch for ch in stem if ch.isalnum())
    if strip_trailing_digits:
        filtered = filtered.rstrip("0123456789")
    return filtered


def summarize_pairs(
    pairs: list[PairCandidate],
    unmatched_raw: list[Path],
    unmatched_ssi: list[Path],
) -> str:
    lines = ["Matched pairs:"]
    if not pairs:
        lines.append("  (none)")
    for pair in pairs:
        lines.append(
            f"  {pair.structure_name}: raw={pair.raw_file.name if pair.raw_file else 'None'} | "
            f"ssi={pair.ssi_file.name if pair.ssi_file else 'None'} | score={pair.score:.3f}"
        )
        for note in pair.notes:
            lines.append(f"    note: {note}")
    lines.append("")
    lines.append("Unmatched raw files:")
    lines.extend([f"  {p.name}" for p in unmatched_raw] or ["  (none)"])
    lines.append("")
    lines.append("Unmatched SSI files:")
    lines.extend([f"  {p.name}" for p in unmatched_ssi] or ["  (none)"])
    return "\n".join(lines)


def fuzzy_match_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def choose_ambiguous_candidate(
    root: tk.Tk | None,
    ssi_file: Path,
    candidates: list[Path],
    scores: list[float],
) -> Path | None:
    indexed = "\n".join(
        f"{i}: {path.name} (score={score:.3f})" for i, (path, score) in enumerate(zip(candidates, scores))
    )
    prompt = (
        f"Ambiguous raw-file match for SSI file:\n{ssi_file.name}\n\n"
        f"Choose candidate index:\n{indexed}\n\n"
        "Cancel to skip this structure."
    )
    answer = popup_ask_string(root, "Resolve ambiguous pair", prompt, initial="0")
    if answer is None or not answer.strip():
        return None
    idx = int(answer)
    if idx < 0 or idx >= len(candidates):
        raise ValueError(f"Invalid ambiguous choice index {idx} for {ssi_file.name}")
    return candidates[idx]


def pair_structure_files(mat_files: list[Path], root: tk.Tk | None = None) -> tuple[list[PairCandidate], dict[str, Any]]:
    all_mats = sorted([p for p in mat_files if p.suffix.lower() == ".mat"])
    ssi_files = [p for p in all_mats if p.name.lower().endswith(SSI_SUFFIX.lower())]
    raw_files = [p for p in all_mats if p not in ssi_files]

    raw_by_key_full: dict[str, list[Path]] = {}
    raw_by_key_loose: dict[str, list[Path]] = {}
    for raw in raw_files:
        key_full = normalize_structure_key(raw.name, strip_trailing_digits=False)
        key_loose = normalize_structure_key(raw.name, strip_trailing_digits=True)
        raw_by_key_full.setdefault(key_full, []).append(raw)
        raw_by_key_loose.setdefault(key_loose, []).append(raw)

    matched_pairs: list[PairCandidate] = []
    used_raw: set[Path] = set()
    unmatched_ssi: list[Path] = []
    ambiguity_log: list[dict[str, Any]] = []

    for ssi in ssi_files:
        ssi_key_full = normalize_structure_key(ssi.name, strip_trailing_digits=False)
        ssi_key_loose = normalize_structure_key(ssi.name, strip_trailing_digits=True)
        candidates = [p for p in raw_by_key_full.get(ssi_key_full, []) if p not in used_raw]
        notes: list[str] = []

        if not candidates:
            candidates = [p for p in raw_by_key_loose.get(ssi_key_loose, []) if p not in used_raw]
            if candidates:
                notes.append("Paired using trailing-digit-insensitive normalized key.")

        if not candidates and raw_files:
            available_raw = [p for p in raw_files if p not in used_raw]
            scores = [fuzzy_match_score(ssi_key_loose, normalize_structure_key(p.name, strip_trailing_digits=True)) for p in available_raw]
            ranked = sorted(zip(available_raw, scores), key=lambda item: item[1], reverse=True)
            if ranked and ranked[0][1] >= 0.72:
                top_score = ranked[0][1]
                top_candidates = [path for path, score in ranked if abs(score - top_score) < 0.03]
                top_scores = [score for path, score in ranked if abs(score - top_score) < 0.03]
                if len(top_candidates) == 1:
                    candidates = top_candidates
                    notes.append("Paired using fuzzy filename similarity.")
                else:
                    chosen = choose_ambiguous_candidate(root, ssi, top_candidates, top_scores)
                    ambiguity_log.append(
                        {
                            "ssi_file": str(ssi),
                            "candidate_files": [str(p) for p in top_candidates],
                            "scores": top_scores,
                            "chosen": str(chosen) if chosen else None,
                        }
                    )
                    if chosen is not None:
                        candidates = [chosen]
                        notes.append("Paired after user-resolved ambiguity.")

        if not candidates:
            unmatched_ssi.append(ssi)
            continue

        raw = candidates[0]
        used_raw.add(raw)
        score = fuzzy_match_score(ssi_key_loose, normalize_structure_key(raw.name, strip_trailing_digits=True))
        matched_pairs.append(
            PairCandidate(
                structure_key=ssi_key_loose,
                structure_name=raw.stem,
                raw_file=raw,
                ssi_file=ssi,
                score=score,
                notes=notes,
            )
        )

    unmatched_raw = [p for p in raw_files if p not in used_raw]
    report = {
        "matched_pairs": [asdict(pair) for pair in matched_pairs],
        "unmatched_raw_files": [str(p) for p in unmatched_raw],
        "unmatched_ssi_files": [str(p) for p in unmatched_ssi],
        "ambiguities": ambiguity_log,
        "summary_text": summarize_pairs(matched_pairs, unmatched_raw, unmatched_ssi),
    }
    return matched_pairs, report


def load_mat_file(path: Path) -> LoadedMat:
    try:
        data = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
        variables = {k: v for k, v in data.items() if not k.startswith("__")}
        loader = "scipy.io.loadmat"
    except NotImplementedError:
        if h5py is None:
            raise RuntimeError(f"{path} appears to require h5py v7.3 support, but h5py is not installed.")
        with h5py.File(path, "r") as h5:
            variables = {k: np.array(h5[k]) for k in h5.keys()}
        loader = "h5py"

    summary = {
        "path": str(path),
        "loader": loader,
        "variables": {},
    }
    for key, value in variables.items():
        entry = {"type": type(value).__name__}
        shape = getattr(value, "shape", None)
        if shape is not None:
            entry["shape"] = tuple(int(x) for x in shape)
        if isinstance(value, np.ndarray):
            entry["dtype"] = str(value.dtype)
        summary["variables"][key] = entry
    return LoadedMat(path=path, variables=variables, loader=loader, summary=summary)


def is_numeric_array(arr: Any) -> bool:
    return isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number)


def squeeze_numeric(arr: Any) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype == object:
        raise ValueError("Encountered object array where numeric array was expected.")
    return np.squeeze(out)


def detect_sampling_frequency(variables: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    candidates: list[tuple[str, float]] = []
    for key, value in variables.items():
        if not is_numeric_array(value):
            continue
        arr = squeeze_numeric(value)
        if arr.ndim == 0 and np.isfinite(arr).all():
            val = float(arr)
            key_l = key.lower()
            if any(tag in key_l for tag in ("fs", "sampling", "sample", "freq")) and 0.1 < val < 1e6:
                candidates.append((key, val))
    choice = candidates[0] if candidates else None
    return (choice[1] if choice else None), {"candidates": candidates, "selected": choice}


def resolve_sampling_frequency(
    variables: dict[str, Any],
    settings: Settings,
) -> tuple[float, dict[str, Any]]:
    fs_detected, fs_debug = detect_sampling_frequency(variables)
    if settings.fs_override is not None:
        resolution = {
            "source": f"{settings.fs_override_source}_fs_override",
            "fs_hz": float(settings.fs_override),
            "warning": None,
            "detection_debug": fs_debug,
        }
        return float(settings.fs_override), resolution
    if fs_detected is not None:
        resolution = {
            "source": "mat_metadata",
            "fs_hz": float(fs_detected),
            "warning": None,
            "detection_debug": fs_debug,
        }
        return float(fs_detected), resolution
    warning = (
        f"Sampling frequency metadata was not found in the raw .mat file. "
        f"Falling back to default {DEFAULT_FIELD_FS_HZ:.1f} Hz."
    )
    print(f"Warning: {warning}")
    resolution = {
        "source": "default_fallback",
        "fs_hz": float(DEFAULT_FIELD_FS_HZ),
        "warning": warning,
        "detection_debug": fs_debug,
    }
    return float(DEFAULT_FIELD_FS_HZ), resolution


def detect_acceleration_matrix(variables: dict[str, Any]) -> tuple[str, np.ndarray, dict[str, Any]]:
    scored: list[tuple[float, str, np.ndarray, dict[str, Any]]] = []
    for key, value in variables.items():
        if not is_numeric_array(value):
            continue
        arr = squeeze_numeric(value)
        if arr.ndim != 2:
            continue
        rows, cols = arr.shape
        score = 0.0
        key_l = key.lower()
        if key == "Data":
            score += 5.0
        if any(tag in key_l for tag in ("acc", "data", "signal", "resp", "response")):
            score += 2.0
        if rows > cols:
            score += 1.5
        if cols >= 3 and cols % 3 == 0:
            score += 1.5
        if rows >= 256:
            score += 1.0
        if cols >= 6:
            score += 0.5
        meta = {"shape": arr.shape, "key": key}
        scored.append((score, key, arr, meta))
        if cols > rows:
            arr_t = arr.T
            meta_t = {"shape": arr_t.shape, "key": key, "transposed": True}
            score_t = score + (2.0 if arr_t.shape[0] > arr_t.shape[1] else 0.0)
            scored.append((score_t, key, arr_t, meta_t))
    if not scored:
        raise RuntimeError("Could not auto-detect a likely raw acceleration matrix in the raw .mat file.")
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_key, best_arr, best_meta = scored[0]
    debug = {
        "selected_key": best_key,
        "selected_shape": best_arr.shape,
        "selected_score": best_score,
        "selected_meta": best_meta,
        "candidates": [
            {"key": key, "shape": tuple(int(x) for x in arr.shape), "score": score, "meta": meta}
            for score, key, arr, meta in scored[:10]
        ],
    }
    return best_key, np.asarray(best_arr, dtype=float), debug


def extract_frequency_column(arr: np.ndarray, key: str) -> tuple[np.ndarray, dict[str, Any]]:
    arr2d = np.asarray(arr, dtype=float)
    if arr2d.ndim == 1:
        return arr2d.reshape(-1), {"column_index": 0, "original_shape": tuple(int(x) for x in arr2d.shape)}
    if arr2d.ndim != 2:
        raise RuntimeError(f"SSI frequency variable '{key}' must be 1D or 2D, got shape {arr2d.shape}.")
    if arr2d.shape[1] == 1:
        return arr2d[:, 0], {"column_index": 0, "original_shape": tuple(int(x) for x in arr2d.shape)}

    plausible_cols: list[tuple[int, float, np.ndarray]] = []
    for col_idx in range(arr2d.shape[1]):
        col = np.asarray(arr2d[:, col_idx], dtype=float).reshape(-1)
        finite = np.all(np.isfinite(col))
        positive = np.all(col > 0.0)
        plausible = finite and positive and np.all(col < 1e3)
        score = 0.0
        if plausible:
            score += 2.0
        if np.all(np.diff(col) >= -1e-9):
            score += 0.5
        if col_idx == 1:
            score += 0.25
        plausible_cols.append((col_idx, score, col))

    plausible_cols.sort(key=lambda item: item[1], reverse=True)
    chosen_idx, chosen_score, chosen = plausible_cols[0]
    debug = {
        "original_shape": tuple(int(x) for x in arr2d.shape),
        "column_index": int(chosen_idx),
        "column_scores": [
            {"column_index": int(col_idx), "score": float(score)}
            for col_idx, score, _ in plausible_cols
        ],
        "decision_note": (
            "Preferred second column when multiple columns looked plausible."
            if arr2d.shape[1] >= 2 and plausible_cols[0][1] == plausible_cols[1][1] + 0.25 and chosen_idx == 1
            else ""
        ),
    }
    return chosen.reshape(-1), debug


def extract_u3_rows_from_mode_shapes(
    mode_shapes: np.ndarray,
    full_channel_count: int,
    expected_n_sensors: int,
    start_index: int,
    stride: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    phi = np.asarray(mode_shapes)
    meta: dict[str, Any] = {
        "original_shape": tuple(int(x) for x in phi.shape),
        "u3_rows_extracted": False,
        "u3_row_indices_zero_based": [],
    }
    if phi.shape[0] == expected_n_sensors:
        return phi, meta
    if phi.shape[0] == full_channel_count:
        u3_rows = np.arange(start_index, full_channel_count, stride, dtype=int)
        phi_u3 = phi[u3_rows, :]
        meta["u3_rows_extracted"] = True
        meta["u3_row_indices_zero_based"] = u3_rows.tolist()
        meta["result_shape"] = tuple(int(x) for x in phi_u3.shape)
        return phi_u3, meta
    raise RuntimeError(
        f"SSI mode-shape row count mismatch: expected either {expected_n_sensors} U3 rows or "
        f"{full_channel_count} full-channel rows, got {phi.shape[0]}."
    )


def detect_ssi_frequency_and_mode_shapes(
    variables: dict[str, Any],
    expected_n_sensors: int,
    full_channel_count: int,
    u3_start_index: int,
    u3_stride: int,
) -> tuple[str, np.ndarray, str, np.ndarray, dict[str, Any]]:
    freq_candidates: list[tuple[float, str, np.ndarray, dict[str, Any]]] = []
    shape_candidates: list[tuple[float, str, np.ndarray, dict[str, Any]]] = []

    for key, value in variables.items():
        if not is_numeric_array(value):
            continue
        arr = squeeze_numeric(value)
        key_l = key.lower()
        if arr.ndim == 1 and arr.size >= 4:
            score = 0.0
            if key == "Extracted_frequencies":
                score += 5.0
            if any(tag in key_l for tag in ("freq", "fn", "hz")):
                score += 3.0
            if np.all(np.isfinite(arr)) and np.all(arr >= 0):
                score += 1.0
            if arr.size <= 500:
                score += 0.5
            freq_candidates.append((score, key, np.asarray(arr, dtype=float).reshape(-1), {"original_shape": tuple(int(x) for x in arr.shape), "column_index": 0}))
        elif arr.ndim == 2 and arr.shape[0] >= 4 and arr.shape[1] >= 1:
            if "freq" in key_l or key == "Extracted_frequencies":
                freqs, freq_meta = extract_frequency_column(arr, key)
                score = 0.0
                if key == "Extracted_frequencies":
                    score += 5.0
                if any(tag in key_l for tag in ("freq", "fn", "hz")):
                    score += 3.0
                if np.all(np.isfinite(freqs)) and np.all(freqs > 0):
                    score += 1.0
                if freqs.size <= 500:
                    score += 0.5
                freq_candidates.append((score, key, freqs, freq_meta))
            if min(arr.shape) < 4:
                continue
            score = 0.0
            meta: dict[str, Any] = {"original_shape": tuple(int(x) for x in arr.shape)}
            arr_use = arr
            if key == "phi_sorted":
                score += 5.0
            if any(tag in key_l for tag in ("phi", "shape", "mode")):
                score += 2.5
            if arr.shape[0] == expected_n_sensors or arr.shape[0] == full_channel_count:
                score += 3.0
            elif arr.shape[1] == expected_n_sensors or arr.shape[1] == full_channel_count:
                score += 3.0
                arr_use = arr.T
                meta["transposed"] = True
            if expected_n_sensors > 0 and expected_n_sensors in arr.shape:
                score += 1.0
            shape_candidates.append((score, key, np.asarray(arr_use), meta))

    if not freq_candidates:
        raise RuntimeError("Could not auto-detect SSI picked frequencies in the SSI .mat file.")
    if not shape_candidates:
        raise RuntimeError("Could not auto-detect SSI mode shapes in the SSI .mat file.")

    freq_candidates.sort(key=lambda item: item[0], reverse=True)
    shape_candidates.sort(key=lambda item: item[0], reverse=True)
    freq_key, freqs, freq_meta = freq_candidates[0][1], freq_candidates[0][2], freq_candidates[0][3]
    shape_key, phi, shape_meta = shape_candidates[0][1], shape_candidates[0][2], shape_candidates[0][3]
    phi_u3, shape_extract_meta = extract_u3_rows_from_mode_shapes(
        phi,
        full_channel_count=full_channel_count,
        expected_n_sensors=expected_n_sensors,
        start_index=u3_start_index,
        stride=u3_stride,
    )

    if phi_u3.shape[1] != freqs.size:
        raise RuntimeError(
            f"SSI frequency/mode count mismatch: freq count={freqs.size}, mode-shape columns={phi_u3.shape[1]} "
            f"for variable '{shape_key}'."
        )

    debug = {
        "selected_frequency_key": freq_key,
        "selected_frequency_meta": freq_meta,
        "selected_mode_shape_key": shape_key,
        "selected_mode_shape_meta": shape_meta,
        "mode_shape_u3_extraction": shape_extract_meta,
        "frequency_candidates": [
            {"key": key, "shape": tuple(int(x) for x in arr.shape), "score": score, "meta": meta}
            for score, key, arr, meta in freq_candidates[:10]
        ],
        "shape_candidates": [
            {"key": key, "shape": tuple(int(x) for x in arr.shape), "score": score, "meta": meta}
            for score, key, arr, meta in shape_candidates[:10]
        ],
    }
    return freq_key, freqs, shape_key, phi_u3, debug


def apply_channel_reorder(acc: np.ndarray, reorder: list[int] | None) -> np.ndarray:
    if reorder is None:
        return acc
    reorder_arr = np.asarray(reorder, dtype=int)
    if reorder_arr.ndim != 1:
        raise ValueError("RAW_CHANNEL_REORDER must be a 1D list of indices.")
    if len(reorder_arr) != acc.shape[1]:
        raise ValueError(
            f"RAW_CHANNEL_REORDER length mismatch: got {len(reorder_arr)}, expected {acc.shape[1]} channels."
        )
    return acc[:, reorder_arr]


def extract_u3_channels(acc: np.ndarray, start_index: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    if acc.ndim != 2:
        raise ValueError(f"Expected 2D acceleration matrix, got shape {acc.shape}")
    idx = np.arange(start_index, acc.shape[1], stride, dtype=int)
    if idx.size == 0:
        raise ValueError("No U3 channels were found with the requested start index and stride.")
    return acc[:, idx], idx


def find_longest_valid_window(X: np.ndarray) -> tuple[int, int, int]:
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array for clean-window search, got shape {X.shape}")
    if X.shape[0] == 0:
        return 0, 0, 0
    valid_rows = np.all(np.isfinite(X), axis=1)
    best_start = 0
    best_end = 0
    current_start: int | None = None

    for idx, is_valid in enumerate(valid_rows):
        if is_valid and current_start is None:
            current_start = idx
        elif not is_valid and current_start is not None:
            if idx - current_start > best_end - best_start:
                best_start, best_end = current_start, idx
            current_start = None
    if current_start is not None and X.shape[0] - current_start > best_end - best_start:
        best_start, best_end = current_start, X.shape[0]
    return best_start, best_end, best_end - best_start


def resolve_clean_window(
    X: np.ndarray,
    fs: float,
    structure_name: str,
    settings: Settings,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_samples, n_channels = X.shape
    valid_rows = np.all(np.isfinite(X), axis=1)
    invalid_rows = ~valid_rows
    invalid_per_channel = np.sum(~np.isfinite(X), axis=0)

    manual_override = None
    if settings.use_manual_window_overrides:
        manual_override = settings.manual_window_overrides.get(structure_name)
        if manual_override is None:
            manual_override = settings.manual_window_overrides.get(sanitize_structure_name(structure_name))

    warning = None
    if manual_override is not None:
        start_idx, end_idx = int(manual_override[0]), int(manual_override[1])
        source = "manual_override"
    elif settings.auto_find_clean_window:
        start_idx, end_idx, _ = find_longest_valid_window(X)
        source = "auto_longest_valid"
    else:
        start_idx, end_idx = 0, n_samples
        source = "full_record"

    start_idx = max(0, min(start_idx, n_samples))
    end_idx = max(start_idx, min(end_idx, n_samples))
    length = end_idx - start_idx
    duration_seconds = float(length / fs) if fs > 0 else float("nan")
    retained_fraction = float(length / n_samples) if n_samples > 0 else 0.0

    if length <= 0:
        raise RuntimeError(
            f"Resolved clean window for structure '{structure_name}' is empty: [{start_idx}:{end_idx})."
        )

    if duration_seconds < settings.min_window_seconds:
        warning = (
            f"Clean window for structure '{structure_name}' is shorter than MIN_WINDOW_SECONDS: "
            f"{duration_seconds:.2f} s < {settings.min_window_seconds:.2f} s."
        )
        print(f"Warning: {warning}")

    clean = X[start_idx:end_idx, :]
    diagnostics = {
        "source": source,
        "original_sample_count": int(n_samples),
        "window_start_index": int(start_idx),
        "window_end_index": int(end_idx),
        "window_length_samples": int(length),
        "window_duration_seconds": duration_seconds,
        "retained_fraction": retained_fraction,
        "number_of_invalid_rows": int(np.sum(invalid_rows)),
        "number_of_invalid_samples_per_channel": invalid_per_channel.astype(int).tolist(),
        "all_channels_finite_in_selected_window": bool(np.all(np.isfinite(clean))),
        "warning": warning,
    }
    return clean, diagnostics


def limit_frequency_range(freq_hz: np.ndarray, arrays: list[np.ndarray], fmin: float | None, fmax: float | None) -> tuple[np.ndarray, list[np.ndarray]]:
    mask = np.ones(freq_hz.shape, dtype=bool)
    if fmin is not None:
        mask &= freq_hz >= fmin
    if fmax is not None:
        mask &= freq_hz <= fmax
    return freq_hz[mask], [arr[..., mask] for arr in arrays]


def compute_spectral_density_matrix(
    acc_u3: np.ndarray,
    fs: float,
    settings: Settings,
) -> SpectralResult:
    _, n_sensors = acc_u3.shape
    freq_hz: np.ndarray | None = None
    spectral_matrix: np.ndarray | None = None

    for i in range(n_sensors):
        for j in range(n_sensors):
            f_ij, pxy = csd(
                acc_u3[:, i],
                acc_u3[:, j],
                fs=fs,
                window=settings.welch_window,
                nperseg=settings.welch_nperseg,
                noverlap=settings.welch_noverlap,
                nfft=settings.welch_nfft,
                detrend=settings.detrend,
                return_onesided=True,
                scaling="density",
            )
            if freq_hz is None:
                freq_hz = f_ij
                spectral_matrix = np.zeros((n_sensors, n_sensors, len(freq_hz)), dtype=np.complex128)
            spectral_matrix[i, j, :] = pxy

    assert freq_hz is not None and spectral_matrix is not None
    raw_psd = np.real(np.diagonal(spectral_matrix, axis1=0, axis2=1)).T

    cpsd_map_full: dict[str, np.ndarray] = {}
    coherence_map_full: dict[str, np.ndarray] = {}
    for i, j in settings.selected_cpsd_pairs[: settings.plot_max_pairs]:
        if i >= n_sensors or j >= n_sensors:
            continue
        key = f"{i}-{j}"
        cpsd_map_full[key] = spectral_matrix[i, j, :].copy()
        if PLOT_COHERENCE:
            _, coh = coherence(
                acc_u3[:, i],
                acc_u3[:, j],
                fs=fs,
                window=settings.welch_window,
                nperseg=settings.welch_nperseg,
                noverlap=settings.welch_noverlap,
                nfft=settings.welch_nfft,
                detrend=settings.detrend,
            )
            coherence_map_full[key] = coh

    full_mask = np.ones(freq_hz.shape, dtype=bool)
    if settings.fmin is not None:
        full_mask &= freq_hz >= settings.fmin
    if settings.fmax is not None:
        full_mask &= freq_hz <= settings.fmax
    freq_hz_trimmed = freq_hz[full_mask]
    spectral_matrix_trimmed = spectral_matrix[..., full_mask]
    raw_psd_trimmed = raw_psd[..., full_mask]
    selected_cpsd = {key: values[full_mask] for key, values in cpsd_map_full.items()}
    selected_coherence = {key: values[full_mask] for key, values in coherence_map_full.items()}

    return SpectralResult(
        freq_hz=freq_hz_trimmed,
        spectral_matrix=spectral_matrix_trimmed,
        psd=raw_psd_trimmed,
        selected_cpsd=selected_cpsd,
        selected_coherence=selected_coherence,
    )


def run_fdd(spectral_matrix: np.ndarray, n_sv_keep: int) -> tuple[np.ndarray, np.ndarray]:
    n_sensors, _, n_freq = spectral_matrix.shape
    n_sv_keep = min(n_sv_keep, n_sensors)
    singular_values = np.zeros((n_sv_keep, n_freq), dtype=float)
    first_singular_vectors = np.zeros((n_sensors, n_freq), dtype=np.complex128)

    for k in range(n_freq):
        u, s, _ = np.linalg.svd(spectral_matrix[:, :, k], full_matrices=False)
        singular_values[:, k] = s[:n_sv_keep]
        first_singular_vectors[:, k] = u[:, 0]
    return singular_values, first_singular_vectors


def normalize_mode_shape(phi: np.ndarray) -> np.ndarray:
    vec = np.asarray(phi, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(vec)
    if norm <= 0.0:
        raise ValueError("Encountered zero-norm mode shape.")
    return vec / norm


def complex_mac_like(a: np.ndarray, b: np.ndarray) -> float:
    aa = normalize_mode_shape(a)
    bb = normalize_mode_shape(b)
    return float(np.abs(np.vdot(aa, bb)) ** 2)


def smooth_mask(mask: np.ndarray, freq_hz: np.ndarray, smoothing_hz: float) -> np.ndarray:
    if smoothing_hz <= 0.0 or mask.size < 3:
        return mask.astype(float)
    df = float(np.median(np.diff(freq_hz)))
    sigma_bins = max(smoothing_hz / max(df, 1e-9), 1.0)
    radius = int(math.ceil(3.0 * sigma_bins))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(mask.astype(float), kernel, mode="same")


def build_target_modes(
    freqs: np.ndarray,
    mode_shapes: np.ndarray,
    mapping: dict[str, int],
) -> list[dict[str, Any]]:
    target_modes: list[dict[str, Any]] = []
    for label in TARGET_MODE_LABELS:
        idx = int(mapping[label])
        if idx < 0 or idx >= freqs.size:
            raise ValueError(f"SSI index {idx} for {label} is out of bounds for {freqs.size} picked modes.")
        target_modes.append(
            {
                "label": label,
                "ssi_index": idx,
                "freq_hz": float(freqs[idx]),
                "mode_shape": np.asarray(mode_shapes[:, idx], dtype=np.complex128),
            }
        )
    return target_modes


def determine_candidate_half_width(target_freq: float, all_target_freqs: np.ndarray, settings: Settings) -> float:
    distance = np.inf
    for other in all_target_freqs:
        if float(other) == float(target_freq):
            continue
        distance = min(distance, abs(float(other) - float(target_freq)))
    if np.isfinite(distance) and distance < 2.0 * settings.band_half_width:
        return settings.close_mode_half_width
    return settings.band_half_width


def build_modal_filter_v1(
    freq_hz: np.ndarray,
    spectral_matrix: np.ndarray,
    singular_values: np.ndarray,
    first_singular_vectors: np.ndarray,
    target_modes: list[dict[str, Any]],
    settings: Settings,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    n_sensors = spectral_matrix.shape[0]
    n_freq = freq_hz.size
    filtered_total = np.zeros_like(spectral_matrix, dtype=np.complex128)
    per_mode_spectral: dict[str, np.ndarray] = {}
    per_mode_masks: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {"version": "Version 1 pragmatic SSI-guided band + FDD filtering", "per_mode": {}}
    target_freqs = np.array([mode["freq_hz"] for mode in target_modes], dtype=float)

    for mode in target_modes:
        label = str(mode["label"])
        target_freq = float(mode["freq_hz"])
        target_shape = normalize_mode_shape(mode["mode_shape"])
        half_width = determine_candidate_half_width(target_freq, target_freqs, settings)
        band_mask = np.abs(freq_hz - target_freq) <= half_width

        similarity = np.zeros(n_freq, dtype=float)
        for k in range(n_freq):
            similarity[k] = complex_mac_like(first_singular_vectors[:, k], target_shape)

        selected = band_mask & (similarity >= settings.mode_similarity_threshold)
        if not np.any(selected):
            selected = band_mask
            fallback = "hard band mask fallback used because no lines exceeded similarity threshold"
        else:
            fallback = ""

        weights = smooth_mask(selected.astype(float), freq_hz, settings.mask_smoothing_hz)
        weights *= band_mask.astype(float)
        weights = np.clip(weights, 0.0, 1.0)

        mode_spectral = np.zeros_like(spectral_matrix, dtype=np.complex128)
        for k in range(n_freq):
            if weights[k] <= 0.0:
                continue
            u = first_singular_vectors[:, k].reshape(n_sensors, 1)
            rank1 = singular_values[0, k] * (u @ np.conjugate(u.T))
            mode_spectral[:, :, k] = weights[k] * rank1

        per_mode_masks[label] = weights
        per_mode_spectral[label] = mode_spectral
        filtered_total += mode_spectral
        diagnostics["per_mode"][label] = {
            "target_freq_hz": target_freq,
            "half_width_hz": half_width,
            "selected_frequency_count": int(np.sum(selected)),
            "band_frequency_count": int(np.sum(band_mask)),
            "similarity_threshold": settings.mode_similarity_threshold,
            "fallback_note": fallback,
            "max_similarity_in_band": float(np.max(similarity[band_mask])) if np.any(band_mask) else None,
        }

    return filtered_total, per_mode_spectral, per_mode_masks, diagnostics


def synthesize_filtered_timeseries(
    acc_u3: np.ndarray,
    fs: float,
    freq_hz: np.ndarray,
    per_mode_masks: dict[str, np.ndarray],
) -> np.ndarray:
    n_samples, n_sensors = acc_u3.shape
    fft_freq = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    combined_mask = np.zeros_like(freq_hz, dtype=float)
    for mask in per_mode_masks.values():
        combined_mask = np.maximum(combined_mask, np.asarray(mask, dtype=float))
    interp_mask = np.interp(fft_freq, freq_hz, combined_mask, left=0.0, right=0.0)

    filtered = np.zeros_like(acc_u3, dtype=float)
    for i in range(n_sensors):
        spec = np.fft.rfft(acc_u3[:, i])
        spec_filtered = spec * interp_mask
        filtered[:, i] = np.fft.irfft(spec_filtered, n=n_samples)
    return filtered


def make_time_axis(n_samples: int, fs: float) -> np.ndarray:
    return np.arange(n_samples, dtype=float) / fs


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, out_path: Path, settings: Settings) -> None:
    if settings.save_png:
        fig.savefig(out_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    if settings.save_svg:
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    if settings.show_plots:
        fig.show()
    else:
        plt.close(fig)


def plot_stacked_timeseries(data: np.ndarray, fs: float, title: str, out_path: Path, settings: Settings) -> None:
    n_samples, n_sensors = data.shape
    keep = min(n_sensors, settings.plot_max_sensors_timeseries)
    x = make_time_axis(n_samples, fs)
    subset = data[:, :keep]
    scale = np.nanmax(np.abs(subset)) or 1.0
    offsets = np.arange(keep, dtype=float) * 2.5 * scale

    fig, ax = plt.subplots(figsize=(12, 7))
    for idx in range(keep):
        ax.plot(x, subset[:, idx] + offsets[idx], lw=0.8, label=f"U3-{idx}")
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude + offset")
    ax.grid(True, alpha=0.3)
    save_figure(fig, out_path, settings)


def plot_psd_overlay(freq_hz: np.ndarray, psd: np.ndarray, title: str, out_path: Path, settings: Settings) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    n_sensors = psd.shape[0]
    for idx in range(n_sensors):
        ax.plot(freq_hz, np.maximum(np.abs(psd[idx]), 1e-18), lw=0.8, alpha=0.8)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD")
    ax.grid(True, alpha=0.3)
    save_figure(fig, out_path, settings)


def plot_selected_cpsd(
    freq_hz: np.ndarray,
    cpsd_map: dict[str, np.ndarray],
    title: str,
    ylabel: str,
    out_path: Path,
    settings: Settings,
) -> None:
    if not cpsd_map:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    for key, values in cpsd_map.items():
        ax.plot(freq_hz, np.maximum(np.abs(values), 1e-18), lw=1.0, label=key)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, out_path, settings)


def plot_selected_coherence(
    freq_hz: np.ndarray,
    coherence_map: dict[str, np.ndarray],
    title: str,
    out_path: Path,
    settings: Settings,
) -> None:
    if not coherence_map:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    for key, values in coherence_map.items():
        ax.plot(freq_hz, values, lw=1.0, label=key)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Coherence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, out_path, settings)


def plot_fdd_singular_values(
    freq_hz: np.ndarray,
    singular_values: np.ndarray,
    out_path: Path,
    settings: Settings,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx in range(singular_values.shape[0]):
        ax.plot(freq_hz, np.maximum(singular_values[idx], 1e-18), lw=1.2, label=f"SV{idx + 1}")
    ax.set_yscale("log")
    ax.set_title("FDD singular values")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Singular value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, out_path, settings)


def plot_raw_vs_filtered_psd(
    freq_hz: np.ndarray,
    raw_psd: np.ndarray,
    filtered_psd: np.ndarray,
    out_path: Path,
    settings: Settings,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for idx in range(raw_psd.shape[0]):
        axes[0].plot(freq_hz, np.maximum(np.abs(raw_psd[idx]), 1e-18), lw=0.8, alpha=0.8)
        axes[1].plot(freq_hz, np.maximum(np.abs(filtered_psd[idx]), 1e-18), lw=0.8, alpha=0.8)
    axes[0].set_title("Raw U3 PSD")
    axes[1].set_title("Filtered U3 PSD")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("PSD")
    axes[1].set_xlabel("Frequency [Hz]")
    save_figure(fig, out_path, settings)


def plot_raw_vs_filtered_timeseries(
    raw: np.ndarray,
    filtered: np.ndarray,
    fs: float,
    out_path: Path,
    settings: Settings,
) -> None:
    keep = min(raw.shape[1], 4)
    x = make_time_axis(raw.shape[0], fs)
    fig, axes = plt.subplots(keep, 1, figsize=(12, 2.5 * keep), sharex=True)
    axes_arr = np.atleast_1d(axes)
    for idx, ax in enumerate(axes_arr):
        ax.plot(x, raw[:, idx], lw=0.8, label="raw", alpha=0.7)
        ax.plot(x, filtered[:, idx], lw=0.8, label="filtered", alpha=0.9)
        ax.set_ylabel(f"U3-{idx}")
        ax.grid(True, alpha=0.3)
    axes_arr[0].legend()
    axes_arr[-1].set_xlabel("Time [s]")
    fig.suptitle("Raw vs filtered time series (visualization only)")
    save_figure(fig, out_path, settings)


def load_mode_assignment_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_mode_assignment_store(path: Path, payload: dict[str, Any]) -> None:
    save_json(path, payload)


def prompt_mode_assignment(
    root: tk.Tk | None,
    structure_name: str,
    ssi_freqs_hz: np.ndarray,
    existing: dict[str, Any] | None,
) -> dict[str, int]:
    freq_lines = "\n".join(f"{i}: {float(f):.4f} Hz" for i, f in enumerate(ssi_freqs_hz))
    default_text = ""
    if existing:
        default_text = ",".join(str(existing.get(label, "")) for label in TARGET_MODE_LABELS)
    prompt = (
        f"Structure: {structure_name}\n\n"
        f"Detected SSI-picked frequencies:\n{freq_lines}\n\n"
        f"Enter indices as B1,T1,B2,T2\n"
        f"Example: 0,1,3,4"
    )

    if existing:
        use_existing = popup_yes_no(
            root,
            "Reuse saved mode mapping?",
            f"{structure_name} already has a saved mapping:\n{existing}\n\nReuse it?",
        )
        if use_existing:
            return {label: int(existing[label]) for label in TARGET_MODE_LABELS}

    answer = popup_ask_string(root, "Assign target modes", prompt, initial=default_text)
    if answer is None:
        raise RuntimeError(f"Mode assignment cancelled for structure '{structure_name}'.")
    parts = [p.strip() for p in answer.split(",")]
    if len(parts) != len(TARGET_MODE_LABELS):
        raise ValueError(f"Expected {len(TARGET_MODE_LABELS)} comma-separated indices, got: {answer}")
    mapping = {label: int(idx) for label, idx in zip(TARGET_MODE_LABELS, parts)}
    return mapping


def convert_complex_dict_for_mat(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            out[key] = convert_complex_dict_for_mat(value)
        else:
            out[key] = value
    return out


def sanitize_structure_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def process_structure(
    pair: PairCandidate,
    settings: Settings,
    root: tk.Tk | None,
    mode_store: dict[str, Any],
) -> dict[str, Any]:
    assert pair.raw_file is not None and pair.ssi_file is not None

    structure_name = sanitize_structure_name(pair.structure_name)
    structure_dir = ensure_dir(settings.output_root / structure_name)
    plots_raw_dir = ensure_dir(structure_dir / "plots_raw")
    plots_filtered_dir = ensure_dir(structure_dir / "plots_filtered")
    data_dir = ensure_dir(structure_dir / "data")
    logs_dir = ensure_dir(structure_dir / "logs")

    raw_mat = load_mat_file(pair.raw_file)
    ssi_mat = load_mat_file(pair.ssi_file)
    save_json(logs_dir / "raw_mat_summary.json", raw_mat.summary)
    save_json(logs_dir / "ssi_mat_summary.json", ssi_mat.summary)

    raw_key, acc_raw, acc_debug = detect_acceleration_matrix(raw_mat.variables)
    fs, fs_resolution = resolve_sampling_frequency(raw_mat.variables, settings)

    acc_reordered = apply_channel_reorder(acc_raw, settings.raw_channel_reorder)
    acc_u3_full, u3_indices = extract_u3_channels(acc_reordered, settings.u3_start_index, settings.u3_stride)
    acc_u3_clean, clean_window_diag = resolve_clean_window(
        acc_u3_full,
        fs,
        structure_name,
        settings,
    )
    freq_key, ssi_freqs_hz, shape_key, ssi_mode_shapes, ssi_debug = detect_ssi_frequency_and_mode_shapes(
        ssi_mat.variables,
        expected_n_sensors=acc_u3_full.shape[1],
        full_channel_count=acc_reordered.shape[1],
        u3_start_index=settings.u3_start_index,
        u3_stride=settings.u3_stride,
    )

    print(
        f"detected raw variable = {raw_key}, shape = {tuple(int(x) for x in acc_raw.shape)}\n"
        f"detected SSI frequency variable = {freq_key}, shape = "
        f"{tuple(int(x) for x in np.asarray(ssi_mat.variables[freq_key]).shape)}\n"
        f"detected SSI mode variable = {shape_key}, shape = "
        f"{tuple(int(x) for x in np.asarray(ssi_mat.variables[shape_key]).shape)}\n"
        f"using fs = {fs:.1f} Hz\n"
        f"extracted U3 channels = {u3_indices.tolist()}\n"
        f"extracted U3 SSI mode shape matrix shape = {tuple(int(x) for x in ssi_mode_shapes.shape)}\n"
        f"auto clean window = [{clean_window_diag['window_start_index']}:{clean_window_diag['window_end_index']}), "
        f"length={clean_window_diag['window_length_samples']} samples, "
        f"duration={clean_window_diag['window_duration_seconds']:.2f} s"
    )

    existing_mapping = mode_store.get(structure_name)
    mapping = prompt_mode_assignment(root, structure_name, ssi_freqs_hz, existing_mapping)
    mode_store[structure_name] = mapping
    save_mode_assignment_store(settings.mode_assignment_json, mode_store)
    save_json(logs_dir / "mode_assignment_used.json", mapping)

    spectral_raw = compute_spectral_density_matrix(acc_u3_clean, fs, settings)
    singular_values, first_singular_vectors = run_fdd(
        spectral_raw.spectral_matrix,
        settings.fdd_n_singular_values,
    )
    target_modes = build_target_modes(ssi_freqs_hz, ssi_mode_shapes, mapping)

    filtered_sdm, per_mode_sdm, per_mode_masks, filtering_diag = build_modal_filter_v1(
        spectral_raw.freq_hz,
        spectral_raw.spectral_matrix,
        singular_values,
        first_singular_vectors,
        target_modes,
        settings,
    )
    filtered_psd = np.real(np.diagonal(filtered_sdm, axis1=0, axis2=1)).T
    filtered_cpsd = {
        key: filtered_sdm[int(key.split("-")[0]), int(key.split("-")[1]), :].copy()
        for key in spectral_raw.selected_cpsd.keys()
    }

    filtered_time = None
    if settings.make_filtered_timeseries:
        filtered_time = synthesize_filtered_timeseries(acc_u3_clean, fs, spectral_raw.freq_hz, per_mode_masks)

    plot_stacked_timeseries(
        acc_u3_clean,
        fs,
        "Raw U3 time series (clean window)",
        plots_raw_dir / "raw_u3_timeseries",
        settings,
    )
    plot_psd_overlay(
        spectral_raw.freq_hz,
        spectral_raw.psd,
        "Raw U3 PSD overlay",
        plots_raw_dir / "raw_psd_overlay",
        settings,
    )
    plot_selected_cpsd(
        spectral_raw.freq_hz,
        spectral_raw.selected_cpsd,
        "Raw selected CPSD magnitude",
        "CPSD magnitude",
        plots_raw_dir / "raw_cpsd_selected",
        settings,
    )
    if PLOT_COHERENCE:
        plot_selected_coherence(
            spectral_raw.freq_hz,
            spectral_raw.selected_coherence,
            "Raw selected coherence",
            plots_raw_dir / "raw_coherence_selected",
            settings,
        )
    plot_fdd_singular_values(
        spectral_raw.freq_hz,
        singular_values,
        plots_raw_dir / "fdd_singular_values",
        settings,
    )

    plot_psd_overlay(
        spectral_raw.freq_hz,
        filtered_psd,
        "Filtered U3 PSD overlay",
        plots_filtered_dir / "filtered_psd_overlay",
        settings,
    )
    plot_selected_cpsd(
        spectral_raw.freq_hz,
        filtered_cpsd,
        "Filtered selected CPSD magnitude",
        "CPSD magnitude",
        plots_filtered_dir / "filtered_cpsd_selected",
        settings,
    )
    plot_raw_vs_filtered_psd(
        spectral_raw.freq_hz,
        spectral_raw.psd,
        filtered_psd,
        plots_filtered_dir / "raw_vs_filtered_psd",
        settings,
    )
    if filtered_time is not None:
        plot_stacked_timeseries(
            filtered_time,
            fs,
            "Filtered U3 time series (visualization only)",
            plots_filtered_dir / "filtered_u3_timeseries",
            settings,
        )
        plot_raw_vs_filtered_timeseries(
            acc_u3_clean,
            filtered_time,
            fs,
            plots_filtered_dir / "raw_vs_filtered_timeseries",
            settings,
        )

    detection_summary = {
        "raw_acceleration_variable": raw_key,
        "sampling_frequency_resolution": fs_resolution,
        "ssi_frequency_variable": freq_key,
        "ssi_mode_shape_variable": shape_key,
        "raw_acceleration_detection": acc_debug,
        "ssi_detection": ssi_debug,
        "channel_extraction": {
            "original_channel_count": int(acc_reordered.shape[1]),
            "u3_channel_indices_zero_based": u3_indices.tolist(),
            "n_u3_sensors": int(acc_u3_full.shape[1]),
        },
        "clean_window": clean_window_diag,
        "pairing": asdict(pair),
    }
    save_json(logs_dir / "detection_summary.json", detection_summary)
    save_json(logs_dir / "filtering_diagnostics.json", filtering_diag)
    save_json(logs_dir / "settings_used.json", asdict(settings))

    target_mode_meta = [
        {
            "label": mode["label"],
            "ssi_index": int(mode["ssi_index"]),
            "freq_hz": float(mode["freq_hz"]),
        }
        for mode in target_modes
    ]
    save_json(logs_dir / "target_modes.json", target_mode_meta)

    npz_payload = {
        "freq_hz": spectral_raw.freq_hz,
        "raw_u3_timeseries_full": acc_u3_full,
        "raw_u3_timeseries_clean": acc_u3_clean,
        "raw_spectral_density_matrix": spectral_raw.spectral_matrix,
        "filtered_spectral_density_matrix": filtered_sdm,
        "raw_psd": spectral_raw.psd,
        "filtered_psd": filtered_psd,
        "singular_values": singular_values,
        "first_singular_vectors": first_singular_vectors,
        "ssi_picked_frequencies_hz": ssi_freqs_hz,
        "ssi_selected_indices": np.array([mapping[label] for label in TARGET_MODE_LABELS], dtype=int),
        "u3_channel_indices_zero_based": u3_indices,
    }
    for label, mode_sdm in per_mode_sdm.items():
        npz_payload[f"filtered_sdm_{label}"] = mode_sdm
    for label, weights in per_mode_masks.items():
        npz_payload[f"mask_{label}"] = weights
    for key, values in spectral_raw.selected_cpsd.items():
        npz_payload[f"raw_cpsd_{key}"] = values
    for key, values in filtered_cpsd.items():
        npz_payload[f"filtered_cpsd_{key}"] = values
    if filtered_time is not None:
        npz_payload["filtered_u3_timeseries_visualization_only"] = filtered_time

    if settings.save_npz:
        np.savez(data_dir / "field_modal_filtering_outputs.npz", **npz_payload)

    if settings.save_mat:
        sio.savemat(
            data_dir / "field_modal_filtering_outputs.mat",
            convert_complex_dict_for_mat(npz_payload),
            do_compression=True,
        )

    structure_summary = {
        "structure_name": structure_name,
        "raw_file": str(pair.raw_file),
        "ssi_file": str(pair.ssi_file),
        "sampling_frequency_hz": fs,
        "n_samples_full": int(acc_u3_full.shape[0]),
        "n_samples_clean": int(acc_u3_clean.shape[0]),
        "n_u3_sensors": int(acc_u3_full.shape[1]),
        "clean_window": clean_window_diag,
        "output_dir": str(structure_dir),
        "target_modes": target_mode_meta,
    }
    save_json(logs_dir / "structure_summary.json", structure_summary)
    return structure_summary


def resolve_input_files(settings: Settings, cli_args: argparse.Namespace | None, root: tk.Tk | None) -> list[Path]:
    if cli_args is not None and cli_args.mat_files:
        return [Path(p) for p in cli_args.mat_files]
    if settings.run_mode.lower() != "interactive":
        raise RuntimeError("No explicit --mat-files were provided and RUN_MODE is not 'interactive'.")
    selected = select_mat_files_interactive(root)
    if not selected:
        raise RuntimeError("No .mat files were selected.")
    return selected


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    settings = build_default_settings()

    cli_args: argparse.Namespace | None = None
    if cli_has_explicit_arguments(argv):
        cli_args = parse_cli_args(argv)
        settings = apply_cli_overrides(settings, cli_args)
        active_mode = "cli"
    else:
        active_mode = settings.run_mode.lower()

    settings.output_root.mkdir(parents=True, exist_ok=True)
    if settings.mode_assignment_json is None:
        settings.mode_assignment_json = settings.output_root / "manual_mode_assignments.json"

    root = init_tk_root() if active_mode == "interactive" else None

    mat_files = resolve_input_files(settings, cli_args, root)
    pairs, pairing_report = pair_structure_files(mat_files, root=root)
    save_json(settings.output_root / "pairing_report.json", pairing_report)
    (settings.output_root / "pairing_report.txt").write_text(pairing_report["summary_text"], encoding="utf-8")

    if active_mode == "interactive":
        popup_show_info(root, "Detected pairs", pairing_report["summary_text"])
        if not popup_yes_no(root, "Continue?", "Continue processing the matched structures?"):
            return 1

    valid_pairs = [pair for pair in pairs if pair.raw_file is not None and pair.ssi_file is not None]
    if not valid_pairs:
        raise RuntimeError("No valid raw/SSI pairs were found to process.")

    mode_store = load_mode_assignment_store(settings.mode_assignment_json)
    summaries = []
    for pair in valid_pairs:
        print(f"Processing structure '{pair.structure_name}'")
        summaries.append(process_structure(pair, settings, root, mode_store))

    save_json(
        settings.output_root / "run_summary.json",
        {"mode": active_mode, "settings": asdict(settings), "structures": summaries},
    )
    if active_mode == "interactive":
        popup_show_info(root, "Processing complete", f"Finished processing {len(summaries)} structure(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
