"""Shared paths and constants for the Final_results_for_paper pipeline.

Everything the paper reports is computed here, from one population generated
inside this folder.  The design was fixed with the author on 2026-09-25, after
the generation audit (see ../README.md for the full list of decisions):

  * one population of 5500 structures, 1375 per girder count (Ng = 2..5);
    the first 1250 per Ng are the training pool, the next 125 per Ng are the
    fixed test set every model is scored on;
  * population-size sweep N = 100 .. 5000, balanced (N/4 per Ng) and nested
    (each N is the first N/4 pool structures of every Ng), split 85/15
    train/validation inside each Ng;
  * broadband response only for training and inference (a paired narrowband
    population is generated but never trained on);
  * 5000 epochs, one seed (123), best-validation checkpoint everywhere.

Stage scripts live in pipeline/ and write their tables to pipeline/results/,
so tools/run_stage.py (which hashes the stage's own folder) never walks the
gigabytes under data/ and runs/.  Run everything from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent
FINAL = PIPELINE.parent                      # Final_results_for_paper/
REPO = FINAL.parent                          # repository root (two parents up)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

DATA = FINAL / "data"
RUNS = FINAL / "runs"
RESULTS = PIPELINE / "results"

POP_BROADBAND_ROOT = DATA / "pop_broadband"      # generator OUT_DIR, broadband arm
POP_NARROWBAND_ROOT = DATA / "pop_narrowband"    # generator OUT_DIR, narrowband arm
POP_BROADBAND = POP_BROADBAND_ROOT / "samples_phase2_bandpass"
POP_NARROWBAND = POP_NARROWBAND_ROOT / "samples_phase2"
SPLITS = DATA / "splits"                     # manifests written by s2_make_splits.py

# --- population design --------------------------------------------------------
NG_VALUES = (2, 3, 4, 5)
N_TOTAL = 5500                               # 1375 per Ng
POOL_PER_NG = 1250                           # training pool: first 1250 per Ng
TEST_PER_NG = 125                            # fixed test set: next 125 per Ng
SWEEP_N = (100, 200, 400, 600, 800, 1000, 2000, 5000)
# The network behind in-domain, the sensor sweep, zero-shot, decomposition and
# fine-tuning: the smallest N in the sweep at which every mode reaches test MAC
# >= 0.9 (author's decision, 2026-09-28; it was 600 until then).
MAIN_N = 2000
# The architecture ablation compares equal-N networks and stays where its GraphSAGE
# arms were trained; its EdgeGAT arm is the N = 600 sweep network.
ABLATION_N = 600
# Since 2026-09-29 the ablation is a full grid, layer x edge feature, every arm trained
# identically at ABLATION_N (fr_train.jobs()).  The 8-d arm (complex coherency at the
# four FE-matched modal frequencies only) sits outside the grid, in the tables only.
ABLATION_LAYERS = ("GraphSAGE", "EdgeGATConv")
ABLATION_FEATURES = ("none", "orientation", "coherency")
ABLATION_GRID = {
    ("GraphSAGE", "none"): f"graphsage_noedge_signed_N{ABLATION_N}",
    ("GraphSAGE", "orientation"): f"graphsage_geomorient_signed_N{ABLATION_N}",
    ("GraphSAGE", "coherency"): f"graphsage_cpsdfull_signed_N{ABLATION_N}",
    ("EdgeGATConv", "none"): f"edgegat_noedge_N{ABLATION_N}",
    ("EdgeGATConv", "orientation"): f"edgegat_geomorient_N{ABLATION_N}",
    ("EdgeGATConv", "coherency"): f"edgegat_N{ABLATION_N}",          # the sweep network
}
ABLATION_EXTRA = (f"graphsage_cpsd8_signed_N{ABLATION_N}",)
FEATURE_LABEL = {"none": "no edge features", "orientation": "edge orientation only (cos, sin), 2-d",
                 "coherency": "full-spectrum complex coherency, 800-d"}
TRAIN_FRAC = 0.85                            # inside each N, per Ng
SPLIT_SEED = 123

# --- generator changes relative to production ----------------------------------
GENERATOR_SEED = 20260507                    # unchanged
DY_MAX_NG3PLUS_MM = 3000.0                   # was 2500.0
SELECTED_MAC_GATE = 0.85                     # reject a structure if any selected mode matches below this

# --- training ----------------------------------------------------------------------
SEED = 123
EPOCHS = 5000
MODE_NAMES = ("B1", "T1", "B2", "T2")

LOOCV = ["Hellesylt", "Hundeidvik", "Leknes", "Linge", "Saebo",
         "Eidsdal", "Liabygda", "Stranda", "Festoya"]

# --- field recordings: sensor order (PROVENANCE Phase P, 2026-11-23) ------------------
# The loggers recorded as x40131 and x40133 - 0-based sensor rows 6 and 8, raw column
# triples 19-21 and 25-27 (1-based) - are interchanged in the raw .mat files at all nine
# bridges: x40131 sat at grid node 11, x40133 at grid node 9.  Every SSI script the labels
# come from swaps them back before identification,
#     y = Data(:, [1:18 25 26 27 22 23 24 19 20 21 28:end]);   % Swap 131 and 133
# (System identification code/Field measurements 26_270625/<bridge>/MAIN_SSI_RUN_SCRIPT*.m;
# Stranda: MAIN_SSI_RUN_SCRIPT_Walk.m, which reads Stranda_Walk.mat).  The GNN field path
# (process_field_modal_filtering.py -> E5's corrected builder) never did; FE geometry
# confirms the SSI order in 36/36 bridge x mode cells (EC_comments T10).  The paper's
# field inputs are rebuilt with the swap by s5a_field_inputs.py; experiments/ keeps the
# legacy raw-order inputs as history.  The labels (SSI_modal_infor/) are not touched.
FIELD_SENSOR_SWAP = (6, 8)                       # 0-based sensor rows, exchanged
FIELD_SENSOR_SERIALS = ("x40131", "x40133")      # their logger serials, checked per bridge
FIELD_CHANNELS_PER_SENSOR = 3                    # raw columns per logger (ch1, ch2, ch3 = u3)


def field_sensor_order(n_sensors: int) -> list[int]:
    """Sensor-row permutation applied to the raw records (identity but FIELD_SENSOR_SWAP)."""
    order = list(range(n_sensors))
    i, j = FIELD_SENSOR_SWAP
    order[i], order[j] = order[j], order[i]
    return order


def field_channel_reorder(n_channels: int) -> list[int]:
    """The same permutation on raw .mat columns (0-based form of the SSI scripts' reorder)."""
    k = FIELD_CHANNELS_PER_SENSOR
    assert n_channels % k == 0, n_channels
    return [k * s + c for s in field_sensor_order(n_channels // k) for c in range(k)]


# raw field records and SSI-picked files the GNN field path was produced from
# (field_modal_filtering_outputs/<S>/logs/structure_summary.json raw_file / ssi_file)
FIELD_RAW_MAT_DIR = (REPO.parents[2] / "System identification code" / "Field measurements 26_270625"
                     / "For GNN")
FIELD_RAW_FILES = {s: (f"{s}.mat", f"{s}_Data_picked.mat") for s in LOOCV}
FIELD_RAW_FILES["Eidsdal"] = ("Eidsdal.mat", "Eisdal_Data_picked.mat")
FIELD_RAW_FILES["Stranda"] = ("Stranda_Walk.mat", "Stranda_Data_picked.mat")
# legacy, raw sensor order: what experiments/ (E2, E5, E8, E10, E11, E17-E19) ran on
LEGACY_FIELD_OUTPUTS = REPO / "field_modal_filtering_outputs"
LEGACY_REAL_FEATURES = REPO / "experiments" / "E5_real_transfer" / "phase3_corrected"
# the paper's field inputs since Phase P, written by s5a_field_inputs.py
FIELD_INPUTS = DATA / "field_inputs_v2"
FIELD_OUTPUTS = FIELD_INPUTS / "field_modal_filtering_outputs"
REAL_FEATURES = FIELD_INPUTS / "phase3_corrected"      # <S>/phase3_dataset_corrected.npz


def keep_awake() -> bool:
    """Ask Windows not to idle-sleep while this process runs.

    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) is a per-process
    request, released automatically when the process exits; no power setting is
    changed.  It does not stop sleep on lid close or on battery exhaustion.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
    except Exception:  # noqa: BLE001
        return False


def run_dir(name: str) -> Path:
    return RUNS / name


def sweep_run_name(n: int) -> str:
    return f"edgegat_N{n}"
