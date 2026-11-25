# Data, networks and field files (not in git)

This repository is public; its inputs are not. Everything below is gitignored. Each file's SHA-256 is in
[DATA_MANIFEST.tsv.gz](DATA_MANIFEST.tsv.gz) and each folder's count, size and folder hash in
[data_summary.csv](data_summary.csv) (`tools/hash_data.py`; `--check` verifies a copy).

"Linked" folders are NTFS hard links to the archive `../Linkspan_GNN_OMA` (same bytes, no extra disk), marked
read-only; "copied" ones are independent copies. `tools/migrate_from_archive.py` made both.

## Inputs

| folder | what | from the archive | how |
|---|---|---|---|
| `data/population/samples_phase2_bandpass/` | the 5500 structures (1375 per Ng 2–5): parameters, FE labels, graph, and features under the **shared** record 12345 | `Final_results_for_paper/data/pop_broadband/` | linked |
| `data/splits/` | train / validation manifests per N, the 500-structure test list | `Final_results_for_paper/data/splits/` | copied |
| `data/T11/trainval_varexc`, `paper_varexc` | N600 train + val, and the paper's test structures, own record each | `EC_comments/data/T11/` | linked |
| `data/T13/rec1..3`, `trainval_N2000`, `trainval_T120`, `paper_varexc_T120` | records 1–3 (N600), pool 0 and validation (N2000), 120 s records | `EC_comments/data/T13/` | linked |
| `data/T17/rec1..3_N2000` | records 1–3 of the N2000 training structures | `EC_comments/data/T17/` | linked |
| `data/T20/{deck,walk}_{p0..p3,test}`, `checks/` | T20's deck-forcing and walker records with a noise floor | `EC_comments/data/T20/` | linked |
| `data/T5/fresh_test`, `fresh_test_paperexc` | 500 new structures (parameter seed 20261123) under one new record / the paper's record | `EC_comments/data/T5/` | linked |
| `data/field_inputs_v2/` | GNN field inputs of the nine bridges, Phase P sensor order (`s5a_field_inputs.py`) | `Final_results_for_paper/data/field_inputs_v2/` | linked |
| `data/legacy_preP/` | the pre-Phase-P field outputs and features (sensors 7 and 9 interchanged): T10 history only | `field_modal_filtering_outputs/`, `experiments/E5_real_transfer/phase3_corrected/` | linked |
| `field_measurements_raw/` | raw 128 Hz recordings and SSI-picked files of the bridges (byte-identical to the archive's external "For GNN" folder) | `field_measurements_raw/` | copied |
| `SSI_modal_infor/` | SSI labels (`mode_selection.json`, `*_Data_picked.mat`, grids) | `SSI_modal_infor/` | copied |
| `phase3_real_graphs/` | the bridges' graph packages and FE targets (with the two T2 overrides) | `phase3_real_graphs/` | copied |
| `experiments/E6_controls/task2_raw/` | FE targets read by s5 / s6 | same | copied |
| `experiments/E19_mode_identification/cache/`, `experiments/E18_population_basis/prior.npz` | the class prior and spectra the template / FDD baselines read | same | linked / copied |

## Derived data (rewritten by stages, not linked)

`data/T5/fdd`, `data/T5/predictions`, `data/T11/predictions`, `data/T13/predictions`, `data/T10`, `data/T15`,
`data/T19/sims`: written by `c_fdd`, `c_rescore` / `h_eval` / `k_eval`, `g_t10_*`, `n_mirror_test`,
`u_decomposition`.

## Networks

| folder | what |
|---|---|
| `runs/ec/ec_varexc_*` | T11 pilot arms (N600, 5000 epochs, one record per structure) |
| `runs/ec/ec_t13_*` | T13 feasibility arms (400 epochs) |
| `runs/ec/ec_t16_*` | T16 sign-test arms (400 epochs) |
| `runs/ec/ec_t17_*` | **T17: the main-network candidates** (N2000, 1000 epochs, seeds 123–125) |
| `runs/ec/ec_t20_*` | T20: deck forcing / walkers + noise floor (N2000, 1000 epochs, seeds 123–125) |
| `runs/paper/` | the archived paper's 14 networks (shared record) and its trainer-check run: history and gate references |
| `runs/repro/` | retrainings made by `tools/verify_training.py` to prove reproduction |

## Seed registry

Never reuse a base. Records are `np.random.SeedSequence([base, structure_index])` unless noted.

| seed | use |
|---|---|
| 20260507 | population parameters (generator, 5500 structures) |
| 12345 | **the archived paper's single excitation record, shared by every structure** (history only) |
| 20261123 | fresh-test parameters (T5); T5 report bootstraps (`c_report.BOOT_SEED`); T4 / T6 random layouts (archive) |
| 20261124 | `fresh_test`'s record: one int seed shared by its 500 structures |
| 20261125 | pool 0 and validation records (T11, T13, T17) |
| 20261126 | test records (`paper_varexc`, `paper_varexc_T120`) |
| 20261127 | real-bridge FE-twin records, T11 (`[base, LOOCV position, k]`) |
| 20261130 | per-epoch record draw (`default_rng([20261130, epoch])`) |
| 20261131–33 | pools 1–3 (T13, T17) |
| 20261140 | T18 FE-twin forces (`[base, bridge, k]`) |
| 20261150 | T19 FE-twin records (`[base, bridge, k]`) |
| 20261160–63, 20261169 | T20 deck forcing: pools 0–3, test |
| 20261170–73, 20261179 | T20 walkers: pools 0–3, test |
| 20261189 | T20 noise-fit noise (`[base, bridge, excitation]`) |

## Folder sizes and hashes

Hashed on 2026-11-24 after the reproduction runs: **51,774 files, 48.2 GB**, in 100 folders. Per folder (file
count, bytes, folder hash): [data_summary.csv](data_summary.csv). Per file: [DATA_MANIFEST.tsv.gz](DATA_MANIFEST.tsv.gz).
The derived folders and the gate runs (`runs/ec/*_gates`) were rewritten by the reproduction itself, so their
hashes are those of this repository's runs. `runs/repro/` (the retrainings) is not hashed.
