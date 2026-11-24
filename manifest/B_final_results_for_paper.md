# Manifest: `Final_results_for_paper/` (excluding `EC_comments/`)

Scope: `Final_results_for_paper/{README.md, pipeline/, data/, runs/}`. Read-only scan, 2026-11-24, repo HEAD `9c1f5da` (Phase P).
Working tree: only `pipeline/s10_figures.py` is modified and uncommitted (+12 lines: a `mean_markers()` helper used by fig7, fig10 and fig12).

**Notation.** *record* says whether the content depends on the shared excitation record: SEED 12345, re-seeded for every structure in `simulate_uniform_inertial_u3_white_noise.py:62/344`. *field*: `none` means no real-bridge inputs; `post-P` means `data/field_inputs_v2` (the SSI sensor order); `pre-P` means the legacy raw order (`field_modal_filtering_outputs/`, `experiments/E5_real_transfer/phase3_corrected/`).
*tracked* refers to git. `data/`, `runs/`, `*.npz`, `*.png` and `*.pdf` are gitignored.

**Main fact for the restructure.** The structure parameters, FE labels, graphs and splits do not depend on the record:
- The parameter RNG is `SeedSequence(20260507).spawn(4)`, one stream per Ng. It is separate from the simulator's RNG.
- All 1224 rejected draws in `failures.log` failed before simulation: 1154 in the MMI matcher, 70 at the MAC ≥ 0.85 gate.
- The simulator uses its RNG once (`ag_inputs`, line 345). Labels come from `PhiN` (lines 382–401).

So the 5500 structures, their labels and the splits carry over. Only five arrays per sample need regenerating.

---

## 1. Inventory

### 1a. Scripts and docs

| path | kind | category | record | field | size | tracked | reason / what the new pipeline needs |
|---|---|---|---|---|---|---|---|
| `Final_results_for_paper/README.md` | doc | KEEP-CHANGE | no (describes the old protocol) | mentions pre-P | 10 KB | yes | Its layout, run order and reproducibility notes still apply. Rewrite the decisions table (record per structure, K records, sign-invariant loss, 802-d edges), drop narrowband and §5, and fix "What it reuses": it predates Phase P and still names `experiments/E5_real_transfer/phase3_corrected/` as the real-bridge inputs. |
| `pipeline/fr_common.py` | module (constants and paths) | KEEP-CHANGE | partly (paths to the shared-record pool) | defines post-P and pre-P paths | 8 KB | yes | **Keep as is:** the population design (NG_VALUES, N_TOTAL, POOL/TEST per Ng, SWEEP_N, TRAIN_FRAC, SPLIT_SEED, GENERATOR_SEED, DY_MAX_NG3PLUS_MM, SELECTED_MAC_GATE, SEED, MODE_NAMES, LOOCV), the Phase P block (`FIELD_SENSOR_SWAP`, `field_sensor_order`, `field_channel_reorder`, `FIELD_RAW_FILES`), `REPO` two parents up, `run_dir`, `sweep_run_name`, `keep_awake`. **Change:** `POP_BROADBAND` to the re-simulated pool (or add a record-pool layout); add the excitation-seed bases and K; redefine `ABLATION_GRID` / `ABLATION_EXTRA` for the new arms; drop `POP_NARROWBAND`; check `MAIN_N` (chosen by the shared-record T1 ≥ 0.9 rule). Decide whether `LEGACY_FIELD_OUTPUTS` / `LEGACY_REAL_FEATURES` point into a copy or into the old repo. `FIELD_RAW_MAT_DIR = REPO.parents[2]/...` resolves to the same folder from the sibling `Linkspan_GNN_OMA_Diff_Exc`, which sits at the same depth. |
| `pipeline/fr_edge_modes.py` | module (edge-mode patch) | KEEP | no | none | 4 KB | yes | `orientation()` (cos, sin of `edge_attr_geom[:,1]`) and `install()` are the base of EC's 802-d coherency + orientation mode (`p_common.install_cohorient`) and of the ablation modes. Leave it unchanged. It is the natural place to add `cpsd_full_orient`, which today lives in EC `p_common.py`. |
| `pipeline/fr_status.py` | tool | KEEP | no | none | 1.5 KB | yes | Reads `fr_train.jobs()` and the resume states. It works unchanged with a new job registry. |
| `pipeline/fr_train.py` | trainer and job registry | KEEP-CHANGE | partly: the code is clean, but `jobs()` names shared-record runs and defaults to `C.POP_BROADBAND` | none | 22 KB | yes | **Keep:** the resumable loop (bit-identical resume on CPU), the explicit manifests and the best-validation rule. **Change `jobs()`:** new `edge_feature_mode` (`cpsd_full_orient`), the sign-invariant loss, and K record pools per training structure. EC currently gets these by wrapping `fr_train.main()` unchanged: `p_common` patches `gs._compute_multitask_loss` and `eg._resolve_edge_attr_array`; `k_train` patches `eg.Phase2AEdgeDataset` and `dgl.dataloading.GraphDataLoader`, and replaces `fr_train.jobs`. Either fold these into fr_train or keep fr_train's call-time look-ups (below) so the EC wrappers keep working. |
| `pipeline/s0_generate_population.py` | stage: population generation | KEEP-CHANGE (or replaced by a re-simulation stage) | **yes**, for features: `gen.main()` simulates with SEED 12345 | none | 9.5 KB | yes | **Keep:** `install_patches()` (spacing up to 3000 mm via exact-string `_rebuild`, and the MAC ≥ 0.85 gate), the Agg backend, and the empty non-TTY stdin. EC `c_gen_fresh_test.py` imports it. **Change:** either add an in-memory per-structure excitation seed (`SIM.SEED = SeedSequence([base, sample_id])` before each `assemble_phase1_sample_payload`), or stop re-running s0 and add a "re-simulate from stored samples" stage (EC `h_common` / `k_prepare`). Either way the FE eigensolve runs again; see §2 Q2. |
| `pipeline/s1_verify_population.py` | stage: population gates G1–G5 | KEEP-CHANGE | partly: `response_content` uses `x_psd`; G1–G5 do not | none (real geometry only) | 12 KB | yes | **Keep:** G1 (counts), G3 (MAC gate), G4 (envelope), G5 (Ng = 2 equal to the old 10k; `OLD_10K = REPO.parent/Linkspan_GNN_FineTuning/...` still resolves from the sibling repo), and the coverage and rejection tables. **Change:** G2 pairs against narrowband, so drop it or pair the re-simulated pool against the stored one. Add gates: non-record keys byte-identical to the paper samples; no two structures, and no train and test structure, share a record. |
| `pipeline/s2_make_splits.py` | stage: splits | KEEP | no | none | 4 KB | yes | Splits depend only on sorted file names and `SeedSequence([123, N, Ng])`. Copy as is. (EC-27 asks for fully nested train/val; EC `t_make_nested_splits.py` does that. See the open questions.) |
| `pipeline/s3_check_trainer.py` | stage: proof that the trainer equals the library loop | KEEP-CHANGE | no (a loop test), but reads the old pool `REPO/population_dataset_modes4/samples_phase2_bandpass` | none | 8 KB | yes | Proves that fr_train is the library's `train_one_run` and that resume is exact. The new trainer (sign-invariant loss, K-record augmentation) needs new proofs: with nothing switched on it equals this loop; the sign-invariant loss equals a brute-force minimum; augmentation with K = 1 equals plain training. EC `p_train --check` and `k_train` "gate" already do these. Repoint `OLD_POOL`, which will not exist in the new repo. |
| `pipeline/s3_train_all.py` | stage: training queue | KEEP-CHANGE | no (it is a queue) | none | 6 KB | yes | The queue, retry and log-slug logic is reusable. Change `ORDER` (hard-coded run names) to the new jobs; 2 threads per process. |
| `pipeline/s4_indomain.py` | stage: in-domain test scoring | KEEP-CHANGE | yes (scores shared-record networks on shared-record test inputs) | none | 8.6 KB | yes | **Keep:** `evaluate` / `summarise` / `bootstrap_ci`. EC `c_*` and `t_*` import `bootstrap_ci`, `evaluate`, `summarise`. **Change:** install the 802-d mode (the sign-invariant loss does not touch evaluation: MAC is sign-blind); read the test set's new-record features, not `C.POP_BROADBAND`; decide whether to score one test record or several. |
| `pipeline/s5_zeroshot.py` | stage: zero-shot on 9 bridges, transfer vs N | KEEP-CHANGE | network only (shared-record checkpoints); inputs are field data | post-P (main); pre-P (E2 gate) | 12 KB | yes | **Keep:** `load_structures`, `evaluate`, `mac`, the feature-convention gate and the input-dependence gate (EC imports `load_structures`, `evaluate`, `mac`, `GRAPH_ROOT`, `MODES`). **Change:** add orientation edges from `phase3_real_graphs/<S>/graph_package.npz` `edge_attr_geom`, as EC `p_eval` / `h_real` do. The E2-faithfulness gate needs `experiments/E2_training_distribution/runs_ep500/bandpass_seed123` and `stage2_zeroshot.csv` plus the pre-P features: drop it or point it at the old repo. The training-reference gate reads `C.POP_BROADBAND`. |
| `pipeline/s5a_field_inputs.py` | stage: post-P real-bridge inputs | KEEP | no (field recordings only) | writes post-P; reads pre-P for the identity gate | 12.5 KB | yes | Correct and independent of the record. To re-run it needs: the raw `.mat` files outside the repo (`FIELD_RAW_MAT_DIR`, 31 MB); `process_field_modal_filtering.py` (root); `experiments/E5_real_transfer/build_phase3_features_corrected.py`; **legacy** `field_modal_filtering_outputs/<S>/logs/{structure_summary,settings_used,mode_assignment_used}.json` and legacy npz for the identity gate; legacy `E5 phase3_corrected` for the identity gate; and `population_dataset_modes4/.../sample_0000_Ng2_phase1_gnn.npz` (the E5 builder's `TRAIN_GRID_SAMPLE`, read only for the 400-bin grid). It does not need re-running if `data/field_inputs_v2` is copied. |
| `pipeline/s6_decomposition.py` | stage: ceiling / propagation / zero-shot decomposition | KEEP-CHANGE | **yes**: `simulate_real()` simulates the FE twins with SEED 12345, and the network is shared-record | post-P (zero-shot variants) | 16 KB | yes | **Keep:** the structure of the experiment, the FE solve, matcher and graph (`_build_params`, `MAC_THR` Eidsdal 0.70, `assemble_deterministic_phase2a_payload` for labels and geometry only) and the checks. EC imports many symbols (§2 Q4). **Change:** give `simulate_real` a per-bridge record (EC T11: base 20261127, index = LOOCV position), possibly several records; make `infer` build 802-d edges (FE graph: orientation from the payload's `edge_attr_geom`; hybrid graph: from `graph_package`). Needs `experiments/E6_controls/task2_raw/<S>.npz` (`fe_target`) for the check. |
| `pipeline/s7_finetune.py` | stage: nested LOOCV fine-tuning | KEEP-CHANGE | network only (shared-record start point) | post-P (main); pre-P (E2 gate) | 13.5 KB | yes | **Keep:** the nested protocol, optimiser groups and schedule (EC T3/T4/T8 import `load_datasets`, `run_folds`, `_predict`, `_sparse_loss`, constants). **Change:** add 802-d edges for real graphs. Decide the fine-tuning loss: `_sparse_loss` is MSE against **signed** SSI shapes, which a network pre-trained with a sign-invariant loss may fight. The E2 gate needs the E2 run, `stage3_finetune.csv` and the pre-P features: drop it or point it at the old repo. Reads `E6_RAW` `fe_target`. |
| `pipeline/s8_sensor_sweep.py` | stage: sensor-count sweep (synthetic) | KEEP-CHANGE | **yes**: re-simulates test structures with the module SEED ("the realisation is the stored one") and gates against stored features | none | 5.5 KB | yes | Uses E14 code (`experiments/E14_corner_sensors/stage1_corner_propagation`, `stage2_n10000_layouts`, `stage3_sensor_sweep`). **Change:** set each test structure's own seed in the worker before `resimulate`; append orientation to the propagated 800-d coherency in E14's `propagated_inputs` / `run_sample`; compare the re-simulation gate (`x_psd_resim_maxdiff`) against the new test features. EC `sp_eval_sweep` / `d_t6_*` already gate against s8's CSVs. |
| `pipeline/s9_report.py` | stage: PAPER_RESULTS.md generator | KEEP-CHANGE | the code no; its prose describes shared-record results | reads post-P tables | 36 KB | yes | The approach (every number read from the tables, nothing transcribed) and the section skeleton are reusable. Rewrite the prose and conventions: narrowband pairing in §1, the E2-proof sentences, the ablation claims in §3, the MAIN_N rule text, and a new excitation-protocol section. |
| `pipeline/s10_figures.py` | stage: figures | KEEP-CHANGE | no (draws from tables) | reads post-P tables | 70 KB | yes (with uncommitted changes) | Reusable. It imports `experiments/E2_training_distribution/figures/{figstyle, plot_loss_curves, plot_synthetic_performance, plot_representative_modeshapes, plot_graph_topology}`: copy them, since the author wants that style. **Change:** the fig10/fig11 grid follows the new `ABLATION_GRID`; fig4's loss label if the loss changes. Decide on the uncommitted `mean_markers` change (commit it in the old repo first, or copy the working-tree file). fig0, fig9 and fig16 do not depend on the record. |
| `pipeline/s11_ablation_cost.py` | stage: ablation cost and contrasts | UNSURE | the code no; the arms are shared-record | none | 9.3 KB | yes | Decided by whether the new paper keeps an architecture ablation. If yes, KEEP-CHANGE: new grid, re-time each arm alone, keep the paired-contrast code (it imports `s4.bootstrap_ci`). If no, ARCHIVE. |
| `pipeline/__pycache__/` | bytecode | ARCHIVE | no | none | small | no | Do not copy. |

### 1b. `pipeline/results/` (tables tracked; `figures/` gitignored)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| `results/PAPER_RESULTS.md` | report | ARCHIVE | yes | post-P | 30 KB | yes | The draft paper's numbers (shared-record networks). Cite it by old-repo path when the new report says what moved. |
| `results/s1_population_checks.json` | gates | KEEP | partly (only `response_content`, the near-target energy of `x_psd`) | none | 2 KB | yes | Certifies the 5500 structures that are reused (G1–G5 PASS). |
| `results/s1_population_summary.csv` | table | KEEP | no (parameters and FE frequencies) | none | 8 KB | yes | Population parameter and frequency distributions, pool and test. It is the paper's Table 2 source. |
| `results/s1_real_bridge_coverage.csv` | table | KEEP | no | none (geometry) | 0.7 KB | yes | Real bridges against the population envelope. |
| `results/s1_rejections.csv` | table | KEEP | no (matcher and gate rejections, before simulation) | none | 2.5 KB | yes | — |
| `results/s2_split_summary.csv` | table | KEEP | no | none | 0.6 KB | yes | Balance, nesting and disjointness of the reused splits. |
| `results/s3_trainer_check.json` | proof | EVIDENCE | no in meaning (it ran on the old `population_dataset_modes4` pool) | none | 1 KB | yes | Proves fr_train equals the library loop and that resume is exact. Still true for an unchanged fr_train; the new trainer needs its own proof. |
| `results/s4_indomain_{per_sample,summary,by_ng}.csv` | tables | ARCHIVE | yes | none | 5.7 MB + 12 KB + 46 KB | yes | Shared-record networks on shared-record test inputs (0.933 etc.). |
| `results/s5_zeroshot.csv`, `s5_transfer_vs_n.csv`, `s5_gates.json` | tables, gates | ARCHIVE | yes (networks) | post-P (the gate's E2 copy is pre-P) | 7 KB, 58 KB, 3 KB | yes | — |
| `results/s5a_field_inputs_checks.json` | gates | KEEP | no | post-P | 58 KB | yes | Provenance of `data/field_inputs_v2` (serials, identity, swap). Copy it with the data. |
| `results/s6_decomposition.csv`, `s6_decomposition_summary.csv`, `s6_checks.json` | tables | ARCHIVE | yes (FE twins and network) | post-P | 22 KB, 2 KB, 3 KB | yes | — |
| `results/s7_finetune.csv`, `s7_checks.json` | tables | ARCHIVE | yes (network) | post-P (the gate is pre-P) | 4 KB, 0.3 KB | yes | — |
| `results/s8_sensor_sweep_{all,outer}.csv`, `s8_diagnostics_{all,outer}.csv` | tables | ARCHIVE | yes | none | 2 × 1.1 MB, 2 × 0.2 MB | yes | — |
| `results/s11_ablation_cost.csv`, `s11_ablation_contrasts.csv`, `s11_main_cost.csv` | tables | ARCHIVE | yes (arms); timings and parameter counts are not | none | 3 KB, 1.5 KB, 0.4 KB | yes | Shared-record arms. Per-epoch cost and parameter counts carry over only for unchanged architectures (the 802-d input changes them). |
| `results/figures/fig0_{frequencies,shape_stats,distributions}_N{2000,5500}.{png,pdf}` | figures | KEEP | no (`y_freq_hz` and `y_mode_shape_signed` labels only) | none | part of 23 MB | no | s10 regenerates them; they stay identical while the labels and the N = 2000 split are unchanged. (The fig0 palette mismatch is still unresolved.) |
| `results/figures/fig9_sensor_layouts.*` | figure | KEEP | no (sensor node sets and geometry) | none | small | no | Drawn from s8's raw files, which hold layouts, coordinates and predictions; it plots only the layouts. |
| `results/figures/fig16_real_graphs.*` | figure | KEEP | no | none (graphs) | small | no | Drawn from `phase3_real_graphs/`. |
| `results/figures/` fig1–fig8, fig10–fig15, `fig13_by_bridge/` | figures | ARCHIVE | yes | post-P (fig5, 6, 12–15) | rest of 23 MB | no | Network results. |

### 1c. `data/` (gitignored)

| path | kind | category | record | field | size | tracked | reason / what the new pipeline needs |
|---|---|---|---|---|---|---|---|
| `data/pop_broadband/samples_phase2_bandpass/` (5500 × `sample_XXXX_NgG_phase1_gnn.npz`, ~1.35 MB each) | training/test samples | RESIMULATE | partly: about 81 % of the bytes are record features | none | 7.4 GB | no | **Reusable, inside each npz:** parameters, matcher record, graph, FE labels (signed and abs shapes, frequencies), sign convention, sensor-domain modes, frequency grid. **Regenerate:** `x_psd`, `x_psd_row_scale`, `edge_attr_cpsd`, `edge_attr_cpsd_resonance`, `edge_attr_cpsd_full` (EC `h_common.EXC_KEYS`). There are no time series and no full FE modes or mass matrix, so re-simulation re-solves FE from `params_json`. See §2 Q2. |
| `data/pop_broadband/samples/` (5500 × `sample_XXXX_NgG.npz`, 113 KB each) | FE-only samples | ARCHIVE (redundant) | no | none | 598 MB | no | Parameters, `phi_u3_4` at the matcher's reference points, frequencies, MAC, matcher JSON. Nothing reads it; everything in it is also in the phase-2 npz. |
| `data/pop_broadband/{summary.csv, filtered_summary.csv, outlier_report.json, failures.log, generation_config.json}` | generator metadata | KEEP | no (parameters, FE frequencies, MAC, rejections) | none | ~5.8 MB | no | s1 reads `failures.log`; EC `c_fresh_checks` reads `generation_config.json`. `filtered_summary.csv` is the generator's outlier filter, never applied (s10 docstring). |
| `data/pop_broadband/plots/` | — | ARCHIVE | — | — | empty | no | — |
| `data/pop_narrowband/` (`samples/`, `samples_phase2/`, metadata) | paired narrowband population | ARCHIVE | yes (features) | none | 8.0 GB | no | Never trained on; §5 dropped. Its structures and labels duplicate the broadband ones (s1 G2). |
| `data/splits/` (`pool.txt`, `test.txt`, `N{100..5000}_{train,val}.txt`) | split manifests | KEEP | no | none | 525 KB | no | Deterministic output of s2. EC uses `N600_*`, `N2000_*` and `test.txt` throughout (T11/T13/T16/T17). |
| `data/field_inputs_v2/phase3_corrected/<S>/phase3_dataset_corrected.npz` (9) | real-bridge network inputs and labels | KEEP | no | **post-P** | 14 MB | no | `C.REAL_FEATURES`, read by s5, s6, s7 and EC. 17 keys: `x_psd`, `edge_attr_cpsd_full`, `edge_index`, `sensor_node_indices`, `y_mode_shape_ssi`, `y_freq_hz`, `fe_target_*`, `n_fe_nodes`, `ssi_mac_vs_fe`, ... It has no `edge_attr_geom`: orientation comes from `phase3_real_graphs/<S>/graph_package.npz`. |
| `data/field_inputs_v2/field_modal_filtering_outputs/<S>/` (+ `manual_mode_assignments.json`, `run_summary.json`) | field records after the reorder, logs, plots | KEEP | no | **post-P** | 174 MB (about 2 MB/bridge of PNGs) | no | Source of `phase3_corrected` (reads `raw_u3_timeseries_clean` and `freq_hz`). Its own `logs/mode_assignment_used.json` makes it self-contained. The PNG plots can be left behind. |

### 1d. `runs/` (gitignored)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| `runs/edgegat_N2000/` | the paper's main network | EVIDENCE | yes | post-P (`decomposition_raw`, `finetune_raw`, rewritten 2026-11-23) | 12 MB (checkpoints 2.1 MB) | no | The "before" network of the record finding (T5 0.934 → 0.377; T15 mirror test). Copy `checkpoints/best_model.pt` and the config only if the new repo should reproduce that before/after itself; otherwise leave it in the old repo. |
| `runs/edgegat_N{100,200,400,600,800,1000,5000}/` | sweep networks | ARCHIVE | yes | none | 6.3–11 MB each (~49 MB) | no | `edgegat_N600` also holds stale `sensor_sweep_raw_*` from 2026-09-25, when N = 600 was the main network. |
| `runs/{graphsage_noedge,graphsage_cpsd8,graphsage_cpsdfull,graphsage_geomorient}_signed_N600/`, `edgegat_{geomorient,noedge}_N600/` | ablation arms | ARCHIVE | yes | none | 5.9–8 MB each (~41 MB) | no | — |
| `runs/_trainer_check/` | s3a scratch | ARCHIVE | yes (old pool) | none | 7.6 MB | no | — |

**Counts (50 rows):** KEEP 17 · KEEP-CHANGE 14 · RESIMULATE 1 · EVIDENCE 2 · ARCHIVE 15 · UNSURE 1.

---

## 2. Answers to the specific questions

### Q1. Stage by stage

| stage | does | reads | writes | record / field | new pipeline |
|---|---|---|---|---|---|
| s0 | Generates 5500 structures (1375 per Ng, seed 20260507) through the production generator, with three in-memory patches (dy ≤ 3000 for Ng ≥ 3, the validator, the selected-MAC gate); arm broadband or narrowband | generator, FE, matcher, simulator, `ref_modes_*.npz` | `data/pop_<arm>/` | features: **yes**; structures and labels: no | Add a per-structure seed, or replace s0 by a re-simulation stage from the stored samples (EC pattern) |
| s1 | Gates G1–G5; parameter, coverage and rejection tables | both populations, `failures.log`, `build_phase3_graphs.STRUCTURES`, the old 10k (`../Linkspan_GNN_FineTuning`) | `s1_*.csv/json` | partly (`response_content`); field none | Drop the narrowband pairing; add record gates |
| s2 | Pool, test set, balanced nested sweep subsets (85/15 per Ng) | sample names | `data/splits/*.txt`, `s2_split_summary.csv` | no | Keep (or adopt EC-27's fully nested splits) |
| s3a (`s3_check_trainer`) | Proves fr_train equals `train_one_run`, and that resume is exact | the old `population_dataset_modes4` pool | `s3_trainer_check.json`, `runs/_trainer_check` | no in meaning | New proofs for the sign-invariant loss and augmentation |
| s3 (`s3_train_all` + `fr_train`) | 14 networks, 5000 epochs, CPU, 2 threads, best validation | `C.POP_BROADBAND`, `splits/` | `runs/<job>/` | **yes** | New jobs: 802-d edges, sign-invariant loss, K records; retrain |
| s4 | In-domain scoring on 500 test structures (best and last checkpoints), bootstrap CI | runs, test features | `s4_*.csv`, `runs/<r>/eval_*/predictions.npz` | **yes** | Score on new-record test features; 802-d mode |
| s5a | Field inputs with loggers 131/133 reordered (Phase P) | raw `.mat` (outside the repo), legacy field logs and npz, E5 builder, `phase3_real_graphs` | `data/field_inputs_v2/`, `s5a_field_inputs_checks.json` | no / writes post-P | Keep (or just copy its outputs) |
| s5 | Zero-shot on 9 bridges (main and all sweep networks), E2-copy gate, feature-convention and input-dependence gates | `REAL_FEATURES`, `E6/task2_raw` `fe_target`, runs, E2 run and CSV (gate) | `s5_*.csv/json` | network **yes**; field post-P | 802-d real edges; drop or repoint the E2 gate |
| s6 | Ceiling / propagation-only / zero-shot decomposition on FE twins and recordings | FE twin (`infer_phase2a_deterministic_edgegat`), `REAL_FEATURES`, `phase3_real_graphs`, `E6_RAW`, the first pool sample for the grid | `s6_*.csv/json`, `runs/<main>/decomposition_raw/` | FE-twin simulation **yes**; field post-P | Per-bridge records for the twins; 802-d edges |
| s7 | Nested LOOCV fine-tuning (full), E2-copy gate | `REAL_FEATURES`, `E6_RAW`, E2 run and CSV | `s7_*.csv/json`, `runs/<main>/finetune_raw/` | network **yes**; field post-P | 802-d edges; a fine-tuning loss compatible with the sign-invariant loss; E2 gate |
| s8 | Sensor sweep (outer / every girder, n = 1–12) through re-simulation, propagation and the network | test samples, E14 code | `s8_*.csv`, `runs/<main>/sensor_sweep_raw_*` | **yes** (re-simulates the shared record) | Test structure's own seed in the worker; 802-d edges |
| s9 | PAPER_RESULTS.md from the tables | all `results/*` | `PAPER_RESULTS.md` | yes (content) | Reuse the skeleton; rewrite the prose |
| s10 | fig0–fig16 | tables, `C.POP_BROADBAND` labels (fig0), predictions, `phase3_real_graphs`, E2 figure modules | `results/figures/` | mixed | Reuse; new grid |
| s11 | Ablation validation loss, parameters, timing alone, contrasts | runs, s4 per-sample | `s11_*.csv` | yes | Only if an ablation is kept |

### Q2. What `data/` stores per sample, and what can be reused

Each `samples_phase2_bandpass/sample_XXXX_NgG_phase1_gnn.npz` (keys listed from `sample_4124_Ng4`; written by `generate_population_dataset_modes4.assemble_phase1_sample_payload`) holds the following.

**Record-free (reuse as is).**
- Identity: `schema_version`, `sample_index` / `sample_id`, `Ng`, `Nx`, `n_graph_nodes`, `n_graph_edges_directed`, `n_freq_bins`.
- Structure and matcher: `params_json` (every sampled parameter), `matcher_config_json`, `matcher_result_json`, `reference_file`.
- Graph: `graph_node_ids`, `coords_xy_mm`, `coords_xyz0_mm`, `edge_index`, `edge_index_undirected`, `edge_length_mm`, `edge_theta_rad`, `edge_attr_geom`, `group_include_*`.
- Grid: `freq_bins_hz`, `zeta`.
- Labels: `target_mode_names`, `y_mode_shape_abs` / `_scale`, `y_mode_shape_signed` / `_scale`, `y_freq_hz`, `matched_mode_indices` / `_freq_hz` / `_mac` / `_composite_score` / `_ambiguity_flag`, `selected_mode_indices_sensor_domain`.
- Sensor-domain modes: `phi_u3_sensor_4` / `_signed` / `_signed_norm`.
- Sign convention: `sign_*` (13 keys), `sign_reference_template_sensor_4`, `S4`.

**Record-dependent (regenerate):** `x_psd` (N × 400, float64), `x_psd_row_scale`, `edge_attr_cpsd` (E × 8), `edge_attr_cpsd_resonance` (E × 8), `edge_attr_cpsd_full` (E × 800, float32). Their `*_feature_names`, `*_description` and `*_shape` metadata are constant. In the sample checked, about 1.10 of 1.35 MB is record-dependent.

**Not stored:**
- raw or filtered acceleration time series;
- the full FE solution (`phi_full` for all 15 modes, `freq_all_hz`, `M_full`, the model).

So re-simulating needs the FE eigensolve run again from `params_json`: `build_solver_params(sampled, n_candidate_modes=15)` and `run_solver(...)`. ARPACK has no fixed start vector, so the result reproduces to about 1e-7. Then `simulate_white_noise_from_modal_model(..., use_all_modes=True, bandpass_hz=(1.5, 20))` runs with a new seed. The broadband response uses all 15 modes, so the matcher does not need to run again.

EC (`h_common`, `k_prepare`, `sp_prepare`) does exactly this:
- it writes a new npz per record with only `EXC_KEYS` replaced and every other key copied byte-for-byte;
- it checks that the re-simulation with SEED 12345 reproduces the stored features to ≤ 1e-5 (≤ 1e-7 measured).

Running s0 again with a per-structure seed would also reproduce the same 5500 structures, in the same order: the parameter RNG is separate, and every rejection happens before simulation. The labels would then come back to about 1e-7, not byte-identical.

The other files:
- `samples/sample_XXXX_NgG.npz` holds parameters, matcher reference-point modes, frequencies and MAC only. It is redundant and unread.
- **Splits** are in `data/splits/*.txt` (gitignored, 525 KB; deterministic from s2). Each run folder also has copies (`runs/<run>/{train,val,test}_split.txt`, `splits.json`). The tracked summary is `results/s2_split_summary.csv`.

### Q3. Field inputs

The post-Phase-P inputs are the 9 files `Final_results_for_paper/data/field_inputs_v2/phase3_corrected/<S>/phase3_dataset_corrected.npz` (= `fr_common.REAL_FEATURES`).
- s5, s6 (its two zero-shot variants) and s7 read them. s8 reads none: it is synthetic only.
- s5a builds them from `data/field_inputs_v2/field_modal_filtering_outputs/<S>/data/field_modal_filtering_outputs.npz` (= `FIELD_OUTPUTS`).
- **They do not depend on the shared record.** They come from the field recordings, the E5 builder (bandpass, Welch, propagation, interpolation onto the 400-bin grid, row normalisation, coherency), and the labels and graph from `phase3_real_graphs/<S>/phase3_dataset.npz`. No simulator is called.

The new pipeline also needs these, all outside my area and all record-free for the parts used:
- `phase3_real_graphs/<S>/graph_package.npz`: hybrid graph, `sensor_node_indices`, `nearest_fe_node_idx`, and `edge_attr_geom`, the source of the 802-d orientation on real bridges;
- `experiments/E6_controls/task2_raw/<S>.npz`: only the key `fe_target` (label-corrected FE target). Its `pred_*` keys are legacy network outputs: ignore them;
- `infer_phase2a_deterministic_edgegat.REAL_STRUCTURE_PARAMS` for s6's FE twins.

What does depend on the record: **s6's `ceiling` and `prop_*` variants.** They simulate the FE twins with SEED 12345, so they must be simulated again.

Pre-P inputs are still read in three places, all as identity or faithfulness gates:
- s5's and s7's E2-copy gates (`LEGACY_REAL_FEATURES`);
- s5a's identity gate (`LEGACY_FIELD_OUTPUTS`, `LEGACY_REAL_FEATURES`).

### Q4. Which of my files `EC_comments/pipeline/*.py` imports (dependencies of the new pipeline)

- **`fr_common`**: directly in `ec_common` (re-exported to every EC script), `e_t7_*`, `e_t8_*`, `t_*`. Constants used: `NG_VALUES`, `MODE_NAMES`, `POP_BROADBAND`(+`_ROOT`), `RUNS`, `RESULTS`, `LOOCV`, `SPLITS`, `REPO`, `PIPELINE`, `MAIN_N`, `sweep_run_name`, `SEED`, `EPOCHS`, `SWEEP_N`, `TRAIN_FRAC`, `SPLIT_SEED`, `GENERATOR_SEED`, `POOL_PER_NG`, `N_TOTAL`, `ABLATION_{N,GRID,LAYERS,FEATURES,EXTRA}`, `REAL_FEATURES`, `FIELD_OUTPUTS`, `keep_awake`.
- **`fr_train`**: `jobs`, `Job`, `main`, `manifest_paths`. Used by `h_train`, `k_train`, `q_train`, `t_train`, `t_jobs`, `t_eval_indomain`, `c_rescore`, `fr_status`. The EC wrappers replace `fr_train.jobs` and call `fr_train.main()` with `--job --threads --pool --train-list --val-list --run-root --epochs`.
- **`fr_edge_modes`**: `install`, `orientation`. Used by `p_common` (802-d `cpsd_full_orient`), `p_eval`, `p_train`, `h_real`, `t_edge_modes`, `c_rescore`.
- **`s0_generate_population`**: `install_patches`, `gen`, `git_head` (`c_gen_fresh_test`).
- **`s4_indomain`**: `bootstrap_ci` (`c_ec23_metrics`, `c_freq_baselines`, `c_report`, `t_report`); `evaluate`, `summarise`, `jobs`, `main` (`c_rescore`, `t_eval_indomain`).
- **`s5_zeroshot`**: `load_structures`, `evaluate`, `mac`, `GRAPH_ROOT`, `MODES`, `STRUCTURES`. Used by `d_t6_field_resample`, `g_t10_*`, `h_real`, `p_eval`, `sp_eval_real`, `t_eval_main`.
- **`s6_decomposition`**: `simulate_real`, `infer`, `mac`, `main`, `_build_params`, `assemble_deterministic_phase2a_payload`, `load_graph_package`, `corrected_node_features` / `corrected_edge_features`, `propagate`, `build_cpsd_full_edge_payload`, `run_solver`, `simulate_white_noise_from_modal_model`, `VARIANTS`, `MAC_THR`, `CORRECTED`, `GRAPH_ROOT`, `PHASE1_FREQ_BINS`, `PHASE1_ZETA`, `WELCH_*`, `REF_X_FRACS`, `REF_SENSOR_TOL_X`. Used by `d_t6_field_control`, `h_real`, `r_excitation_realism`, `sp_eval_real`, `t_eval_main`.
- **`s7_finetune`**: `load_datasets`, `run_folds`, `_predict`, `_mac`, `_mean_sensor_mac`, `_sparse_loss`, `STRUCTURES`, `MODES`, `CORRECTED`, `E6_RAW`, and the hyperparameter constants. Used by `e_t7_stored_outputs`, `e_t8_finetune_variants`, `sp_eval_finetune`, `t_eval_main`.
- **Not imported, but read as files:**
  - s8's `s8_sensor_sweep_{outer,all}.csv` (gates in `d_t6_*`, `sp_eval_sweep`);
  - s11's `s11_ablation_cost.csv` / `_contrasts.csv` (`c_report`, `t_gates`, `t_report`);
  - `s4_indomain_per_sample.csv` (`e_t7_*`, `c_freq_baselines`);
  - `data/pop_broadband` samples (re-simulation source in `h_prepare`, `k_prepare`, `sp_prepare`, `c_fdd`, `p_eval`, ...);
  - `data/splits/` (`N600_*`, `N2000_*`, `test.txt`);
  - `runs/edgegat_N2000` and the other paper runs (`c_rescore`, `e_t7`, `e_t8`, `d_t6_common`, `m_*`, `n_mirror_test`).
- s5a, s1, s2, s3, s9, s10 and s11 are not imported by EC.

### Q5. `results/`: paper-only versus record-free

- **Record-free facts (KEEP):**
  - `s1_population_summary.csv` (parameter and FE-frequency distributions; the source of Table 2);
  - `s1_real_bridge_coverage.csv`, `s1_rejections.csv`, `s1_population_checks.json` (except its `response_content` entry);
  - `s2_split_summary.csv`, `s5a_field_inputs_checks.json`;
  - figures fig0 (label distributions), fig9 (layouts), fig16 (real graphs).
- **Evidence:** `s3_trainer_check.json` (loop identity).
- **Paper-only (ARCHIVE):** `PAPER_RESULTS.md`, s4, s5, s6, s7, s8 and s11 tables, and figures fig1–fig8 and fig10–fig15.
- The SSI-versus-FE agreement (Table 8, `ssi_mac_vs_fe`, 0.856) is record-free but is not in my area. It lives in `experiments/E2_training_distribution/stage5_ssi_vs_fe.*` and in the `ssi_mac_vs_fe` key of the field features. Report §9 lists it as "carried over unchanged".

---

## 3. Dependencies the new pipeline must not break

1. **The library modules stay flat and unedited, and the pipeline patches them in memory.**
   - `s0._rebuild` substitutes exact strings in the generator source: `dy_high = min(2500.0, 12000.0 / (ng - 1))` and `if not (2000.0 <= dy <= 2500.0):`. It hard-fails if the source changes.
   - `fr_edge_modes.install` patches `gs._resolve_edge_attr_array`, `gs._edge_feature_dim_for_mode`, `eg._resolve_edge_attr_array` and `*.EDGE_FEATURE_MODES`.
   - EC patches `SIM.SEED` (the simulator reads its global at call time), `gs._compute_multitask_loss`, `eg.EdgeGATModalGNN.forward`, `eg.Phase2AEdgeDataset` and `dgl.dataloading.GraphDataLoader`.
   - If the new repo edits the simulator to take a seed instead, these patches and EC's re-simulation gates change with it.
2. **`fr_train.main()` must keep its call-time look-ups**, or EC's wrappers break: `jobs()`, `GraphDataLoader` imported inside `main`, `eg.Phase2AEdgeDataset`, and `gs._compute_multitask_loss`. It must also keep its CLI options `--pool --train-list --val-list --run-root --epochs --threads --stop-after --resume-every`.
3. **Sample naming and order.**
   - Names follow `sample_{id:04d}_Ng{g}_phase1_gnn.npz`; `NAME_RE` reads them in s1, s2 and s4.
   - Sorted name order is draw order within each Ng. s2 relies on this, and EC's record seeds (`SeedSequence([base, sample_index])`) use the 4-digit index.
   - The test set is the 125 structures per Ng after the first 1250.
4. **The split rule:** `SeedSequence([123, N, Ng])`, `n_val = floor(0.15 k + 0.5)`. Select sweep networks by run name (`edgegat_N{N}`), never by architecture.
5. **Path walks.**
   - `fr_common.REPO` is two parents up.
   - `FIELD_RAW_MAT_DIR = REPO.parents[2]/"System identification code"/...` and s1's `OLD_10K = REPO.parent/"Linkspan_GNN_FineTuning"/...` both still resolve from a sibling repo at the same depth.
   - These do **not** resolve unless copied or repointed: `REPO/population_dataset_modes4/...` (s3a `OLD_POOL`; the E5 builder's `TRAIN_GRID_SAMPLE`, used only for the frequency grid) and `REPO/field_modal_filtering_outputs/<S>/logs` (s5a).
6. **Files outside this folder that the stages import or read:**
   - root modules: `generate_population_dataset_modes4`, `simulate_uniform_inertial_u3_white_noise`, `EdgeGATConv_Modal_GNN`, `GraphSAGE_Modal_GNN`, `infer_phase2a_unseen_structure(_edgegat)`, `infer_phase2a_deterministic_edgegat`, `export_projected_graph`, `standalone_mode_matching_mmi`, `phase3_utils`, `build_phase3_graphs`, `grid_modal_Kinematic_MPC`, and `process_field_modal_filtering.py` (not in CLAUDE.md's closed set of 14);
   - `ref_modes_*.npz` (`REF_DIR = "."`, the matcher templates for s0 and the s6 twins);
   - `phase3_real_graphs/`, `SSI_modal_infor/` (via `phase3_dataset.npz`);
   - `experiments/E5_real_transfer/build_phase3_features_corrected.py`, `experiments/E6_controls/task2_raw/*.npz` (`fe_target`), `experiments/E14_corner_sensors/stage{1,2,3}*.py` (s8), `experiments/E2_training_distribution/figures/*.py` (s10);
   - E2 `runs_ep500/bandpass_seed123` plus `stage2_zeroshot.csv` / `stage3_finetune.csv` (s5 and s7 gates only);
   - `tools/run_stage.py`, which sets CWD to the repo root and hashes the stage's own folder (the reason scripts sit apart from `data/` and `runs/`), and the `FR` tag in `diff_against_reference` / `snapshot_baseline`.
7. **Reproducibility conventions.**
   - CPU only: the library picks `cuda:0` whenever CUDA is visible.
   - Threads: 2 for training; torch's default for s5 and s7 (bit-reproducibility of the E2 gates); `OMP=1` per s8 worker.
   - The s0 wrapper must keep the Agg backend and the empty non-TTY stdin, or the generator hangs.
8. **Feature conventions** (CLAUDE.md "Fixed conventions"): row-normalised Welch PSD with 400 bins; interleaved Re/Im coherency (800-d); 128 Hz field records interpolated onto the 200 Hz grid; bandpass 1.5–20 Hz; propagation over 50 iterations in order A. The 802-d edge is the 800-d coherency followed by (cos, sin), in that order (EC `p_common`).
9. **Phase P stays in one place:** `FIELD_SENSOR_SWAP = (6, 8)` and `field_channel_reorder` in `fr_common`. Never mix pre-P and post-P field numbers.

---

## 4. Open questions

1. **Re-simulate or regenerate?** Re-simulate from the stored samples, the EC pattern (labels byte-identical; only the five feature keys replaced; the FE solve runs again anyway). Or run s0 again with a per-structure seed (one pipeline, labels about 1e-7 off). The seed scheme also needs fixing. EC used bases 20261125 (train/val, record 0), 20261131–33 (records 1–3), 20261126 (test) and 20261127 (real FE twins, index = LOOCV position).
2. **Edit the simulator, or keep patching in memory?** The new repo could edit `simulate_uniform_inertial_u3_white_noise.py` to take a seed argument, or keep the in-memory `SIM.SEED` patch. CLAUDE.md rule 3 forbids edits here; the new repo can set its own rule.
3. **A known label defect is inherited by reusing the labels.** In all 568 two-girder balcony decks the matcher labels a torsion mode "B2" and a bending mode "T2". This is recorded in EC T5/T7/T9/T15 READMEs and project memory, and as "noticed" on the old pool in PROVENANCE Phase J, but not for this population (Phase K). T16 also found that the B1 label sign flips with Ng. Reuse the labels as they are, or relabel (a column swap that does not depend on the record or on FE) before training? Note that `s1` G3 cannot catch it.
4. **The s7 fine-tuning loss.** A network trained with the sign-invariant loss has an arbitrary sign per mode, while `_sparse_loss` is MSE against signed SSI shapes. A sign-invariant sparse loss, or sign alignment, needs deciding. (EC T8 / `sp_eval_finetune` may already cover this; not checked, since that is outside my area.)
5. **Scope.** Keep the narrowband population (8 GB, unused)? Keep an ablation (s11, fig10, fig11), and with which arms? Retrain the full sweep with K records? Does the `MAIN_N = 2000` rule (in-domain T1 ≥ 0.9, measured on the shared record) still pick N = 2000?
6. **Splits.** Keep s2, which redraws train/val for every N, or adopt EC-27's fully nested `t_make_nested_splits`?
7. **The E2 faithfulness gates in s5 and s7.** They need E2 checkpoints, E2 CSVs and the pre-P features. Copy those into the new repo, point the gates at the old repo, or retire the gates?
8. **s5a in the new repo.** Copy `data/field_inputs_v2` (188 MB) with its checks json, or also make s5a runnable? Running it needs the legacy `field_modal_filtering_outputs/<S>/logs` and npz, the legacy E5 features, and a grid sample.
9. **Test protocol.** Score one record per test structure, or several, to report the spread over records?
10. **The uncommitted `s10_figures.py` change** (`mean_markers`, +12 lines). Commit it in the old repo before freezing it, or carry the working-tree file over?
