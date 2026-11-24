# Restructure manifest - repository root (excluding experiments/ and Final_results_for_paper/)

Repo: `c:\MyFolder\Individual_Study_Syllabus\GNN_PBSHM\Linkspan_GNN\Linkspan_GNN_OMA` (HEAD 9c1f5da, Phase P).
Scanned read-only on 2026-11-24. "record" = depends on the shared excitation record (SEED 12345 re-seeded per
structure). "field" = real-bridge GNN inputs used: none / pre-P (loggers x40131/x40133 = sensor rows 6/8
interchanged) / post-P / raw (raw recordings in logger order; the swap is applied downstream).
Sizes from `du -sh`; tracked = `git ls-files`.

## 1. Inventory

### 1a. Root code (24 .py)

| path | kind | category | record | field | size | tracked | reason / what the new pipeline needs |
|---|---|---|---|---|---|---|---|
| grid_modal_Kinematic_MPC.py | code | KEEP | no | none | 69K | yes | FE solver. Needed by generator, export_projected_graph, build_phase3_graphs; EC n_mirror_test / r_excitation_realism / x_plot_excitation import it directly. `eigsh` at :802 has no `v0` (ARPACK noise <= 7.8e-07). Lazy `from build_phase3_graphs import` at :1623 (CLI path only). |
| export_projected_graph.py | code | KEEP | no | none | 20K | yes | FE model -> projected 2-D graph (topology, edge_attr_geom). Generator, s6, EC h_common/d_t6_truncation/r_excitation_realism. |
| standalone_mode_matching_mmi.py | code | KEEP | no | none | 55K | yes | Mode matcher (decides which FE modes become B1/T1/B2/T2). Generator, s6, deterministic/unseen inference. |
| simulate_uniform_inertial_u3_white_noise.py | code | KEEP-CHANGE | yes - this is where the shared record is made (:62, :344-345) | none | 33K | yes | Must draw a different record per structure: take an explicit excitation seed per call (see Q2). Everything else (influence matrix with one independent record per support DOF, Newmark, bandpass, Welch, row max-abs normalisation) stays. Its `main()` with `CAPTURE_REF = True` overwrites `ref_modes_Ng{Ng}.npz` in the CWD - never run it as a script in the new repo. |
| generate_population_dataset_modes4.py | code | KEEP-CHANGE | partly: parameter draws, FE solve, matching, labels, graph = no; `x_psd`, `x_psd_row_scale`, `edge_attr_cpsd`, `edge_attr_cpsd_resonance`, `edge_attr_cpsd_full` = yes | none | 105K | yes | Pass a per-structure excitation seed (from `sample_id`) through `assemble_phase1_sample_payload` (:965, simulate call :989) from the main loop (:2259); optionally K records per structure (EC T13/T17). Keep the exact lines `OUT_DIR = Path("population_dataset_modes4")` / `N_STRUCTURES_TOTAL = 800` (:53-54) and the two strings FR s0 patches by exact match (:215, :260). |
| GraphSAGE_Modal_GNN.py | code | KEEP | no | none | 63K | yes | Base trainer/model; EdgeGATConv imports 25 symbols (11 private). `DEVICE` picks cuda:0 if visible (:74); module-level `torch.manual_seed(SEED)` (:70) is part of reproducibility. CLI defaults `population_dataset_modes4/samples_phase2a`, `runs_phase2` (CWD-relative, CLI only). |
| EdgeGATConv_Modal_GNN.py | code | KEEP | no | none | 36K | yes | Main architecture. The fix's sign-invariant loss and coherency+orientation (802-d) edges are patched in memory by EC (`p_common.install_signinv`, `t_edge_modes`) and FR `fr_edge_modes`; the file itself need not change unless the author wants those folded in (then KEEP-CHANGE). |
| phase3_utils.py | code | KEEP | no | none | 3.8K | yes | Bandpass constants (1.5-20 Hz), `feature_propagation`, `load_graph_package` (default `phase3_real_graphs`, CWD-relative). `load_training_freq_grid` defaults to `population_dataset_modes4/samples_phase2_bandpass/sample_0000_Ng2_phase1_gnn.npz` (reads only `freq_bins_hz`). |
| infer_phase1_unseen_structure.py | code | KEEP | no (helpers); its own unseen-structure CLI simulates with the simulator's seed | none | 13K | yes | Transitive dependency of all infer_phase2a_* modules (`normalize_json_value`, `write_json`). |
| infer_phase2a_unseen_structure.py | code | KEEP | no (helpers); CLI path simulates (:129) | none | 24K | yes | s5/s6/s7 and EC use `load_frequency_normalization`; also plotting/saving helpers for the deterministic module. |
| infer_phase2a_unseen_structure_edgegat.py | code | KEEP | no | none | 7.3K | yes | s5/s7/EC: `apply_input_normalization`, `load_input_normalization`. CLI default output points into experiments/E1 (irrelevant to pipeline use). |
| infer_phase2a_deterministic_edgegat.py | code | KEEP-CHANGE | partly: `REAL_STRUCTURE_PARAMS`, `mode_col_swap`, the two T2 `mode_candidate_override`s, SSI loading/sign alignment = no; the FE-twin simulated features from `assemble_deterministic_phase2a_payload` -> `assemble_phase1_sample_payload` (:830) = yes | none (SSI labels only) | 50K | yes | Needed by s6 (ceiling, FE-twin variants) and `build_phase3_real_dataset`. Thread a per-bridge excitation seed into the FE-twin simulation (EC h_real uses `SeedSequence([20261127, LOOCV position, k])`). Keep `REAL_STRUCTURE_PARAMS` in sync with `build_phase3_graphs.STRUCTURES`. |
| build_phase3_graphs.py | code | KEEP | no | none (reads `SSI_modal_infor/grid_*.txt`) | 20K | yes | Canonical real-bridge geometry (`STRUCTURES`) and writer of `phase3_real_graphs/<S>/graph_package.npz`. s1, EC f_t9, deterministic module. CWD-relative `SENSOR_GRID_DIR`, `OUTPUT_ROOT`. |
| analyze_phase1_run.py | code | KEEP | no | none | 18K | yes | Needed by s10 through `experiments/E2_training_distribution/figures/plot_representative_modeshapes.py`. |
| structures.py | code | KEEP | no | none | 2.8K | yes | `LOOCV` (order matters; Festoya last), `ALL`, `EXCLUDED`. Imported by the E5 corrected builder (s5a, s6) and E2 `figstyle` (s10). |
| process_field_modal_filtering.py | code | KEEP | no | raw -> pre-P by default (`RAW_CHANNEL_REORDER = None`, :125); post-P only when the caller sets `raw_channel_reorder` (s5a does) | 64K | yes | s5a imports it unmodified. Run directly it opens tkinter dialogs and writes pre-P outputs into CWD `field_modal_filtering_outputs/` - never run it bare. |
| build_phase3_real_dataset.py | code | KEEP-CHANGE | partly: FE targets + SSI labels = no; it also packs the legacy root-builder features | pre-P (packs `node_features.npz`/`edge_features.npz`) | 7.6K | yes | Only needed to rebuild the label carrier `phase3_dataset.npz` (geometry or override change). Change: stop packing legacy x_psd/edge features (or leave them out). Raises on Sykkylven. |
| build_phase3_node_features.py | code | ARCHIVE | no | pre-P, and legacy convention (no row normalisation) | 4.5K | yes | Root "production" builder behind the 2026-08-21 failure; superseded by `experiments/E5_real_transfer/build_phase3_features_corrected.py`. Not imported by FR or EC. |
| build_phase3_edge_features.py | code | ARCHIVE | no | pre-P, raw CPSD (not coherency) | 5.2K | yes | As above. |
| inspect_phase3_propagation.py | code | ARCHIVE | no | pre-P (reads legacy features) | 27K | yes | Diagnostic figures on legacy real features. |
| prepare_edgegat_cpsd_benchmark.py | code | ARCHIVE | yes (simulates) | none | 6.5K | yes | Builds the E1 ablation corpora only. |
| analyze_phase2a_run.py | code | ARCHIVE | no | none | 22K | yes | Not imported by FR/EC; only by `paper_figures/plot_mac_frequency_summary.py` (archived). Harmless to keep if wanted. |
| run_all.py | code | KEEP-CHANGE | n/a (driver) | n/a | 9.9K | yes | The CWD and no-CUDA guards are reusable; every STAGES entry is an experiments/ stage -> rewrite for the new stages. |
| verify_reproduction.py | code | KEEP-CHANGE | gate D reads E2/E4/E5 CSVs = yes | pre-P (gate D) | 12K | yes | Gates A (interpreter/CPU), B (paths), C (diff) are reusable as is; gate D asserts old-pipeline claims (E2 broadband margin, E4 flat transfer, T1 < 0.15...) -> rewrite for the new claims. |

### 1b. Root data files

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| ref_modes_Ng2_Balcony.npz, ref_modes_Ng2_No_bal.npz, ref_modes_Ng3.npz, ref_modes_Ng4.npz, ref_modes_Ng5.npz, ref_modes_Ng6.npz | data | KEEP | no | none | 224K total | yes (`!ref_modes_*.npz`) | FE mode-shape templates (reference library) for the matcher and the label sign convention; read from CWD by the generator (`REF_DIR = Path(".")`). See Q4. Copy byte-identical: any change relabels the whole population. Ng6 is not in `NG_CHOICES = [2,3,4,5]` but is harmless. |

### 1c. Root folders

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| population_dataset_modes4/samples_phase2_bandpass/ | data | ARCHIVE | yes (features); params/FE/labels reusable in principle | none | 1.1G, 800 files | no | Old-generation broadband 800 (E2 arm), superseded by the FR 5500 population. BUT one file, `sample_0000_Ng2_phase1_gnn.npz`, is read for the 400-bin frequency grid by `phase3_utils.load_training_freq_grid` and E5 builder `TRAIN_GRID_SAMPLE` (s5a, s6), and FR `s3_check_trainer.OLD_POOL` reads the folder. Replace the grid read by a constant (`k*200/2048`, k < 400) or a grid-only file rather than copying a shared-record sample. |
| population_dataset_modes4/samples_phase2a/ | data | ARCHIVE | yes | none | 88M, 200 files | no | E1 ablation pool; CLI default of the GNN modules. |
| phase3_real_graphs/<9 LOOCV>/graph_package.npz | data | KEEP | no | none | 140K total | no | Real-bridge graph topology, node coords, sensor-node indices, nearest FE node. Read by E5 builder (s5a), s5, s6, EC h_real/f_t9/g_t10. |
| phase3_real_graphs/<9 LOOCV>/phase3_dataset.npz | data | KEEP-CHANGE | no for the label keys; its `x_psd`/`edge_attr_cpsd_full` are legacy features | pre-P in the two feature keys | ~1.4M each (~14M with Sykkylven) | no | The label carrier: `y_mode_shape_ssi`, `y_freq_hz` (curated SSI), `fe_target_mode_shape`, `fe_target_freq_hz` (with T2 overrides, rebuilt 2026-09-18), `ssi_mac_vs_fe`. The E5 builder copies every key and overwrites the four feature keys. Copy, but strip or rename the stale feature keys so nothing can read them as inputs. `fe_target_mode_shape` equals E6 `task2_raw` `fe_target` to 2.9e-08 (Hellesylt), so it can replace the E6_RAW dependency of s5/s6/s7/s10. |
| phase3_real_graphs/<9 LOOCV>/node_features.npz, edge_features.npz | data | ARCHIVE | no | pre-P + legacy convention | ~0.5M per bridge | no | Root-builder features; the 2026-08-21 failure convention. |
| phase3_real_graphs/<9 LOOCV>/structure_name.txt | data | KEEP | no | none | 9B each | no | Trivial sidecar written beside phase3_dataset.npz. |
| phase3_real_graphs/Sykkylven/ | data | ARCHIVE | no | pre-P | 2.6M | no | Excluded structure; its phase3_dataset.npz is stale (no longer builds). |
| field_measurements_raw/*.mat (10 raw + 10 `_Data_picked`) and grid_*.txt | data | KEEP | no | raw | 31M, 32 files | no | Raw recordings; irreplaceable. Byte-identical (cmp, 32/32) to `C:\MyFolder\Individual_Study_Syllabus\System identification code\Field measurements 26_270625\For GNN\` (that folder also has grid_Rykkjem.txt). The pipeline reads the EXTERNAL copy (`fr_common.FIELD_RAW_MAT_DIR`), not this one. |
| field_measurements_raw/build_real_linkspan_graphs.py, phase3_graph_io.py | code | KEEP | no | n/a | 68K | no | Part of the byte-identical snapshot of the external folder; not imported by anything in the pipeline. Copy with the folder. |
| field_modal_filtering_outputs/manual_mode_assignments.json | data | KEEP | no | n/a | 784B | no | 40 hand-picked SSI mode indices, no script produces them (PROVENANCE "What will never regenerate" #2). |
| field_modal_filtering_outputs/<9>/logs/*.json | data | KEEP | no | pre-P run metadata | 310K total | no | s5a reads `structure_summary.json` (asserts raw_file/ssi_file paths), `settings_used.json` (settings gate) and `mode_assignment_used.json` (reused mode assignment) from `LEGACY_FIELD_OUTPUTS`. Without them s5a cannot run. |
| field_modal_filtering_outputs/<9>/data/field_modal_filtering_outputs.npz | data | ARCHIVE (UNSURE if s5a's identity gate is kept) | no | pre-P | 176M total | no | Raw-order field outputs incl. FDD fields. Regenerated post-P by s5a into `Final_results_for_paper/data/field_inputs_v2/`. Needed only if the new s5a keeps its "identity reproduces legacy bit for bit" gate. |
| field_modal_filtering_outputs/<9>/plots_raw/, plots_filtered/ | figure | ARCHIVE | no | pre-P | 20M | no | Diagnostic PNGs. |
| field_modal_filtering_outputs/pairing_report.json, pairing_report.txt, run_summary.json | data | ARCHIVE | no | pre-P | 4K | no | Legacy run metadata (s5a writes its own run_summary). |
| field_modal_filtering_outputs/Sykkylven/ | data | ARCHIVE | no | pre-P | 22M | no | Excluded structure. |
| SSI_modal_infor/<10>_Data_picked.mat, mode_selection.json, grid_<10>.txt | data | KEEP | no | n/a (SSI order, always correct) | 54K | yes | Authoritative curated labels and sensor grids (build_phase3_graphs reads the grids). |
| SSI_modal_infor/build_ssi_modal_infor.py, README.md | code/doc | KEEP | no | n/a | 17K | yes | Rebuilds the curated .mat from mode_selection.json + the external field folder (`source_dir` in the json). |
| SSI_modal_infor/relabel_phase3_datasets.py | code | KEEP-CHANGE | no | pre-P (rewrites legacy dataset labels) | 8.7K | yes | Only needed if a selection changes; its targets (`phase3_real_graphs`, `experiments/E5_real_transfer/phase3_corrected`) must be re-pointed. |
| SSI_modal_infor/legacy_labels_backup.npz | data | EVIDENCE | no | n/a | 20K | yes | Pre-2026-08-26 frequency-blind labels, for `--restore`. |
| SSI_modal_infor/compare_ssi_relabel_impact.py, RELABEL_IMPACT.md, relabel_impact.csv | code/doc | EVIDENCE | yes (scores of E2/E5 shared-record networks) | pre-P | 16K | yes | Documents the 11-of-36 relabel impact. |
| runs_phase2/ | runs | ARCHIVE | yes | none | 188M, 10396 files | no | Old checkpoints (500/800-sample runs, 800_Samples_Ng2Split, edgegat_20260507_123555, 34 GPU-trained E3 sensitivity runs). |
| runs_phase3/ | runs | ARCHIVE | yes | pre-P + legacy-convention features | 38M, 245 files | no | Old zero-shot and LOOCV fine-tuning outputs, pre-audit. |
| results/baseline/FR__s1_population_summary.csv, FR__s1_rejections.csv, FR__s1_real_bridge_coverage.csv, FR__s2_split_summary.csv | results | KEEP | no (parameters, matcher rejections, splits) | none | small | yes | Regression targets for the new population: a per-structure-seed regeneration from the same generator seed must reproduce these exactly (proves the parameter draws did not desync). |
| results/baseline/FR__s4_*, s5_*, s6_*, s7_*, s8_*, s11_* | results | ARCHIVE | yes | s5/s6/s7 post-P; others none | ~8M | yes | Scores of shared-record networks. |
| results/baseline/E*__*.csv, SSI__relabel_impact.csv, MANIFEST.sha256 | results | ARCHIVE | yes | pre-P | ~1.9M | yes | Old-pipeline regression baseline. |
| results/reference/ | results | ARCHIVE | yes | pre-P | 2.1M | yes | Frozen 8-structure migration reference. |
| results/RUN_LEDGER.jsonl | results | ARCHIVE | n/a | n/a | 2.3M | yes (has uncommitted lines) | Old ledger; the new repo starts its own (run_stage.py creates it). |
| results/COPY_MANIFEST.sha256, GATE1_MIGRATION_REPORT.md, RESULTS.md | results/doc | ARCHIVE | yes (RESULTS.md) | pre-P | 2.6M | yes | Migration evidence and generated E-results summary. |
| tools/run_stage.py | code | KEEP | n/a | n/a | 6.6K | yes | Fully generic: REPO from `__file__`, ledger `results/RUN_LEDGER.jsonl`, logs `logs/`, interpreter from `tools/INTERPRETER` / `$LINKSPAN_PY`. |
| tools/diff_against_reference.py | code | KEEP-CHANGE | n/a | n/a | 12K | yes | Generic engine (IDENTICAL/NUMERIC/DIFFERS/STALE via ledger); `TAGS` folder map (:70-76) and the reference/KNOWN_STALE mode are old-repo specific. |
| tools/snapshot_baseline.py | code | KEEP-CHANGE | n/a | n/a | 3.4K | yes | Generic; same hard-coded `TAGS` map (:31-40). |
| tools/check_paths_resolve.py | code | KEEP-CHANGE | n/a | n/a | 5.6K | yes | Generic loader + interpreter re-exec; `TARGETS` lists experiments/ and fr_common constants -> rewrite. |
| tools/gate2_compare_populations.py | code | KEEP-CHANGE | n/a | n/a | 5.6K | yes | Exactly the check the re-simulation needs: FE fields (`y_freq_hz`, `y_mode_shape_signed`, `edge_index`, ...) must stay identical while response fields change. Re-point `DEFAULT_A/B` to the FR pool and the re-simulated pool; consider adding `params_json`, `matched_mode_indices`. |
| tools/make_results_summary.py | code | ARCHIVE | yes | pre-P | 9.3K | yes | Writes results/RESULTS.md from E1/E2/E4/E5 CSVs. |
| tools/make_paper_results.py | code | ARCHIVE | yes | pre-P | 84K | yes | Writes PAPER_RESULTS_OLD.md. |
| tools/migration_path_repair.py, migration_phase_c.py, migration_phase_c2.py | code | ARCHIVE | n/a | n/a | 21K | yes | One-shot migration scripts. |
| tools/INTERPRETER | config | KEEP | n/a | n/a | 113B | no (gitignored) | Points at `..\Linkspan_GNN_FineTuning\.venv-dgl\Scripts\python.exe` (Python 3.11.6, CPU torch 2.3.0, dgl 2.2.1); valid from the sibling folder too. |
| docs/TORSION_TRANSFER_FINDING.md | doc | EVIDENCE | yes (E2 shared-record networks) | pre-P (cross-deck numbers for pairs with rows 6/8) | 21K | yes | Key finding with provenance; caveat both flaws. |
| docs/LITERATURE_BASE_GNN_Modal_PBSHM.md | doc | KEEP | no | none | 71K | yes | Literature base for the paper. |
| docs/academic-writing-style-guide.md | doc | KEEP | no | none | 7.4K | yes | Author's style guide. |
| docs/PAPER_PROJECT_INSTRUCTIONS.md, docs/PAPER_SECTION_KICKOFF.md | doc | KEEP-CHANGE | conventions quote N=2000 / best-val / shared-record numbers | post-P | 12K | yes | Reusable claude.ai Project instructions; conventions and file pointers must be rewritten for the new pipeline. |
| docs/gnn_anatomy/ code (export_data.py, build.mjs, calc.mjs, live.js, prose.mjs, scene.mjs, shared.mjs, page.css, kit/, README.md) | code/doc | KEEP-CHANGE | the prose quotes numbers of the shared-record main network | none | ~0.2M | yes | Explainer generator; re-export with the new main network (reads `Final_results_for_paper/runs/edgegat_N2000` and the FR pool). |
| docs/gnn_anatomy/data.json, gnn_anatomy.html, gnn_anatomy-dark.html | figure | ARCHIVE | yes (activations of `edgegat_N2000` on a shared-record test deck) | none | 6.0M | yes | Built output of the old network. |
| docs/CLAUDE_source_repo.md, docs/progress_source_repo.md, docs/PROJECT_UNIVERSAL_CONTEXT.md, docs/JOURNAL_PAPER_HANDOVER.md, docs/PHASE3_REAL_TRANSFER_HANDOVER.md, docs/PLAN.md, docs/README.md | doc | ARCHIVE | yes (numbers) | pre-P | ~360K | yes | Development history / superseded handovers (docs/README.md is a stale note about the MATLAB port). |
| docs/PIPELINE_AUDIT_PHASE1.md, docs/PIPELINE_TASK*.md (13), docs/GEOMETRY_ONLY_PLOTS_EXP3.md, docs/SIGNED_MS_STUDY_EXP1-3*.md | doc | ARCHIVE | yes | pre-P | ~110K | yes | March 2026 task notes and early studies. |
| docs/progress_update_context/ | doc | ARCHIVE | yes | pre-P | 108K | yes | Supervisor-briefing context files. |
| CLAUDE.md | doc | KEEP-CHANGE | partly | mentions both | 23K | yes | Hard rules 1-3, conventions, SSI-label, structure-set and "easy to get wrong" sections stay valid; everything about experiments/, 500 epochs, E-numbers and shared-record results must be rewritten or moved to history. |
| README.md | doc | KEEP-CHANGE | yes (headline outline) | pre-P | 11K | yes | Rewrite for the new pipeline. |
| PROVENANCE.md | doc | EVIDENCE | yes | both | 114K | yes | History of the old repo (Phases A-P); the new repo needs a fresh PROVENANCE that points here. |
| PAPER_RESULTS_OLD.md | doc | ARCHIVE | yes | pre-P | 51K | yes | Previous populations' report. |
| requirements.txt | config | KEEP | n/a | n/a | 864B | yes | Pinned CPU environment. |
| .gitattributes | config | KEEP | n/a | n/a | 708B | yes | `* -text`: byte-level contract (no CRLF rewriting; never `sed -i`). |
| .gitignore | config | KEEP-CHANGE | n/a | n/a | 2.2K | yes (uncommitted E15 line) | Re-list the new bulk-data folders; keep the `!ref_modes_*.npz`, `!SSI_modal_infor/*.mat` exceptions. |
| figures/ | figure | ARCHIVE | n/a | n/a | empty | no | Nothing in it. |
| paper_figures/ (plot_mac_frequency_summary.py + output/) | code/figure | ARCHIVE | yes (E3 N2000 ep2000 sensitivity run) | none | 348K | 1 of 3 | Old figure script and its output. |
| logs/ | logs | ARCHIVE | n/a | n/a | 13M, 291 files | no | Run logs of the old repo (new logs/ is created by run_stage.py). |
| graphify-out/ | derived | ARCHIVE | n/a | n/a | 9.3M | no | Regenerable code graph (`python -m graphify update .`); regenerate in the new repo. |
| __pycache__/ (root, tools, SSI_modal_infor) | derived | ARCHIVE | n/a | n/a | 1.8M | no | Do not copy. |
| .claude/settings.local.json | config | UNSURE | n/a | n/a | 1.5K | no | Local Claude Code permission list (stage names, memory path). Copy only if the author wants the same allow-list; it is not part of the pipeline. |

Category counts (84 rows above): KEEP 29, KEEP-CHANGE 17, RESIMULATE 0, PROMOTE 0 (EC only), EVIDENCE 4, ARCHIVE 33 (one of them, the legacy field npz, is UNSURE if the s5a identity gate is kept), UNSURE 1.

Why no RESIMULATE row: the only simulated data in the root area are the two `population_dataset_modes4/` pools, which belong to the previous population generation and are superseded by the FR 5500 population (other agent's area). Their non-record parts are valid but nothing in the new pipeline needs them.

## 2. Answers to the specific questions

### Q1. Root modules imported by Final_results_for_paper/pipeline/*.py and EC_comments/pipeline/*.py

Direct imports, FR pipeline:
- `EdgeGATConv_Modal_GNN`, `GraphSAGE_Modal_GNN` (fr_edge_modes, fr_train, s3, s4, s5, s6, s7, s11)
- `generate_population_dataset_modes4` (s0, s6), `build_phase3_graphs` (s1), `process_field_modal_filtering` (s5a)
- `infer_phase2a_unseen_structure` (s5, s6, s7), `infer_phase2a_unseen_structure_edgegat` (s5, s7), `infer_phase2a_deterministic_edgegat` (s6)
- `export_projected_graph`, `phase3_utils`, `simulate_uniform_inertial_u3_white_noise`, `standalone_mode_matching_mmi` (s6)

Direct imports, EC pipeline (adds nothing new beyond): `grid_modal_Kinematic_MPC` (n_mirror_test, r_excitation_realism, x_plot_excitation), `build_phase3_graphs` (f_t9_bridges), plus the same GNN, generator, simulator, export_projected_graph, phase3_utils, infer_phase2a_unseen_structure(_edgegat) modules.

Through experiments/ modules that FR stages import (sys.path inserts):
- s5a, s6 -> `experiments/E5_real_transfer/build_phase3_features_corrected.py` -> `structures`, generator, simulator, phase3_utils
- s8 -> `experiments/E14_corner_sensors/stage1/2/3_*.py` -> GNN modules, generator, simulator, export_projected_graph, phase3_utils, infer_phase2a_unseen_structure(_edgegat)
- s10 -> `experiments/E2_training_distribution/figures/{figstyle, plot_loss_curves, plot_synthetic_performance, plot_representative_modeshapes}` -> `structures`, `analyze_phase1_run`

Transitive closure inside the root: generator -> export_projected_graph, grid_modal_Kinematic_MPC, simulator, standalone_mode_matching_mmi; simulator -> grid_modal_Kinematic_MPC, phase3_utils; infer_phase2a_* -> infer_phase1_unseen_structure; deterministic -> build_phase3_graphs; grid_modal_Kinematic_MPC -> build_phase3_graphs (lazy, CLI path).

**Needed by the new pipeline (16):** all 14 library modules of CLAUDE.md (grid_modal_Kinematic_MPC, standalone_mode_matching_mmi, export_projected_graph, simulate_uniform_inertial_u3_white_noise, generate_population_dataset_modes4, GraphSAGE_Modal_GNN, EdgeGATConv_Modal_GNN, phase3_utils, infer_phase1_unseen_structure, infer_phase2a_unseen_structure, infer_phase2a_unseen_structure_edgegat, infer_phase2a_deterministic_edgegat, build_phase3_graphs, analyze_phase1_run) **plus `structures.py` and `process_field_modal_filtering.py`**. Not needed: analyze_phase2a_run, build_phase3_node_features, build_phase3_edge_features, inspect_phase3_propagation, prepare_edgegat_cpsd_benchmark (build_phase3_real_dataset only to rebuild labels).

Also needed from experiments/ (flag for the experiments agent): E5 `build_phase3_features_corrected.py`, E14 `stage1/2/3`, E2 `figures/` (4 modules), and E6 `task2_raw/*.npz` (`fe_target`, read by s5/s6/s7/s10; replaceable by `phase3_dataset.npz['fe_target_mode_shape']`, which agrees to 2.9e-08).

### Q2. Where the shared seed is used, and what a per-structure fix touches

The record:
- `simulate_uniform_inertial_u3_white_noise.py:62` `SEED = 12345` (module global).
- `simulate_uniform_inertial_u3_white_noise.py:344` `rng = np.random.default_rng(SEED)` inside `simulate_white_noise_from_modal_model` (def at :272), called afresh for every structure.
- `:345` `ag_inputs = sigma_ag_mmps2 * rng.standard_normal((R.shape[1], nt))`: one independent white-noise record per support DOF (x, y, z of every support node), nt = 12000 (60 s at 200 Hz). The same `ag_inputs` feed the modal forces (`p_all_sim`) and the rigid-body part of `a_abs_full`. The same seed serves the broadband (`use_all_modes=True`, bandpass) and narrowband arms.

Callers that therefore inherit 12345:
- `generate_population_dataset_modes4.py:989` (in `assemble_phase1_sample_payload`, def :965), called from the main loop at :2259 with the global `sample_id` in hand (file `sample_{sample_id:04d}_Ng{ng}_phase1_gnn.npz`).
- `infer_phase2a_deterministic_edgegat.py:830` (`assemble_deterministic_phase2a_payload` -> `assemble_phase1_sample_payload`): the FE twins of the real bridges (s6 ceiling and FE-graph variants, E6 task2).
- `infer_phase1_unseen_structure.py:131`, `infer_phase2a_unseen_structure.py:129` (their unseen-structure CLIs), `prepare_edgegat_cpsd_benchmark.py`, the simulator's own `main()` :585.
- FR `s6_decomposition.py:135` calls the simulator directly; EC callers (h_common, h_real, k_common, c_fdd, c_gen_fresh_test, d_t6_*, sp_prepare, sp_eval_real, x_plot_excitation, n_mirror_test, r_excitation_realism) set `SIM.SEED` in memory.

Not the record: the generator's own `SEED = 20260507` (:55) -> `SeedSequence(SEED).spawn(4)` (:2172-2174), one parameter stream per Ng. The excitation `default_rng` is a separate generator, so changing the excitation seed does not shift any parameter draw.

A per-structure fix touches:
1. Simulator: an explicit `exc_seed` argument (int or `SeedSequence`) to `simulate_white_noise_from_modal_model`, used at :344. Safest: no silent default (raise if missing), so no caller can fall back to a shared record.
2. Generator: `assemble_phase1_sample_payload` gains `exc_seed` and passes it at :989; `main()` passes `SeedSequence([EXC_BASE, sample_id])` at :2259. Use distinct bases for train/val, test and real-bridge twins (EC used 20261125 train/val, 20261126 test, 20261127 real bridges with the LOOCV position, 20261131-33 for extra records k = 1..3). Several records per training structure (the fix) means either K feature sets per sample file or the EC route: re-simulate stored samples from their `params_json` (`h_common.resimulate_payload`, `q_prepare`), which keeps structures, labels and graphs bit-identical by construction.
3. `infer_phase2a_deterministic_edgegat.assemble_deterministic_phase2a_payload`: thread a per-bridge seed to the FE-twin simulation.
4. Caveat: acceptance in the generator loop sits inside a `try` that includes the simulation (:2208-2304), so a simulation-time exception would reject a structure and shift `sample_id`. Verify the regenerated population with `tools/gate2_compare_populations.py` (FE fields bit-identical) and `results/baseline/FR__s1_*`/`FR__s2_*` (identical).
5. Compatibility: EC h_common/k_common/c_gen_fresh_test assert `SIM.SEED == 12345` on entry and that `gen.simulate_white_noise_from_modal_model.__globals__ is SIM.__dict__`. Promoted EC code breaks if the module global disappears; either keep the global as the override hook or update those guards.

### Q3. Field-input data: who holds what, pre- or post-P

| file(s) | content | sensor order |
|---|---|---|
| `field_measurements_raw/<S>.mat` (`Stranda_Walk.mat`) | raw field recordings, 128 Hz, 3 channels per logger | raw logger order (x40131/x40133 interchanged at rows 6/8 relative to SSI). KEEP: irreplaceable. Byte-identical to the external `For GNN` folder that `fr_common.FIELD_RAW_MAT_DIR` actually reads. |
| `field_measurements_raw/<S>_Data_picked.mat` (Eidsdal: `Eisdal_Data_picked.mat`) | original SSI picks (all modes) | SSI order. KEEP. |
| `SSI_modal_infor/<S>_Data_picked.mat`, `mode_selection.json` | curated SSI labels, 4 modes B1/T1/B2/T2, `mode_names` marks them pre-ordered | SSI order: the authoritative labels, never affected by Phase P. |
| `field_modal_filtering_outputs/<S>/data/field_modal_filtering_outputs.npz` | filtered field outputs (`raw_u3_timeseries_clean`, PSD/SDM, FDD, modal-filter masks) | **pre-P** (`raw_channel_reorder = None`). |
| `field_modal_filtering_outputs/<S>/logs/*.json`, `manual_mode_assignments.json` | settings, clean window, mode assignment | metadata; the hand-picked assignments are irreplaceable. |
| `phase3_real_graphs/<S>/node_features.npz`, `edge_features.npz` | real-bridge GNN inputs from the root builders | **pre-P** and the legacy (wrong) convention. |
| `phase3_real_graphs/<S>/phase3_dataset.npz` | the same legacy features **plus** the labels (curated SSI, FE targets with overrides) | features pre-P; labels correct. |
| `phase3_real_graphs/<S>/graph_package.npz` | graph + sensor positions | not an input feature; valid. |
| (other agents' areas) `experiments/E5_real_transfer/phase3_corrected/` | corrected-convention features | pre-P (`LEGACY_REAL_FEATURES`). |
| (other agents' areas) `Final_results_for_paper/data/field_inputs_v2/` | s5a outputs: field outputs + `phase3_corrected` | **post-P**: the only post-P real-bridge inputs in the repo. |

So in the root area there is no post-P input file. Every real-bridge network number computed from root files (runs_phase3/, phase3_real_graphs features) is pre-P.

### Q4. ref_modes_*.npz

Mode-shape templates for the mode matcher, one per girder-count variant (Ng2 with balcony, Ng2 without, Ng3, Ng4, Ng5, Ng6). Keys: `Phi_u3_ref`/`Phi_u2_ref` (sensor DOFs x 20 FE modes of one reference deck at 20 x-stations per girder), `freq_hz_ref`, `ref_mode_indices_0based` (4), `Phi_u3_target_library` (sensors x 4 x 1), `sensor_node_ids`, `sensor_xy`, Ng/Nx/L_mm. Written by `simulate_uniform_inertial_u3_white_noise._save_reference_capture` from `main()` with `CAPTURE_REF = True`, which captures the FE eigenvectors and returns before any simulation (`CONTINUE_AFTER_CAPTURE = False`). They are pure FE, so they do not depend on the record. The generator loads them from the CWD (`REF_DIR = Path(".")`, patterns at :62-65). They decide which FE modes are labelled B1/T1/B2/T2 and, through `compute_reference_sign_alignment`, the label sign convention. Tracked in git; last changed in the consolidation commit f18f7fe. KEEP byte-identical.

### Q5. tools/

- Reusable as is: `run_stage.py` (and `tools/INTERPRETER`).
- Reusable after re-pointing a hard-coded map: `diff_against_reference.py` (`TAGS`), `snapshot_baseline.py` (`TAGS`), `check_paths_resolve.py` (`TARGETS`), `gate2_compare_populations.py` (`DEFAULT_A/B`; it is the right gate for "FE fields unchanged, response fields re-simulated").
- Old-pipeline specific: `make_results_summary.py`, `make_paper_results.py`, `migration_path_repair.py`, `migration_phase_c.py`, `migration_phase_c2.py`.

## 3. Dependencies the new pipeline must not break

- **CWD-relative paths** (run everything from the repo root): `generate_population_dataset_modes4.OUT_DIR`, `REF_DIR = Path(".")` (ref_modes in the root), `build_phase3_graphs.SENSOR_GRID_DIR = SSI_modal_infor`, `OUTPUT_ROOT = phase3_real_graphs`, `build_phase3_real_dataset.SSI_DIR/GRAPH_ROOT`, `phase3_utils.load_graph_package(graph_root="phase3_real_graphs")`, `phase3_utils.load_training_freq_grid` (population_dataset_modes4 sample), `process_field_modal_filtering.OUTPUT_ROOT`, GNN CLI defaults (`population_dataset_modes4/samples_phase2a`, `runs_phase2`).
- **`__file__`-relative paths**: `fr_common.REPO` (two parents up), `FIELD_RAW_MAT_DIR = REPO.parents[2] / "System identification code" / "Field measurements 26_270625" / "For GNN"` (resolves the same from `Linkspan_GNN_OMA_Diff_Exc`, same depth), `s1_verify_population.OLD_10K = REPO.parent / "Linkspan_GNN_FineTuning" / ...`, E5 builder `REPO = parent.parent.parent`, `FIELD_ROOT`, `GRAPH_ROOT`, `TRAIN_GRID_SAMPLE`; E2 `figstyle` `parents[3]`.
- **s5a path asserts:** s5a asserts that `field_modal_filtering_outputs/<S>/logs/structure_summary.json` `raw_file`/`ssi_file` (absolute paths to the external `For GNN` folder) equal `FIELD_RAW_MAT_DIR/<file>`. Re-pointing `FIELD_RAW_MAT_DIR` at the in-repo `field_measurements_raw/` breaks that assert unless it is relaxed. s5a also needs the legacy `logs/settings_used.json` and `logs/mode_assignment_used.json`, and its identity gate needs the legacy `data/*.npz` plus `experiments/E5_real_transfer/phase3_corrected/`.
- **E5 corrected builder inputs:** `field_modal_filtering_outputs/<S>/data/field_modal_filtering_outputs.npz` layout (`raw_u3_timeseries_clean`, `freq_hz`), `phase3_real_graphs/<S>/graph_package.npz`, `phase3_real_graphs/<S>/phase3_dataset.npz` (it copies every key, then overwrites `x_psd`, `x_psd_order_B`, `x_psd_norm_full`, `edge_attr_cpsd_full`), and the frequency-grid sample.
- **Exact-string patches:**
  - `OUT_DIR = Path("population_dataset_modes4")` and `N_STRUCTURES_TOTAL = 800` (generator :53-54), matched by E2 stage0 and E3 (only if they are copied). FR s0 replaces both as module attributes instead.
  - FR s0 recompiles `sample_params_for_ng`/`validate_sampled_params` from `inspect.getsource` after replacing `dy_high = min(2500.0, 12000.0 / (ng - 1))` (:215) and `if not (2000.0 <= dy <= 2500.0):` (:260). Each must occur exactly once, and the generator must stay a real .py file.
  - The generator safety guard `'population_dataset_modes4' not in str(OUT_DIR)` before `clean_output_directory`; s0 and c_gen_fresh_test refuse to write into `population_dataset_modes4/`.
- **Module-global patch hooks:** FR s0 replaces `gen.OUT_DIR`, `gen.N_STRUCTURES_TOTAL`, `gen.sample_params_for_ng`, `gen.validate_sampled_params`, `gen.match_target_modes`. EC sets `SIM.SEED` and checks `gen.simulate_white_noise_from_modal_model.__globals__ is SIM.__dict__` (so the generator must import the function from the simulator module, not copy it). FR `fr_edge_modes` and EC `p_common`/`t_edge_modes` patch edge modes and the loss into the GNN modules in memory: keep the patched names stable.
- **Flat imports:** every library module is imported as `from <module> import ...` with the repo root on sys.path. Keep the 16 modules flat at the root with their current filenames.
- **Consistency pairs:** `build_phase3_graphs.STRUCTURES` and `infer_phase2a_deterministic_edgegat.REAL_STRUCTURE_PARAMS` must stay in sync. `mode_col_swap` and `mode_candidate_override` live only in the latter and must be carried by anything that builds FE params from the former.
- **Byte-level:** `.gitattributes` `* -text`, and `requirements.txt` / `tools/INTERPRETER` (CPU torch; bit-reproducibility depends on the thread count).
- **Ordering:** `structures.LOOCV` order (nested LOOCV partner = next in list).

## 4. Open questions

1. Seed the FE eigensolver (`eigsh` `v0`, as the public package does with `EIGSH_START_SEED = 0`)? Everything is being regenerated anyway. It would move FE numbers by about 1e-07 and make FE re-runs bit-exact.
2. Keep s5a's identity gate (it needs the 176 MB of legacy pre-P field npz plus E5's legacy phase3_corrected), or run s5a once more in the new repo and freeze `field_inputs_v2` as the new baseline?
3. Point `FIELD_RAW_MAT_DIR` at the in-repo `field_measurements_raw/` (self-contained) and relax the s5a path assert, or keep reading the external folder?
4. Strip the legacy feature keys from `phase3_real_graphs/<S>/phase3_dataset.npz`, or keep the file byte-identical and rely on the builder overwriting them?
5. Fold the sign-invariant loss and the 802-d coherency+orientation edge mode into `EdgeGATConv_Modal_GNN.py`, or keep in-memory patches as FR/EC do now? This decides KEEP versus KEEP-CHANGE for both GNN modules.
6. Generate a population with per-structure seeds directly, or re-simulate the existing FR 5500 from their `params_json` (EC route)? The second guarantees identical structures and labels.
7. Not record-related but label-related, from project notes: all 568 Ng=2 balcony structures label a torsion mode "B2" and a bending mode "T2" (matcher plus `ref_modes_Ng2_Balcony.npz`). This is not in PROVENANCE. Fix it before regenerating?
8. Outside this area but dangerous: the public package `..\linkspan-modal-gnn` (published on GitHub 2026-11-05) has the same shared-record bug (`linkspan_gnn/excitation.py:25` `SEED = 12345`, `:75` `default_rng(SEED)` per call). Its measured-data path has no 131/133 reorder.
