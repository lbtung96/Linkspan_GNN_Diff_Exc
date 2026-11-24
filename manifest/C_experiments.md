# Manifest - `experiments/` (E1-E19)

Scope: `c:\MyFolder\Individual_Study_Syllabus\GNN_PBSHM\Linkspan_GNN\Linkspan_GNN_OMA\experiments\`, read-only scan, 2026-11-24.
Paths below are relative to `experiments/` unless they start with `Final_results_for_paper/` (= FR) or a root module name.

Legend
- **record**: does the content depend on the shared excitation record (`simulate_uniform_inertial_u3_white_noise.SEED = 12345`, re-seeded per structure)? yes / no / partly.
- **field**: real-bridge inputs. `pre-P` = GNN field inputs with loggers x40131/x40133 (0-based rows 6 and 8) interchanged; `post-P` = corrected; `raw rec` = reads `raw_u3_timeseries_clean` (row k is mapped to sensor node k, so **column order matters**); `SSI` = SSI labels only (always correct order, unaffected by Phase P).
- **tracked**: git-tracked (T), untracked but not ignored (U, shows as `??`), gitignored (I).

Headline facts that drive most rows
1. Every simulated response in `experiments/` (E1 pool, E2 narrowband/broadband, E3 10k, E7 m30, E10 augmented, E12 derived, and every "re-simulate from params_json" path in E5/E6/E14/E16/E17/E18/E19) was produced with SEED 12345. Every network trained or scored there is a shared-record network.
2. Nothing in `experiments/` was re-run after Phase P. Every real-bridge result that reads recordings is pre-P. One constant controls most of it: `build_phase3_features_corrected.FIELD_ROOT` (E5) and `stage1_holdout.FIELD_ROOT` (E17, which E18 and E19 read through `E17.FIELD_ROOT`). Both point at the legacy `field_modal_filtering_outputs/`.
3. Record-free pieces exist and carry the transfer story forward: the FE-shape population prior (E18), the class template (E19), the imputers and their true-shape ceiling (E16), FE-vs-SSI label tables (E2 stage5, t2_mac_diagnostic, E8 b1), the propagation operator on FE shapes (E6 task1, E8 b6, E9 s1 surrogate).
4. EC_comments T5 already re-ran the E18/E19 machinery on **fresh records** (own excitation per set): the class template is unchanged (0.946 both), FDD + E18 fill gives 0.832 / 0.974 (two corners / four sensors) on fresh records against 0.838 / 0.965 on the paper set. So the E18/E19 synthetic findings survive the shared record; the network's 0.933 -> 0.377 does not.

---

## 1. Per-experiment summary

| experiment | tested | finding (as recorded) | record dependence | field | re-run in new pipeline? | category |
|---|---|---|---|---|---|---|
| E1 architecture_ablation | Node PSD vs scalar CPSD vs 800-d coherency + EdgeGAT (and geometry edges) for signed torsion, 200-sample March pool, Ng 3-5, narrowband. | T1 at 100 ep 0.010 / 0.016 / 0.809. At 500 ep 0.518 / 0.438 / 0.901. The geometry arm is abs-only and does not train signed. | **yes**. The phase-blind arms' torsion came from the record (E12). EC T15 shows the coherency arms used it as a left/right label too. | none | No. Superseded by the FR section 3 grid, which itself must be re-run on the new data (EC T1/T16). | ARCHIVE |
| E2 training_distribution | Paired narrowband vs broadband (800 structures x 5 seeds), zero-shot, LOOCV fine-tuning in legacy and nested protocols. Also the SSI-vs-FE label tables and the T2 diagnostic. | Narrowband is better in-domain (0.943 vs 0.881) and broadband better zero-shot (0.220 vs 0.182; structure p = 0.26). The legacy LOOCV protocol is inflated by +0.041 to +0.077. Table 8 (SSI vs FE) 0.856. | Networks: **yes**. The LOOCV leak is a protocol property and survives in kind. Table 8 / t2 diagnostic: **no**. | Networks pre-P. Table 8: SSI and FE only (unaffected). | No for the network arms (FR is broadband with nested LOOCV). Keep the label tables and the figure modules. | ARCHIVE + KEEP/KEEP-CHANGE pieces |
| E3 population_sensitivity | Synthetic MAC vs N = 100-10000 and epochs, GPU-trained, 10k dataset (data lives in the source repo only). | Synthetic MAC rises 0.726 -> 0.953, but N <= 2000 is Ng = 2 only (confound). | **yes** | none | No. Superseded by the FR balanced sweep. | ARCHIVE |
| E4 transfer_sweep | Real transfer vs N on the E3 checkpoints, 8 structures. | Flat (0.264 -> 0.248) while synthetic rises: the decoupling. | **yes** | pre-P | No. FR s5 transfer-vs-N replaces it and must be re-run with new networks. Whether "flat in N" survives is open. | ARCHIVE (report EVIDENCE) |
| E5 real_transfer | Corrected real feature builder (PSD row-norm, coherency, Welch, 128 -> 200 Hz grid interpolation). Zero-shot, controls, fine-tuning and decomposition on the legacy and E2 checkpoints. | The convention fix made the network input-dependent (MAC to the zero-input response 0.999 -> 0.28). Zero-shot about 0.25. "Deterministic GNN MAC" in the handover was SSI-vs-FE. | Builder: **no**. Network numbers: **yes**. | Builder: raw rec (order matters; default points at pre-P outputs). Results pre-P. | The builder yes, as pipeline code (FR s5a/s6 already import it). The stages no. | KEEP-CHANGE (builder) + ARCHIVE |
| E6 controls | task1: FP operator on FE shapes. task2: propagation-only control on the legacy checkpoint. task2b: input-dependence (scale degeneracy). | task1: FP reconstructs FE shapes to MAC 0.89-0.97 (Linge, Hellesylt). task2b: the published zero-shot equalled the zero-input response (0.999). | task1 **no**; task2 **yes**; task2b **no** (a convention bug) | task1: none (graph + FE). task2: FE twin + SSI labels. task2b: the old production real features (pre-P, pre-convention). | No. task1 is record-free and cited by s9. task2 is replaced by FR s6. | KEEP (task1), KEEP-CHANGE (task2_raw), EVIDENCE / ARCHIVE |
| E7 fe_mode_count | 30 vs 15 FE candidate modes (800, broadband, 3 seeds). | Worse in-domain (-0.050, p = 0.015) and no transfer gain. Mechanism: 30 candidates let 30-47 Hz modes win the target assignment in 2.1 % of samples (label contamination). | Label mechanism **no**. Network numbers **yes**. | pre-P (stage4/5) | No. The label argument for keeping 15 stands. | EVIDENCE + ARCHIVE |
| E8 domain_gap | Why torsion does not transfer (legacy 800_Samples_Ng2Split checkpoint). | SSI and FE agree that T1/T2 are antisymmetric. The cross-deck cue is in the recordings (Re gamma -0.785) and arrives inverted (+0.584). 0/70 transverse sensor pairs are joined by an edge. 50 FP iterations erase the sign. The network relies on the transverse-edge sign (B4), and adding an edge at inference does not help (B7). | **partly**. b1, b6, the topology count and b3 raw coherency: no. b4/b7 (the network's reliance on edge sign): yes, now readable as the record label. | b1: SSI. b3/b3b: raw rec pre-P (pairs with rows 6/8 affected). b2/b5/b7: pre-P features. | Only b3/b3b, post-P, if `docs/TORSION_TRANSFER_FINDING.md` numbers are quoted. | EVIDENCE |
| E9 extension_operators | Screen of ten extension operators, no network. | Diffusion loses about 0.26 (T1) / 0.39 (T2) of the cue relative to alternatives. The current edges carry 0.197 of the T1 amplitude (they sit on the twist axis). | **no** (s1 uses its own surrogate RNG; s2 uses recordings) | s2: pre-P (features + raw rec) | No. Re-run s2 post-P only if quoted. | EVIDENCE |
| E10 cross_deck_edges | Cross-deck edges added to the 800 population and the real graphs, retrained 5 seeds. | In-domain T1 +0.057. Real: null overall. T2 +0.143 with symantisym. T1 unmoved. | **yes** | pre-P (real graphs) | No. EC T16's orientation edges supersede it. Revisit only if cross-deck edges come back. | EVIDENCE (docs) + ARCHIVE |
| E11 epoch500 | Every paper number at 500 vs 100 epochs. | In-domain up, transfer slightly down. The phase-blind arms learn torsion at 500 epochs. | **yes** | pre-P | No | ARCHIVE |
| E12 phase_blind_sign | Where the phase-blind arms get the torsion sign. | From a record-specific left/right spectral asymmetry (bending x torsion interference at f_T1), not from the graph. EdgeGAT "gets the sign through the edges". | **yes**. This is the first visible symptom of the shared record. The phase-blind explanation survives and is strengthened: with own records the asymmetry is no longer a consistent label (EC T13/T16). **The EdgeGAT-robustness claim does not survive**: EC T15 found that mirroring the support records flips 100 % of coherency nets' torsion. | none | No. EC T13-T16 replace it. | EVIDENCE |
| E13 epoch5000 | E2 seed-123 runs continued to 5000 epochs. | The best validation epoch never moves (296 / 356). The frequency head overfits mildly. | **yes** | none | No. The new pipeline trains 5000 epochs with best-val and produces its own curves. | ARCHIVE (README EVIDENCE) |
| E14 corner_sensors | Network on propagated sparse inputs (2 corners, 4 sensors, n-station sweep) using the E3 N600/N10000 nets. Its stage modules are the library behind FR s8. | 2 corners -> MAC 0.06-0.48, 4 sensors 0.22-0.67, every girder n = 8 -> 0.835. "The limit is dense-input training, not observability." It also noticed the Ng = 2 balcony B2/T2 label swap. | Results **yes**. Code **partly**: `resimulate()` uses the module-level SEED. The label-swap observation **no**. | none | As s8's library, yes. Standalone, no. | KEEP-CHANGE (3 modules) + ARCHIVE |
| E15 geometry_encoding | Old geometry edge feature: millimetre length vs angle code (2 GraphSAGE arms, N = 600). | Never completed: runs stopped at `resume.pt`, with no history, no evaluation and no report. Its README holds a record-free measured mechanism: raw-mm lengths saturate the sigmoid, so the edge path is dead (abs) or the edge weight is exactly 0 (signed). | Training **yes**. Mechanism **no**. | none | Optional. **Whole folder untracked.** | EVIDENCE (README) + ARCHIVE |
| E16 fp_imputers | FR s8 with 7 other imputers. Also the imputers applied to the true mode shapes (no network). | No imputer changes the network curve (PCFI <= +0.018). Imputing true shapes beats the network (outer n = 4: 0.818 vs 0.579). | Network part **yes**. Ceiling **no**. | none | Ceiling: no need (record-free, identical if the population is the same). Network part: only if the paper keeps it. | KEEP (imputers.py, ceiling.csv) + UNSURE / ARCHIVE |
| E17 real_holdout | Hide real sensors and fill them from the rest. Score against the recordings and against the FE twins. | Real decks fill worse than their twins (8/9). Coherence > 0.95 still gives 21 % wrong sign. Network vs SSI: 0.233 with all sensors, 0.180 with one. | Fills: **no** (the twin realisation is incidental, nothing is learned). Network rows: **yes**. | raw rec, **pre-P** (layouts pick sensors by index, so order matters) | Yes, post-P (cheap), if quoted. It is also a required library for E18/E19 and EC T5. | KEEP-CHANGE + EVIDENCE |
| E18 population_basis | Legendre tensor basis + Gaussian population prior over FE mode shapes; kriging from few sensors. | 4 sensors: 0.980 synthetic, 0.92-0.94 real vs SSI. 2 corners: 0.84 synthetic, 0.81 real (below the E19 template). | Prior and shape route **no**. ODS route: shared-record re-simulation, but EC T5 shows it unchanged on fresh records. Network column **yes**. | stage3: raw rec **pre-P** (via E17) | Yes: stage1 (identical if the population is the same), stage2 ODS with own records, stage3 post-P. | KEEP / KEEP-CHANGE |
| E19 mode_identification | FDD peaks + 3 labellers from 2/4 sensors. Class-template baseline (no measurement). | Synthetic: 87 / 92 % of modes identified. Real: 64 / 69 %. Template: 0.946 synthetic (beats the network's 0.933), 0.879 vs SSI on real. | Template and frequency prior **no**. Synthetic FDD: shared record, but T5 fresh_test reproduces it (0.832 / 0.974; template 0.946 identical). | real: raw rec **pre-P** (via E17) | Yes. The template is EC's baseline (T5/T16/T17). Re-tune settings on own-record validation decks; real post-P. | KEEP / KEEP-CHANGE / RESIMULATE (cache) |

---

## 2. Detailed table

Sizes are for data, run and result folders. Scripts and small files are left blank.

### E1_architecture_ablation (691 MB)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E1/run_ablation.py | script | ARCHIVE | yes | none | | T | Old ablation on the March pool. Superseded by the FR full grid (`fr_common.ABLATION_GRID`, `fr_edge_modes.py`). Still called by the old `run_all.py`. |
| E1/audit_and_prepare_signed_targets.py, prepare_cpsd_edge_benchmark.py, prepare_cpsd3_edge_benchmark.py | scripts | ARCHIVE | yes | none | | T | Dataset derivation. Cannot run any more: the pool predates the E/rho parameters. |
| E1/generate_detailed_comparison.py, init_experiment.py | scripts | ARCHIVE | n/a | none | | T | Scaffold and report helpers. |
| E1/ablation_summary.csv | result | ARCHIVE | yes | none | 1 KB | T | Superseded numbers. Read by the old `tools/make_paper_results.py`. |
| E1/README.md, docs/ (8 ABLATION_TASK*.md) | docs | ARCHIVE | yes | none | 76 KB | T | Development notes. The ablation story is retold in FR section 3 and E12. |
| E1/datasets/ (4 `*_ready` corpora) | data | ARCHIVE | yes | none | 594 MB | I/U | Pre-E/rho pool, cannot be rebuilt, shared record. |
| E1/runs/ (e100 / e500 arms, smoke runs) | checkpoints | ARCHIVE | yes | none | 21 MB | U | Shared-record networks. |
| E1/analysis/ | results | ARCHIVE | yes | none | 25 MB | 108/225 T | 100-epoch analyses. |
| E1/comparisons/ | results | ARCHIVE | yes | none | 0.8 MB | 21/37 T | Pairwise comparison reports. |
| E1/inference/800_Samples_Ng2Split/ | results | ARCHIVE | yes | FE twin + SSI | 51 MB | U | Legacy checkpoint deterministic inference (the source of the "SSI-vs-FE published as GNN" finding). The library default `--out` root of `infer_phase2a_deterministic_edgegat.py` points here. |
| E1/manifests/ | metadata | ARCHIVE | n/a | none | 48 KB | T | Bookkeeping. |

### E2_training_distribution (1.3 GB)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E2/README.md, TRAINING_DISTRIBUTION_REPORT.md, ..._legacySSI.md | docs | EVIDENCE | yes (numbers) | pre-P | | T | Keeps the LOOCV-leak finding (legacy selects on the test fold, +0.04 to +0.08). The leak is a protocol property and should survive; the nested protocol lives on in FR s7. The narrowband/broadband trade-off needs re-testing if it is ever claimed again. |
| E2/stage0_generate_narrowband.py | script | ARCHIVE | yes | none | | T | Narrowband is not in the new pipeline. Note: it is one of the four scripts that match `N_STRUCTURES_TOTAL = 800` / `OUT_DIR = Path(...)` by exact string (CLAUDE.md rules 4/5). |
| E2/stage0b_verify_pairing.py | script | ARCHIVE | partly | none | | T | Pairing proof (labels identical, features differ). |
| E2/stage1_train_pair.py | script | ARCHIVE | yes | none | | T | Trains the 5 x 2 E2 networks. |
| E2/stage2_zeroshot.py | script | ARCHIVE | yes | pre-P | | T | Its code was copied into FR s5. |
| E2/stage3_finetune.py | script | ARCHIVE | yes | pre-P | | T | Its code was copied into FR s7 (nested LOOCV). |
| E2/stage4_report.py | script | ARCHIVE | yes | pre-P | | T | Report generator. |
| E2/stage5_ssi_vs_fe_tables.py | script | KEEP-CHANGE | no | SSI only | | T | Generates Table 8 (SSI vs FE target). It reads `phase3_real_graphs/<S>/phase3_dataset.npz` labels and asserts them against `E5/phase3_corrected`, which is pre-P and will not be copied. Repoint the assert to the post-P tree (labels are byte-identical) or drop it. |
| E2/stage5_ssi_vs_fe.csv, stage5_ssi_vs_fe.md | result | KEEP | no | SSI only | | T | Table 8. EC T10 confirmed it is unaffected by Phase P. Cited by FR s9 and read by EC `e_t7_stored_outputs.py`. |
| E2/stage0b_pairing.csv, stage1_synthetic_test.csv, stage2_zeroshot*.csv, stage3_finetune*.csv, stage4_summary*.csv | results | ARCHIVE | yes | pre-P | | T | **FR s5 and s7 faithfulness gates read `stage2_zeroshot.csv` and `stage3_finetune.csv`.** |
| E2/*.log (5) | logs | ARCHIVE | yes | | | U | |
| E2/data/nb/ | data | ARCHIVE | yes | none | 1.2 GB | I | Narrowband 800, shared record. |
| E2/runs/ | checkpoints | ARCHIVE | yes | none | 22 MB | I | 100-epoch E2 networks. |
| E2/runs_ep500/ | checkpoints | ARCHIVE | yes | none | 24 MB | I | **FR s5/s7 gates load `runs_ep500/bandpass_seed123`.** Default `RUNS` of three figure modules (unused by s10). |
| E2/t2_mac_diagnostic/ (inspect_fe_vs_ssi_modes.py, CSVs, SUMMARY.md, README.md, PNGs) | diagnostic | KEEP | no | SSI + FE (reads `field_measurements_raw` SSI mats only) | 3.7 MB | 32/48 T | Evidence and the scan behind the T2 FE-target overrides (Hellesylt #7, Leknes #6). Error messages in `infer_phase2a_deterministic_edgegat.py` and FR s9 point here. Could equally sit in docs/. |
| E2/figures/figstyle.py | style module | KEEP-CHANGE | no | none | | T | Imported by s10 (MODE_SHAPE_CMAP, MEDIAN_COLOR, save) and by every old plot module. Change: line 32 reaches the repo root with `parents[3]` to import `structures.LOOCV`; re-count if moved. The author wants this style: do not alter colours or functions. |
| E2/figures/plot_loss_curves.py | style module | KEEP-CHANGE | no | none | | T | s10 uses load_history, save, _style, _legend, A, A4_TEXT_WIDTH_IN, FONT_TITLE, INK_PRIMARY, LINE_W, SPLITS, SPLIT_COLORS, TERMS, TERM_COLORS. Default `RUNS = PARENT / "runs_ep500"` is unused by s10 but points at shared-record runs. Imports figstyle by bare name. |
| E2/figures/plot_synthetic_performance.py | style module | KEEP-CHANGE | no | none | | T | s10 uses mac_panel and freq_panel. Default RUNS as above. |
| E2/figures/plot_representative_modeshapes.py | style module | KEEP-CHANGE | no | none | | T | s10 uses mac, _style, A, AXIS_COLOR, FONT_*, INK_PRIMARY, NODE_EDGE. `REPO = PARENT.parent.parent` (rule 6) and it imports root `analyze_phase1_run`. |
| E2/figures/plot_graph_topology.py | style module | KEEP-CHANGE | no | none (geometry only) | | T | **Also imported by s10 (fig16_real_graphs: fig_real_grid, save). CLAUDE.md lists only four modules.** Uses `REPO = PARENT.parent.parent`, imports `build_phase3_graphs`, reads `phase3_real_graphs/<S>/graph_package.npz`. `SYNTH_ROOT` (fig0a only) points at the shared-record `population_dataset_modes4/samples_phase2_bandpass` (geometry read only). |
| E2/figures/README.md | docs | KEEP-CHANGE | no | | | T | Explains the style. Copy with the modules and trim the old-figure numbers. |
| E2/figures/plot_population_sweep.py, plot_real_frequencies.py, plot_real_modeshapes.py, plot_transfer_summary.py | scripts | ARCHIVE | yes | pre-P | | T | Not imported by s10. They plot old numbers. |
| E2/figures/plot_synthetic_members.py | script | UNSURE | no | none | | T | Record-free geometry figure (fig0a members). Not imported by s10. Copy only if the new paper keeps that figure. |
| E2/figures/*.png, *.pdf, graphs/, modeshapes/ | figures | ARCHIVE | mostly yes | pre-P | ~17 MB | U/I | Old paper figures. fig0a/fig0b/fig0c are geometry-only and regenerable. |

### E3_population_sensitivity (376 KB; its 10k dataset is in the source repo only, its checkpoints in `runs_phase2/`)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E3/*.py (8: sensitivity_config, generate_sensitivity_dataset, run_*, analyze_*) | scripts | ARCHIVE | yes | none | | T | The sweep confounds N with Ng and was GPU-trained. Superseded by the FR balanced sweep. `generate_sensitivity_dataset.py` matches `N_STRUCTURES_TOTAL = 800` by exact string. |
| E3/plots/ (2 CSV + 4 PNG) | results | ARCHIVE | yes | none | 308 KB | 2/6 T | Copy-only inheritance. `sensitivity_mac_vs_N.png` is the known wrong-epoch Figure 8. |

### E4_transfer_sweep (2.3 MB)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E4/stage0..3 *.py, make_report.py | scripts | ARCHIVE | yes | pre-P (E5 phase3_corrected), 8 structures | | T | Pinned to 8 structures. FR s5 transfer-vs-N replaces it. |
| E4/TRANSFER_SWEEP_REPORT*.md, stage1/2/3 *.md | docs | EVIDENCE | yes | pre-P | | T | The "transfer is flat in N" claim. It must be re-tested with record-varied networks; whether it survives is unknown. |
| E4/*.csv, stage0_reference.json, *.png, stage1_run.log | results | ARCHIVE | yes | pre-P | ~1.5 MB | T (png/log U) | |

### E5_real_transfer (50 MB)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E5/build_phase3_features_corrected.py | library | KEEP-CHANGE | no | raw rec (row k -> sensor node k; order matters) | | T | **Imported by FR s5a and s6 (sys.path insert), by E17 (and through it E18, E19, EC c_fdd), EC d_t6_field_control/resample and g_t10.** Changes: (1) `FIELD_ROOT` defaults to the legacy pre-P `field_modal_filtering_outputs` (s5a overrides it at run time); make post-P the default. (2) `TRAIN_GRID_SAMPLE` reads `population_dataset_modes4/samples_phase2_bandpass/sample_0000` (a shared-record file) only for the frequency grid; replace it with the constant grid. (3) `OUT_ROOT` is inside experiments/. Move it into the pipeline/library. |
| E5/stage0_checks.py | script | KEEP-CHANGE | no | none | | T | Keep the `REAL_STRUCTURE_PARAMS` vs `build_phase3_graphs.STRUCTURES` sync diff (CLAUDE.md relies on it). Drop the scan of legacy E1 inference `summary.json` files. |
| E5/stage0_discrepancies.md, stage0_tables.md | docs | EVIDENCE | no | | | T | Shows that the published "deterministic GNN MAC" was SSI-vs-FE (transcription error). Survives. |
| E5/stage1_feature_stats.py (+ .csv/.md) | gate | ARCHIVE | partly (synthetic reference) | pre-P | | T | The gate is re-implemented in FR s5 (`feature_convention`). |
| E5/stage2_roundtrip_validation.py (+ .csv/.md) | gate | KEEP-CHANGE | partly | none (FE twin) | | T | Proves the corrected builder equals the production feature path on full FE responses (code equivalence, record-agnostic as a check). It depends on E6 task2's `resimulate_deterministic` and the module SEED. Port it as a gate with a per-structure seed. `verify_reproduction` gate D reads its CSV. |
| E5/stage3_zeroshot_corrected.py, stage4_controls.py, stage5_finetuning.py, stage6_e2_decomposition.py | scripts | ARCHIVE | yes | pre-P | | T | Legacy and E2 checkpoints. Replaced by FR s5/s6/s7. |
| E5/RERUN_REPORT.md | docs | EVIDENCE | no (the finding) | | | T | The feature-convention bug (zero-input response). Survives as the reason for the s5 gates. |
| E5/stage3..6 *.csv/.md, _freq_table.md, _tables_s3s4.md, stage3_modeshapes_*.png, stage5_run.log | results | ARCHIVE | yes | pre-P | ~1 MB | T (png/log U) | `verify_reproduction` gate D reads `stage6_*.csv`. |
| E5/phase3_corrected/ (9 structures) | data | ARCHIVE | no for labels; features are field-derived | **pre-P** | 14 MB | I | Pre-P real features. Labels are byte-identical to post-P. **Read by FR `fr_common.LEGACY_REAL_FEATURES` (s5a identity gate, s5/s7 E2-faithfulness proofs) and by EC `f_t9_*`.** |
| E5/stage3_raw/, stage6/ | results | ARCHIVE | yes | pre-P | 0.1 MB / 30 MB | I | |

### E6_controls (7.4 MB)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E6/task1_propagation_operator.py, task1_propagation_operator_mac.csv, task1_propagation_reconstruction.png | diagnostic | KEEP | no | none (FE shapes + graph_package) | | T (png U) | FP on FE target shapes, no spectra and no network. Cited in FR s9. Record-free. |
| E6/task2_propagation_only_control.py | script | ARCHIVE | yes | FE twin + SSI | | T | Legacy checkpoint. Replaced by FR s6. E5 stage2/4/6 import its `resimulate_deterministic`. |
| E6/task2_raw/ (9 npz) | data | KEEP-CHANGE | partly | SSI labels | 176 KB | **I** | **FR s5 (`AUDIT_RAW`), s6 (gate `fe_target_vs_E6_mac`), s7, s10 and EC e_t8/g_t10 read `fe_target` from it.** It holds label/geometry arrays (`fe_target` with the T2 override, `coords_fe`, `y_ssi`, `nearest_fe`, `sensor_xy`: record-free) and legacy predictions (`pred_*`: shared record). Copy only the label arrays, or regenerate `fe_target` in the pipeline (s6 already rebuilds it) and repoint `E6_RAW`. Gitignored: lost if only git is copied. |
| E6/task2_propagation_only_control.csv, task2_diagnostics.csv, task2_modeshapes_*.png | results | ARCHIVE | yes | FE twin + SSI | ~1 MB | T (png U) | |
| E6/task2b_scale_degeneracy_check.py (+ .csv, .png) | diagnostic | EVIDENCE | no (the finding) | pre-P, old real features | | T (png U) | The input-dependence failure. Survives. The gate lives on in FR s5. |
| E6/AUDIT_REPORT.md | docs | EVIDENCE | partly | | | T | The propagation vs convention audit that started everything. |

### E7_fe_mode_count (1.3 GB)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E7/RESULTS_E7.md | docs | EVIDENCE | partly | pre-P | | T | The label-contamination mechanism with 30 candidates is FE-level and survives. The -0.050 in-domain cost is record-dependent. |
| E7/stage1_pairing_check.py, stage1b_label_qa.py, stage1_*.csv | label QA | EVIDENCE | no | none | | T | Label-level checks. |
| E7/stage0_generate_modes30.py, stage2_feature_delta.py, stage3_train.py, stage4_zeroshot.py, stage5_finetune.py, stage6_report.py, stage2/4/5 CSVs | scripts / results | ARCHIVE | yes | pre-P | | T | stage0 matches the generator lines by exact string. |
| E7/data/m30/ | data | ARCHIVE | yes | none | 1.2 GB | 4 T (summaries) | |
| E7/runs/ | checkpoints | ARCHIVE | yes | none | 11 MB | 50 T | |

### E8_domain_gap (852 KB)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E8/REPORT.md | docs | EVIDENCE | partly | pre-P / SSI | | T | Survives: label agreement, 0/70 transverse pairs, propagation erasing the sign. Re-read: B4 "the model uses the transverse-edge sign" on a shared-record checkpoint. That is exactly how a record-as-label network would behave (EC T15). |
| E8/b1_target_side.py + b1_*.csv | diagnostic | EVIDENCE | no | SSI only | | T | FE vs SSI antisymmetry, sensor-pair geometry. |
| E8/b6_propagation_selectivity.py + csv | diagnostic | EVIDENCE | no | none | | T | Operator on FE shapes (bending kept 1.18x, torsion halved). |
| E8/b3_transverse_chain.py, b3b_propagation_sweep.py + csvs | diagnostic | EVIDENCE | no | raw rec + features, **pre-P** | | T | The source of the Re gamma -0.785 -> +0.584 numbers. Pairs involving rows 6/8 are affected. Re-run post-P before quoting. |
| E8/b2_feature_side.py, b4_edge_occlusion.py, b5_frequency_coverage.py, b7_counterfactual_probe.py + csvs | diagnostics | ARCHIVE | yes | pre-P | | T | Legacy and E2 checkpoints, shared-record synthetic references. |
| E8/e8_common.py, make_report.py | helpers | ARCHIVE | partly | pre-P | | T | e9_common imports e8_common. |

### E9_extension_operators (3.8 MB)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E9/README.md, REPORT.md, s0/s1/s2 CSVs | docs / results | EVIDENCE | no | s2 pre-P | | T | The operator screen. s1 uses a surrogate field with its own RNG (record-free). s2 is pre-P real. |
| E9/e9_common.py, e9_operators.py, s0_operator_checks.py, s1_synthetic_screen.py, s2_real_delivered.py, make_report.py | scripts | ARCHIVE | no | pre-P | | T | Re-runnable from the archive if an operator change is ever adopted. |

### E10_cross_deck_edges (1.3 GB)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E10/README.md, REPORT.md | docs | EVIDENCE | yes | pre-P | | T | The numbers will not carry over. Keep the design (the girder-line rule, the balcony-bracket pitfall). |
| E10/e10_augment.py | library | ARCHIVE | no | none | | T | Record-free cross-deck edge rule with a span assertion. Reuse it if cross-deck edges are retried. |
| E10/stage0..4 *.py, make_report.py, stage*.csv | scripts / results | ARCHIVE | yes | pre-P | | T | stage0 patches the generator by exact-string anchors. |
| E10/data/ (augmented 1.3 GB, real 42 MB) | data | ARCHIVE | yes | pre-P (real) | 1.3 GB | I | |
| E10/runs/ | checkpoints | ARCHIVE | yes | none | 11 MB | I | |

### E11_epoch500 (103 MB)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E11/README.md, PAPER_RESULTS_ep500.md, STATUS.md, e11_common.py, e11_stages.py, run_mirror.py, make_report.py, *.csv | scripts / docs | ARCHIVE | yes | pre-P | | T (STATUS U) | Superseded twice (Phase G, then FR). |
| E11/ablation/, decomposition/, runs/, figures/, _state/, finetune_chunks/, gates/, paper_ep100/ | data / results | ARCHIVE | yes | pre-P | 5 / 59 / 24 / 12 / 4 MB ... | partly T | |

### E12_phase_blind_sign (901 MB)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E12/README.md, REPORT.md, s1-s4 CSV/JSON | docs / results | EVIDENCE | yes | none | | T | The earliest measurement of the record acting as a left/right label ("record-specific interference signature"). The phase-blind part survives as explanation. **Mark the EdgeGAT "sign through the edges, robustly" conclusion as contradicted by EC T15.** |
| E12/e12_common.py, s1_symmetry.py, s2_probes.py, s3_controls.py, s4_interference.py, make_report.py | scripts | ARCHIVE | yes | none | | T | |
| E12/data/ | data | ARCHIVE | yes | none | 893 MB | I | Derived from the E1 datasets. |
| E12/runs/ | checkpoints | ARCHIVE | yes | none | 6.3 MB | I | |

### E13_epoch5000 (9.8 MB)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E13/README.md, summary.csv | docs | EVIDENCE | yes | none | | T | "More epochs never lower validation loss" was measured on shared-record data. Treat it as a prior, not a result. |
| E13/run_e13.py, gates/, runs/, _state/, figures/ | scripts / data | ARCHIVE | yes | none | 8 MB runs | gates T, rest I | |

### E14_corner_sensors (72 MB)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E14/stage1_corner_propagation.py | library | KEEP-CHANGE | partly | none | | T | Provides resimulate, node_psd_features, edge_features, propagate, infer and mac, used by FR s8, EC T4/T5/T6/sp and E18/E19. **`resimulate()` uses the module-level `SIM.SEED`.** In the new pipeline it must reproduce each structure's own record (EC `h_common` sets `SIM.SEED = SeedSequence([base, idx])` per structure). `RUN_DIR` defaults to the shared-record `runs_phase2/.../sensitivity_N600_ep500`. `main()` reads `dataset_root` from config.json, which points into the source repo. |
| E14/stage2_n10000_layouts.py | library | KEEP-CHANGE | partly | none | | T | FR s8 patches `S2.RUN_DIR`, `S2.CHECKPOINTS` and calls `S2._init_worker`. It provides `propagated_inputs`. `RUN_DIR` defaults to `sensitivity_N10000_ep500`. |
| E14/stage3_sensor_sweep.py | library | KEEP-CHANGE | partly | none | | T | FR s8 calls `S3.run_sample(levels, family, raw_dir)`. It provides `station_nodes`. It records `x_psd_resim_maxdiff` but s8 only prints it. With per-structure seeds a wrong seed would **silently** pair sparse inputs from one record with the full input from another. Make it an assert. |
| E14/README.md, REPORT.md, REPORT_N10000.md, REPORT_SWEEP.md | docs | EVIDENCE | yes | none | | T | "Dense-input training is the limit" must be re-tested. The Ng = 2 balcony B2/T2 label swap observation is record-free, survives, and is still unresolved. |
| E14/make_report*.py, stage*.csv, summary*.csv, fig_*.png | scripts / results | ARCHIVE | yes | none | ~23 MB CSV | T (png U) | |
| E14/raw/, raw_n10000/ | results | ARCHIVE | yes | none | 1.2 / 32 MB | I | Predictions plus targets. |
| E14/raw_stage3_all/, raw_stage3_outer/ | results | ARCHIVE | no | none | 6.7 / 6.5 MB | I | Sensor index layouts only. Trivially regenerable. |

### E15_geometry_encoding (4.9 MB, **entire folder untracked**)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E15/README.md | docs | EVIDENCE | no (mechanism) | none | | **U** | Record-free mechanism: raw-mm edge lengths saturate the sigmoid. It also notes `fr_train.UNRUNNABLE` says 5/508 where 11/508 is measured. |
| E15/e15_common.py, stage1_train.py, stage2_evaluate.py | scripts | ARCHIVE | yes | none | | **U** | Incomplete: there is no make_report.py, and stage2 never produced output. |
| E15/runs/ | checkpoints | ARCHIVE | yes | none | 4.9 MB | **U** | Partial (resume.pt), no history. |

### E16_fp_imputers (86 MB)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E16/imputers.py | library | KEEP | no | none | | T | Pure numpy. Imported by E17 and E18 stage2, and transitively by E19 and EC c_fdd. |
| E16/stage1_imputer_sweep.py | script | UNSURE | partly | none | | T | The ceiling path is record-free. The network path needs the new network. Decided by whether the new paper keeps the imputer comparison. |
| E16/ceiling.csv | result | KEEP | no | none | 19 MB | T | True-shape imputer ceiling. The gate reference for E18 stage2 (FP/PCFI rows). Valid only for the same 500 test decks. |
| E16/per_sample.csv, diagnostics.csv, gates.json, fig1-3.png, make_report.py | results | ARCHIVE | yes | none | 18 MB | T (png U) | Network on the shared record. |
| E16/README.md, REPORT.md | docs | EVIDENCE | partly | none | | T | The imputer ranking on shapes survives. The "network curve unchanged" result needs re-testing. |
| E16/explorer/ (build_explorer.py, template.html, test_explorer.js, decks/) | visualisation | ARCHIVE | yes | none | 50 MB | **U** (all 30 files, including the scripts) | Decks were re-simulated with the shared record. |

### E17_real_holdout (9.5 MB)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E17/stage1_holdout.py | library + script | KEEP-CHANGE | partly | raw rec, **pre-P** (order matters) | | T | **Imported by E18 stage3/make_pages and E19 stage2, and so by EC c_fdd.** It imports FR `s6_decomposition` (twins), the E5 builder and the E16 imputers. Changes: `FIELD_ROOT` -> post-P outputs (`fr_common.FIELD_OUTPUTS`). Twins come from s6's simulation (SEED 12345) -> per-structure seed. Network `RUN` -> the new main network. |
| E17/make_report.py | script | KEEP-CHANGE | partly | pre-P | | T | Re-run with stage1. |
| E17/README.md, REPORT.md, fill.csv, fill_sensors.csv, psd.csv, network.csv, checks.json, fig*.png | docs / results | EVIDENCE | partly | pre-P | ~9.5 MB | T (png U) | The fill findings should mostly survive (only 2 of 14-16 sensors are affected, and no learning is involved). The network rows do not. `fill.csv` is E18 stage3's gate reference. |

### E18_population_basis (22 MB)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E18/popbasis.py | library | KEEP | no | none | | T | Basis, class rule, kriging. Imported by E19, EC c_fdd and make_pages. |
| E18/stage1_prior.py | script | KEEP | no | none | | T | Reads only `y_mode_shape_signed`, `y_freq_hz`, coords and params of the FR N = 2000 training split. Re-point it if fr_common paths or the main split change. |
| E18/prior.npz, prior_fit.csv | data | KEEP | no | none | 224 KB / 0.7 MB | npz **I**, csv T | **Required by E19 (`init_worker` asserts the basis degrees), EC c_fdd and q_template_compare (class template).** Identical if the new pipeline keeps param seed 20260507 and the same training split. Regenerate in about 30 s. |
| E18/stage2_synthetic.py | script + library | KEEP-CHANGE | partly | none | | T | `segment_spectra` is imported by E19 stage1 and EC c_fdd. Shape route record-free. ODS route re-simulates through E14 (shared record) -> per-structure seed. Gates against E16 `ceiling.csv`. |
| E18/stage3_real.py | script | KEEP-CHANGE | partly | raw rec **pre-P** (via `E17.FIELD_ROOT`) | | T | Re-run post-P. |
| E18/make_report.py | script | KEEP-CHANGE | partly | | | T | |
| E18/synthetic.csv, classes.csv, synthetic_checks.json | results | EVIDENCE | partly | none | 13 MB | T | EC T5 confirms the ODS-route result on fresh records. |
| E18/real.csv, real_sensors.csv, real_checks.json | results | EVIDENCE | no / partly | **pre-P** | 5 MB | T | Expected to survive approximately. Re-run before quoting. |
| E18/README.md, REPORT.md, fig1/2.png | docs | EVIDENCE | partly | pre-P | | T (png U) | |
| E18/make_pages.py, pages/, e18_population_prior.html, e18_shape_explorer.html | explainer | UNSURE | partly | pre-P | 3.2 MB | **U** (not ignored) | Decided by whether the explainer is maintained. If so: KEEP-CHANGE and rebuild after E18/E19 re-run. It needs the E19 cache and the field records. |

### E19_mode_identification (64 MB)

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E19/ident.py | library | KEEP | no | none | | T | Peaks, ACG likelihood, labellers, FreqPrior. Imported by EC c_fdd. |
| E19/stage1_spectra.py | script + library | KEEP-CHANGE | partly | none | | T | Re-simulates the validation/test decks through E14 (shared record) -> per-structure seed. `features`, `resimulate`, `F_MAX_HZ` and `X_TOL_FRAC` are imported by EC c_fdd. |
| E19/stage2_identify.py | script + library | KEEP-CHANGE | partly | raw rec **pre-P** (via E17) | | T | `prior`, `leading` (**the class template**), `freq_prior` (from FE frequencies in E18 `prior_fit.csv`), `evaluate` and `init_worker` are imported by EC c_fdd. The template and frequency prior are record-free. Real path -> post-P. |
| E19/make_report.py | script | KEEP-CHANGE | partly | | | T | |
| E19/checks.json | settings | KEEP-CHANGE | partly | pre-P (real rows) | 1 KB | T | **EC c_fdd reads `chosen_on_val`** (peak prominence and lambda per layout). They were tuned on shared-record validation decks. Re-tune on own-record validation decks (T5 suggests they transfer). |
| E19/test.csv | result | EVIDENCE | partly | none | 2.8 MB | T | EC c_fdd gate `paper_outer_reproduces_E19_test_csv` compares against it. In the new pipeline that gate compares shared-record sims, so drop or relabel it. |
| E19/grid.csv, real.csv, stage1_checks.json, README.md, REPORT.md, fig1_identification.png | docs / results | EVIDENCE | partly | pre-P (real) | | T (png U) | The template numbers (0.946 / 0.879) are record-free. Note: 0.879 is vs SSI at the real sensors, so the order of the field records does not enter it. |
| E19/cache/ (val 300 + test 500 npz) | data | RESIMULATE | partly | none | 61 MB | I | Reusable FE part: y, classes, f_true, freq_all, matched, support, coords, geometry features. The cross-spectra come from shared-record sims. Cheap to regenerate entirely. |

### Housekeeping

| path | kind | category | record | field | size | tracked | reason |
|---|---|---|---|---|---|---|---|
| E*/__pycache__/ | bytecode | ARCHIVE | n/a | n/a | ~1.5 MB | I | Never copy. |

### Category counts (rows of section 2)

| category | rows |
|---|---|
| KEEP | 9 |
| KEEP-CHANGE | 23 |
| RESIMULATE | 1 |
| EVIDENCE | 25 |
| ARCHIVE | 59 |
| UNSURE | 3 |
| total | 120 |

By volume nearly all of the 6.0 GB is ARCHIVE: E1 datasets 594 MB, E2 data 1.2 GB, E7 data 1.2 GB, E10 data 1.3 GB, E12 data 893 MB. What is worth copying is about 30 MB plus the 61 MB E19 cache, which is RESIMULATE and need not be copied.

---

## 3. Answers to the specific questions

### Q1. Per experiment
See section 1. In short:
- **Findings that survive the flaw (record-free):**
  - E7's label contamination with 30 FE candidates.
  - E6 task1 (FP on FE shapes) and E8 b6 (the operator favours bending about 2x).
  - E8's 0/70 transverse sensor pairs and FE/SSI agreement on torsion.
  - E9's operator screen (surrogate field).
  - E16's true-shape imputer ceiling.
  - E18's population prior and kriging. EC T5 re-confirmed the ODS route on fresh records.
  - E19's class template (0.946 / 0.879) and FDD identification (re-confirmed by T5).
  - E5/E6's feature-convention bug and the E2 LOOCV-leak protocol finding.
  - E15's mm-length saturation mechanism.
  - The Table 8 label quality (E2 stage5, t2_mac_diagnostic).
- **Findings that do not survive as numbers:**
  - Every trained-network number in E1-E4, E7 (in-domain), E10-E14 and E16 (network part).
  - E12's "EdgeGAT signs torsion through the edges robustly", contradicted by EC T15.
  - E8 B4/B7's reading of the network.
- **Re-run in the new pipeline:**
  - E18 and E19, needed for the template/FDD baselines EC uses.
  - E17 if the real hold-out is quoted.
  - E14 modules as s8's library.
  - The E5 builder as pipeline code.
  - Nothing else needs re-running. E8 b3/b3b and E9 s2 only post-P, and only if quoted.

### Q2. `E2_training_distribution/figures/` style modules
These are classified apart from E2's data. FR s10 loads them by name through `old(module)` (sys.path insert of the folder plus `importlib.import_module`). **Five are used, not four.**

| module | category | what s10 uses | what must change on copy |
|---|---|---|---|
| figstyle.py | KEEP-CHANGE | MODE_SHAPE_CMAP, MEDIAN_COLOR, save | `parents[3]` walk to import `structures`. Nothing else. |
| plot_loss_curves.py | KEEP-CHANGE | load_history, save, _style, _legend and layout constants | Its `RUNS` default points at E2 runs_ep500 (unused by s10). |
| plot_synthetic_performance.py | KEEP-CHANGE | mac_panel, freq_panel | Its `RUNS` default (unused). |
| plot_representative_modeshapes.py | KEEP-CHANGE | mac, _style and font/colour constants | `REPO = PARENT.parent.parent`; it imports root `analyze_phase1_run`. |
| plot_graph_topology.py | KEEP-CHANGE | fig_real_grid, save (fig16_real_graphs) | `REPO = PARENT.parent.parent`; it imports `build_phase3_graphs`. `SYNTH_ROOT` points at the shared-record population (geometry only). |

None of them contains a result or reads a record. They are pure plotting and style code. Their palette and function names must stay unchanged, because the author wants the style and s10 binds to the attribute names. The other five plot_*.py are ARCHIVE (plot_synthetic_members UNSURE). The PNG/PDF outputs are ARCHIVE. CLAUDE.md should be corrected to name plot_graph_topology too.

### Q3. Experiment code imported or read by FR/pipeline and FR/EC_comments/pipeline

**`Final_results_for_paper/pipeline/`**

| consumer | experiment dependency | how |
|---|---|---|
| s5a_field_inputs.py | E5 `build_phase3_features_corrected.py` | sys.path insert. Monkeypatches `FIELD_ROOT`. Its identity gate reads the E5 `phase3_corrected/` (pre-P legacy). |
| s5_zeroshot.py | E6 `task2_raw/<S>.npz` (fe_target) | `AUDIT_RAW` |
| s5_zeroshot.py | E2 `runs_ep500/bandpass_seed123` + `stage2_zeroshot.csv` | Faithfulness gate. Code copied from E2 stage2. |
| s6_decomposition.py | E5 builder | sys.path insert |
| s6_decomposition.py | E6 `task2_raw` | Gate `fe_target_vs_E6_mac > 0.999999` |
| s7_finetune.py | E6 `task2_raw` | |
| s7_finetune.py | E2 `runs_ep500/bandpass_seed123` + `stage3_finetune.csv` | Faithfulness gate |
| s8_sensor_sweep.py | E14 `stage2_n10000_layouts`, `stage3_sensor_sweep` (and so `stage1_corner_propagation`) | sys.path insert. Patches `S2.RUN_DIR` / `S2.CHECKPOINTS`. |
| s10_figures.py | E2 `figures/` (5 modules) | `old()` |
| s10_figures.py | E6 `task2_raw` | |
| fr_common.py | E5 `phase3_corrected` | `LEGACY_REAL_FEATURES` |
| s9_report.py | E2 `stage5_ssi_vs_fe.csv`, `t2_mac_diagnostic/`, E5 builder, E6 `task1_propagation_operator_mac.csv`, E2 figures | Cited in text only |

**`Final_results_for_paper/EC_comments/pipeline/`**

| consumer | experiment dependency |
|---|---|
| c_fdd.py (T5: template + FDD) | E14 stage1/stage3, E18 `popbasis` + `stage2_synthetic.segment_spectra`, E19 `ident`, `stage1_spectra`, `stage2_identify`. Transitively E17 `stage1_holdout`, which pulls in FR s6, the E5 builder and E16 `imputers`. Reads E18 `prior.npz` / `prior_fit.csv` (via E19), E19 `checks.json` (`chosen_on_val`) and E19 `test.csv` (gate). |
| q_template_compare.py (T17) | No direct E-import. Reads T5 outputs (the template rows come from c_fdd). |
| d_t6_common.py, sp_prepare.py, sp_eval_sweep.py | E14 stage1/2/3 |
| d_t6_field_control.py, d_t6_field_resample.py, g_t10_swapped_inputs.py | E5 builder (sys.path) |
| f_t9_bridges.py, f_t9_data_checks.py, f_t9_sign_checks.py | Read E5 `phase3_corrected/*.npz` (pre-P; labels and features) |
| e_t7_stored_outputs.py | Reads E2 `stage5_ssi_vs_fe.csv` |
| e_t8_finetune_variants.py, g_t10_channel_order.py | E6 `task2_raw` (via `S7.E6_RAW`) |

**Root files**
- The library module `infer_phase2a_deterministic_edgegat.py` references E2 `t2_mac_diagnostic/` (messages) and E1 `inference/` (argparse default out-root).
- The old contract files (`run_all.py`, `verify_reproduction.py`, `tools/check_paths_resolve.py`, `snapshot_baseline.py`, `make_paper_results.py`) reference E1-E6 throughout. They belong to the archive, not the new pipeline.

### Q4. E18 and E19: simulated records vs FE shapes only

**E18**

| component | inputs | verdict |
|---|---|---|
| stage1 prior (`prior.npz`, `prior_fit.csv`) | FE target shapes, frequencies and geometry of the FR N = 2000 training split | **No record.** KEEP. |
| stage2 shape route (`route = shape`) | True shapes at the sensors | **No record.** |
| stage2 ODS route (`route = ods`, `dense`) | Re-simulated records via E14 `resimulate` (SEED 12345) | **Shared record.** EC T5 shows the result holds on fresh records (outer n = 2: 0.965 paper -> 0.974 fresh). |
| stage3 real | Real recordings via `E17.FIELD_ROOT` (**pre-P**). FE twins via FR s6 (SEED 12345; one realisation per twin, nothing learned). Labels SSI / FE. | Mixed. |
| "network" columns in its reports | Copied from s8 / E17 | Shared record and pre-P. |

**E19**

| component | inputs | verdict |
|---|---|---|
| **class template** (`stage2_identify.leading` = leading eigenvector of E18's class Sigma; method `template`) | FE shapes only, no measurement, class given from the true shape | **Record-free and field-input-free.** On real bridges it is scored vs SSI at the sensors. T5: 0.946 on both the paper and fresh test sets. |
| frequency prior (`freq_prior`) | FE frequencies (`prior_fit.csv`) + geometry | **No record.** |
| FDD / labellers on synthetic decks | Cross-spectra of re-simulated records (stage1 cache, shared record) | Shared record. T5 reproduced on fresh records. |
| real path | Recordings via `E17.FIELD_ROOT` | **pre-P**. |
| `checks.json` `chosen_on_val` | Tuned on shared-record validation decks | Used by EC c_fdd. |

The class template used in EC_comments (T5 `shape_baselines.csv`, T16 README "class template, no sensors (E19) 0.946 / 0.879", T17 `q_template_compare.py`) therefore depends only on `popbasis.py`, `prior.npz` and the FE target shapes. It carries over unchanged as long as the training split's structures are unchanged.

---

## 4. Dependencies the new pipeline must not break

1. **The s8 sensor sweep is E14 code.**
   - Copy `stage1_corner_propagation.py`, `stage2_n10000_layouts.py` and `stage3_sensor_sweep.py` together, keeping their **module names**: s8, EC and E18/E19 import them by bare name, and `stage3_sensor_sweep` does `RUN_DIR = S2.RUN_DIR` at import.
   - `resimulate()` must be given each structure's own excitation seed.
   - Promote `x_psd_resim_maxdiff` from a printed number to an assert. Otherwise a seed mismatch builds sparse inputs from a different record than the dense input, with no error.
2. **The real feature builder is E5 code.**
   - s5a monkeypatches `B.FIELD_ROOT`; E17 imports `corrected_node_features`, `corrected_edge_features`, `propagate` and `to_training_grid`.
   - Keep those names. Make post-P the default, and remove the hard dependency on `population_dataset_modes4/samples_phase2_bandpass/sample_0000_Ng2_phase1_gnn.npz` for the frequency grid.
3. **`E6_controls/task2_raw/<S>.npz["fe_target"]`** is read by s5, s6 (gate), s7, s10 and EC.
   - It is **gitignored**, so a git-only copy loses it.
   - Either copy the label arrays or generate `fe_target` inside the pipeline and repoint every `E6_RAW` / `AUDIT_RAW`.
4. **The s10 figure style.**
   - Five modules, loaded by bare name from one folder: figstyle, plot_loss_curves, plot_synthetic_performance, plot_representative_modeshapes and plot_graph_topology.
   - Their internal root walks (`parents[3]`, `PARENT.parent.parent`) must be re-counted at their new depth.
   - figstyle needs root `structures.py`; plot_representative_modeshapes needs root `analyze_phase1_run.py`; plot_graph_topology needs root `build_phase3_graphs.py` and `phase3_real_graphs/`.
5. **The EC T5 template/FDD chain:**
   - `c_fdd` -> `E19.stage2_identify` -> `E17.stage1_holdout` -> FR `s6_decomposition` + E5 builder + E16 `imputers`;
   - plus E18 `popbasis` and `stage2_synthetic`, E14 stage1/stage3, E19 `ident`/`stage1_spectra`;
   - plus data files E18 `prior.npz` (gitignored, regenerable), `prior_fit.csv`, E19 `checks.json` and `test.csv`.
   - Each script inserts sibling experiment folders into sys.path with `HERE.parent.parent` / `REPO / "experiments" / "E1x_..."`. Moving them needs every walk re-counted (rule 6). Bare names must stay unique: `stage1_*`, `stage2_*` and `stage3_*` currently do not collide across E14/E17/E18/E19, but would if two folders were merged carelessly.
6. **Legacy faithfulness gates in FR.**
   - s5/s7 "copied code reproduces E2's published rows" needs E2 `runs_ep500/bandpass_seed123` + `stage2_zeroshot.csv` / `stage3_finetune.csv` + E5 `phase3_corrected`.
   - s5a "identity with the legacy outputs" needs the legacy `field_modal_filtering_outputs` + E5 `phase3_corrected`.
   - In the new pipeline these gates either go, or point at the frozen archive by absolute path. Do not copy the shared-record checkpoints just to keep them green.
7. **Simulator change ripple.** When the simulator stops re-seeding with 12345, every "re-simulation reproduces the stored features" gate changes meaning. That covers E14 `x_psd_resim_maxdiff`, E16, E18 stage2, E19 stage1, E17 twin gate vs s6, E5 stage2 round-trip and E6 task2. Each must re-simulate with the record seed the stored sample was generated with. EC `h_common.exc_seed` (`SeedSequence([base, sample_index])`) is the pattern.
8. **Exact-string generator anchors.** E2 stage0, E3 `generate_sensitivity_dataset`, E7 stage0 and E10 stage0 match `N_STRUCTURES_TOTAL = 800` and `OUT_DIR = Path("population_dataset_modes4")`. None is proposed for copying. If any is, the new generator must keep those lines (CLAUDE.md rule 4).
9. **`structures.py` / `build_phase3_graphs.STRUCTURES` / `REAL_STRUCTURE_PARAMS`** sync check (E5 `stage0_checks.py`) and the T2 overrides (E2 `t2_mac_diagnostic`): keep both, record-free.

---

## 5. Open questions

1. **Will the new population keep the same structure draws** (param seed 20260507, the same 5500 structures, the same N = 2000 training split and 500 test decks)?
   - If yes, E18 `prior.npz` / `prior_fit.csv`, E16 `ceiling.csv`, E2 Table 8 and the E19 template carry over bit-identically.
   - If the split or N changes, refit the prior on the new training split. Otherwise the template leaks test shapes or misses training ones.
2. **Legacy faithfulness gates (s5/s7 vs E2, s5a vs legacy field outputs):** keep them pointed at the archive, or drop them? This decides whether any E2 run or E5 `phase3_corrected` needs to be reachable from the new repo.
3. **`E6/task2_raw`:** copy the label arrays, or regenerate `fe_target` in a new pipeline stage? s6 already rebuilds it and gates against E6.
4. **E19 settings (`checks.json` `chosen_on_val`):** re-tune on own-record validation decks, or freeze the shared-record choice? T5 shows the latter works on fresh records, but tuning on a different distribution than the test is a protocol wrinkle.
5. **E16 network part, E17 network rows, E14 standalone:** does the new paper still compare the network with sparse layouts or imputers? If not, those scripts stay archived.
6. **Untracked, not ignored, work** exists only on disk: the whole of E15; E18 `make_pages.py`, `pages/` and the two HTML explainers; E16 `explorer/` (including its scripts); E14 figures; E11 `STATUS.md`. If the frozen archive is a git commit, commit or copy these first, or they are lost.
7. **E12 / E8 re-reading.** Should the docs/history copies of E12 REPORT and E8 REPORT carry a header note? E12's EdgeGAT robustness claim and E8 B4's "the network uses the transverse sign" are contradicted, or reinterpreted, by EC T14/T15 (record as left/right label).
8. **The Ng = 2 balcony B2/T2 label swap** (568/5500, noticed in E14, in memory, not in PROVENANCE) is record-free. It will carry into any new population generated by the same matcher. It is not part of the record fix, and someone must decide whether to fix it before regenerating.
9. **`docs/TORSION_TRANSFER_FINDING.md` numbers** (from E8 b3/b3b): re-run post-P before the history doc is quoted? Pairs involving sensor rows 6/8 are affected.
10. **E4's "transfer flat in N"** and **E3's synthetic learning curve** have no record-free core. Whether they hold for record-varied networks is untested, apart from the FR sweep, which shares the flaw.
