# Manifest: `Final_results_for_paper/EC_comments/` for the restructure into `Linkspan_GNN_OMA_Diff_Exc`

Scanned 2026-11-24, read-only. Unless a path starts with `Final_results_for_paper/` or the repository root, it is
relative to `Final_results_for_paper/EC_comments/`.

**Out of scope.** T19 and T20 work was being written during the scan and is excluded. So far that is
`pipeline/u_decomposition.py` (T19) and `logs/T19_*`. No `results/T19_*` or `results/T20_*` folder existed yet.

**How the categories are applied**
- **PROMOTE** marks the T11 → T13 → T16 → T17 chain that becomes the new main pipeline. Every PROMOTE script
  still has to be refactored before it moves. Its in-memory patches must become real code, and its paths must
  be re-pointed (§3 Q1, Q6). The "reason / role" column names that work.
- **KEEP-CHANGE** marks support code outside that core which the new pipeline still needs once changed: a
  baseline, a check, or a tool for answering a review comment.
- **record** says which excitation realisation the numbers come from:
  - **shared**: the paper's one record, simulator seed 12345 for every structure;
  - **own**: one record per structure, `SeedSequence([base, sample_index])`;
  - **one new shared**: T5's `fresh_test`, 500 new structures all under seed 20261124;
  - **none**.
- **field** says which field inputs a number used:
  - **pre-P**: loggers x40131 and x40133 interchanged;
  - **post-P**: `Final_results_for_paper/data/field_inputs_v2`;
  - **none**.
- **tracked**: only `README.md`, `.gitignore`, `pipeline/ec_common.py`, `pipeline/g_t10_*.py` and
  `pipeline/results/T10_channel_order/` are in git. Everything else under `pipeline/` is untracked but not
  ignored. `data/`, `runs/` and `logs/` are gitignored.

Totals on disk:

| folder | size | notes |
|---|---|---|
| `data/` | 28 GB | of which `T4/variants` is 13 GB |
| `runs/` | 116 MB | |
| `logs/` | 3.7 MB | |
| `pipeline/results/` | ~57 MB | of which T5 is 25 MB |

---

## 1. Per-task summary

| task | comments answered | what it showed | excitation | field | category | re-run in new pipeline? |
|---|---|---|---|---|---|---|
| T1 ablation × seeds, phase arms, record-wise norm | EC-04, EC-20, EC-21, EC-08 | Nothing new: the queue was stopped at 0/49 jobs, and three seed runs stopped at epochs 225–275. Only gates and the reused seed-123 rows exist. | shared | none | ARCHIVE | **Yes.** The phase ablation (Re γ, abs γ), seeds and parameter cost have to be redone for the new network. T16's amplitude arm already covers part of EC-08 on own records. |
| T2 nested sweep | EC-27, EC-20, BS-41 | Quantifies the split leak: the paper's validation members train at a larger N (all 92 at N = 600). Nested manifests built; runs pending. | shared (runs pending) | none | EVIDENCE (leak) + KEEP (manifests) | **Yes**, if the paper keeps a population-size sweep. The nested manifests can be reused as they are. |
| T3 main network × seeds | EC-20, EC-38, EC-40 | Gates only: s4, s5, s6 and s7 reproduce on a mirror. Seeds 124–127 never trained. | shared | pre-P (gates) | ARCHIVE | **Yes.** T17 already has three seeds; the final protocol decides how many. |
| T4 sparse-mask training | EC-30, EC-41, EC-38 | Design, the 13 GB of variants and the trainer checks exist. Training was never launched; only pilot and smoke evaluations ran. | shared (variants carry 12345) | pre-P (smoke) | ARCHIVE (data, results) / KEEP-CHANGE (code) | **Yes.** EC-41 calls this central. It must be redone on own records with the sign-invariant loss and coherency + orientation edges. |
| T5 fresh test set, metrics, baselines | EC-15, EC-23, EC-25 | **The fatal flaw.** The main network falls from 0.933 to 0.377 on a new record, through the edge coherency. FDD on the same records scores 0.969; the class template 0.946. Frequency baselines; EC-23 metrics. | shared, one new shared (`fresh_test`), none (template, kNN) | none | EVIDENCE + PROMOTE (baselines, `fresh_test`) | **Partly.** The template and frequency-baseline numbers hold as they are. FDD has to be run on own-record sets. EC-15 needs a test set nobody has looked at, and EC-23 has to be re-scored on the new network. |
| T6 sensitivity (inference) | BS-43, EC-28, EC-29, EC-17, EC-33 | T6.1: 50 propagation iterations have not converged for sparse layouts. T6.2 and T6.3 were stopped with no output. T6.4a was run on pre-P inputs. T6.4b covers the FE twins on the 128 Hz path. | shared | pre-P (T6.4a), none (T6.4b) | ARCHIVE / KEEP-CHANGE (code) | **Yes**, on the new network: EC-28, EC-29 and EC-33; and EC-17's network part. |
| T7 stored-output analyses | BS-40, JX-10, EC-26, EC-34, EC-35, EC-39, BS-38 | Statistics of the paper network's stored outputs. Gates pass. | shared | pre-P | ARCHIVE (§3, the FE-vs-SSI agreement, is network-free: KEEP) | **Yes**, recomputed from the new outputs. |
| T8 fine-tuning variants | EC-37, EC-38 | Epoch 0 eligible, rotated validation bridge, head-only, frozen processor. The v0 gate reproduces s7. | shared (network) | pre-P | ARCHIVE / KEEP-CHANGE (code) | **Yes.** The s7 loss is MSE against SSI and is not sign-invariant (§5). |
| T9 facts from code and data | EC-09, 10, 11, 12, 17, 18, 19, 20, 31, 32, 37 | Facts read from the code. They include the excitation being the same for every structure (EC-20), that there is no measurement noise, and the coherency convention. `requirements_lock.txt`. Side finding: a possible 7/9 sensor swap, which became T10. | none (code facts); field facts on legacy inputs | pre-P (field facts) | KEEP | **Partly.** The field-derived facts need re-checking on `field_inputs_v2`. Parameter counts change with 802-d edges. |
| T10 channel order | EC-31, EC-34; every field number | The GNN inputs are mis-ordered and the SSI labels are right (36/36 cells). Correcting them moves zero-shot 0.233 → 0.254. This led to Phase P (commit 9c1f5da). | shared (network) | pre-P vs swapped | EVIDENCE (tracked) | No. The order check should become a standing gate in the field-input builder. |
| T11 varied-excitation pilot | follow-up of the T5 finding | With an own record per structure, N = 600 and 5000 epochs, coherency scores 0.41 everywhere (T1 ≈ 0.01) and orientation 0.71–0.73. Real bridges: orientation 0.49–0.51, coherency 0.22–0.23. | own (train base 20261125, test base 20261126, ceiling base 20261127) | post-P | EVIDENCE; data PROMOTE (`paper_varexc`) | No. It is superseded by T16 and T17. |
| T12 excitation plots | (T5 finding) | Figures of the shared record against per-structure records, and the coherency fingerprint. | shared vs own | none | EVIDENCE | No |
| T13 feasibility levers | (T5 finding) | Four records, 120 s records and N = 2000 all leave coherency T1 ≤ 0.01 at 400 epochs. Orientation with four records reaches 0.758. Diagnostic: the MSE networks output roughly zero for torsion. | own (bases 20261125, 20261131–33; 120 s variants) | post-P | EVIDENCE; data/code PROMOTE | No. Its data, trainer wrapper and data job carry into T17. |
| T14 input anatomy | (T5 finding) | Under the paper record a fixed template finds the deck side in 95–100 % of decks; with own records, at chance. The physical relative-sign cue is the same under every record. Occlusion: the paper network scores 0.44 with correct resonance bins and a new record elsewhere. | shared vs own | none | EVIDENCE | No |
| T15 mirror test | (T5 finding) | Mirroring the support records flips 100 % of the torsion predictions of the coherency networks and 0 % of the orientation networks'. The shared record was a left/right label. | shared (mirror variants) | none | EVIDENCE | No |
| T16 sign test | EC-10 (the sign-invariant loss it suggested); T5 finding | N = 600, 400 epochs, four own records. With the sign-invariant loss, coherency lifts T1 from 0.007 to 0.45. Coherency + orientation + sign-invariant: 0.901 synthetic, 0.560 against SSI. Also: the B1 label sign flips with Ng. | own | post-P | EVIDENCE; code PROMOTE (`p_common`, `p_eval`) | No. Superseded by T17; the arm choice comes from here. |
| T17 scale-up | (the fix) | N = 2000, 1000 epochs, seeds 123–125, four own records. Coherency + orientation + sign-invariant: **0.948 ± 0.004** (equal to the template, 0.020 below FDD). Orientation: 0.916 ± 0.008. Real bridges against SSI: 0.525 ± 0.068 and 0.524 ± 0.018. | own (`paper_varexc`), one new shared (`fresh_test`), shared (`paper` control) | post-P | **PROMOTE** | It *is* the new pipeline's starting point. A re-run with the final protocol (epochs, seeds) is an open question (§5). |
| T18 excitation realism | relates to EC-17, EC-36 (not run) | Nine FE twins under six excitations, compared with the field. The paper's support excitation is the least field-like: T2 takes 60 % of the mode peaks against 6 % in the field, and the sensor-power pattern correlates at r = −0.04. Deck forcing comes closer. The FE frequency error dominates. | none (no network); six synthetic excitation models | post-P | EVIDENCE | Only if the excitation model changes (§5). |

---

## 2. Detailed inventory

### 2a. Pipeline scripts (`pipeline/*.py`; 73 files, plus `u_decomposition.py` for T19, excluded)

| path | kind | category | record | field | size | tracked | reason / role in the new pipeline |
|---|---|---|---|---|---|---|---|
| pipeline/ec_common.py | paths/config | KEEP-CHANGE | none | none | 60 lines | yes | Root of every import. It sets REPO three parents up, puts REPO, the paper pipeline and itself on sys.path, and re-exports `fr_common`. It also holds EC_DATA/RUNS/RESULTS and the seed constants. Becomes the new repo's path/config module; the parent count must be redone. |
| pipeline/c_gen_fresh_test.py | data gen (T5) | KEEP-CHANGE | one new shared / shared | none | 129 | no | Generates NEW structures with s0's patches plus `gen.SEED`, `SIM.SEED` and a guarded `clean_output_directory`, Agg and a non-TTY stdin. It sets **one** `SIM.SEED` for the whole set. A test set nobody has looked at needs a per-structure seed inside the generator loop, and its file names must not collide with the paper's (§4). |
| pipeline/c_fresh_checks.py | checks (T5) | KEEP-CHANGE | none | none | 211 | no | Novelty, distribution, rejection and cost checks for a new test set. Re-point it to the new set. |
| pipeline/c_rescore.py | eval (T5) | KEEP-CHANGE | shared / one new shared | none | 275 | no | `set_files()` is imported by `p_eval` (`fresh_test`), so it is a **T17 dependency**. Its import has a side effect: `E.task_dir("T5")` makes a directory. The rest scores the paper runs. Keep only the test-set registry. |
| pipeline/c_ec23_metrics.py | metrics (T5) | KEEP-CHANGE | shared / one new shared | none | 126 | no | EC-23 metrics: sign-aligned RMSE, cross-MAC, success rates. Re-run on the new networks' predictions. |
| pipeline/c_freq_baselines.py | baseline (T5) | PROMOTE | none (the baselines); shared (the network column) | none | 141 | no | **Baseline.** Mean, per-Ng mean and geometry kNN of frequencies, fitted on the N2000 training split. Record-free, so its numbers hold. Re-point the network column. |
| pipeline/c_fdd.py | baseline (T5) | PROMOTE | shared (`paper`, `val`), one new shared (`fresh_test`) | none | 293 | no | **Baseline.** Class template (record-free) and FDD (E19 labeller). Its worker sets one `SIM.SEED` per set, so own-record sets need per-structure seeding: **FDD was never run on `paper_varexc`**. Imports `experiments/E14`, `E18` (`popbasis`) and `E19` (`ident`, `stage1_spectra`, `stage2_identify`) and E19's `stage2_synthetic`. |
| pipeline/c_excitation_diag.py | diagnosis (T5) | EVIDENCE | shared vs one new shared | none | 113 | no | Input swap: new edges alone give 0.410, new nodes alone 0.927. Shows the edge coherency carried the dependence on the record. |
| pipeline/c_report.py | report (T5) | ARCHIVE | — | none | 604 | no | Writes T5's README, which is itself EVIDENCE. |
| pipeline/d_t6_common.py | helpers (T6) | KEEP-CHANGE | shared | none | 140 | no | s8/E14 helpers; `runs/edgegat_N2000` is hard-wired. Needed again if T6 is re-run on the new network. |
| pipeline/d_t6_propagation.py | sensitivity (T6.1) | KEEP-CHANGE | shared | none | 210 | no | EC-28 and BS-43: iteration count, convergence ρ, convexity. The method is reusable; the network and inputs need coherency + orientation. |
| pipeline/d_t6_layouts.py | sensitivity (T6.2) | KEEP-CHANGE | shared | none | 130 | no | EC-29 equal-budget layouts. Never completed. |
| pipeline/d_t6_truncation.py | sensitivity (T6.3) | KEEP-CHANGE | shared (uses the module SEED) | none | 159 | no | EC-17 modal truncation. Never completed. Must seed per structure. |
| pipeline/d_t6_field_resample.py | sensitivity (T6.4a) | KEEP-CHANGE | none | pre-P | 133 | no | EC-33: 128 → 200 Hz. Imports `experiments/E5` builders. Re-run on post-P inputs with the new network. |
| pipeline/d_t6_field_control.py | sensitivity (T6.4b) | KEEP-CHANGE | shared (FE twins at 12345) | none | 169 | no | EC-33 control: FE twins through the 128 Hz path. Copies s6's loop. |
| pipeline/d_t6_s8_replay.py | gate support (T6) | ARCHIVE | shared | none | 66 | no | Measured s8's own noise against the paper CSVs. Never run. |
| pipeline/d_t6_report.py | report (T6) | ARCHIVE | — | — | 330 | no | Gates against the paper's CSVs. |
| pipeline/e_t7_stored_outputs.py | analysis (T7) | ARCHIVE | shared | pre-P | 546 | no | Tied to the paper's CSV formats and `runs/edgegat_N2000`. Its §3, FE vs SSI agreement, is network-free and worth re-deriving. |
| pipeline/e_t7_report.py | report (T7) | ARCHIVE | — | — | 369 | no | |
| pipeline/e_t8_finetune_variants.py | fine-tuning (T8) | KEEP-CHANGE | shared (network) | pre-P (at run time) | 229 | no | Variants v0–v4, copying s7's loop. Needs the new network. **The s7 loss (MSE against SSI) must become sign-invariant** for networks trained with the sign-invariant loss. |
| pipeline/e_t8_report.py | report (T8) | ARCHIVE | — | — | 425 | no | |
| pipeline/f_t9_bridges.py | facts (T9) | KEEP-CHANGE | none | pre-P (legacy E5 features) | 39 | no | Per-bridge table. Uses `Path.cwd()`, legacy `experiments/E5_real_transfer/phase3_corrected` and `field_modal_filtering_outputs`. |
| pipeline/f_t9_channel_order_check.py | facts (T9) | EVIDENCE | none | pre-P (raw order) | 27 | no | The first hint of the 7/9 swap. Superseded by T10 and Phase P. |
| pipeline/f_t9_data_checks.py | facts (T9) | KEEP-CHANGE | shared (stored samples) | pre-P | 76 | no | Coherency convention, node counts, Hermitian checks. Uses `Path.cwd()` and legacy field paths. |
| pipeline/f_t9_loss_weight.py | facts (T9) | KEEP-CHANGE | none | none | 18 | no | Node counts per Ng in N2000 training. Record-free, so the result holds; only the paths change. |
| pipeline/f_t9_params.py | facts (T9) | KEEP-CHANGE | none | none | 27 | no | Parameter counts by s7 group for `edgegat_N2000`. The new 802-d edge encoder changes the counts. |
| pipeline/f_t9_sign_checks.py | facts (T9) | KEEP-CHANGE | none | pre-P | 38 | no | Sign-alignment margins: the label sign convention that sign-invariant training sidesteps. Legacy E5 path. |
| pipeline/g_t10_channel_order.py | check (T10) | KEEP-CHANGE | shared (network) | pre-P | 197 | yes | Three-way sensor-order test (FE, FDD, roughness). Should become a standing gate on the new field inputs. After Phase P, s5's stored inputs are already in SSI order, so the "stored vs swapped" expectation flips. |
| pipeline/g_t10_swapped_inputs.py | fix prototype (T10) | ARCHIVE | shared (network) | pre-P → swapped | 114 | yes | Superseded by `Final_results_for_paper/pipeline/s5a_field_inputs.py` (Phase P). |
| pipeline/h_common.py | data gen core (T11) | **PROMOTE** | own | none | 211 | no | **Data generation.** `resimulate_payload`: the generator's broadband path with `SIM.SEED = SeedSequence([base, idx])` patched per call. Also `EXC_KEYS`, the seed bases, `support_rows`, and the csv/json/bootstrap helpers. The patch must become a real `seed=` argument. |
| pipeline/h_prepare.py | data gen + gates (T11) | **PROMOTE** | own / shared (gate) | none | 310 | no | **Data checks.** Gate (a) reproduces the paper features at 12345 to ≤ 1e-5. Gate (b) checks record independence and determinism. Every written sample is read back and checked. `_load`, `compare_to_stored`, `max_abs_corr` and `_maxdiff` are imported by `k_prepare`. |
| pipeline/h_train.py | training (T11) | ARCHIVE | own | none | 148 | no | Superseded by `k_train` and `q_train`. |
| pipeline/h_eval.py | eval (T11) | ARCHIVE | own / shared / one new shared | none | 210 | no | Re-points `c_rescore` globals in memory. Superseded by `p_eval`. |
| pipeline/h_real.py | real bridges (T11) | ARCHIVE | own (ceiling base 20261127) | post-P | 277 | no | `k_real` imports it. Orientation inputs on the real bridges are re-implemented in `p_eval.real_zeroshot`; the own-record ceiling is T19's territory. |
| pipeline/h_report.py | report (T11) | ARCHIVE | — | — | 450 | no | |
| pipeline/x_plot_excitation.py | figures (T12) | EVIDENCE | shared vs own | none | 206 | no | |
| pipeline/k_common.py | data registry (T13) | **PROMOTE** | own | none | 158 | no | **Data registry.** `BASES` (20261125, 20261131–33), `EPOCH_SEED` 20261130, `DATASETS`, `sdir`. `resimulate` patches `SIM.T_TOTAL` per call. |
| pipeline/k_prepare.py | data gen (T13) | **PROMOTE** | own | none | 362 | no | **Data generation.** `data_job` (used by `q_prepare`): re-simulate, replace `EXC_KEYS`, read back and check. Also the gates and the cross-check that every non-excitation key is byte-identical across all variants. |
| pipeline/k_train.py | training wrapper (T13) | **PROMOTE** | own | none | 290 | no | **Training.** `install_augmentation` (K records per structure, one drawn per epoch with `rng([20261130, epoch])`, resume-aware, logs `variant_draws.csv`) and `run_one` (around `fr_train`). Patches → real code. |
| pipeline/k_eval.py | eval (T13) | ARCHIVE | own / one new shared | none | 201 | no | |
| pipeline/k_real.py | real bridges (T13) | ARCHIVE | own | post-P | 51 | no | |
| pipeline/k_diag.py | diagnosis (T13) | EVIDENCE | own | none | 57 | no | MSE networks output near-zero torsion (norm ratio). This is why the sign-invariant loss. |
| pipeline/k_report.py | report (T13) | ARCHIVE | — | — | 492 | no | Reads the root `results/RUN_LEDGER.jsonl`. |
| pipeline/m_input_anatomy.py | analysis (T14) | EVIDENCE | shared vs own | none | 439 | no | |
| pipeline/m_occlusion.py | analysis (T14e) | EVIDENCE | shared vs own | none | 145 | no | |
| pipeline/n_mirror_test.py | analysis (T15) | EVIDENCE | shared (mirrored) | none | 395 | no | Patches the simulator's `build_influence_matrix` in memory; evidence only. |
| pipeline/p_common.py | loss/edges/nodes (T16) | **PROMOTE** | own | none | 168 | no | **Training.** Sign-invariant shape loss, coherency + orientation 802-d edges, amplitude node input, `install_for`. All are monkeypatches → real code (§3 Q1). |
| pipeline/p_train.py | checks + launcher (T16) | KEEP-CHANGE | own | none | 183 | no | Its `--check` suite becomes unit tests: loss = brute-force sign minimum, 802-d layout, amplitude input, trainer unchanged. The launcher is superseded by `q_train`. |
| pipeline/p_eval.py | evaluation (T16) | **PROMOTE** | own / one new shared / shared | post-P | 220 | no | **Evaluation.** `score`, `summarise`, `contrasts`, `set_files`, and `real_zeroshot` (orientation and coherency + orientation edge inputs on the hybrid graphs, then `s5_zeroshot.evaluate`). Imported by `q_eval` and `q_template_compare`. |
| pipeline/q_common.py | registry (T17) | **PROMOTE** | own | none | 58 | no | **Config.** Runs, seeds 123–125, 1000 epochs, `POOLS`. Patches `k_common.DATASETS` and `k_common.sdir` in memory (an EC-internal monkeypatch). |
| pipeline/q_prepare.py | data gen (T17) | **PROMOTE** | own | none | 72 | no | **Data generation.** Records 1–3 for the 1700 N2000 training structures: 426 hard-linked from T13, 3822 simulated. |
| pipeline/q_train.py | training (T17) | **PROMOTE** | own | none | 125 | no | **Training.** `run()`: `install_for`, then `fr_train.jobs` replaced by a lambda, `install_augmentation`, `sys.argv`, and finally `fr_train.main()`. `--check` reproduces `ec_t13_coh_K4_N600` with `t_gates.same_history`. |
| pipeline/q_eval.py | evaluation (T17) | **PROMOTE** | own / one new shared / shared | post-P | 107 | no | **Evaluation.** Synthetic and real, per seed and across seeds. Its gates re-score the T16 references against T16's CSVs. |
| pipeline/q_template_compare.py | baseline comparison (T17) | **PROMOTE** | own / one new shared | none | 167 | no | **Baseline comparison.** Deck-node MAC paired per structure against T5's template and FDD rows. Reads `data/T5/fdd/*_decks.npz` (support masks) and T5's `fdd_rows.csv` and `shape_baselines.csv`. |
| pipeline/r_excitation_realism.py | analysis (T18) | EVIDENCE | none (six synthetic models) | post-P | 401 | no | Its `deck_noise` and `walkers` excitation models would become KEEP-CHANGE data-generation options if the new pipeline changes its excitation (§5). |
| pipeline/sp_common.py | sparse training (T4) | KEEP-CHANGE | shared | none | 258 | no | EC-30/41 design, storage format, arms. Must move to own records and the sign-invariant loss with coherency + orientation. |
| pipeline/sp_prepare.py | sparse data (T4) | KEEP-CHANGE | shared (module seed) | none | 325 | no | Propagated sparse variants. Re-simulates with the module SEED, so it must seed per structure. |
| pipeline/sp_train.py | sparse training (T4) | KEEP-CHANGE | shared | none | 496 | no | A copy of `fr_train`'s loop with variant slots. Merge it with `k_train`'s augmentation. |
| pipeline/sp_eval_sweep.py | eval (T4) | KEEP-CHANGE | shared | none | 219 | no | Sensor sweep: dense against sparse networks. |
| pipeline/sp_eval_real.py | eval (T4) | KEEP-CHANGE | shared (FE twins) | `paper` source (follows `fr_common`, so post-P now) | 240 | no | |
| pipeline/sp_eval_finetune.py | eval (T4) | KEEP-CHANGE | shared | `paper` source | 116 | no | Uses s7, so it shares the sign-invariance issue (§5). |
| pipeline/sp_check_trainer.py | checks (T4) | ARCHIVE | shared | none | 179 | no | Checks identity with `fr_train`. |
| pipeline/sp_train_all.py | launcher (T4) | ARCHIVE | — | — | 87 | no | |
| pipeline/sp_report.py | report (T4) | ARCHIVE | — | — | 403 | no | |
| pipeline/t_edge_modes.py | input variants (T1) | KEEP-CHANGE | none | none | 144 | no | `cpsd_re`, `cpsd_abs` (phase ablation, EC-04) and `cpsd_full_recnorm` (EC-08). Library patch → real code if the ablation is re-run. `c_rescore` installs it when the file exists. |
| pipeline/t_gates.py | trainer gates (T1–T3) | KEEP-CHANGE | shared | none | 274 | no | Only `same_history()` is needed. It is imported by the `--check`/gate modes of `h_train`, `k_train`, `p_train` and `q_train`. Extract it; archive the rest. |
| pipeline/t_queue.py | job queue (T1–T3) | UNSURE | — | — | 271 | no | A memory-aware multi-process queue with resume and adoption. Keep it if the new pipeline runs many seed or sweep jobs; otherwise archive. |
| pipeline/t_jobs.py | registry (T1–T3) | ARCHIVE | shared | none | 198 | no | |
| pipeline/t_train.py | training (T1–T3) | ARCHIVE | shared | none | 117 | no | |
| pipeline/t_make_nested_splits.py | splits (T2) | KEEP | none | none | 147 | no | Nested train/validation manifests (EC-27). Record-free; reusable once the paths are re-pointed. |
| pipeline/t_eval_indomain.py | eval (T1–T3) | ARCHIVE | shared | none | 222 | no | |
| pipeline/t_eval_main.py | eval (T3) | ARCHIVE | shared | pre-P (gate run) | 248 | no | |
| pipeline/t_report.py | report (T1–T3) | ARCHIVE | — | — | 716 | no | |

Script counts: PROMOTE 14, KEEP 1, KEEP-CHANGE 27, EVIDENCE 8, ARCHIVE 22, UNSURE 1. Total 73.

### 2b. Results folders (`pipeline/results/T*/`)

| path | kind | category | record | field | size | tracked | reason / role |
|---|---|---|---|---|---|---|---|
| results/T1_ablation_seeds_phase/ | tables (mostly pending) | ARCHIVE | shared | none | 76 KB, 10 files | no | Only gates and the paper's seed-123 rows. `t_gates_ram.csv` (RAM per process against N) may help sizing. |
| results/T2_sweep_nested/ | split tables | EVIDENCE | none | none | 28 KB | no | Quantifies the EC-27 leak (`t2_split_summary.csv`, `t2_split_checks.json`). Runs pending. |
| results/T3_main_seeds/ | gates | ARCHIVE | shared | pre-P | 99 KB | no | Gates only. |
| results/T4_sparse_training/ | pilot/smoke | ARCHIVE | shared | pre-P | 309 KB | no | The README's design text is worth reading when T4 is redone. |
| results/T5_fresh_test_baselines/ | finding + baselines | EVIDENCE (with PROMOTE files) | shared / one new shared / none | none | 25 MB, 25 files | no | **The finding** (README, `excitation_diag.csv`, `fresh_vs_fixed.csv`). PROMOTE: `fdd_rows.csv` and `shape_baselines.csv`, which `q_template_compare` reads (it depends on them); the template and kNN rows of `freq_baselines*.csv`, which are record-free; `fresh_checks.json`. |
| results/T6_sensitivity/ | tables | ARCHIVE | shared | pre-P (T6.4a) | 3.3 MB | no | |
| results/T7_stored_outputs/ | tables | ARCHIVE | shared | pre-P | 726 KB | no | `t7_3_fe_vs_ssi_per_bridge.csv` and `t7_3_fe_vs_ssi_summary.csv` are network-free (KEEP). |
| results/T8_finetune_variants/ | tables | ARCHIVE | shared | pre-P | 284 KB | no | The finding on "two folds stopped at epoch 1" is protocol knowledge. |
| results/T9_facts/ | facts | KEEP | none | pre-P (field facts) | 73 KB | no | `requirements_lock.txt` (the environment). Code facts on the simulator: one SEED, R is a selection matrix, no measurement noise. Re-check the field facts. |
| results/T10_channel_order/ | finding | EVIDENCE | shared (network) | pre-P vs swapped | 80 KB | **yes** | Why Phase P. `PROVENANCE_phaseL_draft.md` is already merged into PROVENANCE §Phase P. |
| results/T11_varied_excitation_pilot/ | finding | EVIDENCE | own | post-P | 5.0 MB | no | Includes `prepare_gates.json`, `prepare_checks.csv` and `prepare_summary.json`: the provenance of `T11/paper_varexc`, which is PROMOTEd. Also the own-record FE ceilings (`real_ceiling.csv`). |
| results/T12_excitation_plots/ | figures | EVIDENCE | shared vs own | none | 2.7 MB, 4 PNG | no | |
| results/T13_feasibility/ | finding | EVIDENCE | own | post-P | 5.9 MB | no | `prepare_*` are the provenance of the PROMOTEd `trainval_N2000` and of the rec1–3 files linked into T17. `p_eval.real_zeroshot` reads `real_zeroshot.csv`. |
| results/T14_input_anatomy/ | finding | EVIDENCE | shared vs own | none | 579 KB | no | |
| results/T15_mirror_test/ | finding | EVIDENCE | shared | none | 872 KB | no | |
| results/T16_sign_test/ | finding | EVIDENCE | own | post-P | 4.9 MB | no | `q_eval`'s gates read `eval_summary.csv` and `real_zeroshot.csv`. |
| results/T17_scale_up/ | reference numbers | **PROMOTE** | own / one new shared / shared | post-P | 7.0 MB, 14 files | no | **Report and regression reference** for the new pipeline: `eval_*`, `real_*`, `baseline_compare.csv`. `train_check.json` and `prepare_summary.json` are the launch preconditions of `q_train`. |
| results/T18_excitation_realism/ | finding | EVIDENCE | none | post-P | 1.1 MB | no | |

Results counts: PROMOTE 1, KEEP 1, EVIDENCE 10, ARCHIVE 6.

**Row counts across the whole manifest** (sections 2a–2e, 130 rows):

| category | rows |
|---|---|
| PROMOTE | 24 (14 scripts, 1 results folder, 7 data folders, 2 run groups holding 6 networks) |
| KEEP | 4 |
| KEEP-CHANGE | 27 |
| EVIDENCE | 21 |
| ARCHIVE | 53 |
| UNSURE | 1 |

### 2c. Data (`data/`, gitignored, 28 GB)

Every own-record sample is the paper's stored sample with the five `EXC_KEYS` replaced: `x_psd`, `x_psd_row_scale`,
`edge_attr_cpsd`, `edge_attr_cpsd_resonance` and `edge_attr_cpsd_full`. Every other key (graph, labels, matcher
record, `params_json`) is byte-identical to `Final_results_for_paper/data/pop_broadband`. All records are 60 s at
200 Hz unless the row says otherwise.

| path | kind | category | record | field | size | tracked | reason / role |
|---|---|---|---|---|---|---|---|
| data/T13/trainval_N2000/samples | training + validation pool 0 | **PROMOTE** | own, `SeedSequence([20261125, i])` | none | 2.7 GB, 2000 npz + check.json | no | **Training data**: pool 0 and the validation set (1700 + 300). Its 600 N600 structures are byte copies of T11's `trainval_varexc`. |
| data/T17/rec1_N2000/samples | training pool 1 | **PROMOTE** | own, `[20261131, i]` | none | 2.3 GB apparent, 1700 npz | no | **Training data.** 426 files are hard links to T13 `rec1` (link count 2, verified). |
| data/T17/rec2_N2000/samples | training pool 2 | **PROMOTE** | own, `[20261132, i]` | none | 2.3 GB apparent, 1700 npz | no | Same as rec1 (426 hard links). |
| data/T17/rec3_N2000/samples | training pool 3 | **PROMOTE** | own, `[20261133, i]` | none | 2.3 GB apparent, 1700 npz | no | Same as rec1 (426 hard links). The three rec folders hold 5.2 GB unique beyond T13. |
| data/T11/paper_varexc/samples | test set | **PROMOTE** | own, `[20261126, i]` | none | 685 MB, 500 npz | no | **Primary own-record test set**: the paper's 500 test structures. Caveat: it was looked at to choose the T16 → T17 arms. |
| data/T5/fresh_test/ | test set | **PROMOTE** | one new shared (int seed 20261124); parameters seed 20261123 | none | 746 MB: 500 `samples_phase2_bandpass` + 500 `samples` + logs/config | no | **Secondary test set**: unseen structures under an unseen record, but one realisation for all of them. File names `sample_0000…0499_NgX` **collide with the paper's names** for different structures. |
| data/T5/fdd/ | baseline support | **PROMOTE** (`*_decks.npz`) / ARCHIVE (grids) | none (masks); shared (`paper`, `val` grids); one new shared (`fresh_test` grid) | none | 3.0 MB: 3 npz + 3 csv.gz | no | `paper_decks.npz` and `fresh_test_decks.npz` (support masks, Ng + 2 per deck) are **read by `q_template_compare`**. The grids record every FDD setting. |
| data/T5/fresh_test_paperexc/ | control set | ARCHIVE | shared (12345) | none | 746 MB | no | The paired control behind the finding. Its numbers are in the T5 results. |
| data/T5/predictions/ | predictions | ARCHIVE | shared / one new shared | none | 70 MB, 42 npz | no | Paper networks. |
| data/T11/trainval_varexc/samples | N600 training + validation | ARCHIVE | own, `[20261125, i]` | none | 822 MB, 600 npz | no | Byte-identical subset of `T13/trainval_N2000`. Needed only to reproduce T13 and T16 at N = 600 (their pool 0). |
| data/T11/predictions/ | predictions | ARCHIVE | mixed | none | 34 MB | no | |
| data/T13/rec1, rec2, rec3 | N600 training pools 1–3 | ARCHIVE | own, `[20261131..33, i]` | none | 695 MB each, 508 npz each | no | 426 per folder live on as T17 hard links. The other 82 serve only the N600 T13 and T16 arms. |
| data/T13/trainval_T120 | lever data | ARCHIVE | own, `[20261125, i]`, 120 s | none | 822 MB, 600 npz | no | The 120 s lever failed. |
| data/T13/paper_varexc_T120 | test set, 120 s | ARCHIVE (UNSURE) | own, `[20261126, i]`, 120 s | none | 685 MB, 500 npz | no | Usable as a record-length robustness test if wanted. |
| data/T13/predictions/ | predictions | ARCHIVE | own | none | 35 MB | no | `k_diag` input. |
| data/T15/{paper, mirror, swap_hinge, swap_ends} | mirror variants | ARCHIVE | shared, re-assigned supports | none | 486 MB (100 / 88 / 88 / 88 npz) | no | Evidence numbers are in the T15 results. |
| data/T4/variants (+ `variants_index*.csv`, `check_lists/`) | sparse variants | ARCHIVE | shared (module seed) | none | **13 GB**, 8017 files | no | Must be rebuilt on own records if T4 is redone. |
| data/T10/ | swapped field inputs | ARCHIVE | — | pre-P rows swapped | 11 MB, 9 npz | no | Superseded by `Final_results_for_paper/data/field_inputs_v2` (s5a). |
| data/T6/ | logs, pilot CSVs | ARCHIVE | shared | none | 232 KB | no | |
| data/indomain_per_sample/ | per-sample CSVs | ARCHIVE | shared | none | 1.4 MB | no | T1–T3 scoring of the paper runs. |
| data/splits_nested/{r0, r1, r2} | manifests | KEEP | none | none | 1.1 MB | no | EC-27 nested train/validation lists over the paper population, so still valid for own-record data of the same structures. |

### 2d. Runs (`runs/`, gitignored, 116 MB)

`run_summary.json` and `config.json` hold absolute paths (`dataset_root`, manifests, `run_root`, the python
interpreter in `Linkspan_GNN_FineTuning/.venv-dgl`). They record the edge mode but **not the loss** (sign-invariant
or MSE) and **not the node input** (amplitude).

| path | kind | category | record | field | size | tracked | reason / role |
|---|---|---|---|---|---|---|---|
| runs/ec_t17_cohorient_signinv_N2000_s{123,124,125} | networks | **PROMOTE** | own | post-P (scored) | 1.8 MB each | no | Main-network candidates and regression references. `cpsd_full_orient` (802-d), sign-invariant, 1000 epochs. Best epochs 679, 754, 784. |
| runs/ec_t17_orient_signinv_N2000_s{123,124,125} | networks | **PROMOTE** | own | post-P | 1.3 MB each | no | Orientation-only comparator: the ablation arm. Best epochs 362, 487, **92**; selection on total loss with the frequency head overfitting. |
| runs/ec_t17_gates | gate scratch | ARCHIVE | own | none | 1.3 MB | no | |
| runs/ec_t16_cohorient_signinv, ec_t16_orient_signinv | networks N600 | EVIDENCE | own | post-P | 1.5 MB + 0.96 MB | no | `q_eval` and `q_template_compare` re-score them as gate references. Carry them only if those gates are kept. |
| runs/ec_t16_coh_signinv, \_none_signinv, \_cohorient, \_coh_signinv_amp | networks N600 | ARCHIVE | own | post-P | ~5.4 MB | no | Numbers in T16. The amplitude arm cannot be loaded correctly without `install_amp`, and nothing in its config says so. |
| runs/ec_t16_gates | gate scratch | ARCHIVE | — | — | 1.3 MB | no | |
| runs/ec_t13_coh_K4_N600 | network N600 | EVIDENCE | own | post-P | 1.5 MB | no | Its `training_history.csv` is the reference of `q_train --check` and `p_train --check`. |
| runs/ec_t13_coh_K1_N600, \_coh_K1_N600_T120, \_coh_K1_N2000, \_orient_K4_N600 | networks | ARCHIVE | own | post-P | ~5.6 MB | no | Failed levers. Numbers in T13. |
| runs/ec_t13_gates | gate scratch | ARCHIVE | mixed | — | 7.7 MB | no | |
| runs/ec_varexc_edgegat_N600, ec_varexc_edgegat_geomorient_N600 | networks (T11) | ARCHIVE | own | post-P | 3.1 + 2.6 MB | no | 5000 epochs, K = 1. Superseded. Numbers in T11 and T15. |
| runs/ec_varexc_gates | gate scratch | ARCHIVE | shared | — | 2.1 MB | no | |
| runs/edgegat_N600_s124, s125, s126 | partial (T1) | ARCHIVE | shared | none | 1.1 MB each | no | Stopped at epochs 225–275. Do not resume. |
| runs/_gates, \_paper_mirror, ec_sparse_checks, ec_sweep_r0_N5000 (empty), \_queue_status.txt, \_queue_stdout*.txt | scratch | ARCHIVE | shared | — | 23 + 16 + 33 MB | no | `_paper_mirror` holds byte copies of paper checkpoints. |

### 2e. Other

| path | kind | category | size | tracked | reason |
|---|---|---|---|---|---|
| README.md | task table + finding blockquotes | EVIDENCE | 8 KB | yes (modified) | The audit narrative for docs/history. |
| .gitignore | — | ARCHIVE | — | yes | |
| logs/ (76 files: T11, T13, T16, T17, T18, sp, trainer, gates) | training/eval logs | ARCHIVE (T17_* logs: EVIDENCE) | 3.7 MB | no (ignored) | |
| pipeline/\_\_pycache\_\_/ | — | ARCHIVE | — | no | |
| (root) results/RUN_LEDGER.jsonl | stage ledger | KEEP (old repo) | — | yes (modified) | Holds every EC stage's ledger entries; `k_report` reads it for compute tables. |

---

## 3. Answers to the specific questions

### Q1. Dependency chain of T17 (`q_train`, `q_eval`, `q_template_compare`) and every in-memory patch

#### Module chain

```
q_train / q_eval / q_template_compare / q_prepare
  -> q_common -> p_common -> k_common -> h_common -> ec_common -> fr_common
                                                   (ec_common puts REPO, Final_results_for_paper/pipeline,
                                                    EC_comments/pipeline on sys.path)
q_prepare    -> k_prepare -> h_prepare -> h_common
q_train      -> k_train (install_augmentation, run_one) -> fr_train (paper trainer) -> fr_edge_modes
             -> t_gates.same_history   (only --check)
q_eval, q_template_compare
             -> p_eval -> c_rescore (set_files for fresh_test; module import also pulls s4_indomain -> fr_train.jobs,
                                     and makes results/T5_fresh_test_baselines/ via E.task_dir)
                       -> s5_zeroshot (real_zeroshot), fr_edge_modes
```

#### Paper-pipeline modules

- `Final_results_for_paper/pipeline/fr_common.py`
- `fr_train.py`
- `fr_edge_modes.py`
- `s4_indomain.py` (import only)
- `s5_zeroshot.py`

`s0_generate_population.py` is not needed for re-simulating stored structures. It is needed to generate **new**
ones (`c_gen_fresh_test`).

#### Root library modules, transitively (12 of the 14)

- `EdgeGATConv_Modal_GNN`, which imports `GraphSAGE_Modal_GNN`
- `infer_phase2a_unseen_structure_edgegat`, which imports `infer_phase2a_unseen_structure`, which imports
  `infer_phase1_unseen_structure` and `generate_population_dataset_modes4`
- `generate_population_dataset_modes4`, which imports `export_projected_graph`, `grid_modal_Kinematic_MPC`,
  `simulate_uniform_inertial_u3_white_noise` and `standalone_mode_matching_mmi`
- `grid_modal_Kinematic_MPC`, which imports `build_phase3_graphs`
- `simulate_uniform_inertial_u3_white_noise`, which imports `phase3_utils`

Not needed: `infer_phase2a_deterministic_edgegat` (s6 only) and `analyze_phase1_run`.

#### Data and results the chain reads

- **Paper data:**
  - `Final_results_for_paper/data/pop_broadband/samples_phase2_bandpass`: the source samples for
    re-simulation, and the `paper` control test set;
  - `Final_results_for_paper/data/splits/{N2000,N600}_{train,val}.txt` and `test.txt`.
- **EC data:**
  - `data/T13/trainval_N2000`;
  - `data/T17/rec{1..3}_N2000`;
  - `data/T13/rec{1..3}` (link source);
  - `data/T11/paper_varexc`;
  - `data/T5/fresh_test/samples_phase2_bandpass`;
  - `data/T5/fdd/{paper,fresh_test}_decks.npz`.
- **Check-only data:** the `--check` and gate modes also need `data/T11/trainval_varexc` and `T13 rec1–3`.
- **Field inputs:**
  - `Final_results_for_paper/data/field_inputs_v2/phase3_corrected/<S>/phase3_dataset_corrected.npz` (post-P);
  - `phase3_real_graphs/<S>/graph_package.npz`;
  - `experiments/E6_controls/task2_raw/<S>.npz` (FE targets, read by s5).
- **Gate references:**
  - `results/T16_sign_test/{eval_summary,real_zeroshot}.csv`;
  - `results/T13_feasibility/real_zeroshot.csv`;
  - `results/T5_fresh_test_baselines/{fdd_rows,shape_baselines}.csv`;
  - `runs/ec_t13_coh_K4_N600/training_history.csv`;
  - `runs/ec_t16_{cohorient,orient}_signinv`.

#### In-memory patches

Each must become real code.

1. **Excitation seed** (`h_common.resimulate_payload`, used by `k_common.resimulate`, `k_prepare.data_job` and
   `q_prepare`).
   - It sets `SIM.SEED = np.random.SeedSequence([base, sample_index])` just before
     `gen.simulate_white_noise_from_modal_model(...)`, which calls `np.random.default_rng(SEED)`
     (`simulate_uniform_inertial_u3_white_noise.py:344`). It restores 12345 afterwards.
   - It checks that `gen.simulate_white_noise_from_modal_model.__globals__ is SIM.__dict__`.
   - It is a module global, so it is not thread-safe. It works only because each process simulates one structure
     at a time.
   - **Real code:** a `seed` (or `rng`) argument to the simulate function, with the generator passing a
     per-structure seed.
2. **Record length** (`k_common.resimulate`): `SIM.T_TOTAL = t_total` for the call, with 60.0 restored after.
   Only the T13 120 s sets need it. **Real code:** a `t_total` argument.
3. **Sample assembly** (`k_prepare.data_job`, `h_prepare`).
   - It is not a patch: it copies the stored paper sample, replaces the five `EXC_KEYS` and checks every other key
     byte-identical on read-back.
   - The new pipeline either keeps this "re-excite stored structures" path, which then depends on `pop_broadband`,
     or has the generator write per-structure records directly.
4. **`fr_edge_modes.install()`**, called by `fr_train` itself and by `p_common.install_for`.
   - It wraps `gs._resolve_edge_attr_array` and `gs._edge_feature_dim_for_mode`, and extends
     `gs.EDGE_FEATURE_MODES` with `cpsd_full` and `geom_orient`.
   - It wraps `eg._resolve_edge_attr_array` and extends `eg.EDGE_FEATURE_MODES` with `geom_orient` (cos and sin
     of `edge_attr_geom[:, 1]`) and `zero`.
5. **`p_common.install_cohorient()`**: wraps `eg._resolve_edge_attr_array` so that `"cpsd_full_orient"` returns
   `hstack(cpsd_full[800], orientation[2])` as float32 (802-d), and adds the mode to `eg.EDGE_FEATURE_MODES`.
6. **`p_common.install_signinv()`.**
   - It wraps `eg.EdgeGATModalGNN.forward` to stash `graph.batch_num_nodes()` in a module-global `_STATE["bnn"]`.
   - It replaces `gs._compute_multitask_loss`. The shape loss becomes
     `Σ_graphs Σ_modes min(‖p−y‖², ‖p+y‖²) / numel`. The frequency loss stays MSE on z-scores, weighted 0.25.
   - `fr_train` calls `gs._compute_multitask_loss` through the module attribute in both its training (l.289) and
     validation (l.311) loops, so the **best-validation checkpoint is selected on the sign-invariant total loss**.
   - `EdgeGATConv_Modal_GNN` imports `_compute_multitask_loss` by name (l.39), so its own trainer is **not**
     patched.
   - **Real code:** a loss option that takes the batch node counts explicitly, recorded in `config.json`.
7. **`p_common.install_amp()`** (T16 amplitude arm only, not T17).
   - It wraps `eg.build_phase2a_graph_from_payload` to replace `x_psd` with √(`x_psd` · row scale), divided by the
     deck-wide maximum, with support rows divided by their own maximum.
   - It is not recorded in `config.json` or `run_summary.json`.
8. **`k_train.install_augmentation(pools, train_list, run_dir)`.**
   - **Dataset:** `eg.Phase2AEdgeDataset` is replaced by a factory. Called first with the training manifest's paths,
     it returns a `VariantSet` of K library datasets, one per pool, and checks that edges, `y`, `graph_node_id` and
     frequency labels are identical. Every other call (validation) gets the library class unchanged.
   - **Loader:** `dgl.dataloading.GraphDataLoader` is replaced by an `EpochLoader` subclass whose `__iter__` calls
     `VariantSet.set_epoch()`. That sets `slots = default_rng([20261130, epoch]).integers(0, K, n_train)`, and the
     epoch starts from `resume.pt`. Each draw is appended to `variant_draws.csv`.
   - **Assumptions about `fr_train`:**
     - it imports `GraphDataLoader` at call time (l.129);
     - it iterates the training loader exactly once per epoch;
     - it builds the training dataset first.
9. **`fr_train` driving** (`q_train.run`, `k_train.run_one`).
   - `fr_train.jobs` is replaced by a lambda returning the single `Job("edgegat", n, edge, epochs, seed, note)`.
   - `sys.argv` is rewritten with `--job`, `--threads 2`, `--resume-every 25`, `--run-root`, `--train-list`,
     `--val-list`, `--pool <pool0>` and `--epochs`, then `fr_train.main()` is called.
   - It refuses to write into the paper's `runs/`.
   - Threads: `OMP_NUM_THREADS` and `MKL_NUM_THREADS` = 2 are set before torch is imported.
10. **EC-internal patches.**
    - `q_common` adds `rec{1..3}_N2000` to `k_common.DATASETS` and replaces `k_common.sdir`.
    - `q_eval` and `q_template_compare` do `P.ALL.update(T17 runs)`.
    - `h_eval` and `k_eval` re-point `c_rescore.set_files` and `PRED_ROOT`.
11. **Real-bridge edge inputs** (`p_eval.real_zeroshot`).
    - It is not a library patch, but s5 lacks it. Orientation comes from each hybrid graph's `edge_attr_geom` in
      `graph_package.npz`, with the edge order checked against the feature file.
    - `cohorient` is `hstack(stored coherency, orientation)`. Then `s5_zeroshot.evaluate` is called unchanged.
    - **Real code:** the field-input builder writes the orientation (and 802-d) edges.
12. **Generating new structures** (`c_gen_fresh_test`, not used by T17).
    - `s0_generate_population.install_patches()` sets:
      - `gen.OUT_DIR` and `gen.N_STRUCTURES_TOTAL`;
      - `gen.sample_params_for_ng`, with dy ≤ 3000 mm for Ng ≥ 3;
      - `gen.validate_sampled_params`;
      - `gen.match_target_modes`, which rejects a structure if any selected mode has MAC < 0.85.
    - It also sets `gen.SEED` (the parameter seed) and `SIM.SEED` (one value), guards `clean_output_directory`,
      selects Agg and gives the generator an empty non-TTY stdin (CLAUDE.md rules 4–5).
    - The new generator needs all of this as real code, **plus a per-structure excitation seed**.

### Q2. Own-record datasets

| set | path | structures | seed | length | size | reusable as is? |
|---|---|---|---|---|---|---|
| T13 `trainval_N2000` | data/T13/trainval_N2000/samples | N2000 split: 1700 train + 300 val | `SeedSequence([20261125, idx])` | 60 s | 2.7 GB | **Yes**: pool 0 and validation |
| T17 `rec1–3_N2000` | data/T17/rec{1,2,3}_N2000/samples | 1700 training each | `[20261131/32/33, idx]` | 60 s | 3 × 2.3 GB apparent (5.2 GB unique) | **Yes**: pools 1–3. Copying them breaks the 426 hard links per folder, which is harmless. |
| T11 `paper_varexc` | data/T11/paper_varexc/samples | the paper's 500 test structures | `[20261126, idx]` | 60 s | 685 MB | **Yes**: primary test set. It has been used for arm selection (EC-15). |
| T11 `trainval_varexc` | data/T11/trainval_varexc/samples | N600: 508 + 92 | `[20261125, idx]` | 60 s | 822 MB | Redundant (inside T13 `trainval_N2000`) |
| T13 `rec1–3` | data/T13/rec{1,2,3}/samples | N600 training, 508 each | `[20261131..33, idx]` | 60 s | 695 MB each | 426 live on in T17; the other 82 are N600-only |
| T13 `trainval_T120` | data/T13/trainval_T120 | N600: 600 | `[20261125, idx]` | 120 s | 822 MB | No (lever failed) |
| T13 `paper_varexc_T120` | data/T13/paper_varexc_T120 | 500 test | `[20261126, idx]` | 120 s | 685 MB | Optional robustness set |
| T5 `fresh_test` | data/T5/fresh_test | 500 **new** structures (parameter seed 20261123) | **one** shared int seed 20261124 | 60 s | 746 MB | Yes, as a secondary test set. Note the shared record and the name collisions. |
| T5 `fresh_test_paperexc` | data/T5/fresh_test_paperexc | the same 500 | 12345 (shared) | 60 s | 746 MB | No |
| T15 variants | data/T15/* | 100 / 88 / 88 / 88 paper test structures | 12345, re-assigned supports | 60 s | 486 MB | No |
| T4 variants | data/T4/variants | N2000 split, sparse | 12345 (module) | 60 s | 13 GB | No |

**Seed registry.** Keep it, so that new sets never reuse a base:

| seed | use |
|---|---|
| 20261123 | fresh parameters (T5), T4 layouts, T6 random layouts |
| 20261124 | `fresh_test` record |
| 20261125 | pool 0 and validation |
| 20261126 | test records (`paper_varexc`) |
| 20261127 | real-bridge FE ceilings, T11 (`[base, LOOCV position, k]`) |
| 20261130 | per-epoch draw |
| 20261131–33 | pools 1–3 |
| 12345 | the paper's shared record |

**What the reuse conditions are.**

- **Labels and graphs.** These are the paper generator's. They carry:
  - the per-Ng B1 sign convention, which the sign-invariant loss neutralises;
  - **the Ng = 2 balcony B2/T2 label swap** (568 of 5500).
- **Excitation model.** It is the paper's support model, which T18 found the least field-like.
- **Changing either** means regenerating:
  - a label fix rewrites the label keys only;
  - a new excitation model regenerates every promoted set.

### Q3. Runs

- **PROMOTE** the six T17 networks.
  - `ec_t17_cohorient_signinv_N2000_s123..125` are the main-network candidates and regression references.
  - `ec_t17_orient_signinv_N2000_s123..125` are the orientation-only ablation arm.
  - Whether they stay as final networks depends on the protocol (1000 against 5000 epochs, the selection rule; §5).
- **EVIDENCE, small.** Carry these only if the gates are kept:
  - `ec_t16_cohorient_signinv` and `ec_t16_orient_signinv` (the `q_eval` gate references);
  - `ec_t13_coh_K4_N600` (the `q_train --check` history reference).
- **ARCHIVE:**
  - the other T16 arms, including `coh_signinv_amp`, which cannot be reloaded correctly without an undocumented
    patch;
  - every other T13 arm;
  - the T11 pilot (`ec_varexc_*`);
  - the partial T1 runs `edgegat_N600_s124–126`;
  - every `*_gates` folder, `_paper_mirror`, `ec_sparse_checks`, and the empty `ec_sweep_r0_N5000`.

### Q4. Baselines (T5)

The shared record is a leak problem for **learned** models only. The template, kNN and FDD learn nothing from the
records, so their shared-record rows are valid numbers on that realisation. They are just not paired with own-record
inputs.

| baseline | record | reusable as the new pipeline's baseline? |
|---|---|---|
| Class template (E18 class prior's leading shape; class given; deck nodes) | none | **Yes, as is**: 0.946 on the paper test structures (so `paper_varexc`) and 0.946 on `fresh_test`. `fdd_rows.csv` holds the per-structure rows. |
| Frequency baselines (mean, per-Ng mean, geometry kNN, without E and ρ) | none | **Yes, as is** (`freq_baselines*.csv`). Only the "network_N2000" rows are the paper network's on the shared record. |
| FDD, every deck node, population labeller | `fresh_test`: one new shared record; `paper` and `val`: shared | `fresh_test` **yes** (0.969/0.970; T17 pairs against it). **Not run on `paper_varexc`.** `c_fdd` needs per-structure seeding to do so. The `paper` rows are valid only for the 12345 records. |
| FDD 2 / 4 sensors + E18 fill (E19 path) | same as above | Same as above. These are the sparse-layout baselines for a redone T4/T6. |
| "Given" FDD (frequency + class given) | same as above | Upper reference only |

Code: `c_fdd.py` and `c_freq_baselines.py` are PROMOTE (baselines). `c_fdd` depends on
`experiments/E14_corner_sensors`, `E18_population_basis` (`popbasis`) and `E19_mode_identification` (`ident`,
`stage1_spectra`, `stage2_identify`, `stage2_synthetic`). These must come along or be vendored.

### Q5. T1–T10

| task | comments | re-run under the new pipeline? |
|---|---|---|
| T1 | EC-04, EC-20, EC-21, EC-08 | **Yes**: phase ablation, seeds, parameter cost (EC-21 needs the 802-d edge count) |
| T2 | EC-27, EC-20, BS-41 | **Yes**, if there is a sweep. The manifests carry over. |
| T3 | EC-20, EC-38, EC-40 | **Yes**: seeds of the new main network and field uncertainty |
| T4 | EC-30, EC-41, EC-38 | **Yes** (EC-41 calls it central). The data must be rebuilt on own records. |
| T5 | EC-15, EC-23, EC-25 | **Partly**: EC-15 needs a test set nobody has looked at; EC-23 metrics on the new network; EC-25's FDD on own records. The template and kNN hold. |
| T6 | BS-43, EC-28, EC-29, EC-17, EC-33 | **Yes**, on the new network; T6.2 and T6.3 were never completed |
| T7 | BS-40, JX-10, EC-26, EC-34, EC-35, EC-39, BS-38 | **Yes**, from the new outputs. EC-34's FE-vs-SSI part holds. |
| T8 | EC-37, EC-38 | **Yes**. A sign-invariant fine-tuning loss is needed first. |
| T9 | EC-09–12, 17–20, 31, 32, 37 | **Partly**: code facts hold; field facts on post-P; parameter counts on the new network |
| T10 | EC-31, EC-34 | **No**: done (Phase P). Keep the order test as a gate. |

### Q6. Hard-coded paths needing re-pointing

1. **Paths derived by counting parents.**
   - `ec_common`: REPO is `Path(__file__).parents` three levels up, from `EC_comments/pipeline`.
   - `fr_common`: two levels up from `Final_results_for_paper/pipeline`.
   - Moving code to another nesting depth breaks both (CLAUDE.md rule 6). `h_common`, `k_common`, `p_common`,
     `q_common`, `sp_common` and `r_` add their own directory to sys.path.
2. **Paper data through `fr_common`.**
   - `C.POP_BROADBAND` (`Final_results_for_paper/data/pop_broadband/samples_phase2_bandpass`), `C.SPLITS`,
     `C.RUNS`, `C.RESULTS`.
   - `C.REAL_FEATURES` and `C.FIELD_OUTPUTS` (`data/field_inputs_v2`, post-P).
   - `C.LEGACY_REAL_FEATURES` and `C.LEGACY_FIELD_OUTPUTS` (pre-P).
   - `C.FIELD_RAW_MAT_DIR = REPO.parents[2] / "System identification code/Field measurements 26_270625/For GNN"`,
     **outside the repository**.
3. **Repository-root data.**
   - `s5_zeroshot`: `GRAPH_ROOT = REPO/phase3_real_graphs`, `AUDIT_RAW = REPO/experiments/E6_controls/task2_raw`.
     Its E2 reference paths are used by the s5 gate.
   - Library modules: `build_phase3_graphs` and the generator use CWD-relative paths (`SSI_modal_infor`,
     `phase3_real_graphs`, `population_dataset_modes4`; rule 1).
4. **The `experiments/` tree, imported or read.**
   - `c_fdd`: E14, E18, E19.
   - `d_t6_*` and `sp_*`: E14, and E5 (`build_phase3_features_corrected`).
   - `g_t10_*`: E5.
   - `f_t9_*`: the legacy E5 `phase3_corrected` features.
   - `e_t7`: E2 `stage5_ssi_vs_fe.csv`.
5. **CWD-based scripts.** All six `f_t9_*.py` use `Path.cwd()` and literal
   `"Final_results_for_paper/EC_comments/pipeline/results/T9_facts"`. They also read `field_modal_filtering_outputs/`
   (legacy, pre-P) and `SSI_modal_infor/`.
6. **EC-internal constants.**
   - `h_common`: `TRAINVAL_DIR`, `TEST_DIR`.
   - `k_common`: `T13_DATA`, `sdir`, `T11_TRAINVAL`, `GATE_ROOT`.
   - `q_common`: `T17_DATA`.
   - `q_template_compare`: `DECKS` (`data/T5/fdd`) and `T5_RES`.
   - `c_rescore`: `T5_DATA`, `PRED_ROOT`, `OUT`.
   - `p_eval.set_files`.
   - The precondition JSONs read by the launchers:
     - `q_train --launch` reads T17's `train_check.json` and `prepare_summary.json`;
     - `k_train --launch arms` reads T13's `train_gate.json`, `prepare_gates.json` and `prepare_summary.json`;
     - `k_prepare --part data` reads T13's `prepare_gates.json`.
7. **Absolute paths inside artefacts.**
   - Every `runs/*/run_summary.json` and `config.json`: `dataset_root`, `train_manifest`, `val_manifest`,
     `run_root`, and `python = …\Linkspan_GNN_FineTuning\.venv-dgl\Scripts\python.exe`.
   - `data/T5/*/generation_config.json` (python path, git head).
   - The T10 README's absolute `C:\MyFolder\...\System identification code\...` path.
   - The T8 README's `LINKSPAN_PY` interpreter.
8. **Ledger and launcher.** `tools/run_stage.py` (root), and `results/RUN_LEDGER.jsonl`, which `k_report` reads.

---

## 4. Dependencies the new pipeline must not break

### Seeding and data

- **Per-structure seeding.**
  - The scheme is `SeedSequence([base, sample_index])`, with `sample_index` the four digits of the paper's file
    name.
  - Test bases must stay disjoint from training bases.
  - `fresh_test` reuses indices 0000–0499 for **different** structures. Any new set keyed by index needs its own
    base, and its own name space or folder, never mixed into a pool by file name.
- **Data invariants that `VariantSet` and `k_prepare`'s cross-check assume.** Across records of a structure, only
  `EXC_KEYS` differ. Edges, `y_mode_shape_signed`, `graph_node_id` and the frequency labels must be byte-identical.
  `x_psd` rows peak at 1. |coherency| ≤ 1. `x_psd_row_scale` is stored.
- **Physics and feature constants.**
  - 60 s records, DT 0.005.
  - Welch: Hamming 2048 / 1024 / 2048, 400 bins.
  - Bandpass 1.5–20 Hz, zero-phase.
  - Coherency: Re/Im interleaved.
  - Node features row-normalised by max-abs.
  - All per CLAUDE.md's fixed conventions.

### Model and loss

- **802-d edge layout.** Coherency (800) first, then (cos θ, sin θ) of `edge_attr_geom[:, 1]`. The checkpoints
  depend on this order and on `edge_feat_dim` 802 in `config.json`. The mode string `cpsd_full_orient` must
  resolve when a checkpoint is loaded.
- **Record what the artefacts don't.** The loss (sign-invariant or MSE) and the node-input mode (row-normalised or
  amplitude) are not in any run artefact. The new config must record both.
- **Checkpoint rule.** Best-validation on the **total** loss: sign-invariant shape loss + 0.25 × frequency MSE on
  z-scores, with strict `<`.
- **Augmentation draw.** `default_rng([20261130, epoch])` is independent of torch's RNG. Bit-identical resume holds
  only at a fixed thread count: 2 threads for training, torch's default for s5 and s7.

### Field and scoring

- **Field path (post-P).**
  - `field_inputs_v2` in SSI sensor order.
  - Orientation from the hybrid graph's `edge_attr_geom`, with the same edge order as the feature file (checked in
    `h_real` and `p_eval`).
  - s5's input-dependence gate.
  - Real recordings at 128 Hz interpolated onto the 400-bin grid.
- **Scoring conventions.**
  - Network MAC is over all graph nodes. The template and FDD are over deck nodes, which excludes the Ng + 2
    degree-1 support joints (masks in `data/T5/fdd/*_decks.npz`). `q_template_compare` re-scores the networks on
    deck nodes for the paired comparison.
  - MAC is sign-blind; the signed cosine is the sign diagnostic (T15).
- **Fine-tuning.** `s7_finetune._sparse_loss` is MSE against the SSI shapes. It is **not sign-invariant**, so it
  conflicts with networks trained with the sign-invariant loss.

### Repository rules

- The 14 flat library modules keep their names. The generator's two protected lines stay untouched.
  `grid_modal_Kinematic_MPC` imports `build_phase3_graphs`. Library constants are CWD-relative.
- **Hard links.** T17's `rec*_N2000` share 426 inodes per folder with T13's `rec*`.
  - Editing either side in place changes both: they are the same files.
  - Deleting one link leaves the other intact.
  - A plain copy into the new repo turns them into independent files, about 1.7 GB extra, which is fine.

---

## 5. Open questions

1. **Protocol of the final main network.**
   - T17 used 1000 epochs, where the paper used 5000.
   - The orientation arm, seed 125, was selected at epoch 92 because its frequency head overfits.
   - Keep the T17 networks, or retrain with more epochs, a different selection rule (shape loss alone?) or more
     seeds?
2. **Test-set hygiene (EC-15 again).** `paper_varexc`, `fresh_test` and `paper` were all looked at when choosing
   the T16 → T17 arms. A final test set nobody has looked at would need new structures with per-structure records:
   `c_gen_fresh_test` with per-structure seeding, a new base and a non-colliding name space.
3. **`fresh_test`'s single shared record.** Keep it as a secondary test set, or regenerate it with a record per
   structure?
4. **Labels.** Fix the Ng = 2 balcony B2/T2 label swap (568 structures) before promoting the data? Only the label
   keys of every promoted set would be rewritten.
5. **Excitation model.**
   - T18: the support-noise model is the least field-like; deck forcing or walkers come closer.
   - If the new pipeline changes the model, every promoted dataset and network must be regenerated, and
     `r_excitation_realism`'s models become generator code.
6. **Fine-tuning a network trained with the sign-invariant loss** (s7, T8 v0–v4, T4 `sp_eval_finetune`): use a
   sign-invariant SSI loss, or align each mode's sign to SSI before the MSE? This must be decided before any field
   fine-tuning number.
7. **FDD on own records.** Run `c_fdd` (with per-structure seeding) on `paper_varexc`, so that the network, FDD and
   the template are paired on identical own-record inputs.
8. **Amplitude node input.** It gave +0.03 in T16 but cannot be used on the field inputs, which carry no PSD
   scale. Include it, or drop it?
9. ~~`results/T17_scale_up/prepare_summary.json` shows empty `linked_from_t13` and `complete` dicts.~~
   **Resolved (checked 2026-11-24 by the coordinating session):** the file records `linked_from_t13` 426 and
   `complete` 1700 for each of `rec1–3_N2000`, and each folder holds 1700 samples. The scan misread it.
10. **T1–T4 queue.** The T1 queue (0/49) and the partial runs `edgegat_N600_s124–126` are on the shared record.
    Confirm they are abandoned, not resumed.
11. **Environment.** All runs used `Linkspan_GNN_FineTuning/.venv-dgl` (torch 2.3.0+cpu). The new repo needs its
    own environment; `results/T9_facts/requirements_lock.txt` is the lockfile to start from.
12. **The `experiments/` modules the baselines import** (E14, E18, E19, E5, E6 `task2_raw`): copy them into the new
    repo, vendor them, or keep a read-only reference to the old repo?
