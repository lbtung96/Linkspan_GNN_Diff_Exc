# Restructure manifest: from `Linkspan_GNN_OMA` to `Linkspan_GNN_OMA_Diff_Exc` (2026-11-24)

**Status: inventory only.** Nothing has been copied, moved or deleted yet. Every action below waits for the
author's approval of this manifest.

**Plan agreed with the author (2026-11-24).**
- `Linkspan_GNN_OMA` stays untouched as a frozen archive.
- `Linkspan_GNN_OMA_Diff_Exc` is built by copying in only what is kept.
- The new main pipeline gives **every structure its own excitation record**. It builds on the
  `Final_results_for_paper/EC_comments` work (T11–T20).

**Appendices** (`manifest/`): the detailed per-file inventories from four read-only scans. They contain about
380 rows, each with a category, record dependence, field-input version, size, git status and reason.

| file | area | rows | KEEP | KEEP-CHANGE | RESIMULATE | PROMOTE | EVIDENCE | ARCHIVE | UNSURE |
|---|---|---|---|---|---|---|---|---|---|
| `A_root.md` | repository root (library modules, real-bridge data, tools, docs) | 84 | 29 | 17 | 0 | — | 4 | 33 | 1 |
| `B_final_results_for_paper.md` | the paper pipeline (s0–s11, data, runs, results) | 50 | 17 | 14 | 1 | — | 2 | 15 | 1 |
| `C_experiments.md` | `experiments/E1–E19` | 120 | 9 | 23 | 1 | — | 25 | 59 | 3 |
| `D_ec_comments.md` | `EC_comments/` T1–T18 | 130 | 4 | 27 | — | 24 | 21 | 53 | 1 |

**Categories:**
- **KEEP:** copy as is.
- **KEEP-CHANGE:** needed, but must change.
- **RESIMULATE:** the structures and labels are valid; the simulated responses must be regenerated.
- **PROMOTE:** EC work that becomes the main pipeline.
- **EVIDENCE:** goes to `docs/history/`, outside the pipeline.
- **ARCHIVE:** stays in the old repository only.

---

## 1. The two defects, and what each one invalidates

### 1.1 One excitation record for every structure

- **The cause.**
  - `simulate_uniform_inertial_u3_white_noise.py:62` sets `SEED = 12345`.
  - Line 344 then calls `np.random.default_rng(SEED)` **for every structure**.
  - So every simulated structure in the repository shares one excitation realisation. That covers the paper's
    5500, `population_dataset_modes4/`, every `E*` dataset, and the FE twins of the real bridges (s6, E6, E17,
    E18).
- **The effect.**
  - The network used the shared record as a left/right label for the deck (T14, T15).
  - On structures with their own record, the paper's main network drops from 0.933 to 0.38–0.41 (T5, T11).
- **What is invalidated:** everything computed from simulated responses:
  - input features (`x_psd`, coherency);
  - every trained network;
  - every synthetic and real-bridge number those networks produced;
  - FDD on simulated records;
  - figures and tables of those numbers.
- **What is not invalidated:**
  - structure parameters, FE models, mode shapes and frequencies;
  - labels and their sign convention;
  - graphs and geometry;
  - field recordings and SSI.
- **Where it is fixed:** EC T11 → T13 → T16 → T17. A record per structure, `SeedSequence([base, sample_index])`.
  Four records per training structure, drawn one per epoch. A sign-invariant shape loss. Coherency +
  orientation edges (802-d).
  - T17 result: **0.948 ± 0.004** on unseen records, level with the class template (0.946).
  - Real bridges: 0.525 ± 0.068 against SSI (the paper reported 0.254).

### 1.2 Loggers x40131 / x40133 (sensors 7 and 9) interchanged in the GNN field inputs (T10, Phase P)

- **The SSI results were always correct.**
  - The SSI scripts reorder these two loggers before SSI.
  - So these are correct and **KEEP**:
    - `SSI_modal_infor/*_Data_picked.mat`;
    - `mode_selection.json`;
    - the SSI frequencies and shapes;
    - Table 8 (SSI vs FE, 0.856).
- **The GNN field-input path never applied that reorder.**
  - Every real-bridge **network** number built from those inputs used a deck with sensors 7 and 9 swapped.
  - The swapped inputs moved zero-shot MAC vs SSI from 0.254 to 0.233.
- **Where it is fixed.** Phase P (commit 9c1f5da) fixed **the paper pipeline only**. `FIELD_SENSOR_SWAP = (6, 8)`
  in `fr_common`; `s5a_field_inputs.py` writes `Final_results_for_paper/data/field_inputs_v2/`. Those are **the
  only post-P real-bridge inputs in the repository** (KEEP).
- **Still pre-P (ARCHIVE, or re-point before reuse):**
  - `field_modal_filtering_outputs/*/data`;
  - the feature arrays in `phase3_real_graphs/<S>/*.npz` (the labels in `phase3_dataset.npz` are correct);
  - `experiments/E5_real_transfer/phase3_corrected/`.
  - The defaults `E5`/`E17 FIELD_ROOT` point there, and through them so do E18 stage3 and E19's real path.
  - Every real-bridge network number in `experiments/` and in EC T3, T4, T6, T7 and T8 is pre-P.
- **For the new pipeline:**
  - **Build from `field_inputs_v2`, and make the T10 order check (FDD vector vs SSI shape) a standing gate of the
    field-input builder.** T18 re-checked it: per-bridge median MAC ≥ 0.989.

### 1.3 Also affected: the public package

`../linkspan-modal-gnn` (on GitHub since 2026-11-05) has **both** defects:
- `linkspan_gnn/excitation.py:25` sets `SEED = 12345`, and `:75` calls `default_rng(SEED)` on every call;
- its measured-data path has no 131/133 reorder.

It needs a fix or a notice. That is the author's decision, and nothing has been done about it.

---

## 2. What goes into the new repository

### 2.1 Library modules (repository root, flat, current file names)

| module(s) | category | note |
|---|---|---|
| `grid_modal_Kinematic_MPC`, `standalone_mode_matching_mmi`, `export_projected_graph`, `phase3_utils`, `build_phase3_graphs`, `infer_phase1_unseen_structure`, `infer_phase2a_unseen_structure`, `infer_phase2a_unseen_structure_edgegat`, `analyze_phase1_run`, `GraphSAGE_Modal_GNN`, `structures.py`, `process_field_modal_filtering.py` | KEEP | `analyze_phase1_run` is needed by the s10 figure modules; `structures.py` and `process_field_modal_filtering.py` are outside CLAUDE.md's 14 but needed |
| `simulate_uniform_inertial_u3_white_noise` | **KEEP-CHANGE** | an explicit per-structure excitation seed (no silent default); deck forcing and a noise floor if T20 is adopted |
| `generate_population_dataset_modes4` | **KEEP-CHANGE** | pass `SeedSequence([base, sample_id])` to the simulator; keep the two exact-string lines (rule 4) only if E2/E3 wrappers come along (they do not) |
| `infer_phase2a_deterministic_edgegat` | **KEEP-CHANGE** | a per-bridge seed for the FE-twin simulation; `REAL_STRUCTURE_PARAMS` stays in sync with `build_phase3_graphs.STRUCTURES` |
| `EdgeGATConv_Modal_GNN` | **KEEP-CHANGE** (or KEEP plus patches) | the sign-invariant loss and the 802-d `cpsd_full_orient` mode: real code, or the current in-memory patches (decision 4) |
| `ref_modes_*.npz` (6) | KEEP, **byte-identical** | the matcher templates: they decide which FE mode is B1/T1/B2/T2 and the label sign; pure FE |
| `analyze_phase2a_run`, `build_phase3_node/edge_features`, `inspect_phase3_propagation`, `prepare_edgegat_cpsd_benchmark` | ARCHIVE | not on any live path; their features are pre-P and use the old convention |

### 2.2 Real-bridge data (all record-free)

| path | category | note |
|---|---|---|
| `field_measurements_raw/` | KEEP | raw 128 Hz recordings, raw logger order; irreplaceable; byte-identical to the external `For GNN` folder `fr_common.FIELD_RAW_MAT_DIR` reads |
| `SSI_modal_infor/` (incl. `mode_selection.json`) | KEEP | the authoritative SSI labels, in SSI order |
| `Final_results_for_paper/data/field_inputs_v2/` (188 MB) | KEEP | **the post-P GNN field inputs**, plus `s5a_field_inputs_checks.json` |
| `field_modal_filtering_outputs/<S>/logs/*.json`, `manual_mode_assignments.json` | KEEP (metadata) | settings and the hand-picked mode assignments; needed to re-run s5a. The `data/*.npz` beside them is pre-P (ARCHIVE) |
| `phase3_real_graphs/<S>/graph_package.npz` | KEEP | hybrid graph, sensor positions, `edge_attr_geom` (the 802-d orientation on real bridges) |
| `phase3_real_graphs/<S>/phase3_dataset.npz` | KEEP-CHANGE | its labels are correct (SSI, FE targets with the two T2 overrides); its features are pre-P. Strip them, or keep the file and rely on the builder overwriting them |
| `experiments/E6_controls/task2_raw/<S>.npz` → `fe_target` | KEEP-CHANGE | **gitignored**; read by s5, s6, s7, s10 and EC. Copy the label arrays, or generate `fe_target` in a stage. `phase3_dataset.npz['fe_target_mode_shape']` agrees to 2.9e-8 |

### 2.3 Population: the structures carry over, the responses do not

- **The structures and their labels are still valid.**
  - `Final_results_for_paper/data/pop_broadband/samples_phase2_bandpass/` holds 5500 structures, 7.4 GB.
  - Parameter draws (seed 20260507) use their own random stream, and every rejection happens before simulation.
  - Labels come from the FE modes only.
  - So the 5500 structures, their labels, graphs and the splits (`data/splits/`, gitignored, 525 KB) are valid.
- **Per sample, only five keys depend on the record:** `x_psd`, `x_psd_row_scale`, `edge_attr_cpsd`,
  `edge_attr_cpsd_resonance` and `edge_attr_cpsd_full`. They are about 1.1 of the 1.35 MB. **RESIMULATE**.
- **Two ways to regenerate** (decision 3):
  - Re-simulate the stored structures from `params_json`, the EC route. Labels and graphs stay byte-identical by
    construction; only the five keys are replaced.
  - Run s0 again with per-structure seeds. One pipeline, but the labels come back about 1e-7 off, because the
    eigensolver is not seeded.
- **Own-record sets that already exist** (`EC_comments/data`; reusable if the support excitation is kept):

| set | structures | seed base | size | role |
|---|---|---|---|---|
| T13 `trainval_N2000` | 1700 train + 300 val | 20261125 | 2.7 GB | pool 0 + validation |
| T17 `rec1–3_N2000` | 1700 each | 20261131–33 | 5.2 GB (426 hard links each to T13) | pools 1–3 |
| T11 `paper_varexc` | the paper's 500 test structures | 20261126 | 685 MB | test (already used to choose arms: not unseen) |
| T5 `fresh_test` | 500 new structures | **one shared new record** (20261124) | 746 MB | secondary test only; its indices collide with the paper's names |
| T20 `deck_*`, `walk_*` (being written) | as T17, plus 500 test each | 20261160–79 | about 20 GB | only if a deck excitation is adopted (decision 1) |

### 2.4 Pipeline code

**Paper stages (`Final_results_for_paper/pipeline/`), KEEP-CHANGE unless stated.** Per-stage detail is in
appendix B §2 Q1.

| stage | change |
|---|---|
| `s2` (splits), `s5a` (field inputs), `fr_edge_modes`, `fr_status` | KEEP |
| `s0` | per-structure seed; or replaced by an EC-style re-simulation stage |
| `s1` | record gates; drop the narrowband pairing |
| `s3` / `fr_train` | jobs with 802-d edges, the sign-invariant loss and K records. Keep `fr_train.main()`'s call-time look-ups, which EC's wrappers rely on |
| `s3a` | new trainer proofs (loss, augmentation) |
| `s4` | own-record test sets; 802-d mode |
| `s5` | 802-d real edges; drop or repoint the E2-copy gate |
| `s6` | a per-bridge record for the FE twins (done in T19); 802-d edges |
| `s7` | **a fine-tuning loss compatible with the sign-invariant training** (decision 7) |
| `s8` | each test structure's own seed in the E14 re-simulation; promote `x_psd_resim_maxdiff` to an assert |
| `s9`, `s10` | reuse the skeleton and figure code; new numbers. s10 has an uncommitted 12-line change (mean markers) |
| `s11` | UNSURE: only if an ablation is kept |

**EC code to PROMOTE.** The T17 chain, appendix D §3 Q1:
- `h_common`, `h_prepare` (re-simulation);
- `k_common`, `k_prepare`, `k_train` (record pools, per-epoch draw);
- `p_common`, `p_eval` (sign-invariant loss, 802-d edges, scoring, real-bridge edges);
- `q_*` (T17);
- the baselines `c_fdd` and `c_freq_baselines`;
- `r_excitation_realism` (T18);
- `u_decomposition` (T19);
- `v_*` (T20).

**Patches that must become real code.** Eight library or `fr_train` patches; appendix D §3 Q1 describes each:
1. excitation seed;
2. record length;
3. edge modes;
4. 802-d coherency + orientation;
5. sign-invariant loss;
6. amplitude node input (T16 only; not recorded in any run config);
7. per-epoch record draw;
8. `fr_train` driving.

The loss and the node-input mode are not recorded in any run artefact; the new config must record both.

**Experiment code the live pipeline imports** (copy, keeping module names; re-count the root walks, rule 6):

| code | used by |
|---|---|
| `E5_real_transfer/build_phase3_features_corrected.py` | s5a, s6, T19: the real-feature builder. Make post-P the default; remove its dependence on `population_dataset_modes4/` for the frequency grid |
| `E14_corner_sensors/stage1–3_*.py` | s8, EC T6, the T4 sparse path |
| `E2_training_distribution/figures/`: `figstyle`, `plot_loss_curves`, `plot_synthetic_performance`, `plot_representative_modeshapes`, **`plot_graph_topology`** (five, not the four CLAUDE.md names) | s10, the author's figure style |
| `E16/imputers.py`, `E17/stage1_holdout.py`, `E18` (`popbasis`, `stage2_synthetic`, `prior.npz` (gitignored), `prior_fit.csv`), `E19` (`ident`, `stage1_spectra`, `stage2_identify`, `checks.json`) | the class template and FDD baselines (EC `c_fdd`) |

**Tools:**
- KEEP: `tools/run_stage.py`.
- Re-point their maps: `diff_against_reference.py`, `snapshot_baseline.py`, `check_paths_resolve.py` and
  `gate2_compare_populations.py`. The last is the right gate for "FE fields unchanged, responses re-simulated".
- ARCHIVE: the old contract (`run_all.py`, `verify_reproduction.py`, `make_paper_results.py`,
  `make_results_summary.py`, `migration_*`).

**Networks:**
- The six T17 runs (`ec_t17_*`, 116 MB with the rest of EC `runs/`) are the candidate main network and the
  regression reference. Whether they become final depends on decision 2.
- T20's six runs are pending.
- Everything else is ARCHIVE. `edgegat_N2000` is kept as EVIDENCE: the "before" network of the record finding.

### 2.5 Evidence for `docs/history/` (findings, not pipeline)

| finding | source | still holds? |
|---|---|---|
| the shared record and its effect | EC T5, T11, T12 | yes: this is the finding |
| the record as a left/right label; the physical cue is the same under every record | EC T14, T15 | yes |
| sign-invariant loss; B1 label sign flips with Ng | EC T16 | yes |
| excitation realism vs field | EC T18 | yes |
| real-bridge gap decomposition of the new networks | EC T19 | yes |
| the 131/133 sensor swap | EC T10 (tracked) | yes (fixed by Phase P) |
| the LOOCV leak (legacy vs nested) | E2 | in kind, yes; the numbers are old |
| the feature-convention bug (zero-input response) | E5, E6 task2b | yes |
| 30 FE candidates contaminate labels | E7 | yes |
| FP operator on FE shapes; E9's operator screen; E16's true-shape ceiling | E6 task1, E9, E16 | yes (record-free) |
| SSI vs FE (Table 8) and the T2 overrides | E2 `stage5_ssi_vs_fe*`, `t2_mac_diagnostic/` | yes (record-free) |
| E18 prior / kriging; E19 class template (0.946 / 0.879) | E18, E19 | yes (record-free; T5 re-confirmed on fresh records) |
| mm-length saturation kills the old geometry edges | E15 README (untracked) | yes |
| **"EdgeGAT signs torsion robustly through the edges"** | E12 | **no**: contradicted by T15. The copy needs a header note |
| E8 B4/B7 ("the network uses the transverse-edge sign") | E8 | to be re-read as the record label. The copy needs a header note |
| `docs/TORSION_TRANSFER_FINDING.md` | E8 b3/b3b | the numbers come from pre-P raw records: re-run post-P before quoting |

---

## 3. What stays in the archive

- **Paper pipeline:**
  - `Final_results_for_paper/pipeline/results/PAPER_RESULTS.md`, and every s4–s8 and s11 table;
  - figures fig1–fig8 and fig10–fig15;
  - every paper run;
  - the narrowband population (8 GB, never trained on);
  - `pop_broadband/samples/` (598 MB, redundant).
- **Repository root:**
  - `population_dataset_modes4/`, `runs_phase2/`, `runs_phase3/`;
  - the pre-P field features;
  - `PAPER_RESULTS_OLD.md`;
  - `results/` (the old regression contract).
- **Experiments:**
  - every trained-network number in E1–E4, E7 (in-domain), E10–E14 and E16 (network part);
  - their datasets and checkpoints, about 6 GB.
- **EC_comments:**
  - T1–T4 and T6–T8 (shared record, partly pre-P; T1 never completed a run);
  - T4's 13 GB of variants;
  - T13's failed levers (120 s);
  - the T11 pilot and the T16 arms not carried forward.

T1–T9 answer review comments and **must be re-run on the new network**; appendix D §3 Q5 lists which comments
each covers.

---

## 4. Easy to lose: do before freezing the archive

1. **Untracked work exists only on disk.**
   - **Almost all of `EC_comments/pipeline/`**: only `README.md`, `ec_common.py`, `g_t10_*.py` and the T10
     results are in git. That includes the T11–T20 code and results that the new pipeline is built on.
   - All of E15.
   - E18's `make_pages.py`, `pages/` and the explainers.
   - E16's `explorer/`.
   - E14's figures.
   - **Commit these to the old repository before freezing it, or they cannot be traced.**
2. **Gitignored but needed:**
   - `E6_controls/task2_raw/` (`fe_target`);
   - E18 `prior.npz`;
   - `Final_results_for_paper/data/splits/`;
   - `field_inputs_v2/`;
   - the own-record datasets.
   - They are copied as files, not through git.
3. **The working tree has uncommitted edits:** `s10_figures.py`, `.gitignore`, two experiment READMEs,
   `EC_comments/README.md` and `RUN_LEDGER.jsonl`.
4. **Disk space.** 48 GB free on C: before T20; about 27 GB after it.
   - Copying the population (7.4 GB) and the promoted own-record sets (about 11 GB, or about 31 GB with T20) as
     plain files would nearly fill the disk.
   - **Use hard links** (same NTFS volume): the files are never edited in place.

---

## 5. Decisions needed before copying (author)

1. **Excitation model of the new population.** Options:
   - support white noise (the paper's model; T17 shows it works synthetically);
   - deck white noise + a noise floor;
   - walkers + ambient + a noise floor.

   T18 found deck forcing more field-like; T20 (running) tests whether it transfers better. **This decides which
   own-record datasets are promoted and whether every set is re-simulated.**
2. **Protocol of the final main network.**
   - Epochs: T17 used 1000, the paper 5000.
   - Number of seeds: T17 used 3; real-bridge numbers vary with the seed, SD 0.068.
   - Checkpoint rule: the total loss picked epoch 92 for one orientation run, because the frequency head
     overfits. Shape loss only?
3. **Re-simulate the stored 5500, or regenerate with s0?** Re-simulating keeps structures and labels
   byte-identical; regenerating is one pipeline, with labels about 1e-7 off.
4. **May library modules be edited in the new repository?**
   - Candidates: a simulator seed argument, the loss and 802-d edge mode as real code, and possibly a seeded
     `eigsh` start vector (makes FE re-runs bit-exact; the public package already does it).
   - This replaces CLAUDE.md rule 3 for the new repository, and decides KEEP versus KEEP-CHANGE for the GNN
     modules.
5. **The Ng = 2 balcony B2/T2 label swap.**
   - All 568 two-girder balcony decks label a torsion mode "B2" and a bending mode "T2". The matcher and
     `ref_modes_Ng2_Balcony.npz` produce it.
   - It is not in PROVENANCE, and it carries into any population built by the same matcher.
   - Fix it (a label-key rewrite) before training?
6. **A test set nobody has looked at (EC-15).** `paper_varexc`, `fresh_test` and `paper` were all used to choose
   the T16 → T17 arms. A final test set needs new structures, a record each, a new seed base and its own name
   space.
7. **Fine-tuning a network trained with the sign-invariant loss.** s7's loss is MSE against signed SSI shapes.
   Use a sign-invariant loss, or align each mode's sign first? This must be decided before any field fine-tuning
   number (s7, T8, T4).
8. **Legacy faithfulness gates** (s5/s7 vs E2 runs, s5a vs the legacy field outputs). Drop them, or point them at
   the archive by absolute path? Do not copy shared-record checkpoints just to keep them green.
9. **Scope of the new paper.** Which analyses are redone on the new network:
   - the ablation grid (T1);
   - the population sweep (T2);
   - sparse-mask training (T4, which EC-41 calls central);
   - the sensor sweep (s8), imputers (E16), the real hold-out (E17).
10. **The public package** (§1.3): fix, or add a notice?
11. **Environment.** Every run used `Linkspan_GNN_FineTuning/.venv-dgl` (torch 2.3.0+cpu). The new repository
    needs its own; `EC_comments/pipeline/results/T9_facts/requirements_lock.txt` is the lockfile to start from.

---

## 6. Proposed layout of the new repository (for discussion)

```
Linkspan_GNN_OMA_Diff_Exc/
  CLAUDE.md, README.md, PROVENANCE.md     new: the per-structure-record rules; history pointer to the archive
  <library modules, flat>                  §2.1, plus ref_modes_*.npz and structures.py
  pipeline/                                s0-s11 rebuilt + the promoted EC chain, as real code
  baselines/                               class template, FDD, frequency baselines (+ the E16-E19 modules they need)
  field/                                   raw recordings, SSI labels, field_inputs_v2, graph packages
  data/                                    population (hard-linked), own-record sets, splits (gitignored)
  runs/                                    networks (gitignored)
  results/                                 tables, figures, PAPER_RESULTS.md (generated)
  docs/history/                            §2.5 evidence, with header notes on E8/E12
  tools/                                   run_stage.py and the re-pointed regression tools
```

The scans were read-only. The appendices list each file's category, record dependence, field-input version
(pre-P / post-P), size and git status, and every path constant that must be re-pointed.
