# Provenance

Where everything in this repository came from, what was changed on the way, and the evidence that this
repository's code reproduces every result it carries over.

## 1. Origin

This repository was built on 2026-11-24 from the earlier pipeline `Linkspan_GNN_OMA` (the *archive*: private,
frozen, not published). `tools/migrate_from_archive.py` copied code and results and hard-linked or copied the
data; every operation is logged in [docs/migration_log.json](docs/migration_log.json).

| commit | what |
|---|---|
| `270ca44` | **Batch 1**: verbatim copy at archive commit `f4e1ad5`. Code exactly as it ran there (library modules, the paper-pipeline modules the review tasks import, the T5 and T10–T20 scripts, the experiment modules the baselines import, `tools/run_stage.py`) and the T5, T10–T20 results as the reference. |
| `d145475` | Paths re-pointed to this layout (`data/`, `runs/ec`, `runs/paper`, `results/`, `logs/`). No numerical code changed. |
| `7f5a063` | The archive's in-memory patches become library and pipeline code (§2). Reproduction harness. |
| `494b1f2` | **Batch 2**: the finished T20 networks and results, at archive commit `6d14d11`. |
| `e212c48` | T10 pinned to its pre-Phase-P inputs; the fresh-test check made sign-aware; CRLF endings restored in ten files. |
| *(this commit)* | The reproduction: every carried-over result re-computed (§3). |

**One edit to history before publication.** `docs/history/EC_comments_README.md` named the three co-authors
whose review comments the tasks answer and gave a local path to the author's comment spreadsheet. The names were
replaced by the comment-ID initials already used throughout (EC, BS, JX) and the path removed, in every commit;
the commits were rewritten before the first push, with their messages, authors and dates unchanged. No other
file differs from the archive copy because of this.

Results under `results/paper/` are the archived paper's tables and figures (shared excitation record), kept as
history and as the reference for gates. They were produced by the archive's own pipeline and its regression
contract, not re-computed here.

## 2. What changed in the code, and why it changes no number

| change | where | proof |
|---|---|---|
| Paths re-pointed (200 plain string replacements; `REPO` one parent up from `pipeline/`) | `d145475` | every stage re-run below reads and writes the new layout |
| Excitation record as an argument: `simulate_white_noise_from_modal_model(..., seed=None)` (None = the module's SEED 12345, the archived shared record); `t_total` likewise. Replaces setting `SIM.SEED` / `SIM.T_TOTAL` in memory | library simulator; `h_common`, `k_common`, `h_real`, `u_decomposition`, `r_excitation_realism`, `s6.simulate_real` | every stored record re-simulated through the argument (§3.2) |
| Edge modes `cpsd_full`, `geom_orient`, `zero`, `cpsd_full_orient` (802-d) and the amplitude node input as library options (were `fr_edge_modes.install`, `p_common.install_cohorient`, `p_common.install_amp`) | `GraphSAGE_Modal_GNN.py`, `EdgeGATConv_Modal_GNN.py` | byte-identical graphs on 5 samples before committing; training histories and weights below |
| Sign-invariant shape loss as `_compute_multitask_loss(..., shape_loss="signinv")` (was `p_common.install_signinv`) | `GraphSAGE_Modal_GNN.py` | bit-equal on 20 random batches; T16/T17 retrained (§3.3) |
| Trainer as a function, `fr_train.train(Job, pools=...)`; the per-epoch record draw in `pipeline/records.py` (the code of `k_train.install_augmentation`, unchanged). Replaces swapping `fr_train.jobs` and `sys.argv` | `fr_train.py`, `records.py`, `h_/k_/p_/q_/v_train.py` | T17 retrained in full, other set-ups as 50-epoch prefixes (§3.3) |
| T10 pinned to the field inputs and the `s5_zeroshot.csv` it ran against (before Phase P interchanged sensors 7/9 back) | `g_t10_*.py`, `results/T10_channel_order/s5_zeroshot_preP.csv` (bytes of archive `9c1f5da^`) | T10 re-run bit-identical |
| `c_rescore` falls back to `fr_edge_modes` when T1's `t_edge_modes.py` is absent (T1 is archived); `fr_edge_modes` provides every mode the 14 paper networks use | `c_rescore.py` (unchanged code) | rescore tables equal except the recorded module name (§3.1) |
| Stages that study the archived shared record (`c_gen_fresh_test`, `c_fdd`, `x_plot_excitation`) keep the module seed on purpose | — | — |

## 3. Reproduction evidence

Everything here was run on 2026-11-24/25 with the interpreter pinned in `requirements_lock.txt` (CPU, torch
2.3.0+cpu), through `tools/run_stage.py` (each run is in `results/RUN_LEDGER.jsonl`).

### 3.1 Result tables: every scoring stage re-run

`tools/reproduce.py` re-ran each stage behind the carried-over results, in the archive ledger's order and with
its arguments, writing over the committed archive copies; `tools/compare_results.py` then classified every file
whose bytes changed. Per task: [results/reproduction/RESULTS_REPRODUCED.md](results/reproduction/RESULTS_REPRODUCED.md);
per file: [results/reproduction/RESULTS_COMPARED.md](results/reproduction/RESULTS_COMPARED.md).

All 44 steps of the plan (`tools/reproduce.py --list`; T5, T10–T20) ran to exit 0 on 2026-11-24, 18:31–22:27,
while other jobs shared the machine. One earlier attempt at T10's second step failed its gate against the
post-Phase-P table; that is what led to pinning T10 (§2), and the step was re-run after the pin.

| task | stages | files rewritten | byte-identical | numbers within 1e-5 | only timing / path / module strings / row order | generated README text | not rewritten | largest numeric difference |
|---|---|---|---|---|---|---|---|---|
| T5 | 10 | 25 | 14 | 6 | 4 | 1 | 0 | 1.8e-7 (FDD MAC) |
| T10 | 2 | 7 | 7 | 0 | 0 | 0 | 2 | 0 |
| T11 | 10 | 15 | 5 | 5 | 4 | 1 | 7 | 9.5e-7 Hz (FE-twin ceiling frequency) |
| T12 | 1 | 4 | 4 (figures) | 0 | 0 | 0 | 0 | 0 |
| T13 | 8 | 14 | 6 | 2 | 5 | 1 | 3 | 5.6e-8 |
| T14 | 2 | 13 | 13 (incl. 5 figures) | 0 | 0 | 0 | 1 | 0 |
| T15 | 1 | 6 | 1 | 5 | 0 | 0 | 1 | 6.6e-7 (mirror-test MAC) |
| T16 | 2 | 7 | 6 | 0 | 1 | 0 | 2 | 0 |
| T17 | 3 | 11 | 9 | 1 | 1 | 0 | 3 | 2.1e-12 |
| T18 | 1 | 7 | 2 (figures) | 5 | 0 | 0 | 1 | 1.7e-7 (peak-to-background, value ≈ 23) |
| T19 | 1 | 5 | 0 | 5 | 0 | 0 | 1 | 1.9e-6 Hz (FE-twin predicted frequency) |
| T20 | 3 | 12 | 8 | 4 | 0 | 0 | 4 | 2.7e-7 (noise-fit peak-to-background) |

- **Every evaluation of a stored network on stored inputs is byte-identical**: T16's, T17's and T20's synthetic
  and real-bridge tables, T14's input anatomy and all its figures, T10 and T12. In T5 and T11 the scored tables
  differ only in the forward-pass timing column.
- **The numbers within 1e-5 all come from stages that re-solve the FE model** (FE twins of the real bridges,
  re-simulated test structures, FDD on re-simulated responses, the noise fit): ARPACK's `eigsh` has no fixed
  start vector, so its eigenvectors move in the 7th–8th digit, and that is what reaches the scores (at most
  1.9e-6 Hz and 1e-6 MAC, through the network).
- **The three generated READMEs** (T5 `c_report`, T11 `h_report`, T13 `k_report`) differ, by reading the diffs, only in
  their compute-time tables (minutes per stage on this run), the re-pointed paths in their usage lines, and
  gate maxima at the FE re-solve level (e.g. 1.1e-8 → 3.0e-8). T5's also carries one corrected sentence (edge
  modes are library options; it named T1's `t_edge_modes.py`).
- **Not rewritten** (25 files): hand-written READMEs, the training launch records, the data-preparation summaries
  (`prepare_summary.json`, `prepare_checks.csv`: written by the data stages, see §3.2), T11's post-processing
  logs and T10's draft provenance note.

### 3.2 Stored excitation records: re-simulated in memory

The record sets are inputs (hard-linked, read-only), written by the archive's data stages. `tools/verify_records.py`
re-simulated every one of them from the stored structure and its seed through this repository's simulator and
compared the five excitation-dependent arrays (tolerance 1e-5: the FE model is re-solved and ARPACK's `eigsh` has
no fixed start vector), and the graph and labels byte for byte.

Summary: [results/reproduction/records_summary.csv](results/reproduction/records_summary.csv); one row per record in
`records_per_sample.csv.gz`. Maximum absolute difference per set (the labels and graph are byte-identical in
every record):

| set | records | seed base | node PSD `x_psd` | coherency `edge_attr_cpsd_full` | PSD peak `x_psd_row_scale` |
|---|---|---|---|---|---|
| T11 `trainval_varexc` | 600 | 20261125 | 4.1e-7 | 3.6e-7 | 7.0e-7 |
| T11 `paper_varexc` | 500 | 20261126 | 2.5e-8 | 1.5e-7 | 4.4e-8 |
| T13 `rec1`–`rec3` | 3 × 508 | 20261131–33 | 2.8e-7 | 1.0e-6 | 8.8e-8 |
| T13 `trainval_N2000` (T17's pool 0 + validation) | 2000 | 20261125 | 4.0e-7 | 7.1e-7 | 7.1e-7 |
| T13 `trainval_T120`, `paper_varexc_T120` (120 s) | 600 + 500 | 20261125 / 20261126 | 1.9e-7 | 4.5e-7 | 2.8e-7 |
| T17 `rec1`–`rec3_N2000` | 3 × 1700 | 20261131–33 | 1.3e-7 | 8.2e-7 | 5.3e-7 |
| T20 `deck_p0`–`p3` | 2000 + 3 × 1700 | 20261160–63 | 4.8e-7 | 1.9e-7 | 1.1e-6 (relative 2.9e-7) |
| T20 `deck_test` | 500 | 20261169 | 9.0e-8 | 8.9e-8 | 1.5e-7 |
| T20 `walk_p0`–`p3` | 2000 + 3 × 1700 | 20261170–73 | 1.9e-7 | 2.3e-7 | 6.9e-3 (relative 2.0e-7) |
| T20 `walk_test` | 500 | 20261179 | 1.7e-7 | 8.9e-8 | 2.5e-3 (relative 1.5e-7) |
| **all 21 sets** | **26,024** | | | | **0 failures** |

**One tolerance was corrected after the first run.** `x_psd_row_scale` is the only key in physical units (each
node's PSD peak before normalisation: about 10 under deck forcing, up to 5×10⁵ under the T20 walkers). The
first full run judged it at 1e-5 absolute like the normalised keys and flagged 578 of the first 2240 walker
records (absolute differences up to 5.7e-3 on values up to 4.8×10⁵), while every other key passed. The key is
now judged at 1e-5 relative above 1 (the rule of `tools/compare_results.py`), and all 15,200 T20 records were
re-simulated under it: the largest relative difference is 2.9e-7. T11/T13/T17 records had already passed the
stricter absolute rule and were kept.

T5's two fresh test sets (500 new structures each) were written by the archive's generator wrapper.
`tools/verify_fresh_test.py` regenerated the first 10 structures per girder count of each set (a true prefix:
each Ng draws from its own stream): 40 + 40 structures pass, max difference 6.0e-8 and 1.5e-7. The raw
eigenvector sign differs from the stored one in 83 and 82 of 160 modes (eigsh's sign is arbitrary per solve);
the four keys taken before sign alignment agree up to that sign, and the aligned labels agree to 2.7e-11.

### 3.3 Networks: retrained through this repository's trainer

`tools/verify_training.py` retrains through the stage wrappers' own entry points (`h_train`, `k_train`,
`p_train`, `q_train`), each run in its own process with 2 torch threads, into `runs/repro/` (never over the
archived runs), and compares with the archived run: the training history number for number
(`t_gates.same_history`: epoch, every loss, MAE, sizes, seed, learning rate) and, for full runs, the weights of
`best_model.pt` and `last_model.pt` bit for bit, the best epoch and the best validation loss.
Summary: [results/reproduction/training_summary.csv](results/reproduction/training_summary.csv).

**T17, the main-network candidates: all six retrained in full** (N = 2000, 4 record pools, 1000 epochs;
2026-11-24 18:03–22:59, four at a time):

| run | history (1000 epochs) | best_model.pt | last_model.pt | best epoch (re-run / archive) |
|---|---|---|---|---|
| `ec_t17_cohorient_signinv_N2000_s123` | equal | bit-identical | bit-identical | 679 / 679 |
| `ec_t17_cohorient_signinv_N2000_s124` | equal | bit-identical | bit-identical | 754 / 754 |
| `ec_t17_cohorient_signinv_N2000_s125` | equal | bit-identical | bit-identical | 784 / 784 |
| `ec_t17_orient_signinv_N2000_s123` | equal | bit-identical | bit-identical | 362 / 362 |
| `ec_t17_orient_signinv_N2000_s124` | equal | bit-identical | bit-identical | 487 / 487 |
| `ec_t17_orient_signinv_N2000_s125` | equal | bit-identical | bit-identical | 92 / 92 |

Identical weights mean every T17, T19 and T20 number computed from these networks follows from this
repository's trainer, not only from the archived checkpoints.

**Every other training set-up: the first 50 epochs** (2026-11-24 23:00–23:26, four at a time). There is no
learning-rate schedule and no early stopping, so a 50-epoch run is the archived run's beginning; all 15
histories are equal to the archived ones number for number:

| task | runs (50 epochs each, history equal) | archived length |
|---|---|---|
| T11 | `ec_varexc_edgegat_N600`, `ec_varexc_edgegat_geomorient_N600` | 5000 epochs |
| T13 | `ec_t13_coh_K1_N600`, `_K4_N600`, `_K1_N600_T120`, `_K1_N2000`, `ec_t13_orient_K4_N600` | 400 |
| T16 | `ec_t16_none_signinv`, `_coh_signinv`, `_coh_signinv_amp`, `_cohorient`, `_cohorient_signinv`, `_orient_signinv` | 400 |
| T20 | `ec_t20_deck_cohorient_signinv_N2000_s123`, `ec_t20_walk_cohorient_signinv_N2000_s123` (one seed per excitation) | 1000 |

Together these cover every training path the archive used: one record per structure (T11), the per-epoch draw
over four record pools (T13, T16, T17, T20), 120 s records (T13), the sign-invariant loss, the 802-d edges and
the amplitude node input (T16), and T20's deck-forcing records.

## 4. Not reproduced here, and why

- **T1–T4 and T6–T9** were not carried over (archived): T1–T3 replicate the archived paper's shared-record
  protocol, T4's training was never launched, and T6–T9 (sensitivity checks, analyses of stored outputs,
  fine-tuning variants, facts read from code) concern the archived paper's networks. Their results stay in the
  archive.
- **Data-writing stages** (`h_prepare`/`k_prepare`/`q_prepare --part data`, `v_prepare --run`,
  `c_gen_fresh_test`) were not re-run into `data/`: their outputs are the hard-linked inputs. They are verified
  in memory (§3.2), the fresh test sets by prefix.
- **Training other than T17** was verified by 50-epoch prefixes, not in full (T11's two arms trained 5000 epochs,
  T13's and T16's 400, T20's 1000 epochs on 2000 structures × 4 record pools). A prefix is the full run's
  beginning number for number, because there is no learning-rate schedule or early stopping; the evaluation
  stages then score the archived checkpoints.
- **Timing columns** (`forward_ms_median`, `forward_s`, minutes in T5's compute table) are machine noise and
  differ on every run.
- **`results/paper/`** is the archived paper's output, carried as history (§1).
