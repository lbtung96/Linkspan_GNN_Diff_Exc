# CLAUDE.md

Guidance for Claude Code working in this repository.

This is the **main pipeline since 2026-11-24**: every simulated structure has **its own excitation record**.
It was built from the frozen archive `../Linkspan_GNN_OMA` (commit `f4e1ad5`), whose paper pipeline gave
every structure the same record (the defect found in the co-author review, EC T5). What was kept, what was
archived and why: [RESTRUCTURE_MANIFEST.md](RESTRUCTURE_MANIFEST.md). What reproduces, and how faithfully:
[PROVENANCE.md](PROVENANCE.md). The archive's own rules are in `docs/history/CLAUDE_archive.md` — read
them only for history; this file is authoritative here.

Start with [README.md](README.md) and [pipeline/README.md](pipeline/README.md).

---

## Hard rules

1. **Run everything from the repository root, through `tools/run_stage.py`** —
   `python tools/run_stage.py pipeline/<stage>.py [args]`. It sets the working directory, logs to `logs/`
   and appends to `results/RUN_LEDGER.jsonl`. Library modules read some paths relative to the working
   directory (`SSI_modal_infor/`, `phase3_real_graphs/`).

2. **Every new simulation passes an explicit excitation seed.** `simulate_white_noise_from_modal_model(...,
   seed=...)`; the pipeline uses `np.random.SeedSequence([base, structure_index])`. `seed=None` falls back to
   the module's `SEED = 12345` — the archived paper's single shared record. Only the history stages that
   *study* that record (T5 `c_gen_fresh_test`, `c_fdd`; T12 `x_plot_excitation`) may rely on the fallback.
   Never reuse a seed base: the registry is in [DATA.md](DATA.md). Test bases stay disjoint from training
   bases.

3. **Never write into a hard-linked data set.** The population (`data/population/`), the own-record sets
   (`data/T11`, `T13`, `T17`, `T20`, `data/T5/fresh_test*`), `data/field_inputs_v2/` and `data/legacy_preP/`
   are hard links to the archive's files, marked read-only. A stage that rewrote one in place would change the
   archive too; that is why they are read-only. New data goes to new folders. Re-simulation checks compare in
   memory (`tools/verify_records.py`).

4. **CPU torch only; 2 torch threads per training process.** Thread count is part of bit-reproducibility
   (inference differs in the last float32 bit across thread counts; the T17 real-bridge rows reproduce only
   at torch's default thread count, s6-style rows only at 2). The GPU does not help this model.

5. **Memory: at most three N = 2000 networks with four record pools training at once.** Each holds about
   7 GB resident / 9.4 GB committed. Six at once (2026-11-24) exhausted the 32 GB of RAM, killed one run and
   grew `pagefile.sys` to 59 GB, filling C:. N = 600 runs and orientation-only runs are small.

6. **Library modules stay flat at the root, with their file names.** Every internal import is
   `from <module> import ...`. Since the restructure they *may* be edited, but only additively and so that
   every archived result still reproduces bit for bit (`tools/reproduce.py`, `tools/verify_*.py`). Do not
   reformat the two lines `OUT_DIR = Path("population_dataset_modes4")` / `N_STRUCTURES_TOTAL = 800` in
   `generate_population_dataset_modes4.py`: `pipeline/s0_generate_population.py` patches them by exact match.

7. **Never hand-edit a results table.** Every table under `results/` is written by a stage.

8. **This repository is public on GitHub (`lbtung96/Linkspan_GNN_Diff_Exc`). Push code, results and docs
   only.** `data/`, `runs/`, `logs/`, `field_measurements_raw/`, `SSI_modal_infor/`, `phase3_real_graphs/`
   and `manifest/cleanup_*.md` are gitignored — keep them so. Check `git status` before every push.

9. **Close out any change that produces numbers:** run it through `run_stage.py`; re-run
   `tools/reproduce.py` for the tasks it touches (or say why not); say which numbers moved and by how much;
   update PROVENANCE.md if what reproduces changed; commit with the numbers in the message.

---

## What the pipeline is (since 2026-11-24)

| | |
|---|---|
| Structures | the archive's 5500 (1375 per Ng 2–5, generator seed 20260507), labels and graphs unchanged |
| Records | own record per structure, `SeedSequence([base, idx])`; **four** per training structure, one drawn per epoch (`pipeline/records.py`, `EPOCH_SEED = 20261130`) |
| Excitation | support white noise (the paper's model) for T11–T19; T20 tests deck forcing + a noise floor |
| Network | EdgeGATConv, edges `cpsd_full_orient` = 800-d complex coherency + 2-d orientation (802-d) |
| Loss | sign-invariant shape loss + 0.25 × frequency MSE (`Job(shape_loss="signinv")`) |
| Main candidate | T17: N = 2000, 1000 epochs, seeds 123–125, best-validation checkpoint (total loss) |
| Field inputs | `data/field_inputs_v2` only (Phase P: loggers x40131 / x40133 in SSI order) |

**The final protocol is not decided** (epochs, seeds, checkpoint rule, excitation model, an unseen test set,
the Ng = 2 balcony label swap — RESTRUCTURE_MANIFEST.md §5). Do not present T17 as final.

## Things that are easy to get wrong

- **The sign-invariant loss and the best-validation rule.** The checkpoint is chosen on the total validation
  loss; the frequency head overfits, so one orientation run picked epoch 92. A shape-only rule is an open
  decision, not a fix to apply silently.
- **Torsion's sign is undetermined without a left/right cue** (T13–T16). MSE on signed torsion labels pulls the
  prediction to zero; that is why the loss is sign-invariant. The B1 label sign also flips with Ng.
- **The shared record was a left/right label** (T14, T15). Any result computed on the paper's record
  (`runs/paper/`, `data/population` features, `fresh_test_paperexc`) is about the old pipeline.
- **Field inputs:** only `data/field_inputs_v2`. `data/legacy_preP/` has sensors 7 and 9 interchanged and is
  read only by T10's history stages and s5a's legacy gate. The E5 and E17 builders still *default* to it —
  pass the post-P root explicitly in any new code.
- **SSI labels and the two T2 FE-target overrides (Hellesylt T2 → #7, Leknes T2 → #6) are label-only**
  (archive CLAUDE.md, still true): never let them change which modes are simulated.
- **The Ng = 2 balcony B2/T2 label swap** (568 of 5500 structures label a torsion mode "B2") is still in the
  labels. Fixing it is a decision (manifest §5.5) and moves every number.
- **Fixed conventions** (unchanged from the archive): modes `["B1", "T1", "B2", "T2"]`; node features Welch PSD
  (Hamming, 2048/1024/2048, first 400 bins) row-normalised by max-abs; coherency `Pxy/sqrt(Pxx Pyy)`
  interleaved Re/Im, 800-d; bandpass 1.5–20 Hz zero-phase; real recordings 128 Hz interpolated onto the
  400-bin training grid (Δf = 0.09765625 Hz). Checkpoints: node dim 400, edge dim 800 / 802 / 2.
- **Tolerances.** Inference over stored inputs is bit-reproducible at a fixed thread count (`--atol 0`). Anything
  that re-solves the FE model (`eigsh`, no fixed start vector) reproduces to about 1e-7; use 1e-5.
- **`results/paper/` and `runs/paper/`** are the archived paper's tables and networks (shared record). They are
  gate references and history, never baselines for a new claim.

## Verification

```bash
python tools/reproduce.py --list            # every analysis stage behind the carried-over results
python tools/reproduce.py --report          # results/reproduction/RESULTS_REPRODUCED.md
python tools/verify_records.py              # every stored record re-simulated in memory (1e-5)
python tools/verify_training.py --report    # retrained networks vs the archived runs
```
