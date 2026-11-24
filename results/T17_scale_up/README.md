# T17 — scale-up of T16's two best set-ups: N = 2000, 1000 epochs, three seeds (2026-11-24)

> EC-comments work (see `../../../README.md`). Not part of `PAPER_RESULTS.md`. The protocol is not the
> paper's: own excitation records per structure, sign-invariant loss, 1000 epochs, seeds 123–125.
> Seed means and spreads are reported here; no seed-123 row reproduces a paper number.

Scripts, all run through `tools/run_stage.py`:
- `q_common.py`
- `q_prepare.py`
- `q_train.py`
- `q_eval.py`
- `q_template_compare.py`

Runs: `EC_comments/runs/ec_t17_*`.

## Design

**Data.**
- The paper's N = 2000 split: 1700 training and 300 validation structures. Every structure is re-simulated under records of its own.
- Each training structure has **four records**, one drawn per epoch (T13's augmentation wrapper):
  - pool 0: T13's `trainval_N2000`;
  - pools 1–3: `SeedSequence([20261131..33, i])`, T13's record bases.
- Validation uses pool 0.
- 426 of the 1700 training structures are also N600 training structures, so their pool 1–3 files are T13's own, hard-linked. The other **3822 records** were simulated here (`prepare_summary.json`: 0 failures, 22 min).

**Arms.** Both use T16's sign-invariant loss (`p_common.py`, patched in memory):
- `cohorient_signinv`: coherency + orientation edges, 802-d;
- `orient_signinv`: orientation edges, 2-d.

**Training.**
- `fr_train`, unchanged, through `k_train.run_one`.
- 1000 epochs; 2 threads per process, six processes side by side.
- Seeds 123, 124 and 125 set torch/numpy; the split and the data are the same for all three.
- Best-validation checkpoint on the total validation loss (shape + frequency), as everywhere else.

**Test sets:**
- `paper_varexc`: the paper's 500 test structures, each under its own record (T11);
- `fresh_test`: T5's 500 untouched structures, all under one new record (seed 20261124), which none of the networks trained on;
- `paper`: the paper's 500 test files under the paper's shared record (seed 12345).

**Real bridges.** `s5_zeroshot` on the nine bridges with the Phase P (corrected) field inputs. MAC is computed against SSI and against the FE target.

**Checks** (all pass):
- `train_check.json`: with nothing switched on, the wrapper reproduces T13's `ec_t13_coh_K4_N600` history for 2 epochs (0 differences).
- `eval_gates.json`: T16's two N600 networks, re-scored here, reproduce T16's synthetic means (difference 0.0) and its real-bridge rows exactly.
- `baseline_gates.json`:
  - the deck-node scoring recomputes `q_eval`'s all-node MAC to 1e-16;
  - every support mask has Ng + 2 nodes and the graph's node count;
  - T5's template and FDD rows reproduce `shape_baselines.csv`.

## Results

### Training

Best epochs out of 1000:

| arm | seed 123 | 124 | 125 |
|---|---|---|---|
| coherency + orientation | 679 | 754 | 784 |
| orientation | 362 | 487 | **92** |

The orientation arm's total validation loss turns up early because its frequency head overfits. Selecting on shape loss alone would pick a later epoch: for seed 125, epoch 584, with validation shape loss 0.0231 against 0.0273 at epoch 92 (`training_history.csv`). The selection rule is left unchanged.

### Synthetic

Mean ± SD over the three seeds (`eval_seeds.csv`; per run: `eval_summary.csv`). MAC is over every graph node.

| arm | test set | mean MAC (min–max) | B1 | T1 | B2 | T2 | f MAE (Hz) |
|---|---|---|---|---|---|---|---|
| **coherency + orientation** | `paper_varexc` | **0.948 ± 0.004** (0.944–0.951) | 0.976 | 0.962 | 0.924 | 0.929 | 0.208 |
| | `fresh_test` | 0.949 ± 0.002 (0.948–0.951) | 0.976 | 0.959 | 0.928 | 0.932 | 0.205 |
| | `paper` | 0.944 ± 0.004 (0.941–0.948) | 0.976 | 0.961 | 0.918 | 0.922 | 0.214 |
| orientation | `paper_varexc` | 0.916 ± 0.008 (0.907–0.922) | 0.968 | 0.946 | 0.868 | 0.881 | 0.263 |
| | `fresh_test` | 0.912 ± 0.007 (0.904–0.916) | 0.965 | 0.943 | 0.868 | 0.872 | 0.263 |
| | `paper` | 0.915 ± 0.007 (0.907–0.922) | 0.969 | 0.946 | 0.868 | 0.877 | 0.261 |

Against T16 (N = 600, 400 epochs, seed 123), on `paper_varexc`:
- coherency + orientation: 0.901 → 0.948;
- orientation: 0.893 → 0.916.

For comparison, the paper's main network (N = 2000, trained on the shared record) scores 0.933 on `paper` and 0.406 on `paper_varexc`.

### Against the class template and FDD

Paired per structure, on deck nodes (`baseline_compare.csv`):
- the class template reads no record, so its `paper` row from T5 applies to `paper_varexc`;
- FDD on `fresh_test` reads the same records the networks see;
- across seeds, each structure's difference is averaged over the three seeds first, then bootstrapped over structures (95 %).

| test set | baseline (MAC) | arm | network MAC | Δ network − baseline | network better on |
|---|---|---|---|---|---|
| `paper_varexc` | class template (0.946) | coherency + orientation | 0.948 | **+0.002** [−0.003, +0.007] | 40 % of structures |
| | | orientation | 0.916 | −0.030 [−0.037, −0.024] | 22 % |
| `fresh_test` | class template (0.946) | coherency + orientation | 0.949 | +0.003 [−0.003, +0.008] | 42 % |
| | | orientation | 0.912 | −0.034 [−0.042, −0.027] | 22 % |
| `fresh_test` | FDD, population labeller (0.970) | coherency + orientation | 0.949 | **−0.020** [−0.028, −0.012] | 9 % |
| | | orientation | 0.912 | −0.057 [−0.068, −0.046] | 7 % |

**Per mode.** The coherency + orientation arm beats the template at T1 (+0.026 on `paper_varexc`). It is level at T2 (+0.003) and slightly below at B1 (−0.008) and B2 (−0.013).

**T16's N600 networks** sit 0.044–0.053 below the template.

**Caveats.**
- The template is told each target's class.
- FDD's population labeller assigns the modes from the population.
- The network infers which output slot is which mode.

### Real bridges

Zero-shot, mean MAC vs SSI over 9 bridges × 4 modes (`real_zeroshot.csv`, `real_summary.csv`, `real_seeds.csv`).

| arm | seed 123 | 124 | 125 | mean ± SD | B1 | T1 | B2 | T2 | vs FE |
|---|---|---|---|---|---|---|---|---|---|
| coherency + orientation | 0.471 | 0.601 | 0.503 | **0.525 ± 0.068** | 0.780 | 0.594 | 0.350 | 0.377 | 0.477 |
| orientation | 0.508 | 0.543 | 0.520 | **0.524 ± 0.018** | 0.800 | 0.638 | 0.255 | 0.402 | 0.517 |
| T16, N = 600, seed 123: coherency + orientation / orientation | | | | 0.560 / 0.547 | | | | | |

For reference:
- the paper's main network: 0.254 (Phase P inputs);
- the class template, which reads no sensors: 0.879 (E19).

**Per bridge**, seed means (coherency + orientation / orientation):

| bridge | coherency + orientation | orientation |
|---|---|---|
| Eidsdal | 0.66 | 0.64 |
| Liabygda | 0.58 | 0.57 |
| Hundeidvik | 0.59 | 0.54 |
| Stranda | 0.49 | 0.61 |
| Linge | 0.49 | 0.53 |
| Saebo | 0.50 | 0.44 |
| Leknes | 0.49 | 0.45 |
| Hellesylt | 0.46 | 0.49 |
| Festoya | 0.46 | 0.45 |

## Reading

- **On synthetic data the scale-up reaches the class template.**
  - Coherency + orientation with the sign-invariant loss scores 0.948 ± 0.004 on records it has never seen.
  - That equals the no-sensor class template (Δ +0.002 [−0.003, +0.007]).
  - It is above what the paper's network reached on its own shared record (0.933).
  - It no longer depends on the record: the three test sets agree within 0.005.
  - It is still 0.020 below FDD on the same records.
- **Coherency now adds clearly to orientation.**
  - The gain is +0.032 (0.948 vs 0.916), and it holds at every seed: the lowest coherency + orientation seed (0.944) beats the highest orientation seed (0.922). At N = 600 and 400 epochs the gain was +0.008.
  - The gain sits in B2 and T2 (+0.056 / +0.048).
  - Frequencies are also better: 0.208 Hz against 0.263 Hz.
- **Seeds agree on synthetic data, not on the real bridges.**
  - Synthetic SD is at most 0.008.
  - On the real bridges the coherency + orientation arm ranges from 0.471 to 0.601 vs SSI (SD 0.068); the orientation arm from 0.508 to 0.543.
  - T16's single-seed 0.560 sits inside that spread, so it should not be read as better than N = 2000.
- **Scale does not move the real bridges.**
  - Both arms average 0.52 vs SSI. That is double the paper's 0.254, but far below the class template's 0.879.
  - Since the synthetic score is already at template level, the remaining gap lies in the real input path, not in what the network has learned:
    - only the sensor nodes are measured, and the rest is filled by propagation;
    - the excitation is not field-like (T18);
    - the FE frequencies do not match the measured ones (T18).
- **B2 is the weakest real mode** (0.35 / 0.26). T18 finds B2 and T2 weakly excited in the field: 7 % and 6 % of the mode-peak height.
- **Caveats.**
  - Three seeds and one N.
  - 1000 epochs, against the paper's 5000.
  - The orientation arm's checkpoints are chosen early by the total-loss rule.
  - The template and FDD are given class information that the network infers.
