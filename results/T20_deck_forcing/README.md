# T20 — deck forcing plus a noise floor: does a more field-like excitation transfer better?

Started 2026-11-24. Scripts `v_common.py`, `v_fit_noise.py`, `v_prepare.py`, `v_train.py`, `v_eval.py`
(run from the repository root through `tools/run_stage.py`). EC protocol, not the paper's.

## Question

T18 found the paper's excitation (independent white base acceleration at every support DOF) the least
field-like of six, and deck forcing closer. T19 found that about half of the T17 networks' real-bridge gap
comes from the recordings themselves rather than from sparse sensing. T20 asks whether training on a more
field-like excitation closes that half: T17's best arm, retrained unchanged on two deck excitations.

## Design

| | |
|---|---|
| Network | EdgeGATConv, coherency + orientation edges (802-d), sign-invariant loss — T17's `cohorient_signinv` arm exactly, through T17's own wrapper (`q_train.run`) |
| Population | the N = 2000 paper split (1700 train / 300 val), labels and graphs unchanged; only the five excitation-dependent keys re-simulated |
| Records | own record per structure, four per training structure drawn one per epoch (pools 0–3), validation on pool 0 |
| Training | 1000 epochs, seeds 123–125, best-validation checkpoint (total loss), 2 threads per process, CPU |
| `deck` | independent white vertical forces at every deck graph node (all nodes except the Ng + 2 supports) |
| `walk` | three 70 kg walkers (T18's `walker_force`) plus deck noise at 30 % of the walkers' in-band RMS |
| noise floor | white noise at every graph node before the 1.5–20 Hz bandpass, σ = ALPHA × in-band deck-node RMS |
| Seeds | `SeedSequence([base, i])`; deck bases 20261160–63 (test 20261169), walk 20261170–73 (test 20261179) |

The FE model fixes the Ng hinge supports vertically, so they carry only the noise floor. The two front supports
(bottom-beam ends) are springs and move under the deck load (RMS 0.23–0.83 of the deck's).

## Noise floor (`v_fit_noise.py`, `noise_fit.json`)

ALPHA is fitted on the nine FE twins (field record length, field sensors) so that the median over bridges of
T18's `peak_to_bg` equals the field's (23.45). The SSI labels are not used.

| excitation | ALPHA | median `peak_to_bg` without floor | with floor |
|---|---|---|---|
| deck | 0.840 | 34.5 | 23.45 |
| walk | 1.597 | 101.1 | 23.45 |

**One number cannot match the field's spectra and its coherency together.** Without a floor, both deck
excitations already match the field's coherency. The floor that matches the peak-to-background ratio pulls it
well below (T18 statistics, nine bridges):

| source | B1 share | T1 share | B2 share | T2 share | peak/bg (median) | coherency at modes | background coherency |
|---|---|---|---|---|---|---|---|
| field | 0.468 | 0.405 | 0.066 | 0.061 | 23.4 | 0.941 | 0.477 |
| deck, no floor (T18) | 0.207 | 0.385 | 0.171 | 0.237 | — | 0.955 | 0.476 |
| walkers, no floor (T18) | 0.246 | 0.395 | 0.217 | 0.142 | — | 0.956 | 0.503 |
| deck + floor | 0.206 | 0.385 | 0.172 | 0.237 | 23.4 | 0.839 | 0.259 |
| walk + floor | 0.246 | 0.385 | 0.211 | 0.157 | 23.4 | 0.719 | 0.214 |

So the field's background is spatially coherent, not independent sensor noise, and the T20 training data
are noisier at the modes than the field. The agreed fit was kept. A floor that matches the background coherency
would have to be correlated across sensors, which is a different noise model.

## Checks

| check | result |
|---|---|
| `v_fit_noise` t18: with ALPHA = 0 the statistics equal T18's | max rel. diff 8.7e-9 (gate 1e-6) |
| `v_fit_noise` fit: fitted median equals the field's | 4.6e-13 |
| `v_prepare --check` supports: under the paper's support excitation and T13's seed, the simulator reproduces T13's five arrays | x_psd 4.2e-9, coherency 6.7e-8 (gate 1e-5, FE re-solve) |
| graph identical to the stored structure; hinges at rest before the noise (rows with no vertical mode content = Ng) | pass, max 0.0 |
| achieved noise / in-band RMS = ALPHA | 1.5e-3 rel. (check), 3.4e-3 (all 15 200 records) |
| x_psd rows peak at 1; every array finite | pass |
| `v_prepare --run`: 2 × (2000 + 3 × 1700) pool records + 2 × 500 test records = 15 200 | all present, pass (`prepare_summary.json`, 86 min on 16 workers, 21 GB) |

## Results

**Training.** Six runs launched 2026-11-24 12:31, side by side. Six N = 2000 processes with four record pools
each exhausted the 32 GB machine: deck s123 died between epochs 275 and 299 (rc 3221225477, access violation), and
walk s123 / s124 were stopped just after their epoch-300 checkpoint. All three were resumed from `resume.pt` (written every 25 epochs) three at a time:
deck s123 from epoch 275, walk s123 / s124 from epoch 300. Each history is contiguous over epochs 0–999, and the
epochs drawn twice across a resume (275–276, 300–301) drew the same records both times (`variant_draws.csv`).
Deck s123's epoch-275 log line is identical before the crash and after the resume, to every printed digit.
Best-validation epochs: deck 307 / 948 / 751, walk 876 / 628 / 913 (seeds 123 / 124 / 125). `train_launch.json`
is the second batch's launch record, `train_launch_batch1.json` the first's.

**Gates** (`eval_gates.json`): T17's three networks reproduce T17's `paper_varexc` means and all 108 real rows
with a max difference of 0.0. Pass.

### Synthetic: every training excitation on every test excitation

Mean MAC over the four modes, the paper's 500 test structures with their own records; mean ± SD over three
seeds (`eval_cross.csv`).

| trained on ↓ / tested on → | supports (`paper_varexc`) | deck (`deck_test`) | walk (`walk_test`) |
|---|---|---|---|
| supports (T17) | **0.948 ± 0.004** | 0.855 ± 0.009 | 0.485 ± 0.097 |
| deck + floor | 0.308 ± 0.005 | **0.950 ± 0.001** | 0.785 ± 0.026 |
| walk + floor | 0.321 ± 0.016 | 0.757 ± 0.011 | **0.931 ± 0.003** |

Each excitation is learned equally well in its own domain (0.93–0.95). The transfer between excitations is
asymmetric: the support-trained networks keep 0.855 on deck forcing, while the deck-trained networks fall to
0.308 on support excitation.

### Real bridges, zero-shot, MAC against SSI

Nine bridges × four modes, mean ± SD over three seeds (`real_arms.csv`, `real_bridges.csv`).

| trained on | mean | per seed | B1 | T1 | B2 | T2 | vs FE target | freq. MAE vs SSI |
|---|---|---|---|---|---|---|---|---|
| supports (T17) | **0.525 ± 0.068** | 0.471, 0.601, 0.503 | 0.780 | 0.594 | 0.350 | 0.377 | 0.477 | 1.60 Hz |
| deck + floor | 0.276 ± 0.015 | 0.293, 0.272, 0.264 | 0.634 | 0.359 | 0.032 | 0.082 | 0.285 | 2.83 Hz |
| walk + floor | 0.212 ± 0.019 | 0.192, 0.230, 0.213 | 0.660 | 0.088 | 0.016 | 0.083 | 0.221 | 1.81 Hz |

Paired over bridges (seed means; 95 % bootstrap over the nine bridges, 2000 resamples): deck − supports
−0.249 [−0.312, −0.192], walk − supports −0.313 [−0.368, −0.264]. **No bridge is better under either
deck excitation** (0 of 9 for both).

### What this says

- **The hypothesis is not supported.** Training on the excitations T18 rated more field-like makes zero-shot
  transfer worse, at every bridge, by about half. B2 and T2 collapse (≤ 0.08), and walker training also loses T1
  (0.088). T18's field-likeness statistics (modal shares, peak-to-background, coherency) did not predict which
  training excitation transfers.
- The support-trained networks are also the most robust across synthetic excitations, so on this evidence the
  support excitation is the better training distribution, not just the paper's default.

### What it does not separate

- **The noise floor and the excitation changed together.** There is no deck arm without the floor. The floor
  was fitted to the field's peak-to-background ratio, and it leaves the training data less coherent at the modes
  than the field (0.84 / 0.72 against 0.94, table above). Either change, or both, could cause the loss.
- The real-bridge comparison is nine bridges and three seeds, on one population (N = 2000 paper split).
- Separating them would take a deck arm without the floor, or a floor that is correlated across sensors like
  the field's background. Neither was run.
