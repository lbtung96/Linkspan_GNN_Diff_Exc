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

Training launched 2026-11-24 12:31 (six runs side by side; T17's took 2.6 h). Scoring by `v_eval.py`:
every network on every test excitation (`paper_varexc`, `deck_test`, `walk_test`, the paper's 500 test
structures with their own records), and on the nine real bridges, zero-shot, against SSI and the FE target.
Gates: T17's three networks must reproduce T17's `paper_varexc` means (1e-9) and real rows exactly.

*Pending.*
