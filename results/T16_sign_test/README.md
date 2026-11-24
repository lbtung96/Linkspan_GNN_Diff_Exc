# T16 — sign test: does coherency carry usable information once the global sign is free? (2026-11-23)

> EC-comments work (see `../../../README.md`). Not part of `PAPER_RESULTS.md`. One seed (123),
> N = 600, 400 epochs: a feasibility screen, not a final protocol.

Scripts: `p_common.py`, `p_train.py`, `p_eval.py` (all through `tools/run_stage.py`). Runs: `EC_comments/runs/ec_t16_*`.

## Design

- **Data.** T13's own-record data: 508 N600 training structures × four own records (one drawn per epoch) and 92 validation structures under record r0.
- **Trainer.** `fr_train` unchanged, through `k_train.run_one`, best-validation checkpoint.
- **In-memory additions.** Three, defined in `p_common.py`; the library modules are not edited:
  - **sign-invariant loss:** per structure and mode, min(‖p − y‖², ‖p + y‖²), normalised like MSE. Used for the training and validation loss. MAC, the metric, is sign-blind.
  - **coherency + orientation edges:** 802-d.
  - **amplitude-keeping node input:** √PSD; deck nodes are divided by the deck-wide maximum, support nodes by their own.

**Checks** (`train_checks.json`, all pass):
- the loss equals a brute-force minimum over sign choices;
- the 802-d edges are exactly coherency followed by orientation;
- the amplitude input is 1 at the deck maximum. At each target mode's frequency it correlates with |φ| at a median of 0.997–0.9998 (B1 to T2);
- with nothing switched on, the wrapper reproduces T13's `ec_t13_coh_K4_N600` history for 2 epochs, number for number.

**Scoring checks** (`eval_gates.json`, all pass): T13's two reference networks reproduce T13's synthetic means (to 1e-8) and its real-bridge rows exactly.

## Results

### Synthetic

500 paper test structures, each under a record of its own (`paper_varexc`). `fresh_test` and `paper` agree within 0.02 (`eval_summary.csv`).

| arm | mean MAC | B1 | T1 | B2 | T2 | f MAE (Hz) | T1 output RMS / B1 |
|---|---|---|---|---|---|---|---|
| coherency, MSE (T13 reference) | 0.423 | 0.80 | 0.008 | 0.79 | 0.09 | 0.305 | 0.01 |
| no edges, sign-invariant (control) | 0.445 | 0.91 | 0.007 | 0.78 | 0.08 | 0.391 | 0.05 |
| **coherency, sign-invariant** | **0.678** | 0.95 | **0.45** | 0.85 | 0.47 | 0.297 | 0.51 |
| coherency, sign-invariant, amplitude nodes | 0.706 | 0.95 | 0.47 | 0.86 | 0.54 | 0.273 | 0.48 |
| orientation, MSE (T13 reference) | 0.758 | 0.77 | 0.68 | 0.74 | 0.84 | 0.357 | 0.66 |
| coherency + orientation, MSE | 0.834 | 0.83 | 0.80 | 0.86 | 0.86 | 0.304 | 0.74 |
| orientation, sign-invariant | 0.893 | 0.96 | 0.93 | 0.83 | 0.85 | 0.375 | 0.76 |
| **coherency + orientation, sign-invariant** | **0.901** | 0.95 | 0.91 | 0.86 | 0.89 | 0.294 | 0.73 |

### Paired contrasts

`paper_varexc`, bootstrap 95 % over structures (`eval_contrasts.csv`):

| contrast | Δ mean MAC | Δ T1 |
|---|---|---|
| sign-invariant: coherency vs no edges | **+0.234** [0.222, 0.244] | +0.446 |
| coherency: sign-invariant vs MSE | +0.256 [0.244, 0.267] | +0.445 |
| MSE: coherency+orientation vs orientation | +0.076 [0.060, 0.092] | +0.123 |
| sign-invariant: coherency+orientation vs orientation | +0.008 [0.000, 0.017] | −0.022 |
| orientation: sign-invariant vs MSE | +0.135 [0.119, 0.150] | +0.253 |
| coherency, sign-invariant: amplitude vs row-normalised nodes | +0.027 [0.016, 0.038] | +0.020 |

### Real bridges

Zero-shot, mean MAC vs SSI over the 9 bridges × 4 modes (`real_zeroshot.csv`, `real_summary.csv`). The amplitude arm is not scored here: the field inputs carry no PSD scale.

| arm | vs SSI | B1 | T1 | B2 | T2 | vs FE |
|---|---|---|---|---|---|---|
| coherency, MSE (T13) | 0.228 | 0.58 | 0.01 | 0.23 | 0.09 | 0.252 |
| no edges, sign-invariant | 0.260 | 0.74 | 0.03 | 0.18 | 0.09 | 0.212 |
| coherency, sign-invariant | 0.352 | 0.75 | 0.23 | 0.38 | 0.05 | 0.305 |
| orientation, MSE (T13) | 0.452 | 0.76 | 0.33 | 0.25 | 0.47 | 0.452 |
| coherency + orientation, MSE | 0.472 | 0.74 | 0.49 | 0.30 | 0.36 | 0.352 |
| orientation, sign-invariant | 0.547 | 0.82 | 0.67 | 0.26 | 0.44 | 0.536 |
| **coherency + orientation, sign-invariant** | **0.560** | 0.81 | 0.59 | 0.41 | 0.43 | 0.523 |

For reference:

| | synthetic, own records | real vs SSI |
|---|---|---|
| the paper's main network (N 2000, trained on the shared record) | 0.406 (T11) | 0.254 |
| class template, no sensors (E19) | 0.946 | 0.879 |
| FDD (T5) | 0.969 | — |

## A second sign problem: the label convention changes with Ng

`compute_reference_sign_alignment` aligns every mode to a fixed per-Ng template. On the paper test set, the B1 labels sum negative for **all** Ng 2–4 decks and positive for **all** Ng 5 decks. A network trained with MSE must therefore work out Ng to get B1's sign. On Ng 5 decks it gets it wrong in 18 % (coherency) to 38 % (orientation) of cases, which shows as negative cosines in `eval_per_sample.csv`. The sign-invariant networks simply choose one sign per Ng group. This is why the sign-invariant loss raises B1 (0.77–0.80 → 0.95–0.96) and helps even the orientation network, which already has a left/right frame.

## Reading

- **Coherency does carry the relative sign.**
  - Once the network no longer has to guess the global sign, coherency lifts T1 from 0.007 (no edges, same loss) to 0.45, and the mean by +0.23.
  - The physical cue is real and the network can use it. It is thin: per T14, T1's cross-deck sign rests on about 1.7 edges per structure.
- **Orientation is the stronger route.** With the sign-invariant loss, orientation alone reaches 0.893. Adding coherency gives +0.008 in shape and better frequencies (0.294 vs 0.375 Hz). Under MSE, coherency adds +0.076 to orientation.
- **The sign-invariant loss is the main fix.**
  - It solves torsion's global sign, which no input settles without a left/right frame.
  - It also solves the per-Ng label convention.
  - On the real bridges the best arms reach 0.55–0.56 against SSI: more than double the paper network's 0.254, but still well below the no-sensor class template's 0.879.
- **Amplitude-keeping nodes** add a little (+0.03) and give the best frequencies (0.273 Hz).
- **Caveats.** One seed, N = 600, 400 epochs. Every best arm is still below the class template and FDD on synthetic data.
