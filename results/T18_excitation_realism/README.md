# T18 — excitation realism: FE twins under six excitations vs the field recordings (2026-11-23)

> EC-comments work (see `../../../README.md`). Not part of `PAPER_RESULTS.md`. No network; simulation
> and signal statistics only.

Script: `r_excitation_realism.py` (through `tools/run_stage.py`, 45 s).

## Design

**FE twin.** For each of the nine LOOCV bridges:
- the twin is solved as in `s6_decomposition` (deterministic parameters, all 15 candidate modes, PHASE1_ZETA);
- it is simulated for **exactly the field record's length** (59–148 s);
- it is read at the field sensors via the graph package's `nearest_fe_node_idx`.

**Excitations:**

| key | excitation |
|---|---|
| `supports` | the paper's model: independent white base acceleration at every support DOF |
| `front_common` | one 3-DOF base motion shared by the two front (bottom-beam-end, x ≈ 10 m) supports |
| `front_indep` | independent 3-DOF base motion at each front support |
| `deck_noise` | independent white vertical forces at every deck node |
| `walkers` | three 70 kg walkers on random paths at 1.1–1.5 m/s, pacing 1.6–2.4 Hz plus harmonics (DLF 0.4 / 0.1 / 0.1), 4 % jitter |
| `walkers_amb` | walkers plus deck noise at 30 % of the walkers' RMS |

**Processing.** Field and simulation are handled identically: 1.5–20 Hz bandpass (the training filter); Welch/CSD with a Hamming window, 0.1 Hz resolution, 50 % overlap. **No measurement noise is added to the simulation.**

**Gates** (`gates.json`, pass):
- this script's modal simulation, under the paper's support excitation and the simulator's own record draw, reproduces `simulate_white_noise_from_modal_model` to 4e-17 (relative);
- the modes are mass-normalised (deviation 9e-16);
- the field columns follow the SSI sensor order: the FDD vector at each SSI frequency against the SSI shape gives a median MAC of 0.989 or more per bridge. The minimum is 0.764, Saebo's B2 alone (`checks.csv`).

## Results

Mean over the nine bridges (`summary.csv`, `distance_to_field.csv`, `per_bridge.csv`, `fig_statistics.png`, `fig_spectra_per_bridge.png`).

**Columns:**
- **share of mode peaks:** each of B1, T1, B2, T2's peak height (summed sensor PSD) as a share of the four;
- **energy > 12 Hz:** share of in-band energy above 12 Hz;
- **sensor pattern r:** correlation over sensors of the log in-band power with the field's;
- **bg coherency:** median |coherency| over sensor pairs at background bins;
- **T1 anti-phase:** fraction of cross-deck sensor pairs at T1 with Re γ < 0.

| source | B1 / T1 / B2 / T2 share of mode peaks | energy > 12 Hz | sensor pattern r | bg coherency | T1 anti-phase |
|---|---|---|---|---|---|
| **field** | **0.47 / 0.41 / 0.07 / 0.06** | **0.34** | 1 | **0.48** | **0.89** |
| supports (paper) | 0.11 / 0.16 / 0.13 / **0.60** | 0.65 | **−0.04** | 0.65 | 1.00 |
| front_common | 0.55 / **0.07** / 0.35 / 0.04 | 0.63 | 0.51 | 0.85 | **0.41** |
| front_indep | 0.13 / 0.15 / 0.11 / 0.61 | 0.61 | −0.07 | 0.77 | 1.00 |
| deck_noise | 0.21 / 0.39 / 0.17 / 0.24 | 0.47 | 0.46 | 0.48 | 1.00 |
| walkers | 0.25 / 0.40 / 0.22 / 0.14 | 0.06 | **0.65** | 0.50 | 1.00 |
| walkers_amb | 0.24 / 0.39 / 0.22 / 0.15 | 0.10 | 0.65 | 0.48 | 1.00 |

Distance to the field (lower is closer, except sensor r): `distance_to_field.csv`.
- deck_noise is closest in energy distribution and mode shares (band L1 0.67, mode-share L1 0.83) and in background coherency (0.04).
- the walkers are closest in which sensors are loud (r 0.65).
- the paper's support model is among the furthest on every measure: mode-share L1 1.34, sensor r −0.04.

## Reading

- **The paper's excitation is the least field-like.**
  - Independent base acceleration at the supports pushes the energy into the high modes: T2 takes 60 % of the mode-peak height against 6 % in the field, and 65 % of the energy lies above 12 Hz against 34 %.
  - Which sensors are loud is unrelated to the field (r −0.04): the support nodes, not the deck, drive the response.
  - Background coherency is too high (0.65 vs 0.48).
- **Deck forcing (noise or walkers) is closer** on mode balance, coherency and the sensor pattern.
  - The walkers concentrate their energy at the pacing harmonics (4–8 Hz takes 74 % against the field's 40 %).
  - They lack the field's flat background above 8 Hz, which looks like ambient or measurement noise. That floor is visible in every panel of `fig_spectra_per_bridge.png` and is not modelled here.
- **Front-beam excitation in unison barely excites torsion** (T1 share 0.07; the T1 anti-phase cue at 0.41 is close to chance). As expected: both front supports move together.
- **In the field, B1 and T1 dominate** (0.47 / 0.41) and **B2 / T2 are weakly excited** (0.07 / 0.06). This may contribute to the networks' weaker B2/T2 transfer. No excitation reproduces B1's dominance except front_common, which loses torsion.
- **The largest visible difference is not the excitation.** The FE twins' modal frequencies differ from the measured ones (e.g. Hellesylt B1: FE 5.4 Hz, SSI 3.7 Hz). That shifts every band statistic whatever the excitation, so the band shares are confounded by FE frequency error. Mode shares, the sensor pattern and coherency are less affected.
- **Not modelled:** measurement noise, ambient (wind, waves, pontoon), moving-mass effects, and the field's strong peaks at 14–18 Hz (Hundeidvik, Linge, Stranda), whose source is unknown.
- **Suggestion for a retrained population:** deck-distributed forcing (`deck_noise`, or walkers plus an ambient floor), plus a measurement-noise floor fitted to the field's background.
