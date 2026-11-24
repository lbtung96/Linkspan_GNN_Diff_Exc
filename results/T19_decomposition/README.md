# T19 — where the T17 networks' real-bridge gap comes from (s6's decomposition, own records) (2026-11-24)

> EC-comments work (see `../../../README.md`). Not part of `PAPER_RESULTS.md`. Inference only, on T17's
> six networks (N = 2000, 1000 epochs, seeds 123–125) and, as the gate, the paper's `edgegat_N2000`.

Script: `u_decomposition.py` (through `tools/run_stage.py`, 3 min). It imports `s6_decomposition`'s own simulation, FE payload, real-data builders and propagation, so the code is not copied.

## Design

Variants: s6's five, per bridge.

| variant | network input | graph |
|---|---|---|
| `ceiling` | the bridge's FE twin, all 15 modes, every FE node, the generator's broadband path | FE |
| `prop_fegraph` / `prop_hybrid` | the same response read **only at the sensors**, through the real-data pipeline (bandpass → PSD → propagation → coherency) | FE / hybrid |
| `zeroshot_hybrid` / `zeroshot_fegraph` | the field recordings: Phase P inputs, as in s5, s6 and T17 | hybrid / FE |

The variants are compared in this order:
- `ceiling` → `prop`: what measuring only the sensors and filling the rest by propagation costs.
- `prop` → `zeroshot`: what the real recordings cost beyond that (excitation, noise, FE-versus-real differences).

**Two changes from s6**, both needed by the T17 networks:
- **Excitation:** three **own** records per bridge, `SeedSequence([20261150, bridge, k])`, 60 s as in training. s6 reads the shared record (12345), which these networks never saw. The simulated variants are averaged over the three records.
- **Edges:** built according to the network's edge mode. Orientation on the hybrid graph comes from the stored angle (as in T16/T17). On s6's deterministic FE graph, which stores no angle, it is the unit vector from source to destination.

**Gates** (`gates.json`, all pass):

| gate | result |
|---|---|
| the paper network under the shared record reproduces `s6_decomposition.csv` | 180 rows; 1.9e-6 overall (the FE eigensolve is re-run), real-recording rows exact |
| the T17 networks' `zeroshot_hybrid` matches T17's `real_zeroshot.csv` | exact |
| orientation from coordinates matches the stored angle | 9e-8 |
| s6's simulation checks | refilter difference 0 |
| the records are different records | yes |

Each network is run at its reference's thread count: 2 for the paper network (as s6), torch's default for the T17 networks (as `q_eval`). At 2 threads the T17 rows differ by 2e-7, the last-float32-bit effect recorded in CLAUDE.md.

## Results

Mean MAC over 9 bridges × 4 modes. T17 rows are the mean ± SD over the three seeds (`summary_arms.csv`; per network: `summary.csv`; rows: `decomposition.csv`).

**Against the FE target, all FE nodes:**

| network | ceiling | prop (FE / hybrid graph) | zeroshot (hybrid) | share of the gap from sensors + propagation |
|---|---|---|---|---|
| paper `edgegat_N2000` (shared record) | 0.772 | 0.369 / 0.379 | 0.267 | 0.80 / 0.78 |
| T17 coherency + orientation | **0.905** ± 0.022 | 0.729 / 0.703 | **0.477** ± 0.065 | 0.41 / 0.47 |
| T17 orientation | 0.839 ± 0.016 | 0.669 / 0.653 | 0.517 ± 0.020 | 0.53 / 0.58 |

**Against SSI, sensor nodes:**

| network | ceiling | prop (FE / hybrid) | zeroshot (hybrid) | share from sensors + propagation |
|---|---|---|---|---|
| paper `edgegat_N2000` | 0.715 | 0.415 / 0.360 | 0.254 | 0.65 / 0.77 |
| T17 coherency + orientation | 0.795 ± 0.025 | 0.646 / 0.601 | 0.525 ± 0.068 | 0.55 / 0.71 |
| T17 orientation | 0.763 ± 0.019 | 0.616 / 0.594 | 0.524 ± 0.018 | 0.62 / 0.71 |

**Per mode**, T17 coherency + orientation, FE frame, ceiling → prop (hybrid) → zeroshot:

| mode | ceiling | prop (hybrid) | zeroshot |
|---|---|---|---|
| B1 | 0.947 | 0.791 | 0.751 |
| T1 | 0.908 | 0.679 | 0.439 |
| B2 | 0.908 | 0.651 | **0.178** |
| T2 | 0.858 | 0.690 | 0.540 |

## Reading

- **The FE frame shows the split most directly.**
  - Sensors + propagation cost the coherency + orientation networks 0.18–0.20.
  - The real recordings cost a further 0.23.
  - So the gap is now about **half sparse sensing and half the recordings themselves**. For the paper network it was 78–80 % propagation.
- **The recordings cost the most at B2** (0.651 → 0.178) and T1 (0.679 → 0.439). B1 barely moves (0.791 → 0.751).
  - In the field, B2 takes 7 % of the mode-peak height (T18), so the recordings carry little B2 information.
  - This is the part a more field-like training excitation could address (T20).
- **Coherency is more sensitive to the recordings than orientation.**
  - From prop to zeroshot, the coherency + orientation networks lose 0.23; the orientation networks lose 0.14.
  - They start higher in the simulated variants and end level on the real bridges (0.477 vs 0.517 in the FE frame).
  - The real coherency differs from the simulated one more than the PSDs do.
- **Sensors + propagation still cost about 0.19 MAC.**
  - The networks are trained on every node measured.
  - Training on masked and propagated inputs (T4's design, never launched) would address this part.
- **The FE-twin ceiling scored against SSI** (0.795 / 0.763) includes the FE-versus-SSI shape difference: T2's ceiling against SSI is 0.57.
- **Caveats.**
  - The ceiling and prop variants use 60 s records; the real records last 59–148 s.
  - There are three records per bridge and three seeds per arm.
