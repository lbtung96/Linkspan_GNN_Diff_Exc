# T15 — mirror test: is the excitation record the only thing that tells the network left from right? (2026-11-23)

> EC-comments work (see `../../../README.md`). Not part of `PAPER_RESULTS.md`. Inference only on
> re-simulated inputs; nothing trained.

Script: `n_mirror_test.py` (run through `tools/run_stage.py`, `--per-ng 25 --workers 6`). Data: `EC_comments/data/T15/`.

## Design

**Structures.** The first 25 paper test structures of each Ng (100 in all), re-simulated with the **paper record** (seed 12345).

**Record assignments.** Four assignments of the support records, patched in memory into the simulator's `build_influence_matrix` (the library module is not edited):

| variant | what moves |
|---|---|
| `paper` | nothing; the paper's own assignment |
| `mirror` | each support's three records go to its mirror-image support, with the y record negated. The excitation becomes the exact mirror image of the paper's. |
| `swap_hinge` | only the x = 0 hinge supports are mirrored |
| `swap_ends` | only the two bottom-beam-end supports are mirrored |

**Exclusions.** 12 structures have supports without a mirror partner and are left out of the three swaps (88 remain). 77 of the 100 decks are fully mirror-symmetric.

**Metric.** Per mode: MAC, and the **signed cosine** between prediction and target. MAC = cos² and cannot see a global sign flip; the cosine can.

**Target sign convention** (`generate_population_dataset_modes4.compute_reference_sign_alignment`): each mode is flipped to a positive dot product with a fixed per-Ng sensor template. Mirror parity of the targets (median): B1 +1, T1 −1, B2 +1, T2 −1. Torsion's sign is therefore tied to a side of the deck.

## Gates (`gates.json`)

- `paper` re-simulation reproduces the stored features: max |Δx| 1.8e-8, max |Δe| 8.9e-8. Predictions on stored and re-simulated inputs differ by at most 4.8e-7 MAC.
- **`mirror` equals the stored features with the nodes permuted left ↔ right**, on every symmetric deck: max |Δx| 1.4e-5, max |Δe| 5.5e-4 (ARPACK level). The mirrored excitation gives exactly the mirrored inputs, so **in these inputs nothing but the record assignment distinguishes left from right**.

## Results

`summary.csv`, `sign_flips.csv` and `per_sample.csv`, figure `fig_mirror_signed_cosine.png`.

Signed cosine of prediction against target, mean over the 88 structures with mirrorable supports:

| network | variant | MAC | B1 | T1 | B2 | T2 |
|---|---|---|---|---|---|---|
| paper main, coherency, N 2000 | paper | 0.918 | +0.97 | +0.94 | +0.95 | +0.95 |
| | **mirror** | 0.925 | +0.97 | **−0.94** | +0.82 | **−0.81** |
| | swap_hinge | 0.656 | +0.92 | +0.66 | +0.68 | +0.49 |
| | swap_ends | 0.660 | +0.92 | −0.66 | +0.60 | −0.33 |
| paper, orientation, N 600 | paper | 0.759 | +0.75 | +0.57 | +0.75 | +0.87 |
| | **mirror** | 0.767 | +0.77 | **+0.58** | +0.78 | **+0.88** |
| T11 own records, orientation, N 600 | mirror | 0.718 | +0.61 | +0.47 | +0.75 | +0.81 |
| T11 own records, coherency, N 600 | paper | 0.412 | +0.79 | +0.03 | +0.81 | +0.01 |

B2's mean includes the Ng = 2 balcony decks, whose "B2" label is a torsion mode (project memory), so it flips with them. `sign_flips.csv` therefore classes targets by their mirror parity, not by mode name. It counts only cells on symmetric decks where the paper prediction has |cos| > 0.3:

| network | variant | antisymmetric targets flipped | symmetric targets flipped |
|---|---|---|---|
| paper coherency N 2000 | mirror | **100 %** (153 / 153) | 0 % |
| | swap_ends | 80 % | 4 % |
| | swap_hinge | 20 % | 4 % |
| paper coherency N 600 | mirror | 100 % (149 / 149) | 0 % |
| paper orientation N 600 | mirror | **0 %** | 0 % |
| T11 orientation N 600 | mirror | 0 % | 0 % |

**Decks that are not mirror-symmetric.** On the 11 such decks with mirrorable supports, where geometry could in principle say which side is which, the paper coherency network (N 2000) still flips T1 and T2 in 11 / 11 cases and keeps B1 and B2. The orientation network flips none.

## Reading

- **Coherency networks.** The paper's coherency networks set torsion's sign from which side received which record. Mirror the record assignment and every torsion prediction flips against its target, while MAC stays at 0.92. Even on asymmetric decks the geometry is not used. Most of the label sits in the two bottom-beam-end records (80 % of flips when only they move).
- **Orientation networks.** The orientation networks set it from the edge directions, and the record assignment does not move them.
- **Symmetry argument.** For a symmetric deck the coherency network's flip follows from symmetry once the gate holds. The gate is the finding: the record is the only left/right cue the coherency networks were given.
- **Varied-excitation training.** With records varied per structure, that cue is gone. A network without coordinates or orientation then has nothing to set the sign of torsion with, which is T11 and T13's T1 ≈ 0.01.
