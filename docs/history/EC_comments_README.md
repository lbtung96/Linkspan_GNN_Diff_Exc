# EC comments — analyses run for the co-author review (started 2026-11-23)

> **Remark.** Everything in this folder was run to answer co-author review comments on the
> 2026-11-16 draft, mainly co-author EC's (`EC-xx`), with related points from co-authors BS
> (`BS-xx`) and JX (`JX-xx`). The comment IDs are those of the author's comment triage
> spreadsheet (not in this repository; names and local path removed for publication).
> **These results are not part of `../pipeline/results/PAPER_RESULTS.md`.** The author decides
> later how, and whether, they enter the paper. Do not quote them beside the paper's numbers
> without saying which protocol each comes from.

The paper's own protocol (one population, seed 123, 5000 epochs, best-validation checkpoint,
fixed 500-structure test set) is unchanged. Nothing here writes into `../pipeline/`, `../data/` or
`../runs/`, or edits a library module. New edge modes and feature variants are patched in
memory, as `../pipeline/fr_edge_modes.py` does.

## Layout

| path | what |
|---|---|
| `pipeline/ec_common.py` | shared paths, seeds, task folders |
| `pipeline/<prefix>_*.py` | stage scripts; the prefix names the task that owns them |
| `pipeline/results/T*_*/` | one folder per task, each with its own `README.md` (tracked) |
| `data/`, `runs/` | generated data and checkpoints (gitignored) |

## Single seed vs several seeds: kept apart

Where a comparison is repeated over seeds, the **seed-123 result** (the paper's protocol, existing
runs reused unchanged) and the **multi-seed result** (seeds 123–127) are reported in separate
tables. A seed-123 number in this folder that reproduces a paper number must equal it exactly.

## Tasks

| task | comments | what runs | results |
|---|---|---|---|
| T1 | EC-04, EC-20, EC-21, EC-08 | ablation grid at N = 600 over seeds 124–127; phase arms (Re γ only, abs γ only) for both layers; record-wise node-PSD normalisation arm; seeds 123–127 | `pipeline/results/T1_ablation_seeds_phase/` |
| T2 | EC-27, EC-20, BS-41 | population-size sweep with nested train **and** validation sets; replicate 0 = paper subsets + seed 123, replicates 1–2 = new subset draws + seeds 124/125 | `pipeline/results/T2_sweep_nested/` |
| T3 | EC-20, EC-38, EC-40 | main network (N = 2000, paper split) over seeds 124–127: in-domain, zero-shot, decomposition, fine-tuning | `pipeline/results/T3_main_seeds/` |
| T4 | EC-30, EC-41, EC-38 | sparse-mask training at N = 2000 (masked + propagated inputs), then sensor sweep, zero-shot and fine-tuning | `pipeline/results/T4_sparse_training/` |
| T5 | EC-15, EC-23, EC-25 | fresh untouched synthetic test set; re-scoring; sign-aligned error, success rates, cross-MAC; baselines (mean frequency, geometry kNN, class template, FDD) | `pipeline/results/T5_fresh_test_baselines/` |
| T6 | BS-43, EC-28, EC-29, EC-17, EC-33 | propagation iteration count; random vs structured layouts at equal sensor counts; modal truncation beyond 15 modes; field records resampled to 200 Hz | `pipeline/results/T6_sensitivity/` |
| T7 | BS-40, JX-10, EC-26, EC-34, EC-35, EC-39, BS-38 | analyses of stored outputs only | `pipeline/results/T7_stored_outputs/` |
| T8 | EC-37, EC-38 | fine-tuning variants: epoch 0 eligible, rotated validation bridge, head-only, frozen processor | `pipeline/results/T8_finetune_variants/` |
| T9 | EC-09, EC-10, EC-11, EC-17, EC-18, EC-19, EC-20, EC-31, EC-32 | facts read from code and data (no runs), lockfile | `pipeline/results/T9_facts/` |
| T10 | (finding from T9; touches EC-31, EC-34 and every field number) | SSI scripts swap sensors 7 and 9 before SSI, the GNN field inputs do not: which order matches the sensor geometry, and the effect on the field results | `pipeline/results/T10_channel_order/` |

> **Excitation finding (T5, 2026-11-23) — all compute paused by the author.** The simulator re-seeds
> its white noise with `SEED = 12345` for every structure (`simulate_uniform_inertial_u3_white_noise.py:62, 344`),
> so all 5500 paper structures share one excitation realisation. On 500 new structures the main network
> scores 0.934 with that realisation and **0.377** with a new one; the dependence enters through the edge
> coherency (new edges only: 0.410; new node PSDs only: 0.927). FDD on the same new records scores 0.969.
> Every run in this folder so far uses the fixed realisation. The training queue is stopped (T1 runs at
> epochs 225–275, resumable); T4's sparse training was never launched; T6 was stopped part-way. Nothing
> restarts until the author decides. Evidence: `pipeline/results/T5_fresh_test_baselines/`.

> **T11, varied-excitation pilot (author-approved 2026-11-23).** The N = 600 training and validation
> structures re-simulated with a different excitation record per structure; EdgeGAT + coherency and
> EdgeGAT + orientation retrained (seed 123, 5000 epochs) and scored on new-excitation test sets and the
> real bridges. Results: `pipeline/results/T11_varied_excitation_pilot/`. Everything else stays paused.
> **Outcome:** the coherency arm scores 0.41 on every test set (T1 ≈ 0.01, like the no-edge arm) — verdict
> "does not"; the orientation arm scores 0.71–0.73 on every excitation; real-bridge zero-shot vs SSI is
> 0.49–0.51 for the orientation arms against 0.22–0.23 for the coherency arms. One seed, N = 600.

> **Field-input finding (T10, 2026-11-23).** The GNN field inputs have sensors 7 and 9 (loggers
> x40131 / x40133) interchanged at all nine bridges; the SSI labels are correctly ordered. Corrected,
> zero-shot MAC vs SSI is 0.254 instead of the paper's 0.233. The production field pipeline has **not**
> been changed (author's decision pending). Until then, every field number in T3, T4, T6 and T8 is computed
> on the paper's current (mis-ordered) inputs so that it can be gated against the paper, and is marked so.

> **T12–T15, why the paper's inputs worked (author-approved 2026-11-23).** T12: plots of the excitation
> (`x_plot_excitation.py`). T13: feasibility levers at 400 epochs (records ×4, 120 s, N = 2000): none lets
> the coherency network learn torsion. T14 (`m_*.py`, no training): under the paper record a fixed template
> tells left from right in 95–100 % of test decks (chance with own records); the physical relative-sign cue at
> the modes is identical under every record; the paper network fails (0.44) when given correct resonance bins
> but a new record elsewhere. T15 (`n_mirror_test.py`): mirroring the support-record assignment flips every
> torsion prediction of the paper coherency networks (100 %), never the orientation networks (0 %).
> Results: `pipeline/results/T13_feasibility/`, `T14_input_anatomy/`, `T15_mirror_test/`.
> T16 (`p_*.py`, 400 epochs, N = 600, own records): with a sign-invariant loss, coherency lifts T1 from 0.007
> (no-edge control) to 0.45 (mean 0.678); orientation + sign-invariant 0.893, coherency + orientation +
> sign-invariant 0.901 synthetic and 0.560 vs SSI on the real bridges. The B1 label sign also flips with Ng
> (negative Ng 2–4, positive Ng 5). `pipeline/results/T16_sign_test/`.
> T17 (`q_*.py`): T16's two best arms at N = 2000, 1000 epochs, seeds 123–125, four own records per
> structure (3822 new records simulated, T13's linked). Synthetic, own records: coherency + orientation
> 0.948 ± 0.004 (equal to the class template, Δ +0.002 [−0.003, +0.007]; 0.020 below FDD), orientation
> 0.916 ± 0.008. Real bridges vs SSI: 0.525 ± 0.068 and 0.524 ± 0.018, no better than T16's N = 600; the
> real-bridge seed spread (0.47–0.60) is far wider than the synthetic one. `pipeline/results/T17_scale_up/`.
> T18 (`r_excitation_realism.py`, no network): the nine FE twins under six excitations vs the field
> recordings. The paper's support excitation is the least field-like (T2 takes 60 % of the mode peaks vs 6 % in
> the field; sensor-power pattern r = −0.04); deck forcing (white noise or walkers) is closer. FE-vs-real
> frequency error dominates the spectra whatever the excitation. `pipeline/results/T18_excitation_realism/`.
> T19 (`u_decomposition.py`, inference only): s6's decomposition for the T17 networks, FE twins under own
> records. Coherency + orientation, FE frame: ceiling 0.905 → sensors + propagation 0.703 → recordings 0.477, so
> about half the real gap is sparse sensing and half the recordings (the paper network: 78–80 % propagation). The
> recordings cost most at B2 (0.65 → 0.18). `pipeline/results/T19_decomposition/`.
> T20 (`v_*.py`, running): T17's best arm retrained on deck white noise and on walkers + ambient, each with a
> white noise floor fitted to the field's peak-to-background ratio (ALPHA 0.84 / 1.60); seeds 123–125.
> `pipeline/results/T20_deck_forcing/`.

Not run, by decision: BS-53 (tower stiffness as an input needs a new population), EC-36 (a
robustness study over excitation, damping, noise, record length and support stiffness).
