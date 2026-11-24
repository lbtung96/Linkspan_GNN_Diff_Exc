# T14 — input anatomy: what differs between the paper's one-record inputs and own-record inputs (2026-11-23)

> EC-comments work (see `../../../README.md`). Not part of `PAPER_RESULTS.md`. No training and no new
> simulation: stored features only, plus one inference pass for part (e).

Scripts: `m_input_anatomy.py` (a–d), `m_occlusion.py` (e). Run through `tools/run_stage.py`.

## Data

- **508 N = 600 training structures**, each under five records:
  - the paper record (seed 12345, shared by every structure);
  - T11's own record r0;
  - T13's r1–r3.
- **500 paper test structures**, under the paper record and under T11's `paper_varexc` records.
- **Analysed band:** the 1.5–20 Hz band. Node input is log10 of the row-normalised PSD; edge input is one direction per undirected edge.
- **Mirror-symmetric decks:** 435 / 508 training and 427 / 500 test.

## (a) Is the input at a frequency set by the structure or by the record?

Correlation across the graph (deck nodes and deck-to-deck edges; the loud support nodes are excluded) of one bridge's input under two records, per bin (`a_structure_vs_record.csv`, `a_profile.csv`, `fig_a_structure_vs_record.png`):

| | node log PSD | edge Re coherency | edge Im coherency |
|---|---|---|---|
| near a target mode (±0.29 Hz), two own records | 0.966 | 0.948 | 0.155 |
| off resonance (> 0.59 Hz from every target mode), two own records | 0.858 | 0.932 | 0.063 |
| paper record vs an own record (same pattern) | 0.968 / 0.860 | 0.947 / 0.931 | 0.153 / 0.064 |

- Node PSD and Re coherency are mostly set by the structure, on and off resonance.
- Im coherency is mostly record noise, except right at the modes.
- The paper record is statistically like any other record. **What made it special is that every structure shared it.**

## (b) A fingerprint shared across the population

Structure-level spectra (mean over support nodes, deck nodes or edges of the detrended input) are correlated across bins, off resonance, for same-Ng pairs. Values are medians, averaged over Ng (`b_shared_fingerprint.csv`, `fig_b_shared_fingerprint.png`):

| | support nodes, log PSD | deck nodes, log PSD | edges, Re coherency | edges, abs coherency |
|---|---|---|---|---|
| different bridges, **same paper record** | **0.996** | 0.154 | **0.278** | 0.173 |
| different bridges, own records | 0.032 | 0.008 | 0.027 | 0.010 |
| same bridge, paper vs own record | 0.057 | 0.725 | 0.302 | 0.310 |
| same bridge, two own records | 0.055 | 0.735 | 0.302 | 0.312 |

- The support nodes carry the record itself, so it is identical across the paper population.
- **Off resonance, in Re coherency, two different bridges shaken by the paper record are as alike (0.28) as one bridge under two records (0.30).**
- `fig_b_example_two_bridges.png` is an example. It is dominated by the resonance peaks and shows the fingerprint poorly.

## (c) The physical relative-sign cue is the same under every record

At each target mode's frequency, on edges where both |φ| ≥ 0.3 (`c_relative_sign_cue.csv`, `fig_c_relative_sign_cue.png`):

- sign(Re γ) = sign(φᵢφⱼ) on 99.9 % (B1, T1), 98.9 % (B2) and 97 % (T2) of edges, **for all five records alike**;
- mean Re γ on anti-phase edges is −0.92 (T1), −0.92 (B2) and −0.99 (T2) under every record.

**The information that defines a torsion shape up to its global sign is equally present with any record.**

Caveat: T1 has only 1.7 anti-phase edges per structure among edges with both |φ| ≥ 0.3, against 20.5 edges in all. The cross-deck cue rests on very few edges.

## (d) A left/right label

On mirror-symmetric decks:
1. The left-minus-right asymmetry spectrum is computed for node log PSD and for edge Re coherency.
2. A template is formed per Ng as the mean over the training structures.
3. Each test structure is scored by its correlation with the template (`d_side_label.csv`, `fig_d_side_label.png`).

| template from | scored on | node log PSD: same side | edge Re coherency: same side |
|---|---|---|---|
| paper-record training | test, paper record | **100 %** (all four Ng) | **95–100 %** |
| paper-record training | test, own records | 50–60 % (chance) | 44–55 % (chance) |
| own-record training (r0) | test, own records | 45–56 % (chance) | 49–58 % (chance) |

- In the paper inputs, a single fixed linear template tells the deck's left half from its right in every test structure.
- With own records there is nothing to find.
- The rows "own-record template scored on paper-record test" in the CSV (35–89 %) are not meaningful. They score one random template against a pattern that all paper-record structures share, so they land wherever that single correlation happens to fall.

## (e) Which bins does the paper network depend on?

Edge coherency is swapped bin by bin between the paper record and the structure's own record (`paper_varexc`), with node PSDs from the paper record (`e_occlusion.csv`):
- **modal bins:** ±3 bins of a target mode, 6.8 % of all bins;
- **peak bins:** the top 15 % of in-band bins by PSD, 7.3 % of all bins;
- **out-of-band:** 52.8 % of bins.

| edge input | N 2000 MAC | T1 | f MAE (Hz) | N 600 MAC |
|---|---|---|---|---|
| all paper | 0.933 | 0.902 | 0.219 | 0.866 |
| all new | 0.428 | 0.267 | 0.514 | 0.427 |
| **modal bins paper, rest new** | **0.442** | 0.285 | 0.507 | 0.439 |
| modal bins new, rest paper | 0.926 | 0.894 | 0.219 | 0.854 |
| peak bins paper, rest new | 0.434 | 0.265 | 0.508 | 0.434 |
| peak bins new, rest paper | 0.930 | 0.898 | 0.220 | 0.859 |
| out-of-band new, in-band paper | 0.888 | 0.857 | 0.255 | 0.796 |
| in-band new, out-of-band paper | 0.540 | 0.418 | 0.460 | 0.505 |

- Gate: the unmixed variants equal T11's `eval_summary` (max diff 1e-9).
- **Correct resonance bins with the paper record's bins elsewhere replaced: the network fails (0.44).** Its shapes and frequencies follow the bins between and outside the resonances, which is where the shared record lives. Most of it is in-band off-resonance; the out-of-band bins carry a smaller part.
- The "modal bins new, rest paper" rows staying high also reflects that modal bins hardly change between records (part a). The informative row is the one above it.
