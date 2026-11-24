# Linkspan GNN — operational modal analysis with an excitation record per structure

A graph neural network (EdgeGATConv) that predicts the first four mode shapes and frequencies (B1, T1, B2, T2)
of linkspan bridge decks from ambient acceleration spectra, trained on a population of finite-element
structures and tested zero-shot on nine instrumented Norwegian linkspans.

This repository is the **main pipeline since 2026-11-24**. It continues an earlier one (archived, not
public) after a co-author review found two defects:

1. **One excitation record for every simulated structure.** The simulator re-seeded its white noise with the
   same seed for every structure, so all 5500 training and test structures shared one excitation realisation.
   The networks used that record as a left/right label for the deck: on structures with records of their own,
   the earlier main network fell from 0.93 to about 0.38–0.41 mean MAC. **Here every structure has its own
   record** (`SeedSequence([base, structure index])`), training structures four of them, drawn one per epoch.
   A sign-invariant shape loss and coherency + orientation edges (802-d) restore synthetic accuracy: 0.948 ±
   0.004 on unseen records (T17), level with a no-sensor class template (0.946).
2. **Two field sensors interchanged in the network's inputs.** Loggers x40131 / x40133 (sensors 7 and 9) were
   swapped in the GNN field inputs at all nine bridges; the SSI labels were always correct. Only the corrected
   inputs are used here.

## Layout

```
<library modules>        flat at the root (FE model, mode matching, simulator, GNN layers, graph builders)
pipeline/                stage scripts; see pipeline/README.md for the task map
results/<task>/          tables and a README per task (T5, T10-T20); results/paper/ = the archived paper's tables
results/reproduction/    evidence that this repository reproduces every carried-over result
experiments/             the few archive experiment modules the baselines import (E5, E14, E16-E19)
tools/                   run_stage.py (run + log + ledger), reproduce.py, verify_records.py, verify_training.py,
                         migrate_from_archive.py (how this repository was built)
docs/history/            the archive's CLAUDE.md and the review-task index
RESTRUCTURE_MANIFEST.md  what was kept, archived or changed, and the open decisions
data/, runs/, logs/      not in git (inputs, networks, logs; see DATA.md)
```

## Running

Python 3.11 with torch 2.3.0+cpu, dgl 2.2.1, numpy 1.26.4, scipy 1.17.1 — the exact environment every result
was computed in is pinned in [requirements_lock.txt](requirements_lock.txt). CPU only. From the repository root:

```bash
python tools/run_stage.py pipeline/q_eval.py --workers 8     # e.g. score the T17 networks
python tools/reproduce.py --list                            # every stage behind the results, in order
```

Data, networks and the field recordings are not in this public repository (DATA.md lists them with hashes).

## Status

- Reproduction of every carried-over result (T5, T10–T20) by this repository's code:
  `results/reproduction/` and PROVENANCE.md.
- Open decisions before the next paper version (final protocol, excitation model, unseen test set, the
  Ng = 2 balcony label swap, fine-tuning loss): RESTRUCTURE_MANIFEST.md §5.
