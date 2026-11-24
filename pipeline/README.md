# pipeline/

Every stage runs from the repository root: `python tools/run_stage.py pipeline/<script> [args]`.
Scripts keep the archive's names: the prefix is the task (T5, T10–T20) that introduced them, and each task's
tables are in `results/<task folder>/` with its own README. `tools/reproduce.py --list` gives the exact order
and arguments in which they produced those tables.

## Shared modules

| module | role |
|---|---|
| `fr_common.py` | paths and constants (population design, LOOCV bridges, field sensor order, seeds) |
| `ec_common.py` | paths of the task folders (`data/`, `runs/ec/`, `results/`, `logs/`) |
| `fr_train.py` | the resumable trainer: `train(Job, pools=...)`; `Job.shape_loss`, `Job.node_feature_mode` |
| `records.py` | several excitation records per training structure, one drawn per epoch |
| `fr_edge_modes.py` | compatibility layer: the edge modes are library code now |
| `s0_generate_population.py` | the archived paper's generator wrapper (generator patches; T5 uses it) |
| `s4_indomain.py`, `s5_zeroshot.py`, `s6_decomposition.py` | scoring: synthetic test sets, real-bridge zero-shot, real-gap decomposition |
| `s5a_field_inputs.py` | builds `data/field_inputs_v2` (Phase P sensor order) from the raw recordings |
| `t_gates.py` | `same_history`: training histories compared number for number |

## Main chain: a record per structure

| task | prefix | what | results |
|---|---|---|---|
| T11 | `h_` | varied-excitation pilot (N = 600): re-simulation per structure, two arms | `T11_varied_excitation_pilot/` |
| T13 | `k_` | feasibility levers: 4 records per structure (per-epoch draw), 120 s, N = 2000 | `T13_feasibility/` |
| T16 | `p_` | sign-invariant loss; coherency + orientation (802-d); amplitude node input | `T16_sign_test/` |
| T17 | `q_` | **the main-network candidate**: N = 2000, 1000 epochs, seeds 123–125 | `T17_scale_up/` |
| T18 | `r_` | excitation realism: nine FE twins under six excitations vs the field | `T18_excitation_realism/` |
| T19 | `u_` | the T17 networks' real-bridge gap: sparse sensing vs recordings | `T19_decomposition/` |
| T20 | `v_` | deck forcing + a fitted noise floor, retrained | `T20_deck_forcing/` |
| T5 | `c_` | baselines: class template, FDD, frequency baselines; the fresh test set | `T5_fresh_test_baselines/` |

## History: findings about the archived paper's shared record

These re-run against the archive's paper networks (`runs/paper/`) and its shared-record features; they explain
why the new pipeline exists.

| task | prefix | finding | results |
|---|---|---|---|
| T5 | `c_` | the shared record (main network 0.934 → 0.377 on a new record) | `T5_fresh_test_baselines/` |
| T10 | `g_t10_` | GNN field inputs had loggers x40131 / x40133 interchanged (SSI correct) | `T10_channel_order/` |
| T12 | `x_` | plots of the shared support excitation | `T12_excitation_plots/` |
| T14 | `m_` | the record as a left/right label; the physical cue is the same under every record | `T14_input_anatomy/` |
| T15 | `n_` | mirroring the record assignment flips 100 % of the coherency networks' torsion | `T15_mirror_test/` |
