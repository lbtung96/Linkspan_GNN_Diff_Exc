<!-- DRAFT for PROVENANCE.md. The brief asked for "Phase L", but PROVENANCE.md already has a
     Phase L (E16, 2026-10-06) and runs through Phase O (E19), so this is written as Phase P.
     The code comments (fr_common, s5a, s5, s6, s7, s9, tools/check_paths_resolve.py) say
     "Phase P" too. Rename everywhere at once if another letter is chosen. -->

## Phase P — the paper's field inputs move to the SSI sensor order, 2026-11-23

**Why.** EC_comments task T10 (`Final_results_for_paper/EC_comments/pipeline/results/T10_channel_order/`)
found that at all nine LOOCV bridges the GNN's field inputs and its SSI labels put two sensors in a
different order. The sensors are 0-based rows 6 and 8, logger serials `x40131` and `x40133`. Each
SSI script the labels come from reorders the raw columns before identification:

```matlab
y = Data(:, [1:18 25 26 27 22 23 24 19 20 21 28:end]);   %Swap 131 and 133
```

These scripts are `System identification code/Field measurements 26_270625/<bridge>/MAIN_SSI_RUN_SCRIPT*.m`.
Stranda uses `MAIN_SSI_RUN_SCRIPT_Walk.m`, which reads `Stranda_Walk.mat`, the record the GNN uses.
Eidsdal's comment reads `%Sensor 131 swap 133`.

The GNN path never applied this reorder: `process_field_modal_filtering.py` ran with
`raw_channel_reorder = None`, and `experiments/E5_real_transfer/build_phase3_features_corrected.py`
maps raw row k to graph sensor node k.

Three tests show that the SSI order is the right one. None of them uses the other side as its reference.

| test, over 36 bridge × mode cells | result |
|---|---|
| FE target vs SSI at the sensors | the SSI order wins 36/36 |
| FE target vs an FDD of the raw records (no SSI involved) | the swapped order wins 36/36 |
| model-free roughness along the girder line | the SSI order is smoother in 32/36 cells; the swapped FDD in 32/36 |

The raw `.mat` `Names` confirm it at every bridge: columns 19–21 are `x40131_ch1..3` and
columns 25–27 are `x40133_ch1..3`. The nominal cabling table in `Modal_info_Summary.xlsx` is the
unswapped plan. The repository holds no written reason for the interchange; the script comment is
the only statement.

The two sensors sit on the same girder line (y = 0), at neighbouring stations 1.8–2.8 m apart. The
swap does not flip a cross-deck torsion sign directly. What it corrupts is the two stations' PSDs
(their FE amplitudes differ by 8–51 % of the mode maximum), their cross-deck pairs, and the edge
coherencies around them.

**What changed.** The author approved this change.

- **`fr_common.py`.** The reorder is declared in one place: `FIELD_SENSOR_SWAP = (6, 8)`,
  `FIELD_SENSOR_SERIALS`, `field_sensor_order(n)`, and `field_channel_reorder(n_channels)`. The
  last one is the 0-based form of the MATLAB line above. The new paths are:
  - `FIELD_RAW_MAT_DIR` and `FIELD_RAW_FILES`: `For GNN/<bridge>.mat` plus `<bridge>_Data_picked.mat`.
    Eidsdal's file is `Eisdal_Data_picked.mat`, and Stranda's is `Stranda_Walk.mat` plus
    `Stranda_Data_picked.mat`.
  - `LEGACY_FIELD_OUTPUTS` and `LEGACY_REAL_FEATURES`, for the raw-order inputs.
  - `FIELD_OUTPUTS` and `REAL_FEATURES`, under the gitignored
    `Final_results_for_paper/data/field_inputs_v2/`.
- **New stage `pipeline/s5a_field_inputs.py`.** It imports `process_field_modal_filtering.py` and
  the E5 builder unmodified and calls `process_structure` with explicit file pairs. No file dialog
  opens: the yes/no popup is answered in memory, and the legacy `mode_assignment_used.json` is
  reused. The stage sets `settings.raw_channel_reorder = field_channel_reorder(n)` and points the
  builder's `FIELD_ROOT` at the new outputs.
  It writes:
  - `data/field_inputs_v2/field_modal_filtering_outputs/<S>/`
  - `data/field_inputs_v2/phase3_corrected/<S>/phase3_dataset_corrected.npz`
  - `pipeline/results/s5a_field_inputs_checks.json`
- **s5, s6 and s7 read `C.REAL_FEATURES`.** The faithfulness proofs in s5 and s7 ("the copied code
  reproduces E2's published rows") now run on `C.LEGACY_REAL_FEATURES`, the inputs E2 ran on, so
  they still mean what they meant. Everything new is scored on the corrected inputs.
- **s9** gains a conventions bullet and an s5a gate line, and the E2-proof sentences now say
  "on E2's own (raw-order) inputs". s10 is unchanged: its field figures read s6's and s7's raw
  outputs.
- **`tools/check_paths_resolve.py`** also checks the five new fr_common paths.
- **Nothing in `experiments/`, `field_modal_filtering_outputs/`, `SSI_modal_infor/` or the library
  modules changed, and nothing was retrained.** The labels the paper scores against
  (`y_mode_shape_ssi` and `y_freq_hz` from `SSI_modal_infor/`), the FE targets and the graphs are
  byte-identical.

**Verification.**

- **s5a gates (`s5a_field_inputs_checks.json`), PASS at all nine bridges.**
  - *serials*: rows 6 and 8 are x40131 and x40133.
  - *settings*: every data-affecting setting equals the legacy run's `settings_used.json`.
  - *identity*: with no reorder, the imported producer reproduces every array of the legacy
    `field_modal_filtering_outputs/<S>/data/*.npz` bit for bit (27 arrays). The builder then
    reproduces all 17 arrays of the legacy `phase3_corrected/<S>` bit for bit.
  - *swap*:
    - The raw u3 records (`_full`, `_clean`) equal the legacy ones with rows 6 and 8 exchanged,
      bit for bit. So do the raw SDM, the raw PSD and the three stored raw CPSD pairs, none of
      which involves rows 6 or 8.
    - The arrays read from the SSI file or the channel layout are unchanged: picked frequencies,
      selected indices, u3 channel indices and the frequency axis.
    - The corrected dataset's non-feature arrays are unchanged.
    - The modal-filter products (`filtered_*` and `mask_*`) compare SSI mode shapes, which were
      always in SSI order, with the FDD of the records. They change in two ways. At eight bridges
      the change is round-off only (≤ 1.2e-06 on the filtered SDM, ≤ 1.4e-14 on the singular
      values). At Festoya the B2 mask changes by up to 0.33, because SSI and FDD are now compared
      in the same order. The GNN features read none of these arrays; they read only
      `raw_u3_timeseries_clean` and `freq_hz`.
- The rebuilt features equal T10's `EC_comments/data/T10/<S>_swapped_inputs.npz` (`x_psd`,
  `edge_attr_cpsd_full`) bit for bit at all nine bridges.
- **s5** gates PASS: the E2 copy on legacy inputs reproduces 36 rows with max diff 0.0, the feature
  convention holds, and input dependence holds for the main network and for all eight sweep
  networks. Its main-network rows equal T10's `t10_swapped_inputs.csv` `inputs_swapped` rows
  exactly (36 rows, all numeric columns, max diff 0.0).
- **s6** checks PASS. Its three FE-simulated variants (ceiling, prop_fegraph, prop_hybrid) moved
  only by ARPACK noise, max 1.9e-06; only the two recording variants moved.
- **s7**: the E2 copy on legacy inputs reproduces 36/36 rows, max diff 0.0. Torch ran at 14 threads, the default. Median best epoch 19 → 18.
- `tools/check_paths_resolve.py`: 73 path constants resolve, 0 problems.
- `diff_against_reference --atol 1e-5`: exactly five tables DIFFER, all FR field tables: `s5_zeroshot`, `s5_transfer_vs_n`, `s6_decomposition`, `s6_decomposition_summary` and `s7_finetune`. 42 are IDENTICAL, and no `experiments/` table moved. The two NO REF lines are E3's copy-only summaries, as before. Not CSVs, so not in the diff: `s5_gates.json` (input dependence 0.380 → 0.388), `s7_checks.json` (adds `inputs`), `PAPER_RESULTS.md`, and the new `s5a_field_inputs_checks.json`. `s6_checks.json` came back byte-identical.
- `verify_reproduction.py`: 11/12 before the snapshot. Gate C ("no unexplained differences") fails on exactly those five intended tables. Gates A, B and D pass, including D's "corrected builder == production features", which still reads the legacy features. It should return to 12/12 after `snapshot_baseline.py --apply`; re-run it to confirm.

**What moved, raw order → SSI order.** The main network is `edgegat_N2000`, best_model, epoch 591.
Values are in the SSI frame unless stated.

| result | raw order (until now) | SSI order |
|---|---|---|
| zero-shot vs SSI, mean | 0.233 | **0.254** (+0.022) |
| zero-shot vs SSI, B1 / T1 / B2 / T2 | 0.549 / 0.117 / 0.213 / 0.052 | 0.579 / 0.147 / 0.218 / 0.074 |
| zero-shot vs FE target (every FE node), mean | 0.257 | 0.267 |
| zero-shot vs FE, B1 / T1 / B2 / T2 | 0.574 / 0.053 / 0.354 / 0.047 | 0.598 / 0.047 / 0.357 / 0.065 |
| zero-shot \|f − f_SSI\| / \|f − f_FE\| | 2.243 / 1.536 Hz | 2.235 / 1.531 Hz |
| input dependence (MAC to the zero-input response) | 0.380 | 0.388 |
| transfer vs N, real MAC vs SSI, N = 100 … 5000 | 0.218 0.273 0.233 0.214 0.253 0.218 0.233 0.202 | 0.217 0.281 0.243 0.221 0.264 0.232 0.254 0.205 |
| transfer vs N, Spearman against N | ρ = −0.26 (p = 0.53) | ρ = −0.19 (p = 0.65) |
| decomposition ceiling, FE / SSI frame | 0.772 / 0.715 | unchanged (FE-simulated) |
| propagation-only, FE graph, FE / SSI | 0.369 / 0.415 | unchanged (FE-simulated) |
| propagation-only, hybrid graph, FE / SSI | 0.379 / 0.360 | unchanged (FE-simulated) |
| zero-shot, hybrid graph, FE / SSI | 0.257 / 0.233 | 0.267 / 0.254 |
| zero-shot, FE-only graph, FE / SSI | 0.234 / 0.214 | 0.239 / 0.221 |
| share of gap from propagation, FE graph, FE / SSI | 78.2 % / 62.3 % | 79.8 % / 65.2 % |
| share of gap from propagation, hybrid graph, FE / SSI | 76.3 % / 73.6 % | 77.8 % / 77.0 % |
| fine-tuned (nested, full) vs SSI, mean | 0.220 (−0.013 vs zero-shot) | **0.248** (−0.006 vs zero-shot) |
| fine-tuned vs SSI, B1 / T1 / B2 / T2 | 0.537 / 0.084 / 0.168 / 0.090 | 0.600 / 0.090 / 0.198 / 0.105 |
| fine-tuned vs FE target, mean | 0.226 | 0.241 |
| fine-tuned \|f − f_SSI\| / \|f − f_FE\| | 1.581 / 1.849 Hz | 1.353 / 1.853 Hz |
| bridges where fine-tuned > zero-shot; Wilcoxon (per-bridge means) | 4/9, p = 0.65 | 4/9, p = 0.82 |

Zero-shot MAC vs SSI per bridge, raw order → SSI order:

| Hellesylt | Hundeidvik | Leknes | Linge | Saebo | Eidsdal | Liabygda | Stranda | Festoya |
|---|---|---|---|---|---|---|---|---|
| 0.446 → 0.512 | 0.103 → 0.116 | 0.283 → 0.295 | 0.280 → 0.279 | 0.193 → 0.200 | 0.223 → 0.266 | 0.283 → 0.311 | 0.095 → 0.101 | 0.190 → 0.208 |

Eight of the nine bridges rise. Per cell the change runs from −0.033 to +0.274. The largest cells
are Hellesylt T1 (0.427 → 0.701) and Eidsdal T2 (0.006 → 0.187).

**What this changes in the story.** Small.

- **Zero-shot transfer stays poor.** It is 0.25 against an in-domain 0.93, and it stays flat in N.
- **Torsion stays far below bending.** T1 is 0.147 and T2 0.074, against B1 0.579. T1 against the
  FE target even dips, from 0.053 to 0.047.
- **Propagation's share of the gap rises 1.5–3.4 points, to 65–80 %.** The previous range was
  62–78 %. The ceiling and propagation-only cases are unchanged; only the zero-shot end moved up.
  Quote the new range.
- **Fine-tuning still shows no reliable gain.** It is 0.248 against a zero-shot 0.254 (−0.006), improves 4/9 bridges, and gives Wilcoxon p = 0.82. Per bridge it reshuffles: Saebo 0.235 → 0.377, Liabygda 0.280 → 0.211. Its frequency error against SSI improves, from 1.58 to 1.35 Hz.
- **Every field number the paper quotes changes in the second or third decimal.** That is why the
  paper adopts the corrected inputs rather than carrying a known mis-order.

**What stays on the old (raw-order) inputs: `experiments/`, as history.** Nothing there was re-run.
It remains reproducible from `field_modal_filtering_outputs/` and
`experiments/E5_real_transfer/phase3_corrected/`, which were not touched. The following used the
raw-order field records and are therefore computed on mis-ordered inputs at rows 6 and 8 (T10's
list):

- E2's and E5's zero-shot, fine-tuning and decomposition, together with E11's 500-epoch mirror;
- E8;
- E10's real graphs;
- E17's real hold-out ("real" source);
- E18 `stage3_real.py` and E19 `stage2_identify.py` ("real" source);
- the cross-deck coherency numbers of `docs/TORSION_TRANSFER_FINDING.md`, for the pairs that
  involve rows 6 and 8;
- the EC_comments tasks T3, T4, T6 and T8, in their field rows;
- the FDD fields of `field_modal_filtering_outputs/*.npz`.

Never quote these beside the paper's field numbers.

These results are *not* affected, because they compare SSI with FE and both are in grid order:

- Table 8 / `ssi_mac_vs_fe`, and E2 stage5 SSI vs FE (0.856 at the sensors, = E17's
  "nominal FE vs SSI");
- the `mode_selection.json` curation;
- the T2 override rule;
- the SSI sign alignment;
- every synthetic result (s0–s4, s8, s11).

**Reproducibility.** The corrected features live in gitignored `Final_results_for_paper/data/`. They
regenerate from the raw records outside the repository (`FIELD_RAW_MAT_DIR`), in about two minutes,
as `field_modal_filtering_outputs/` always did. Without those records they cannot be regenerated.
Add them to *What will never regenerate* on the same terms as the legacy field outputs.

**Not determined.** Why the loggers were interchanged in the field; whether any other channel pair
is interchanged. The SSI scripts correct only this one, and only this one was tested.
