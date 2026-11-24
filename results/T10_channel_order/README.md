# T10 - sensor channel order: SSI labels vs GNN field inputs

Written 2026-11-23. Read-only: nothing trained, nothing in `SSI_modal_infor/`, field data, library modules or
`Final_results_for_paper/pipeline/` changed. Scripts: `../../g_t10_channel_order.py` (questions 1, 2, 3a, 3c) and
`../../g_t10_swapped_inputs.py` (question 3b), run through `tools/run_stage.py`. Swapped features were written to
`Final_results_for_paper/EC_comments/data/T10/<S>_swapped_inputs.npz`.
Rows are 0-based sensor rows: rows 6 and 8 are sensors 7 and 9 (1-based), logger serials `x40131` and `x40133`,
and grid nodes 9 and 11 of `SSI_modal_infor/grid_<S>.txt`.

## Verdict

**The GNN field inputs are the side in the wrong order. The SSI labels are right.** All three tests agree, and
none of them uses the other as its reference:

| test (36 bridge x mode cells) | result |
|---|---|
| FE target at the sensors vs SSI: stored order vs rows 6/8 swapped | stored order better in **36/36** cells (mean MAC 0.856 vs 0.825) |
| FE target vs FDD first singular vector of the raw records (no SSI involved): raw vs swapped | swapped better in **36/36** cells (0.800 vs 0.830) |
| Model-free roughness along the y = 0 girder line (second differences) | SSI stored smoother in 32/36 cells. FDD swapped smoother in 32/36 cells |

The per-bridge margin (FE vs SSI, stored minus swapped) is positive at every bridge: mean / minimum
Hellesylt 0.057/0.032, Hundeidvik 0.035/0.022, Leknes 0.024/0.013, Linge 0.020/0.007, Saebo 0.019/0.004,
Eidsdal 0.024/0.003, Liabygda 0.016/0.007, **Stranda 0.013/0.006**, Festoya 0.072/0.022 (`t10_geometry_tests.csv`).
The FDD margin matches it closely. So the problem holds for all nine bridges, Stranda included.

**Documentation.** Every SSI script the bridges' labels come from swaps the two sensors, with the comment
`%Swap 131 and 133` (Eidsdal: `%Sensor 131 swap 133`). Each line below is in
`C:\MyFolder\Individual_Study_Syllabus\System identification code\Field measurements 26_270625\<dir>\`:

- `Hellesylt/MAIN_SSI_RUN_SCRIPT.m:59-60`
- `Hundeidvik/...:59-60`, which overrides an unswapped line at `:56`
- `Leknes/...:59-60`
- `Linge/...:60-61`
- `Saebo/...:59-60`
- `Eisdal/...:58-59`
- `Liabygda/...:59-60`
- `Festoya/...:59-60`
- **Stranda**, found here (T9 had not found it): `Stranda/MAIN_SSI_RUN_SCRIPT_Walk.m:60-61`. That script reads
  `Stranda_Walk.mat`, the record the GNN uses (`field_modal_filtering_outputs/Stranda/logs/structure_summary.json`
  `raw_file`), and its line `:215` is `% save Stranda_Data_picked.mat ...`. `MAIN_SSI_RUN_SCRIPT_Tower_Adjusting.m:60-61`
  swaps as well.
- Sykkylven, too.

The swap moves raw column triples 19-21 (serial 40131) and 25-27 (serial 40133). The nominal table in
`Modal_info_Summary.xlsx` (sheet1, columns N-R) maps serial 131 to node 9 / columns 19-21 and serial 133 to node 11 /
columns 25-27. In other words it is the *unswapped* cabling plan. The scripts therefore correct a mounting or
cabling interchange: the logger recorded under x40131 actually sat at grid node 11, and the one under x40133 at
grid node 9. The repository has no written record of *why* the sensors were swapped; the comment is the only
statement. The data say the correction is right.

**The GNN pipeline** reads ch3 of every triple in raw column order. `raw_u3_timeseries_full` equals
`Data[:, 2::3]` of the raw `.mat` exactly at all nine bridges (`t10_checks.json`), with `raw_channel_reorder = null`
(`field_modal_filtering_outputs/run_summary.json`). It then maps raw row k to graph sensor node k
(`build_phase3_features_corrected.py:208-213,236`, `sensor_physical_ids` 125, 126, ... in order). The stored SSI
labels equal |`phi_sorted` uz| row for row (`y_ssi_equals_phi_sorted_uz_up_to_sign`), and `ssi_mac_vs_fe` is
reproduced to 3e-8.

## Where the two sensors are (`t10_sensor_positions.csv`)

Both sensors sit on the **same girder line, y = 0**, at neighbouring stations about 1.8-2.8 m apart in the outer
half of the span:

| bridge | x of the two stations (m) |
|---|---|
| Hellesylt | 11.33 / 13.21 |
| Hundeidvik | 9.34 / 11.34 |
| Leknes | 9.52 / 11.52 |
| Linge | 11.32 / 13.12 |
| Saebo | 11.16 / 13.02 |
| Eidsdal | 14.96 / 17.78 |
| Liabygda | 11.33 / 13.15 |
| Stranda | 11.31 / 13.03 |
| Festoya | 9.31 / 12.08 |

The swap is **not across the deck**, so it does not flip a torsion sign directly. The FE amplitude still differs
between the two stations by 8-51 % of the mode maximum (`rel_diff_fe_rows_6_8`). Their cross-deck pairs (with the
y = W sensors at the same x) and the edge coherencies around them are corrupted.

## Effect sizes (main network edgegat_N2000, best_model, s5 path)

Gates:

- The stored scoring reproduces `s5_zeroshot.csv` exactly (max abs diff 0.0).
- The unswapped rebuild equals the stored corrected features bit for bit and reproduces s5 exactly.
- Torch threads: 14, as in s5.

**(b) Inputs corrected** (raw rows 6 and 8 swapped before the builder, labels unchanged). This is the physically
right fix (`t10_swapped_inputs.csv`):

| | as published | inputs corrected |
|---|---|---|
| MAC vs SSI mean | 0.233 | **0.254** (+0.022) |
| B1 | 0.549 | 0.579 |
| T1 | 0.117 | 0.147 |
| B2 | 0.213 | 0.218 |
| T2 | 0.052 | 0.074 |
| MAC vs FE (all nodes) | 0.257 | 0.267 |
| frequency error vs SSI | 2.243 Hz | 2.235 Hz |

Per bridge, MAC vs SSI before and after:

| bridge | as published | inputs corrected |
|---|---|---|
| Hellesylt | 0.446 | 0.512 |
| Hundeidvik | 0.103 | 0.116 |
| Leknes | 0.283 | 0.295 |
| Linge | 0.280 | 0.279 |
| Saebo | 0.193 | 0.200 |
| Eidsdal | 0.223 | 0.266 |
| Liabygda | 0.283 | 0.311 |
| Stranda | 0.095 | 0.101 |
| Festoya | 0.190 | 0.208 |

Eight of nine bridges rise. Per cell the change runs from -0.033 to +0.274, median +0.003. The large cells are
Hellesylt T1 0.427 -> 0.701, Eidsdal T2 0.006 -> 0.187, Hundeidvik B1 +0.103 and Liabygda B1 +0.101. The
headline story (zero-shot transfer is poor, about 0.23-0.25) does not change, but the paper's field numbers are
computed on mis-ordered inputs.

**(a) Labels swapped instead** (stored predictions scored against SSI with rows 6/8 swapped;
`t10_zeroshot_rescore.csv`). This is *not* the right correction, because the labels are the correct side. It is
reported for completeness: mean 0.233 -> 0.235, per cell -0.060 to +0.156.

**(c) FE vs SSI at the sensors**: 0.856 as stored and 0.825 with SSI swapped. The stored value is the one already
published. It equals E17's "nominal FE vs SSI 0.856".

## Which stored results are affected (not recomputed)

Affected, because they use the raw field records in raw order as input or as reference:

- s5 zero-shot, and s5 transfer-vs-N.
- The s6 decomposition cases that use field records. The FE-simulated and propagated-FE cases build their inputs
  from FE, which is in grid order, so they are unaffected.
- s7 fine-tuning: its inputs are mis-ordered against its correctly ordered SSI targets at two sensors.
- s8, only if it ever feeds real records (`s8_sensor_sweep.py` has no field-record reference; it appears synthetic).
- E17 real hold-out ("real" source: the fill and score use the recordings).
- E18 `stage3_real.py` and E19 `stage2_identify.py` ("real" source: `raw_u3_timeseries_clean` mapped to the sensor
  positions).
- E5 / E2 / E11 field zero-shot and fine-tuning history.
- EC tasks T3, T4, T6 and T8 field rows.
- `docs/TORSION_TRANSFER_FINDING.md` cross-deck coherency numbers, for the pairs that involve rows 6/8.
- E10 real graphs.
- The FDD fields in `field_modal_filtering_outputs/*.npz`.

Not affected, because they compare SSI with FE, and both are in grid order:

- E2 stage5 SSI vs FE.
- Table 8 / `ssi_mac_vs_fe`.
- The `mode_selection.json` curation.
- The T2 override rule (it scans FE against SSI).
- The SSI sign alignment.
- The synthetic results.

## Not determined

- Why the sensors were interchanged in the field. The repository only has the script comment.
- Whether any other channel pair is also interchanged. The SSI scripts correct only this one; this study tested
  only this swap.
