# Final results for the paper

*Generated 2026-11-23 12:34 by `Final_results_for_paper/pipeline/s9_report.py` from the tables in `pipeline/results/`. Do not edit by hand - re-run the stage.*

**One population, one protocol.** Every number below comes from the population generated in `Final_results_for_paper/data/` and the networks trained on it. Conventions, fixed with the author on 2026-09-25:

- **Population**: 5500 structures (1375 per girder count, Ng = 2-5), generator seed 20260507; the first 1250 per Ng are the training pool, the next 125 per Ng (500) are the **fixed test set** every network is scored on. Girder spacing for Ng >= 3 is drawn from 2000-3000 mm (was 2000-2500), so the real bridges at 3000 mm lie inside the population. A structure is rejected if any selected mode matches its reference template with MAC < 0.85. E and rho are sampled for every structure.
- **Response**: broadband (all 15 solved FE modes, 1.5-20 Hz bandpass) for every network. A paired narrowband population exists but nothing trains on it.
- **Training**: EdgeGATConv on full-spectrum complex coherency, 5000 epochs, seed 123, CPU. The population-size sweep uses balanced, nested subsets (N/4 per girder count), split 85/15 train/validation inside each girder count. The **main network is N = 2000**, the smallest N in the sweep at which every mode reaches test MAC >= 0.9 (section 5); in-domain, sensor-sweep, zero-shot, decomposition and fine-tuning results all come from it. The architecture ablation compares equal-N networks at N = 600. (Until 2026-09-28 the main network was N = 600; PROVENANCE Phase K records what moved.)
- **Checkpoint**: the best-validation checkpoint everywhere. One seed, so the spread quoted is over test structures (bootstrap 95 % interval) or over the nine bridges, never over seeds.
- **Field inputs** (since 2026-11-23, PROVENANCE Phase P): sensor rows 6 and 8 (loggers x40131 / x40133) are exchanged in the raw field records of all nine bridges, and every SSI script the labels come from swaps them back (`% Swap 131 and 133`). The real-bridge features are rebuilt in that order by `s5a_field_inputs.py` into `Final_results_for_paper/data/field_inputs_v2/`; the labels are unchanged. `experiments/` keeps the raw-order inputs as history.

## 1. The population

Gates (`s1_population_checks.json`): G1_broadband PASS, G1_narrowband PASS, G2_pairing PASS, G3_labels PASS, G4_envelope PASS, G5_fidelity_ng2 PASS.

The broadband and narrowband populations hold the same 5500 structures: identical parameters in 5500, identical graphs in 5500; frequency labels agree to 9.5e-07 Hz and signed mode shapes to 1.6e-07 (the eigensolve has no fixed start vector).

Two-girder decks are untouched by the spacing change, so they must equal the previous 10k generation's two-girder decks with the gated ones removed: 1375 compared, first mismatch None -> **identical**.

Sampled parameters over the training pool (min / median / max as drawn):

| parameter | Ng | min | median | max |
|---|---|---|---|---|
| L_mm | 2-5 | 14000.8 | 17523.4 | 21999.9 |
| Nx | 2-5 | 15 | 20 | 25 |
| dy_mm | 2 | 4000.75 | 4255.15 | 4499.75 |
| dy_mm | 3 | 2250.18 | 2632.75 | 2999.25 |
| dy_mm | 4 | 2002.81 | 2519.83 | 2998.44 |
| dy_mm | 5 | 2000.88 | 2504.32 | 2998.28 |
| width_mm | 2 | 4000.75 | 4255.15 | 4499.75 |
| width_mm | 3 | 4500.35 | 5265.5 | 5998.51 |
| width_mm | 4 | 6008.43 | 7559.49 | 8995.31 |
| width_mm | 5 | 8003.53 | 10017.3 | 11993.1 |
| E | 2-5 | 160001 | 196037 | 229986 |
| rho | 2-5 | 7.00029e-09 | 7.50014e-09 | 7.99984e-09 |
| kx | 2-5 | 1750.39 | 2005.53 | 2249.79 |
| kz | 2-5 | 17500.4 | 19950.4 | 22499.2 |
| nsm | 2-5 | 25.0029 | 30.0551 | 34.9998 |

Median FE target frequency (Hz) by girder count, training pool:

| Ng | B1 | T1 | B2 | T2 |
|---|---|---|---|---|
| 2 | 2.9925 | 4.5852 | 5.7678 | 8.0927 |
| 3 | 3.2538 | 4.4528 | 6.2744 | 9.3973 |
| 4 | 3.1576 | 4.3648 | 6.1228 | 9.1684 |
| 5 | 3.0400 | 4.4457 | 5.9261 | 8.8100 |

The real bridges against the population:

| bridge | Ng | dy (mm) | L (mm) | node spacing (mm) | dy inside | pool share within 150 mm of its dy |
|---|---|---|---|---|---|---|
| Hellesylt | 2 | 4000.0 | 14920.0 | 994.7 | True | 0.2822 |
| Hundeidvik | 4 | 2333.0 | 14555.0 | 970.3 | True | 0.2749 |
| Leknes | 2 | 4500.0 | 14900.0 | 993.3 | True | 0.3076 |
| Linge | 5 | 3000.0 | 18000.0 | 620.7 | True | 0.16 |
| Saebo | 4 | 2333.0 | 18000.0 | 1000.0 | True | 0.2749 |
| Eidsdal | 4 | 3000.0 | 22000.0 | 628.6 | True | 0.1665 |
| Liabygda | 4 | 3000.0 | 18090.0 | 623.8 | True | 0.1665 |
| Stranda | 4 | 3000.0 | 18090.0 | 623.8 | True | 0.1665 |
| Festoya | 5 | 3000.0 | 15000.0 | 833.3 | True | 0.16 |

Rejected draws, broadband run (the narrowband run rejects the same draws):

| reason | Ng 2 | Ng 3 | Ng 4 | Ng 5 |
|---|---|---|---|---|
| No feasible candidate survives MMI scoring for target B1 | 198 | 102 | 63 | 1 |
| No feasible candidate survives MMI scoring for target B2 | 79 | 41 | 39 | 5 |
| No feasible candidate survives MMI scoring for target T1 | 25 | 64 | 108 | 77 |
| No feasible candidate survives MMI scoring for target T2 | 296 | 12 | 29 | 15 |
| selected-mode MAC below gate | 70 | 0 | 0 | 0 |

Sweep subsets (`s2_split_summary.csv`) - balanced by girder count, nested in N, disjoint from the test set:

| N | train | val | train per Ng (2/3/4/5) | nested | disjoint from test |
|---|---|---|---|---|---|
| 100 | 84 | 16 | 21/21/21/21 | True | True |
| 200 | 168 | 32 | 42/42/42/42 | True | True |
| 400 | 340 | 60 | 85/85/85/85 | True | True |
| 600 | 508 | 92 | 127/127/127/127 | True | True |
| 800 | 680 | 120 | 170/170/170/170 | True | True |
| 1000 | 848 | 152 | 212/212/212/212 | True | True |
| 2000 | 1700 | 300 | 425/425/425/425 | True | True |
| 5000 | 4248 | 752 | 1062/1062/1062/1062 | True | True |

## 2. Training

The trainer (`fr_train.py`) reproduces the library's own `train_one_run` bit for bit - every loss in the history and every weight in best/last checkpoints, for both EdgeGAT and GraphSAGE - and a run stopped and resumed equals an uninterrupted one: **PASS** (`s3_trainer_check.json`).

| run | N | best epoch of 5000 | train time (h) | device |
|---|---|---|---|---|
| edgegat_N100 | 100 | 176 | 0.9 | cpu |
| edgegat_N200 | 200 | 195 | 1.5 | cpu |
| edgegat_N400 | 400 | 257 | 2.8 | cpu |
| edgegat_N600 | 600 | 141 | 4.2 | cpu |
| edgegat_geomorient_N600 | 600 | 477 | 4.3 | cpu |
| edgegat_noedge_N600 | 600 | 572 | 2.9 | cpu |
| edgegat_N800 | 800 | 1896 | 5.6 | cpu |
| edgegat_N1000 | 1000 | 3773 | 6.7 | cpu |
| edgegat_N2000 | 2000 | 591 | 9.5 | cpu |
| edgegat_N5000 | 5000 | 1249 | 16.4 | cpu |
| graphsage_cpsd8_signed_N600 | 600 | 113 | 3.3 | cpu |
| graphsage_cpsdfull_signed_N600 | 600 | 3606 | 2.7 | cpu |
| graphsage_geomorient_signed_N600 | 600 | 4964 | 1.7 | cpu |
| graphsage_noedge_signed_N600 | 600 | 4277 | 3.3 | cpu |

## 3. Architecture ablation (N = 600, fixed test set)

A full grid: two message-passing layers x three edge features (none, edge orientation, full-spectrum complex coherency), every arm trained on the same N = 600 split with the same seed and epochs and scored from its best-validation checkpoint on the same 500 test structures. The EdgeGATConv / coherency arm is the N = 600 network of the sweep, not the main network (N = 2000). Four arms were added on 2026-09-28/29 to complete the grid; their edge-feature modes are patched in by `fr_edge_modes.py` (the library modules are unmodified). One training seed per arm: the spread quoted is over test structures.

How the two layers use an edge (read from the code that runs). GraphSAGE (`GraphSAGEEdgeWeighted`, three SAGEConv `pool` layers) maps each edge's features, once, to one positive weight (edge MLP, sigmoid, EdgeWeightNorm by the weighted degrees of both ends); the same weight scales neighbour j's message, every channel alike, in all three layers, before an element-wise max over the neighbours. DGL's EdgeGATConv, in every layer and for each of four heads, scores each neighbour from the edge embedding and both nodes' current states, normalises the scores by a softmax over the neighbours, and sums the messages W h_j + W_e e_ij - the edge embedding is added to what the neighbour sends, not only to how much it counts. A positive weight cannot reverse a neighbour's sign, yet GraphSAGE given the coherency recovers most of the torsion, so it is not true that GraphSAGE cannot use phase. Neither orientation arm has any phase input; E12 found that this excitation leaves a left/right asymmetry in the node spectra, which a network able to tell the deck's halves apart can use to sign torsion, and edge orientation supplies that. Neither mechanism is tested here.

*Mean MAC over B1, T1, B2, T2 (T1 in brackets), 500 test structures:*

| layer | no edge features | edge orientation only (cos, sin), 2-d | full-spectrum complex coherency, 800-d |
|---|---|---|---|
| GraphSAGE | 0.513 (0.157) | 0.781 (0.688) | 0.814 (0.773) |
| EdgeGATConv | 0.408 (0.006) | 0.770 (0.718) | 0.866 (0.798) |

| arm | B1 | T1 | B2 | T2 | mean MAC | freq MAE (Hz) | layer, edge feature |
|---|---|---|---|---|---|---|---|
| graphsage_noedge_signed_N600 | 0.741 | 0.157 | 0.669 | 0.483 | 0.513 | 0.390 | GraphSAGE, no edge features |
| graphsage_geomorient_signed_N600 | 0.831 | 0.688 | 0.783 | 0.821 | 0.781 | 0.341 | GraphSAGE, edge orientation only (cos, sin), 2-d |
| graphsage_cpsdfull_signed_N600 | 0.862 | 0.773 | 0.795 | 0.828 | 0.814 | 0.313 | GraphSAGE, full-spectrum complex coherency, 800-d |
| edgegat_noedge_N600 | 0.803 | 0.006 | 0.693 | 0.129 | 0.408 | 0.412 | EdgeGATConv, no edge features |
| edgegat_geomorient_N600 | 0.845 | 0.718 | 0.703 | 0.814 | 0.770 | 0.440 | EdgeGATConv, edge orientation only (cos, sin), 2-d |
| edgegat_N600 | 0.903 | 0.798 | 0.875 | 0.889 | 0.866 | 0.304 | EdgeGATConv, full-spectrum complex coherency, 800-d |
| graphsage_cpsd8_signed_N600 | 0.792 | 0.023 | 0.690 | 0.278 | 0.446 | 0.382 | GraphSAGE, complex coherency (Re, Im) at the 4 FE-matched modal frequencies only, 8-d - outside the grid |

*Paired contrasts over the same 500 test structures (bootstrap 95 %), from `s11_ablation_cost.py`:*

| contrast | mean MAC | T1 MAC |
|---|---|---|
| layer, none: EdgeGATConv - GraphSAGE | -0.105 [-0.121, -0.088] | -0.151 [-0.172, -0.130] |
| layer, orientation: EdgeGATConv - GraphSAGE | -0.011 [-0.028, +0.007] | +0.030 [-0.004, +0.065] |
| layer, coherency: EdgeGATConv - GraphSAGE | +0.052 [+0.037, +0.067] | +0.025 [-0.001, +0.050] |
| feature, GraphSAGE: orientation - none | +0.268 [+0.250, +0.287] | +0.531 [+0.497, +0.564] |
| feature, GraphSAGE: coherency - orientation | +0.034 [+0.018, +0.049] | +0.085 [+0.056, +0.117] |
| feature, EdgeGATConv: orientation - none | +0.363 [+0.349, +0.376] | +0.712 [+0.689, +0.738] |
| feature, EdgeGATConv: coherency - orientation | +0.096 [+0.081, +0.111] | +0.080 [+0.053, +0.104] |
| interaction: (coherency - orientation) in EdgeGATConv minus in GraphSAGE | +0.062 [+0.042, +0.083] | -0.005 [-0.046, +0.034] |

*Validation and cost. Validation losses at the best-validation epoch (92 validation structures; total = mode shape + 0.25 x frequency); spread = IQR of the validation mode-shape loss over the last 1000 epochs. Time per epoch is measured by `s11` with each arm trained alone at 2 threads - the recorded run times are not comparable, because those runs shared the machine with one to six others.*

| arm | best epoch | val loss total | val loss mode shape | spread | parameters (edge path) | s / epoch, alone | 5000 epochs, alone (h) |
|---|---|---|---|---|---|---|---|
| graphsage_noedge_signed_N600 | 4277 | 0.1683 | 0.1047 | 0.0045 | 92,809 (4,417) | 0.74 | 1.0 |
| graphsage_geomorient_signed_N600 | 4964 | 0.0898 | 0.0414 | 0.0034 | 92,809 (4,417) | 0.74 | 1.0 |
| graphsage_cpsdfull_signed_N600 | 3606 | 0.0992 | 0.0412 | 0.0027 | 143,881 (55,489) | 0.95 | 1.3 |
| edgegat_noedge_N600 | 572 | 0.1958 | 0.1182 | 0.0033 | 30,376 (4,320) | 0.81 | 1.1 |
| edgegat_geomorient_N600 | 477 | 0.1297 | 0.0563 | 0.0033 | 30,376 (4,320) | 0.81 | 1.1 |
| edgegat_N600 | 141 | 0.0718 | 0.0290 | 0.0033 | 55,912 (29,856) | 0.97 | 1.4 |
| graphsage_cpsd8_signed_N600 | 113 | 0.1657 | 0.1024 | 0.0038 | 93,193 (4,801) | 0.72 | 1.0 |

Timed on Intel64 Family 6 Model 186 Stepping 2, GenuineIntel / 20 logical CPUs, 40 epochs per arm. Timings carry machine noise and are not bit-reproducible.

**Not run: `graphsage_geom_abs_N600`.** Fails before training: GraphSAGEEdgeWeighted passes the learned edge weight through sigmoid and EdgeWeightNorm(norm='both'), which rejects a zero weight. edge_attr_geom is the raw edge length in mm; the new pool's two-girder decks carry full-length diagonal braces of 19.4-19.9 m, and at initialisation (seed 123) 5 of the 508 N = 600 training decks already give a pre-sigmoid value near -90, which is exactly 0 in float32. The previous pool had no two-girder decks (longest edge 9.6 m). Same failure class as the signed geometry arm the previous ablation reported; not forced to run by changing the architecture or the feature.

## 4. In-domain performance of the main network (N = 2000, 500 test structures)

|  | B1 | T1 | B2 | T2 | mean |
|---|---|---|---|---|---|
| MAC | 0.957 | 0.902 | 0.932 | 0.939 | 0.933 [0.926, 0.939] |
| freq abs error (Hz) | 0.085 | 0.250 | 0.245 | 0.295 | 0.219 [0.201, 0.238] |

Brackets: bootstrap 95 % interval of the mean over the 500 test structures.

By girder count:

| Ng | n | MAC B1 | MAC T1 | MAC B2 | MAC T2 | mean MAC | freq MAE (Hz) |
|---|---|---|---|---|---|---|---|
| 2 | 125 | 0.949 | 0.894 | 0.849 | 0.902 | 0.898 | 0.353 |
| 3 | 125 | 0.958 | 0.898 | 0.964 | 0.951 | 0.943 | 0.171 |
| 4 | 125 | 0.970 | 0.921 | 0.961 | 0.959 | 0.953 | 0.158 |
| 5 | 125 | 0.952 | 0.893 | 0.954 | 0.945 | 0.936 | 0.193 |

For the record only: the last-epoch checkpoint (epoch 4999) scores mean MAC 0.937 and frequency MAE 0.231 Hz against 0.933 / 0.219 Hz for the best-validation checkpoint (epoch 591).

*Training cost, measured by `s11` with the network trained alone at 2 threads (40 epochs timed, Intel64 Family 6 Model 186 Stepping 2, GenuineIntel / 20 logical CPUs):* 3.30 s per epoch over 1700 training and 300 validation structures, i.e. 4.6 h for 5000 epochs and 0.5 h to the best-validation epoch (591); 55,912 trainable parameters. The recorded run time, 9.5 h, is not comparable: the run shared the machine with other trainings. Timings carry machine noise and are not bit-reproducible.

## 5. Population-size sweep (fixed test set)

| N | MAC B1 | MAC T1 | MAC B2 | MAC T2 | mean MAC [95 %] | freq MAE Hz [95 %] | best epoch |
|---|---|---|---|---|---|---|---|
| 100 | 0.638 | 0.415 | 0.669 | 0.725 | 0.612 [0.593, 0.630] | 0.561 [0.525, 0.600] | 176 |
| 200 | 0.800 | 0.619 | 0.759 | 0.814 | 0.748 [0.732, 0.764] | 0.503 [0.471, 0.538] | 195 |
| 400 | 0.867 | 0.750 | 0.842 | 0.868 | 0.832 [0.817, 0.844] | 0.369 [0.344, 0.397] | 257 |
| 600 | 0.903 | 0.798 | 0.875 | 0.889 | 0.866 [0.855, 0.877] | 0.304 [0.283, 0.328] | 141 |
| 800 | 0.937 | 0.874 | 0.905 | 0.909 | 0.906 [0.896, 0.915] | 0.264 [0.245, 0.285] | 1896 |
| 1000 | 0.935 | 0.872 | 0.919 | 0.927 | 0.913 [0.904, 0.922] | 0.258 [0.239, 0.278] | 3773 |
| 2000 (main) | 0.957 | 0.902 | 0.932 | 0.939 | 0.933 [0.926, 0.939] | 0.219 [0.201, 0.238] | 591 |
| 5000 | 0.961 | 0.918 | 0.942 | 0.948 | 0.942 [0.936, 0.947] | 0.187 [0.170, 0.205] | 1249 |

Spearman against N over 8 points: mean MAC rho = +1.00 (p = 0.000); frequency MAE rho = -1.00 (p = 0.000). One network per N.

**Choice of the main network.** The smallest N at which every mode reaches test MAC >= 0.9 is **N = 2000** (lowest mode T1 0.902, bootstrap 95 % [0.893, 0.909]); at N = 1000 the lowest mode is T1 0.872. The rule reads the same fixed test set that section 4 reports, so the main network's in-domain numbers carry a small selection effect.

## 6. The same sweep on the real bridges (zero-shot, 9 bridges)

| N | synthetic MAC (test set) | real MAC vs SSI | real MAC vs FE target | real |f - f_SSI| (Hz) | real |f - f_FE| (Hz) |
|---|---|---|---|---|---|
| 100 | 0.612 | 0.217 | 0.206 | 2.053 | 1.557 |
| 200 | 0.748 | 0.281 | 0.295 | 2.067 | 1.479 |
| 400 | 0.832 | 0.243 | 0.228 | 1.917 | 1.407 |
| 600 | 0.866 | 0.221 | 0.237 | 1.707 | 1.304 |
| 800 | 0.906 | 0.264 | 0.221 | 2.189 | 1.424 |
| 1000 | 0.913 | 0.232 | 0.227 | 2.459 | 1.506 |
| 2000 (main) | 0.933 | 0.254 | 0.267 | 2.235 | 1.531 |
| 5000 | 0.942 | 0.205 | 0.185 | 1.706 | 1.282 |

Spearman, real MAC vs SSI against N: rho = -0.19, p = 0.651 (8 networks).

## 7. The real bridges

Field inputs (`s5a_field_inputs_checks.json`): PASS - at all 9 bridges the swapped rows carry the logger serials x40131, x40133; without the reorder the imported producer and builder reproduce the legacy field outputs and features bit for bit; with it the raw records equal the legacy ones with rows 6 and 8 exchanged, and the labels, geometry and FE targets in the features are unchanged.

Gates (`s5_gates.json`): copied scoring code reproduces E2's published rows on E2's own (raw-order) inputs (max diff 0.0) - PASS; feature convention PASS; input dependence PASS (mean MAC to the zero-input response 0.388, gate < 0.9 per bridge).

### 7.1 Zero-shot, main network

MAC against SSI at the sensor nodes (and, last column, the mean against the FE target at every FE node):

| bridge | B1 | T1 | B2 | T2 | mean vs SSI | mean vs FE |
|---|---|---|---|---|---|---|
| Hellesylt | 0.901 | 0.701 | 0.427 | 0.019 | **0.512** | 0.438 |
| Hundeidvik | 0.399 | 0.021 | 0.014 | 0.030 | **0.116** | 0.089 |
| Leknes | 0.267 | 0.002 | 0.588 | 0.322 | **0.295** | 0.306 |
| Linge | 0.694 | 0.415 | 0.004 | 0.005 | **0.279** | 0.304 |
| Saebo | 0.627 | 0.011 | 0.150 | 0.012 | **0.200** | 0.296 |
| Eidsdal | 0.825 | 0.028 | 0.025 | 0.187 | **0.266** | 0.338 |
| Liabygda | 0.580 | 0.003 | 0.596 | 0.067 | **0.311** | 0.259 |
| Stranda | 0.242 | 0.079 | 0.075 | 0.006 | **0.101** | 0.144 |
| Festoya | 0.673 | 0.064 | 0.081 | 0.015 | **0.208** | 0.230 |
| **mean** | **0.579** | **0.147** | **0.218** | **0.074** | **0.254** | 0.267 |

Frequencies, mean over the nine bridges - against SSI and against the FE target (CLAUDE.md: SSI sits above FE by ~2 Hz at B2/T2 through genuine FE softness, so both are reported):

| mode | predicted (Hz) | SSI (Hz) | FE target (Hz) | |pred - SSI| (Hz) | |pred - FE| (Hz) |
|---|---|---|---|---|---|
| B1 | 4.209 | 4.061 | 3.828 | 0.771 | 0.759 |
| T1 | 5.801 | 6.095 | 5.711 | 1.551 | 1.455 |
| B2 | 5.374 | 8.734 | 6.541 | 3.360 | 1.354 |
| T2 | 8.411 | 11.669 | 9.602 | 3.258 | 2.556 |

### 7.2 Where the gap comes from (broadband decomposition)

*MAC vs FE target, all FE nodes*

| variant | B1 | T1 | B2 | T2 | mean |
|---|---|---|---|---|---|
| ceiling: FE-simulated broadband response, every FE node | 0.888 | 0.682 | 0.768 | 0.749 | 0.772 |
| propagation-only: simulated, sensors only, extended - FE graph | 0.632 | 0.314 | 0.279 | 0.251 | 0.369 |
| propagation-only - hybrid graph | 0.638 | 0.332 | 0.302 | 0.244 | 0.379 |
| zero-shot on the recordings - hybrid graph | 0.598 | 0.047 | 0.357 | 0.065 | 0.267 |
| zero-shot on the recordings - FE-only graph | 0.582 | 0.035 | 0.304 | 0.036 | 0.239 |
| share of the ceiling-to-zero-shot gap reproduced by propagation alone (FE graph) |  |  |  |  | 79.8 % |
| share of the ceiling-to-zero-shot gap reproduced by propagation alone (hybrid graph) |  |  |  |  | 77.8 % |

*MAC vs SSI, sensor nodes*

| variant | B1 | T1 | B2 | T2 | mean |
|---|---|---|---|---|---|
| ceiling: FE-simulated broadband response, every FE node | 0.933 | 0.703 | 0.651 | 0.574 | 0.715 |
| propagation-only: simulated, sensors only, extended - FE graph | 0.698 | 0.351 | 0.316 | 0.294 | 0.415 |
| propagation-only - hybrid graph | 0.693 | 0.274 | 0.245 | 0.230 | 0.360 |
| zero-shot on the recordings - hybrid graph | 0.579 | 0.147 | 0.218 | 0.074 | 0.254 |
| zero-shot on the recordings - FE-only graph | 0.566 | 0.071 | 0.173 | 0.073 | 0.221 |
| share of the ceiling-to-zero-shot gap reproduced by propagation alone (FE graph) |  |  |  |  | 65.2 % |
| share of the ceiling-to-zero-shot gap reproduced by propagation alone (hybrid graph) |  |  |  |  | 77.0 % |

Checks (`s6_checks.json`): PASS - the unfiltered and filtered simulations share one excitation realisation exactly, the ceiling features equal the corrected real-data builder's with every node measured, and the FE targets equal E6's.

### 7.3 Fine-tuned (LOOCV, nested, full)

MAC against SSI at the sensor nodes on the held-out bridge (validation bridge = the next in the list):

| held out | validation | B1 | T1 | B2 | T2 | mean | zero-shot mean | best epoch |
|---|---|---|---|---|---|---|---|---|
| Hellesylt | Hundeidvik | 0.870 | 0.145 | 0.528 | 0.036 | **0.395** | 0.512 | 18 |
| Hundeidvik | Leknes | 0.468 | 0.009 | 0.087 | 0.049 | **0.153** | 0.116 | 2 |
| Leknes | Linge | 0.287 | 0.001 | 0.130 | 0.429 | **0.212** | 0.295 | 11 |
| Linge | Saebo | 0.706 | 0.258 | 0.306 | 0.008 | **0.319** | 0.279 | 77 |
| Saebo | Eidsdal | 0.883 | 0.019 | 0.573 | 0.031 | **0.377** | 0.200 | 82 |
| Eidsdal | Liabygda | 0.824 | 0.048 | 0.042 | 0.267 | **0.295** | 0.266 | 3 |
| Liabygda | Stranda | 0.689 | 0.135 | 0.005 | 0.015 | **0.211** | 0.311 | 68 |
| Stranda | Festoya | 0.000 | 0.133 | 0.049 | 0.099 | **0.070** | 0.101 | 53 |
| Festoya | Hellesylt | 0.674 | 0.061 | 0.059 | 0.014 | **0.202** | 0.208 | 1 |
| **mean** |  | **0.600** | **0.090** | **0.198** | **0.105** | **0.248** | 0.254 |  |

Fine-tuned frequency error, mean over bridges:

| mode | |pred - SSI| (Hz) | |pred - FE| (Hz) |
|---|---|---|
| B1 | 0.751 | 0.710 |
| T1 | 1.443 | 1.350 |
| B2 | 1.739 | 1.661 |
| T2 | 1.479 | 3.690 |

The copied fine-tuning code reproduces E2's nested/full broadband seed-123 rows on E2's own (raw-order) inputs (max diff 0.0): PASS.

## 8. How many measured nodes does the main network need? (synthetic test set)

*Stations on two outermost girders only (2n sensors); mean over the 500 test structures:*

| layout | sensors | share of nodes | B1 | T1 | B2 | T2 | mean MAC | |freq error| (Hz) |
|---|---|---|---|---|---|---|---|---|
| full | 85.1 | 100 % | 0.957 | 0.902 | 0.932 | 0.939 | 0.933 | 0.219 |
| n1 | 2.0 | 3 % | 0.475 | 0.294 | 0.319 | 0.047 | 0.284 | 1.167 |
| n2 | 4.0 | 5 % | 0.661 | 0.418 | 0.423 | 0.306 | 0.452 | 0.795 |
| n3 | 6.0 | 8 % | 0.672 | 0.434 | 0.458 | 0.428 | 0.498 | 0.668 |
| n4 | 8.0 | 11 % | 0.741 | 0.480 | 0.566 | 0.529 | 0.579 | 0.643 |
| n5 | 10.0 | 13 % | 0.786 | 0.515 | 0.627 | 0.565 | 0.623 | 0.601 |
| n6 | 12.0 | 16 % | 0.803 | 0.535 | 0.674 | 0.587 | 0.650 | 0.569 |
| n8 | 16.0 | 22 % | 0.821 | 0.577 | 0.731 | 0.613 | 0.685 | 0.515 |
| n10 | 20.0 | 27 % | 0.826 | 0.590 | 0.751 | 0.625 | 0.698 | 0.486 |
| n12 | 24.0 | 32 % | 0.831 | 0.597 | 0.764 | 0.634 | 0.706 | 0.465 |
| zeros | 0.0 | 0 % | 0.407 | 0.001 | 0.397 | 0.063 | 0.217 | 1.181 |

*Same, by girder count - mean MAC (mean number of sensors):*

| layout | Ng = 2 | Ng = 3 | Ng = 4 | Ng = 5 |
|---|---|---|---|---|
| full | 0.898 (49) | 0.943 (74) | 0.953 (95) | 0.936 (122) |
| n1 | 0.298 (2) | 0.345 (2) | 0.252 (2) | 0.241 (2) |
| n2 | 0.393 (4) | 0.634 (4) | 0.379 (4) | 0.401 (4) |
| n3 | 0.414 (6) | 0.698 (6) | 0.463 (6) | 0.418 (6) |
| n4 | 0.537 (8) | 0.787 (8) | 0.571 (8) | 0.423 (8) |
| n5 | 0.611 (10) | 0.823 (10) | 0.627 (10) | 0.432 (10) |
| n6 | 0.676 (12) | 0.834 (12) | 0.653 (12) | 0.436 (12) |
| n8 | 0.765 (16) | 0.849 (16) | 0.686 (16) | 0.442 (16) |
| n10 | 0.803 (20) | 0.852 (20) | 0.696 (20) | 0.442 (20) |
| n12 | 0.824 (24) | 0.857 (24) | 0.700 (24) | 0.445 (24) |
| zeros | 0.231 (0) | 0.224 (0) | 0.212 (0) | 0.202 (0) |

*Stations on every girder (Ng x n sensors); mean over the 500 test structures:*

| layout | sensors | share of nodes | B1 | T1 | B2 | T2 | mean MAC | |freq error| (Hz) |
|---|---|---|---|---|---|---|---|---|
| full | 85.1 | 100 % | 0.957 | 0.902 | 0.932 | 0.939 | 0.933 | 0.219 |
| n1 | 3.5 | 4 % | 0.494 | 0.406 | 0.301 | 0.049 | 0.312 | 1.120 |
| n2 | 7.0 | 8 % | 0.743 | 0.540 | 0.500 | 0.432 | 0.554 | 0.761 |
| n3 | 10.5 | 13 % | 0.779 | 0.603 | 0.584 | 0.647 | 0.653 | 0.597 |
| n4 | 14.0 | 17 % | 0.848 | 0.694 | 0.700 | 0.769 | 0.753 | 0.523 |
| n5 | 17.5 | 21 % | 0.888 | 0.763 | 0.751 | 0.814 | 0.804 | 0.451 |
| n6 | 21.0 | 25 % | 0.908 | 0.799 | 0.778 | 0.818 | 0.826 | 0.407 |
| n8 | 28.0 | 34 % | 0.928 | 0.848 | 0.819 | 0.832 | 0.857 | 0.344 |
| n10 | 35.0 | 42 % | 0.932 | 0.867 | 0.834 | 0.839 | 0.868 | 0.312 |
| n12 | 42.0 | 50 % | 0.937 | 0.877 | 0.846 | 0.841 | 0.875 | 0.292 |
| zeros | 0.0 | 0 % | 0.407 | 0.001 | 0.397 | 0.063 | 0.217 | 1.181 |

*Same, by girder count - mean MAC (mean number of sensors):*

| layout | Ng = 2 | Ng = 3 | Ng = 4 | Ng = 5 |
|---|---|---|---|---|
| full | 0.898 (49) | 0.943 (74) | 0.953 (95) | 0.936 (122) |
| n1 | 0.298 (2) | 0.351 (3) | 0.295 (4) | 0.305 (5) |
| n2 | 0.393 (4) | 0.649 (6) | 0.516 (8) | 0.657 (10) |
| n3 | 0.414 (6) | 0.732 (9) | 0.687 (12) | 0.780 (15) |
| n4 | 0.537 (8) | 0.829 (12) | 0.821 (16) | 0.823 (20) |
| n5 | 0.611 (10) | 0.876 (15) | 0.876 (20) | 0.854 (25) |
| n6 | 0.676 (12) | 0.879 (18) | 0.884 (24) | 0.865 (30) |
| n8 | 0.765 (16) | 0.889 (24) | 0.895 (32) | 0.878 (40) |
| n10 | 0.803 (20) | 0.889 (30) | 0.899 (40) | 0.881 (50) |
| n12 | 0.824 (24) | 0.892 (36) | 0.901 (48) | 0.884 (60) |
| zeros | 0.231 (0) | 0.224 (0) | 0.212 (0) | 0.202 (0) |

## 9. Carried over unchanged

These do not depend on the synthetic population and are not recomputed here: the curated SSI labels (`SSI_modal_infor/mode_selection.json`), the SSI-vs-FE agreement table and the T2 override rule (`experiments/E2_training_distribution/stage5_ssi_vs_fe.csv`, `t2_mac_diagnostic/`), the corrected real-bridge feature builder (`experiments/E5_real_transfer/build_phase3_features_corrected.py`, run by `s5a_field_inputs.py` on the reordered records) and the propagation operator on FE mode shapes with no network (`experiments/E6_controls/task1_propagation_operator_mac.csv`). Wherever the FE-target frame appears above, Hellesylt T2 and Leknes T2 use the overridden FE modes (best FE mode within 0.5-2x the SSI frequency, adopted if it beats the matcher by >= 0.05 MAC) - those two cells are maximised by construction.

## 10. Figures

Drawn by `s10_figures.py` into `pipeline/results/figures/` (PNG and PDF; binaries, gitignored, rebuilt by the stage). fig4, fig7, fig8 and fig10-fig16 use the previous paper's own figure modules (`experiments/E2_training_distribution/figures/`), in its style and mode palette.

| figure | content | section | on disk |
|---|---|---|---|
| `fig0_frequencies_N2000` | FE target frequency distribution per mode, the main network's 2000 structures | 1 | yes |
| `fig0_frequencies_N5500` | the same over all 5500 structures (same bins and axes) | 1 | yes |
| `fig0_shape_stats_N2000` | FE target mode shapes per mode, main network's structures: the anchor paper's S = sum (phi/max|phi|)^2 (sign-blind; grows with node count) and the sign balance R = |sum phi| / sum |phi| (1 for any absolute shape) | 1 | yes |
| `fig0_shape_stats_N5500` | the same over all 5500 structures | 1 | yes |
| `fig0_distributions_N2000` | for the paper: frequency, S and R in one row (the three panels above, one mode legend), main network's structures | 1 | yes |
| `fig0_distributions_N5500` | the same over all 5500 structures | 1 | yes |
| `fig1_sweep` | in-domain MAC per mode and frequency MAE against N; MAC 0.9 and the main network marked | 5 | yes |
| `fig2_transfer_vs_n` | synthetic test-set MAC and real-bridge MAC (vs SSI) against N | 6 | yes |
| `fig3_sensor_sweep` | MAC and frequency error against the number of measured nodes | 8 | yes |
| `fig4_loss_curves` | the main network's training and validation losses, 5000 epochs (old fig1c style) | 4 | yes |
| `fig5_real_bridges` | zero-shot and fine-tuned MAC vs SSI, per bridge | 7.1, 7.3 | yes |
| `fig6_decomposition` | ceiling, propagation-only and zero-shot, both MAC frames | 7.2 | yes |
| `fig7_synthetic_performance` | main network: MAC box plot per mode, predicted vs true frequency (old fig1 style) | 4 | yes |
| `fig8_modeshapes` | main network: signed mode shapes on a representative test deck (old fig1b style) | 4 | yes |
| `fig9_sensor_layouts` | fig3's companion: the nodes each sensor layout measures, n = 1-3 shown | 8 | yes |
| `fig10_ablation_mac` | the ablation grid: MAC per mode, rows layer, columns edge feature | 3 | yes |
| `fig11_ablation_modeshapes` | the ablation grid's predicted mode shapes on one test deck | 3 | yes |
| `fig12_real_cases_fe` | the nine bridges, three cases side by side - simulated, every FE node; simulated, sensors + propagation; measured, sensors + propagation (7.2's ceiling, propagation-only and zero-shot on the hybrid graph): MAC vs FE target per mode and predicted vs FE target frequency (fig7 style) | 7.2 | yes |
| `fig12_real_cases_ssi` | the same, MAC vs SSI at the sensor nodes and predicted vs SSI frequency | 7.2 | yes |
| `fig12_real_cases_hybrid` | the same, each case against its own reference: the two simulated cases vs the FE target, the measured case vs SSI. Its last step changes input and reference at once - quote a drop within one frame, from the two figures above | 7.2 | yes |
| `fig13_real_modeshapes` | those three cases' predicted mode shapes beside the FE target and the SSI shapes, with MAC in both frames, on the representative bridge; every bridge in `figures/fig13_by_bridge/` | 7.2 | yes |
| `fig14_finetune` | zero-shot (fig12_real_cases_hybrid's third column) against the nested leave-one-out fine-tuned network: MAC vs SSI per mode and predicted vs SSI frequency, nine bridges | 7.3 | yes |
| `fig15_finetune_modeshapes` | fig13's bridge before and after fine-tuning: FE target, SSI, zero-shot and fine-tuned predictions, MAC in both frames | 7.3 | yes |
| `fig16_real_graphs` | the nine bridges' graphs as the network receives them: FE nodes, the sensor nodes snapped onto them, and the edges (the previous paper's fig0b) | 7 | yes |

