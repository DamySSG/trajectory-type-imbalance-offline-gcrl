# Weighted-sampling experiments

The revised study covers seven algorithms in FetchPush, PointMaze and OGBench AntMaze. The prespecified experiments use three training/dataset seed pairs (11/1011, 29/1029 and 47/1047). The implementation and S1–S6 archives below cover that part of the study. Figure 3 also uses the intermediate sampling levels, and Table 4 uses an independent set of ten seed pairs.

## Figure 3 and Table A2

[Figure 3 source data (CSV)](https://raw.githubusercontent.com/DamySSG/trajectory-type-imbalance-offline-gcrl/1ac95e3ff9a2fe06b34fc4f6482e55a8a3cbe2c3/figure_data/Dose_profiles_source.csv) contains the final success rate and paired change for each observation. The seven settings are −10, −5, −2.5, 0, +2.5, +5 and +10 percentage points relative to the natural sampling share. Each environment has three target labels: distant goals, sparse spatial support and positive action direction (movement direction in AntMaze). The method ID `ddpg` denotes HER+DDPG.

Run the following command from the repository root.

```bash
python revision_20261118/summarize_figure3.py --output figure3_cells.csv
```

The script checks the three seed pairs in each of the 441 cells and writes means and sample standard deviations for both final success (Table A2a) and change from natural sampling (Table A2b). Across the seven levels, mean success rises and falls in 57 of 63 profiles. Six profiles remain at zero. The older `dose_source_data.csv` in this directory contains 378 rows for two algorithms only.

## Table 4. Independent FetchPush experiment

Both GoFAR and SMORe use ten new seed pairs and the same 1,024 evaluation cases. Within each pair, both learners use the same 10,000 stored trajectories under five settings: natural sampling, a ten-percentage-point increase in the sparse-support share, and separate removal of distant-goal, sparse-support or positive-action trajectories. See the [experiment setup and weight builder](fetchpush_replication/DESIGN.md).

[Seed-level data](table4_paired_counts.csv) contain the eight within-algorithm changes in successful episode counts in Table 4. Each column is a training seed; its dataset seed is 1,000 higher. The [analysis script](reproduce_table4.py) calculates the ninth contrast by subtracting GoFAR's change at +10 points from SMORe's change within each pair.

```bash
python revision_20261118/reproduce_table4.py
```

The output gives the mean change in percentage points, its sample standard deviation, an individual 95% t interval, an exact two-sided paired permutation *p* value and a Holm-adjusted *p* value. Each test enumerates all 1,024 sign assignments for the ten paired differences. The Holm correction covers these nine tests together. All nine adjusted values are 0.017578125 (0.0176 in Table 4).

## Check the sampling probabilities

The public workbook covers 522 sampling settings. In 441 settings the weights preserve the joint shares of source, final success, return group and the two other trajectory labels. The other 81 settings control source alone. These are probabilities of drawing trajectories during training.

```bash
python -m pip install openpyxl
python revision_20261118/verify_sampling.py
```

The script reads the largest change in any controlled joint group from the workbook's `Views` sheet. Across the 441 jointly matched settings this is below 0.00000001 percentage points. It also uses the two-variable tables on the `Cross-tabs` sheet to independently recalculate the before-and-after share of each controlled label. The full five-variable distribution cannot be reconstructed from pairs of variables alone.

## Training and sampling implementation

Extract [implementation_snapshot.zip](implementation_snapshot.zip) to inspect the source. [implementation_manifest.json](implementation_manifest.json) records file hashes. The archive includes the modules listed below.

| Files inside the archive | Purpose |
| --- | --- |
| `revision/measurements.py` | Calculate trajectory descriptors and fit category thresholds. |
| `revision/experimental_controls.py`, `revision/secondary_controls.py`, `revision/reciprocal_controls.py` | Construct weighted training views, including source-matched and jointly matched changes, category removal and direction swaps. |
| `revision/legacy_control.py`, `rl_modules/`, `revision/modern_agent.py`, `revision/ogbench_native.py` | Learners, training adapters and the AntMaze environment interface. |
| `revision/episode_evaluation.py`, `revision/seed_statistics.py` | Episode evaluation and paired-seed statistics. |
| `revision/vendor/ADAPTATIONS.md` | Third-party revisions, licences and study-specific adaptations. |

The snapshot includes historical launchers for several experimental designs. Use the run configurations and condition records for the relevant study component. The release does not bundle the complete historical AntMaze worker pipeline. Retraining also requires the original offline buffers, simulator environments and reset-bank artifacts. The public release contains processed results and analysis code. It excludes the raw buffers and model checkpoints. Saved validation curves are available for FetchPush and PointMaze; complete training-loss histories and a complete AntMaze validation-curve collection are unavailable.

## Supporting material S1–S6

| Material | Files |
| --- | --- |
| S1. Prespecified outcomes and effects | [FetchPush and PointMaze runs and contrasts](S1_Complete_effects.zip) |
| S2. Validation | [FetchPush and PointMaze validation success](S2_Complete_validation.zip) |
| S3. Dataset construction | [Trajectory counts, selection probabilities and run-to-dataset mapping](S3_Construction.zip) |
| S4. Diagnostics | [Sampling](S4_Baseline_sampling.zip), [configurations](S4_Configuration.zip), [trajectory descriptors](S4_Descriptor_diagnostics.zip), [evaluation cases](S4_Evaluation_banks.zip), [source pools](S4_Source_pools.zip) |
| S5. AntMaze results | [Outcomes, contrasts and summaries](S5_Ant_complete.zip) |
| S6. AntMaze construction and training draws | [Weighted views and exposure records](S6_Ant_construction_and_full_exposure.zip) |
| Sampling workbook | [Counts, sampling probabilities and label associations for 522 settings](https://raw.githubusercontent.com/DamySSG/trajectory-type-imbalance-offline-gcrl/7743d3b825a92e9b64cea4c7342c0d4ca70fefd1/figure_data/Sampling_cross_tabs.xlsx) |

[evidence_summary.json](evidence_summary.json) covers the 1,773 prespecified runs (591 per environment). [weighted_contrasts_audited.json](weighted_contrasts_audited.json) contains 546 contrasts in 24 families for that three-pair study. The Holm-adjusted values are 1.00 within those families. The independent ten-pair tests in Table 4 use their own data and correction. To regenerate the workflow and weighted removal/reciprocal panels from the archived outcomes, install NumPy and Matplotlib and run `python revision_20261118/plot_weighted.py` from the repository root.

[release_manifest.json](release_manifest.json) records the original evidence release. Its README hash refers to the text at the time of that release.
