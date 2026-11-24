# Trajectory-Type Imbalance in Offline Goal-Conditioned Reinforcement Learning

Code and results for *Trajectory-Type Imbalance in Offline Goal-Conditioned Reinforcement Learning: A Controlled Dataset Analysis*.

The study tests how the composition of offline training data affects policy success in **FetchPush, PointMaze and OGBench AntMaze**. It evaluates **GCSL, WGCSL, HER+DDPG, GoFAR, HIQL, offline contrastive RL (CRL) and SMORe**. Within each seed pair, the stored dataset contains 10,000 trajectories. Stratified weighted sampling changes how often a trajectory category is drawn during training. The main matched shifts preserve the sampling probabilities of groups defined by source (expert or random), success or failure, return group and the two other trajectory labels. Complete category removal is a separate test that sets the target weight to zero and redistributes its probability within each source.

## Reproduce the manuscript results

Run these commands from the repository root with Python 3. Both use the standard library and the data in this repository.

```bash
python revision_20261118/summarize_figure3.py --output figure3_cells.csv
python revision_20261118/reproduce_table4.py
```

| Result | Source and interpretation |
| --- | --- |
| Figure 3 and Table A2 | [Download the Figure 3 CSV](https://raw.githubusercontent.com/DamySSG/trajectory-type-imbalance-offline-gcrl/1ac95e3ff9a2fe06b34fc4f6482e55a8a3cbe2c3/figure_data/Dose_profiles_source.csv). The first command calculates mean success, mean change from natural sampling and their sample standard deviations for 441 cells. Each cell has three seed pairs. The 1,323 observations include a shared natural-sampling baseline repeated across the three trajectory labels. |
| Table 4 | [Paired changes in successful episode counts](revision_20261118/table4_paired_counts.csv). The second command reproduces all nine tests for the independent FetchPush experiment with GoFAR and SMORe. It prints means, sample standard deviations, individual 95% confidence intervals and two-sided paired permutation *p* values before and after Holm correction. |
| Sampling probabilities and trajectory labels | [Download the sampling workbook](https://raw.githubusercontent.com/DamySSG/trajectory-type-imbalance-offline-gcrl/7743d3b825a92e9b64cea4c7342c0d4ca70fefd1/figure_data/Sampling_cross_tabs.xlsx). It shows how often pairs of labels occur and their sampling probabilities before and after weighting. The [sampling check](revision_20261118/README.md#check-the-sampling-probabilities) verifies which shares stay fixed. |
| Training, weighting and other analyses | [Implementation and S1–S6 evidence](revision_20261118/README.md), including sampling code, learner implementations, validation data and instructions for the ten-seed FetchPush experiment. |

The manuscript covers **2,413 training runs**. The prespecified experiments account for 1,773 runs, intermediate sampling levels add 540, and the independent FetchPush experiment adds 100 (two algorithms, five settings and ten new seed pairs). The nine tests in Table 4 apply to this last experiment.

## Earlier experiments

The root-level training code, `scripts/run_experiments.py`, `analysis/` and `results/processed/` preserve the earlier two-environment, four-algorithm study using trajectory removal and duplication. The [original reproduction guide](REPRODUCIBILITY.md) documents its 532-condition matrix.

## Attribution and licence

The original training code derives from [GoFAR](https://github.com/JasonMa2016/GoFAR). See the [notice](NOTICE.md) and [licence](LICENSE). The revised implementation archive also includes the licences and attribution for its third-party components.
