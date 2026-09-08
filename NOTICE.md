# Notice

This repository contains research code derived from
[JasonMa2016/GoFAR](https://github.com/JasonMa2016/GoFAR). The experimental
checkout was based on source revision
`cfb2d4e36d69c5ec16edf6d127464eb263925e70` and included study-specific local
changes to the training entry point, buffer loading, evaluation compatibility,
and HER+DDPG implementation. GoFAR is distributed under the MIT License
retained in `LICENSE`.

Study-specific additions include trajectory-type classification and controlled
ratio interventions for FetchPush-v1 and PointMaze-Medium-v3, experiment-matrix
orchestration, processed result tables, analysis scripts, and reproducibility
documentation.

The repository does not redistribute raw offline trajectory buffers, trained
model checkpoints, or W&B run directories.

The reviewer release routes the manuscript's HER+DDPG condition to
`rl_modules/ddpg_agent.py`, the GoFAR DDPG implementation that invokes the HER
sampler. A later local exploratory module named `ddpg_agent1.py` is excluded:
its recorded modification timestamp postdates the FetchPush HER+DDPG runs and
its TD3-style twin-critic objective is outside the methods reported in the
manuscript.
