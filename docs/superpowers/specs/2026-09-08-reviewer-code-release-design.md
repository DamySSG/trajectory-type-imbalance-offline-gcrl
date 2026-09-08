# Reviewer Code Release Design

## Purpose

Create a compact, private GitHub repository containing the code and processed results needed to inspect and reproduce the experiments reported in *Trajectory-Type Imbalance in Offline Goal-Conditioned Reinforcement Learning: A Controlled Dataset Analysis*.

The repository begins from pre-generated GoFAR-format expert and random trajectory files. Raw offline buffers are intentionally excluded from Git because they are large binary research data and will be distributed separately through an archival download location.

## Scope

The release includes:

- the GoFAR-derived training implementation used for GCSL, WGCSL, GoFAR, and HER+DDPG;
- FetchPush-v1 and PointMaze-Medium-v3 environment support used by the study;
- deterministic trajectory-type classification and ratio intervention code;
- validation scripts for the generated dataset variants;
- a compact experiment-matrix launcher with the manuscript settings;
- processed result tables and the script used to regenerate appendix figures;
- dependency specifications, provenance, citation metadata, and end-to-end reproduction instructions;
- focused tests for intervention invariants and release hygiene.

The release excludes:

- raw `.pkl` trajectory buffers and trained checkpoints;
- W&B run caches, logs, media, and generated figures;
- duplicated one-off launch-command dumps and exploratory scripts;
- unrelated upstream utilities and local IDE/cache files.

## Scientific Contract

For each environment, an expert pool and a random-action pool are supplied separately. Training constructs a shuffled 50:50 mixture. Dataset interventions act on one trajectory subtype at a time at ratios `0.1, 0.2, 0.5, 0.8, 1, 2, 5, 10`; non-target trajectories are retained unchanged. Ratios below one downsample the selected subtype, ratio one is the unchanged reference, and ratios above one duplicate selected trajectories using a fixed random seed.

FetchPush-v1 classifications use initial gripper, object, and goal geometry. PointMaze-Medium-v3 classifications use initial spatial support, net horizontal displacement, and realized path length. The exact constants and decision rules remain in executable code and are summarized in the data documentation.

## Repository Layout

```text
.
|-- README.md
|-- REPRODUCIBILITY.md
|-- CITATION.cff
|-- LICENSE
|-- NOTICE.md
|-- environment.yml
|-- train.py
|-- envs/
|-- her_modules/
|-- mpi_utils/
|-- rl_modules/
|-- scripts/
|   |-- data/
|   `-- run_experiments.py
|-- analysis/
|-- data/README.md
|-- results/processed/
`-- tests/
```

## Reproducibility and Safety

The public-facing commands must not depend on absolute server paths, hard-coded W&B entities, or untracked local state. Dataset paths, output directories, seeds, W&B mode, and project names are command-line parameters. Every tracked file is scanned for credentials and large/binary artifacts before push. The GitHub repository is created as private and populated from a fresh history rather than exposing the full experimental server history.

## Provenance

The learning implementation is derived from the MIT-licensed GoFAR repository by Ma et al. The release retains the upstream license and adds a NOTICE identifying the source and describing study-specific additions. Processed result tables are outputs of the experiments reported in the manuscript.
