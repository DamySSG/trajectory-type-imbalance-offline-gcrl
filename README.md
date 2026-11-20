# Trajectory-Type Imbalance in Offline Goal-Conditioned Reinforcement Learning

Code and processed results for *Trajectory-Type Imbalance in Offline
Goal-Conditioned Reinforcement Learning: A Controlled Dataset Analysis*.

The study examines three forms of trajectory-level dataset composition:
reachability, spatial coverage, and behavioral tendency. Deterministic
classifiers identify task-grounded trajectory subtypes in FetchPush-v1 and
PointMaze-Medium-v3, after which one subtype is removed, downsampled, retained,
or duplicated while all non-target trajectories remain unchanged. GCSL, WGCSL,
GoFAR, and HER+DDPG are evaluated on the resulting datasets.

## Repository Contents

- `scripts/data/`: executable subtype definitions, ratio interventions, dataset builders, and validation.
- `train.py`, `rl_modules/`, `her_modules/`, `mpi_utils/`: GoFAR-derived offline GCRL training code.
- `scripts/run_experiments.py`: the 532-condition matrix represented in the cleaned manuscript results.
- `results/processed/`: cleaned row-level metrics and appendix summary tables.
- `analysis/`: deterministic regeneration of the six appendix tables and 96 response-curve figures.
- `tests/`: synthetic tests of subtype definitions, resampling invariants, experiment coverage, and analysis output.

Raw offline buffers and checkpoints are deliberately not stored in Git. Their
required layout and provenance are documented in [`data/README.md`](data/README.md).

## Quick Check

```bash
conda env create -f environment.yml
conda activate trajectory-imbalance
python -m unittest discover -s tests -v
python -m scripts.run_experiments --output outputs/experiment_matrix.jsonl
python -m analysis.generate_appendix_figures --output-root outputs/appendix
```

The experiment launcher is dry-run by default. It writes commands but does not
start training unless `--execute` is supplied.

Detailed, ordered instructions are in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Provenance

The learning implementation is derived from the MIT-licensed
[GoFAR codebase](https://github.com/JasonMa2016/GoFAR) by Ma et al. The source
checkout and study-specific changes are identified in [`NOTICE.md`](NOTICE.md).

## License

This repository is distributed under the MIT License. See [`LICENSE`](LICENSE).
