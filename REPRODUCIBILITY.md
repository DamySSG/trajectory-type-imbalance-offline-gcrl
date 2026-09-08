# Reproducibility Guide

## 1. Software Environment

The recorded experiments ran with Python 3.8.19, NumPy 1.24.4, PyTorch 2.4.0,
Gymnasium 0.29.1, gymnasium-robotics 1.2.4, mpi4py 4.0.0, and W&B 0.17.5.
The dependency file reproduces these principal versions.

The manuscript name PointMaze-Medium-v3 corresponds to the registered runtime
environment ID `PointMaze_Medium-v3` in gymnasium-robotics 1.2.4.

```bash
conda env create -f environment.yml
conda activate trajectory-imbalance
python -m unittest discover -s tests -v
```

MuJoCo-capable Linux and a working MPI installation are required for full
training. Plot and dataset-construction tests do not require a GPU.

## 2. External Offline Buffers

Place the four source files as follows after obtaining the separately
distributed data archive:

```text
data/raw/
|-- fetchpush/
|   |-- expert.pkl
|   `-- random.pkl
`-- pointmaze/
    |-- expert.pkl
    `-- random.pkl
```

The archive location is intentionally not hard-coded in this repository. Add
the persistent repository or reviewer-only download URL to `data/README.md`
before distributing the review package.

## 3. Construct Dataset Variants

Run each builder once for the expert source and once for the random-action
source. The default seed is 0 and the default ratio grid is
`0, 0.1, 0.2, 0.5, 0.8, 1, 2, 5, 10`.

```bash
python -m scripts.data.make_fetchpush_imbalances \
  --input data/raw/fetchpush/expert.pkl \
  --output-root data/generated/fetchpush/expert

python -m scripts.data.make_fetchpush_imbalances \
  --input data/raw/fetchpush/random.pkl \
  --output-root data/generated/fetchpush/random

python -m scripts.data.make_pointmaze_imbalances \
  --input data/raw/pointmaze/expert.pkl \
  --output-root data/generated/pointmaze/expert \
  --base-episodes 10000

python -m scripts.data.make_pointmaze_imbalances \
  --input data/raw/pointmaze/random.pkl \
  --output-root data/generated/pointmaze/random \
  --base-episodes 10000
```

Each output root contains one `manifest.csv`. Validate all generated files:

```bash
python -m scripts.data.validate_variants data/generated/fetchpush/expert
python -m scripts.data.validate_variants data/generated/fetchpush/random
python -m scripts.data.validate_variants data/generated/pointmaze/expert
python -m scripts.data.validate_variants data/generated/pointmaze/random
```

### Intervention Definitions

For FetchPush-v1, the gripper projection is read from observation coordinates
`0:3`, the object position from the initial achieved goal, and the desired
target from the initial goal. Behavioral tendency uses the sign of the planar
cross product with `epsilon = 1e-8`. Spatial coverage uses the mean initial
gripper projection as the center; the inner radius is the `0.01` quantile of
initial object-gripper distances. Reachability uses the weighted initial effort
score with `alpha = 1`, `beta = 1`, and `gamma = 0.5`, split at its median.

For PointMaze-Medium-v3, 10,000 base episodes are selected with seed 0.
Behavioral tendency uses final-minus-initial horizontal displacement with
`epsilon = 1e-3`. Spatial coverage uses the 25th and 75th percentiles of initial
achieved-goal coordinates to define four corner regions and a center region.
Reachability is the accumulated Euclidean achieved-goal path length; its lower
and upper tertiles define short and long trajectories, while the middle group
remains non-target data.

In every intervention, `r` is the new target-subtype count divided by its count
in the base buffer. Non-target trajectories are retained. Values below one
downsample the target subtype, `r = 1` preserves its count, values above one
resample target trajectories, and `r = 0` removes the subtype.

## 4. Generate and Run the Experiment Matrix

Write the complete dry-run matrix:

```bash
python -m scripts.run_experiments \
  --data-root data/generated \
  --seed 0 \
  --output outputs/experiment_matrix.jsonl
```

The matrix contains 532 runs: 54 conditions per algorithm for FetchPush and 79
per algorithm for PointMaze. PointMaze reachability has no recorded `r = 0`
condition; this scope matches the processed results rather than silently adding
unreported runs.

Execute the complete matrix only after inspecting the generated commands:

```bash
python -m scripts.run_experiments --data-root data/generated --seed 0 --execute
```

All runs use `expert_percent = 0.5`, `random_percent = 0.5`, seed 0, 800 epochs,
10 cycles per epoch, 20 updates per cycle, batch size 512, discount 0.98, and
Polyak coefficient 0.95. HER+DDPG uses a future-goal relabeling proportion of
0.5. GCSL and WGCSL use 1.0. GoFAR uses its discriminator-derived reward with
20 discriminator iterations and coefficient 0.01.

W&B logging is configurable:

```bash
python train.py ... \
  --wandb-mode online \
  --wandb-project trajectory-type-imbalance-offline-gcrl
```

Use `--wandb-mode disabled` for local checks without network logging.

## 5. Rebuild the Appendix

The repository contains the 532 cleaned records used for descriptive analysis.

```bash
python -m analysis.generate_appendix_figures \
  --input results/processed/Table_A0_cleaned_raw_results.csv \
  --output-root outputs/appendix
```

The command writes six CSV files and 96 response-curve figures. Generated media
are excluded from Git because they are deterministic outputs of the tracked
script and result table.

## 6. Statistical Scope

The processed package contains one recorded result for each dataset-algorithm
condition. Seed 0 fixes dataset selection, resampling, training initialization,
and evaluation. The tables and figures therefore support descriptive
condition-level comparisons, not seed-level confidence intervals or
inferential claims across independent training repetitions.
