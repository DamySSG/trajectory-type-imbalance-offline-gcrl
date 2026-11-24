# Offline buffers for FetchPush and PointMaze

This page documents inputs to the original two-environment code. The revised study also includes AntMaze. Its [data and code guide](../revision_20261118/README.md) links the public results, source summaries and construction records.

## Provenance

For FetchPush and PointMaze, each environment has 10,000 expert-source episodes and 10,000 random-action episodes. Collection records describe a DDPG-HER collector trained for 1,000 epochs with GoFAR-derived code. Its checkpoint and full training configuration are unavailable. The [public source summaries](../revision_20261118/S4_Source_pools.zip) give the measured outcomes. The source labels identify the policy that collected each trajectory.

## Expected Pickle Schema

Each file is a Python dictionary of NumPy arrays with one shared leading
episode dimension `N`:

| Key | Meaning | Expected shape |
| --- | --- | --- |
| `o` | observations | `(N, H + 1, observation_dim)` |
| `ag` | achieved goals | `(N, H + 1, goal_dim)` |
| `g` | desired goals | `(N, H, goal_dim)` or `(N, H + 1, goal_dim)` |
| `u` | actions | `(N, H, action_dim)` |

Additional metadata entries are preserved if they do not use the episode
dimension. The four required arrays must have matching episode counts.

## Placement

```text
data/raw/fetchpush/expert.pkl
data/raw/fetchpush/random.pkl
data/raw/pointmaze/expert.pkl
data/raw/pointmaze/random.pkl
```

The raw offline buffers are not included in this repository.
