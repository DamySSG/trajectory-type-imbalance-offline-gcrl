# Offline Data

Raw trajectory buffers are distributed separately and are ignored by Git.

## Provenance

For each environment, the expert pool was collected by evaluating a pretrained
HER+DDPG agent for 10,000 episodes. The agent was trained for 1,000 epochs with
the official GoFAR implementation and its original hyperparameter settings.
The random pool contains 10,000 evaluation episodes collected by applying
uniformly sampled random actions in the same environment. This procedure was
used independently for FetchPush-v1 and PointMaze-Medium-v3.

During offline learning, the expert and random sources are sampled at equal
episode quotas and shuffled, yielding the 50% expert and 50% random mixture
reported in the manuscript. Source labels are used only to construct that
mixture and are not supplied as policy inputs.

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

The long-term archive URL or reviewer-only Drive URL should be recorded here
before external distribution. No URL is currently embedded because the data
have not yet been deposited.
