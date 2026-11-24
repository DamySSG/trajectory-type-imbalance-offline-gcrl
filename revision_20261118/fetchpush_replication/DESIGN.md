# FetchPush replication design

GoFAR and SMORe each used the same five sampling settings in each of ten new seed pairs (100 training runs). Each pair used one FetchPush dataset of 10,000 distinct trajectories, drawn without replacement from the expert and random pools (5,000 each). The data seed equals the training seed plus 1,000. The ten pairs and the hashes of their ordered trajectory IDs are in [cohorts_verified.json](cohorts_verified.json).

| Setting in the files | Probability of selecting a trajectory during training |
| --- | --- |
| `natural` | All 10,000 trajectories have equal probability. |
| `plus` | The total probability of selecting sparsely supported trajectories increases by 10 percentage points relative to the natural setting. The weights preserve the probability of each joint group defined by source, final success, return group, goal-distance category and action-direction category. |
| `goal_separation` | Trajectories starting with the object far from the goal receive zero probability. |
| `spatial_support` | Sparsely supported trajectories receive zero probability. |
| `action_direction` | Trajectories with a positive action-direction label receive zero probability. |

In each removal setting, the weights assign half the probability to each source and divide that half equally among its eligible trajectories. GoFAR used 100,000 updates with batches of 512. SMORe used 200,000 updates with batches of 256. The final policy in every run was evaluated on the same 1,024 FetchPush scenarios. Table 4 uses only the ten new pairs.

## Rebuild the training weights

The code calls the original `build_cohort` and `removal_cohort` functions in [implementation_snapshot.zip](../implementation_snapshot.zip). It requires Python 3.9 or later, NumPy 1.26.4, SciPy and the original per-trajectory FetchPush label file. The labels depend on the offline source pools and are not in this public release. The expected SHA-256 of `labels_FetchPush.npz` is `96ced21c25447d8a6c50c3d5e4da0e705f993c46cf03b11cfa53deb9ddc01b7a`.

Run the following commands from the repository root, replacing the label path with the path to the original file.

```sh
python -m zipfile -e revision_20261118/implementation_snapshot.zip implementation
python revision_20261118/fetchpush_replication/prepare_views.py --implementation implementation --labels /path/to/labels_FetchPush.npz --output fetchpush_replication_weights
```

Each output file contains the ordered trajectory IDs (`global_ids`), source IDs (`source`), within-source episode IDs (`source_episode`) and five probability arrays named after the settings above. Array positions refer to the same trajectories in all five settings. Generating the weights does not train a policy. Training additionally requires the original offline trajectories, training configurations, simulator and evaluation scenarios.
