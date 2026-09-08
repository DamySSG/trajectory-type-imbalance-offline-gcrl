# Reviewer Code Release Implementation Plan

**Goal:** Build and privately publish a minimal, auditable code package for the paper's trajectory-type imbalance experiments.

**Architecture:** Preserve the tested GoFAR training modules and environment wrappers with minimal behavioral edits. Place study-specific dataset construction, experiment orchestration, and result analysis behind small command-line interfaces. Keep large offline buffers external and make their required schema and placement explicit.

**Tech Stack:** Python 3.8, NumPy, pandas, PyTorch, Gymnasium, gymnasium-robotics, mpi4py, W&B, unittest.

---

### Task 1: Establish clean release history and provenance

**Files:** `.gitignore`, `LICENSE`, `NOTICE.md`, `CITATION.cff`, design and plan documents.

1. Initialize a fresh Git repository on branch `release/private-review`.
2. Retain the upstream MIT license and record the GoFAR source commit provenance available from the source checkout.
3. Ignore raw buffers, checkpoints, logs, W&B state, generated figures, and Python caches.
4. Commit the release design and plan before importing implementation files.

### Task 2: Import the minimum scientific implementation

**Files:** `train.py`, `envs/**`, `her_modules/**`, `mpi_utils/**`, `rl_modules/**`.

1. Copy only the training and environment modules used by the reported methods and environments.
2. Remove cache and generated files from the imported tree.
3. Confirm Python syntax with `python -m compileall`.
4. Commit the preserved training implementation separately from study tooling.

### Task 3: Package deterministic dataset interventions using TDD

**Files:** `tests/test_dataset_interventions.py`, `scripts/data/interventions.py`, `scripts/data/make_fetchpush_imbalances.py`, `scripts/data/make_pointmaze_imbalances.py`, validation entry points.

1. Write tests for ratio-one identity, non-target preservation, deterministic downsampling, and exact target duplication counts.
2. Run the tests and confirm failure because the shared intervention module does not exist.
3. Implement the smallest reusable intervention functions and rerun until tests pass.
4. Adapt the environment-specific classifiers from the source scripts without changing their scientific definitions.
5. Add synthetic schema tests for both environment-specific classifiers.
6. Commit the dataset tooling and tests.

### Task 4: Make experiment execution reviewer-facing

**Files:** `tests/test_experiment_matrix.py`, `scripts/run_experiments.py`, `train.py`.

1. Write a test asserting the expected environments, methods, ratios, subtype conditions, seed, and 50:50 mixture arguments.
2. Confirm the test fails before the matrix generator exists.
3. Implement dry-run command generation and optional execution without absolute paths.
4. Make W&B mode/project/entity configurable while preserving training defaults used in the study.
5. Run unit tests and syntax checks, then commit.

### Task 5: Package processed outputs and analysis

**Files:** `results/processed/*.csv`, `analysis/generate_appendix_figures.py`, `analysis/README.md`.

1. Copy the cleaned experimental matrix, baseline, full-result, and best/worst-condition tables.
2. Copy the appendix figure-generation script and replace machine-specific path assumptions with relative CLI arguments if present.
3. Run the analysis in a temporary output directory and verify expected artifacts are produced.
4. Commit result and analysis assets.

### Task 6: Write reproduction documentation

**Files:** `README.md`, `REPRODUCIBILITY.md`, `data/README.md`, `environment.yml`.

1. Document installation, external data placement, expected pickle schema, dataset generation, validation, dry-run experiment generation, training, and result regeneration.
2. State the expert/random data provenance and 50:50 mixture precisely without claiming raw data are hosted in Git.
3. Record exact environment and package versions from a successful experiment run.
4. Add a machine-readable citation file and repository provenance.
5. Commit documentation.

### Task 7: Audit and privately publish

**Files:** entire repository.

1. Run all unit tests, syntax compilation, CLI help commands, and analysis smoke tests.
2. Verify no tracked `.pkl`, checkpoint, log, cache, generated media, credential, or oversized file exists.
3. Review the final Git diff and tracked-file manifest against this plan.
4. Create `DamySSG/trajectory-type-imbalance-offline-gcrl` on GitHub with private visibility.
5. Push the verified release as the remote `main` branch over SSH.
6. Confirm the remote branch and private visibility, then report the repository URL and exact included/excluded scope.
