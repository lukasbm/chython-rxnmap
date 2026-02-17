# AGENTS.md

Rules and conventions for coding agents working in this repository.
Follow these unless explicitly instructed otherwise.

---

## Core Design Principles

### No Configuration in Code

* **Do not use config objects** in application code.
* Write all code **as if configuration does not exist**.
* Every class and function must take **explicit parameters**.
* Pass fully constructed objects as arguments instead of instantiating them ad-hoc.
* Hydra (or Fire) exists **outside** the application layer to wire things together.

---

### Modularity & Clarity

* Avoid global variables.
* Pass all required data via function parameters or class constructors.
* Each function or method should do **one clear thing**.
* Break large functions into smaller, focused ones.
* Use descriptive, explicit names for variables, functions, classes, and methods.
* Avoid unnecessary abstractions; stick to frameworks already in use.
* Exclude error handling unless it is **critical for correctness**.

---

## Output & File System Rules (Critical)

* All outputs must go through:
  ```python
  graph_learning.utils.get_output_dir()
  ```
* Treat the returned `Path` as the **single output root**.
* Organize outputs using subdirectories under this root.
* Prefer simple files in directories.
* Avoid databases or lock-heavy binary formats (e.g. sqlite, rocksdb).
* Package outputs (zip, `.h5`) **only for publication or distribution**.
* When writing files, do **not** assume directories exist.
* Always print out the output paths being written to.
* Ad-hoc scripts for debugging, validating, sanity-checking or quick experiments should go into `scratch/`

---

## Execution & Structure

* Always run code from the **project root**.
* Use:

  ```bash
  python -m package.module
  ```
* Do not rely on `cd`-based execution or `sys.path` hacks.

---

## Pipelines, CLIs, and Orchestration

* Do **not** embed pipeline logic in scripts.

    * No checking for file existence
    * No orchestration logic
* Do **not** orchestrate jobs (SLURM, retries, batching) inside scripts.
* These belong to external layers (Snakemake, submitit, wrappers).

### CLIs

* Do not write custom CLIs.
* Use **Hydra auto-CLI** or **Google Fire** (installed).
* Do **not** use `argparse`.

---

## Parallelism

* Prefer **GNU Parallel** when invocations are independent.
* If needed inside Python, use **Joblib** only for:

    * Parallel execution
    * Disk caching
    * Serialization
* Do not misuse Joblib for I/O-bound or distributed workloads.

---

## Repository Philosophy

* Use a **monorepo** for all research code, notes, and papers.
* Keep a **single Python environment**.
* Updating code is preferred over reviving old environments.

### External Code

* Do not clone external repos and set them up.
* Copy relevant files into this repo and adapt them.
* Always include the **paper** as context (prefer Markdown).

---

## Testing & Validation

* Do not write unit tests.
* Prefer:

    * Assertions
    * Hypothesis / property-based tests
* Focus on:

    1. Does it crash?
    2. Does it violate invariants?

---

## Static Analysis & Invariants

* Use:

    * `ruff` for linting
    * `pyright` for typing (preferred over mypy)
* Enforce runtime invariants:

    * Correct working directory
    * Output paths restricted to output root

---

## PyTorch Rules

* Use `torch.compile` (PyTorch ≥ 2.0) at least once to surface bugs.
* Prefer **assertions over tests** inside `forward` and critical code paths.
* Use `torch.autograd.set_detect_anomaly(True)` in debug mode only.
* When saving models:

  ```python
  torch.save(model, path)
  ```

Do **not** save `state_dict` only; it does not help if architectures change.

### Incompatible Shapes

If there are incompatible shapes they almost ALWAYS have a good reason for it.
E.g. the dataset features count does not match the model input features count.
(Happens often when loading from checkpoints.)

NEVER fix these by silently reshaping or squeezing tensors, it's a major cause of errors which can derail my research!
Always raise an explicit error explaining the mismatch.

---

## Model API Standard: BaseGraphLightning

* All new GNN models MUST inherit from `graph_learning.models.base_lightning.BaseGraphLightning`.
* This ensures a consistent interface for training, evaluation, and explanation tools.
* **Unified Forward:** Use `model(data)` where `data` is a PyG `Data` or `Batch` object.
* **Explainer Compatibility:** Implement/use `forward_spread(x, edge_index, batch)` for compatibility with standard PyG explainers.
* **Task Type:** Explicitly set `task_type` to either `"classification"` or `"regression"`.

---

**Summary:**
Write simple, explicit, modular code with no configs, no globals, no orchestration, and file-based outputs rooted in a
single directory. Configuration, pipelines, and parallelization belong outside the core logic.

