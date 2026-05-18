# Change Report

## Scope reviewed

Reviewed the combined diff in `changes` and corresponding current files:

- `chytorch/__init__.py`
- `chytorch/zoo/__init__.py`
- `chytorch/zoo/rxnmap/__init__.py`
- `datasets.py`
- `nora.py`
- `nora_eval.py`
- `nora_optuna.py`
- `run_ring_comparison.py`
- `sanity.py`

## Executive summary

This diff is mostly an **experiment/training/evaluation extension** around the existing RXNMap model, with small core model packaging/loading updates.  

The most important functional change is that new mapping evaluation/training utilities in `nora.py` use an **embedding-similarity + greedy assignment** formulation, while the original package mapping flow is based on the model’s `mapping_task=True` attention outputs.

## Code + functional changes (combined view)

| Area | Code change | Functional impact | Status |
|---|---|---|---|
| Namespace packaging | Added namespace package shims in `chytorch/__init__.py` and `chytorch/zoo/__init__.py` | Allows this repo to overlay/extend installed `chytorch` namespace packages | Expected/OK |
| Pretrained loading | `pkg_resources.resource_stream(...)` replaced with `importlib.resources.files(...).joinpath(...)` in `chytorch/zoo/rxnmap/__init__.py:100-104` | Still loads pretrained checkpoint by package resource path; modernized API | Behavior mostly same; potential packaging edge risk |
| Scheduler location | `WarmUpCosine` moved inline into `chytorch/zoo/rxnmap/__init__.py:36-68` | Removes external dependency on missing local `...optim.lr_scheduler` module in this repo layout | Expected/OK |
| Core model forward API | `Model.forward(..., mapping_task=False)` retained in `chytorch/zoo/rxnmap/__init__.py:119-122` | Original mapping-task entry point still exists | Same as original |
| Dataset layer | New `datasets.py` with ring + Schneider loaders and split logic | Adds parsing, split selection, stats, and dataset combining support for experiments | New functionality |
| Training/eval stack | New `nora.py`, `nora_eval.py`, `nora_optuna.py`, `run_ring_comparison.py`, `sanity.py` | Adds CLI-based training/eval/optuna workflows and output artifacts | New functionality |
| Mapping metric implementation | `nora.py:290-417` computes mapping metrics from embeddings (`model(batch)`) + cosine + greedy assignment (`_greedy_assignment`) | Mapping metrics now represent a different evaluation protocol than original attention-based mapping API | **Functional change (important)** |
| Optional supervised loss | `SupervisedMappingModel` in `nora.py:46-177` adds mapping loss term over aligned reactant/product embeddings | Changes training objective when enabled (`use_supervised_loss=True`) | New optional behavior |
| Trainer callbacks | `run_training_experiment` uses only progress callback (`nora.py:484`) and manual final checkpoint save (`nora.py:501`) | No periodic checkpoint callback / LR monitor in these scripts | Functional change in logging/checkpoint behavior |

## Confirmed issues / broken behavior

1. **`data_root` is ignored by implementation**
   - `datasets.py:149` explicitly does `del root`.
   - Multiple public CLIs still expose/pass `data_root`/`root` (e.g. `nora.py`, `nora_eval.py`, `nora_optuna.py`).
   - **Impact:** user-provided dataset root has no effect; this is a real interface/behavior mismatch.

2. **`nora_eval.py` usage docs are currently wrong**
   - Docstring says `dataset` is a CSV path and provides examples like `--dataset=test.csv` (`nora_eval.py:106,115,118,121`).
   - Actual code expects dataset name (`ringreactions`/`uspto50k`) and routes CSV via `--csv_path`.
   - **Impact:** documented commands can fail with unknown dataset errors.

## High-impact functional changes to be aware of

1. **Mapping quality numbers are not directly comparable to original package mapping behavior**
   - Original mapping call path remains `model(batch, mapping_task=True)` (`chytorch/zoo/rxnmap/__init__.py:119-122`, README usage).
   - New evaluation in `nora.py` does **not** use that path; it uses embedding similarity + atom-type mask + greedy assignment (`nora.py:290-417`).
   - **Impact:** “mapping accuracy/exact/top-k” from new scripts are a different metric family and can diverge from original AAM behavior.

2. **Training objective can differ substantially**
   - With `use_supervised_loss=True`, total loss becomes `mapping_loss + mlm_weight * mlm_loss` (`nora.py:169-174`).
   - **Impact:** model optimization target and resulting behavior differ from original MLM-only training.

3. **Checkpoint/logging semantics changed in experiment scripts**
   - Periodic `ModelCheckpoint` and `LearningRateMonitor` callback behavior from model config is not used in these script-level trainer runs.
   - **Impact:** fewer automatic checkpoints/metrics unless manually added; reproducibility/debugging UX changes.

## Potential risk (not confirmed broken in normal local installs)

1. **Pretrained resource loading in non-standard packaging**
   - `importlib.resources.files(...).joinpath("weights.pt")` converted to `str(...)` path before `load_from_checkpoint` (`chytorch/zoo/rxnmap/__init__.py:101-104`).
   - Usually fine for standard wheel installs; may be fragile in some zipped/importer contexts.

2. **Dataset parse error handling is permissive**
   - Parser catches broad exceptions and skips invalid reactions with warnings (`datasets.py:89-93`).
   - This is useful operationally, but can hide upstream data quality regressions if warning output is not monitored.

## What appears unchanged in core behavior

- Core `Model.training_step` in `chytorch/zoo/rxnmap/__init__.py` remains MLM-based.
- `mapping_task=True` forward path remains present for original attention-style mapping outputs.
- Public pretrained entry point `Model.pretrained()` remains available.

## Bottom line

The diff is a substantial **research workflow expansion** rather than a rewrite of the core model package.  
The main things to treat as functionally changed/broken are:

1. `data_root` argument behavior mismatch (ignored in code).
2. `nora_eval.py` CLI docs/examples mismatch.
3. New mapping metrics/training options are **not equivalent** to original mapping-task behavior and should be interpreted as a separate evaluation/training regime.
