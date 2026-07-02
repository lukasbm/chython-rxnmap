# AAM comparison playbook

This document describes the comparison strategy used in this repository for
LocalMapper active-learning experiments and the constraints another agent should
preserve when comparing a different atom-to-atom mapping (AAM) system.

It is written as a practical handoff document, not as a paper summary.

## Goal

The goal is to compare AAM systems under a controlled active-learning setup
while keeping the evaluation aligned with chemistry-aware mapping equivalence.

There are two distinct comparison targets:

1. Compare LocalMapper configurations against each other.
2. Compare LocalMapper against another AAM system or equivalence backend.

Those are not the same experiment. The first changes training and annotation
budget. The second should change only the mapper or equivalence method while
holding everything else fixed.

## Core principle

Only one major factor should change at a time.

- If the question is "does pretrained initialization help?", keep dataset,
  seed, budget, splits, and evaluation backend fixed.
- If the question is "does a different AAM system help?", keep dataset, seed,
  sampled reactions, splits, and metric definitions fixed.
- If the question is "does another equivalence backend help?", keep the mapped
  reactions fixed and re-score them under the alternate backend.

If multiple axes change at once, the comparison stops being interpretable.

## Current LocalMapper comparison design

The default sweep is implemented in
[scripts/experiments/comparison_sweep.sh](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/experiments/comparison_sweep.sh:1).

The experiment varies these factors:

- Dataset: `USPTO_50K`, `Golden`, `ringreactions`, `metAMDB`
- Initialization mode: `scratch`, `pretrained`
- Annotation budget:
  - most datasets: `low = 50 x 3`, `standard = 200 x 5`
  - `ringreactions`: `low = 5 x 5`, `standard = 10 x 10`
- Seed: currently `0` in the default comparison run

The active-learning loop is always:

1. `Sample`
2. `Train`
3. `Test`

That loop is orchestrated in
[scripts/experiments/run_active_learning.sh](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/experiments/run_active_learning.sh:43).

## What must remain fixed for a fair comparison

Another agent should treat the following as controlled variables unless the
comparison explicitly targets them:

- Dataset and split fractions
- Seed
- Sampled reactions per iteration
- Sample candidate factor
- Number of active-learning iterations
- Epoch budget and patience
- Training batch size
- Confidence selection logic
- Exact metric definitions
- Equivalence backend and its timeout / parser settings

In this repository, the run metadata already records these controls in
`run_metadata.json`, including:

- active-learning budget and total annotation budget
- initialization mode
- pretrained checkpoint
- split fractions
- equivalence backend
- CGRTools timeout and parser handling
- software versions
- git commit and dirty state

See [scripts/RunMetadata.py](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/RunMetadata.py:80).

## Annotation model used here

This repository does not use human corrections during comparison runs.

Instead, the active-learning workflow emulates annotation with dataset ground
truth:

- selected reactions are still chosen by the model
- the "correction" written back to training data is the dataset mapping
- the annotation source is recorded as `dataset_ground_truth`

This matters because any new AAM system must be compared under the same
annotation regime. Do not mix human-corrected rounds with dataset-ground-truth
rounds in the same comparison table.

## Exact AAM metric

The primary metric is:

- `aam.equiv_exact_match_accuracy`

This is the paper-aligned exact mapping metric. It does not use raw mapped-SMILES
string equality. Instead it checks whether the predicted and reference mappings
are chemically equivalent under the configured backend.

In the current implementation, the intended backend is `cgrtools`.

Secondary AAM metrics are:

- `aam.atom_accuracy`
- `aam.raw_exact_match_accuracy`
- `aam.invalid_mapping_count`
- `aam.total`

Interpretation:

- `equiv_exact_match_accuracy` is the main reported metric.
- `atom_accuracy` is useful when exact equivalence is too strict to show partial
  progress.
- `raw_exact_match_accuracy` is diagnostic only. It will underestimate quality
  whenever two mappings are equivalent but serialized differently.
- `invalid_mapping_count` is a stability signal. A method that wins only by
  failing on fewer rows should be understood differently from one that improves
  actual mapping quality.

Metric construction lives in
[scripts/Test.py](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/Test.py:117).

## Confidence and calibration metrics

The comparison does not only measure final mapping correctness. It also checks
whether the system's confidence signal is useful.

### Score calibration metrics

These use the scalar mapping score from the model output:

- `score_calibration.ap`
- `score_calibration.f1`
- `score_calibration.accuracy`
- `score_calibration.precision`
- `score_calibration.recall`
- `score_calibration.mcc`
- `score_calibration.threshold`
- `score_calibration.confusion.*`

These are computed at the threshold that maximizes F1 on the evaluated rows.
That threshold source is recorded as `best_f1_on_evaluation`.

There is also a fixed-threshold diagnostic:

- `score_uncalibrated.*`

This uses threshold `0.5` and should be treated as a comparability check, not
as the main score metric.

Definitions are in
[scripts/Test.py](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/Test.py:245)
and
[scripts/Test.py](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/Test.py:269).

### Template-confidence metrics

These use the binary `confident` prediction flag:

- `template_confidence.coverage`
- `template_confidence.confident_accuracy`
- `template_confidence.unconfident_accuracy`
- `template_confidence.f1`
- `template_confidence.mcc`
- `template_confidence.precision`
- `template_confidence.recall`
- `template_confidence.confusion.*`

These metrics matter because the active-learning loop depends on confidence
signals, not only final mapped reactions.

If another AAM system does not produce an equivalent confidence output, the
comparison must state that clearly. Do not silently compare a system with a
confidence signal against one without it and then draw conclusions about sample
selection quality.

## Equivalence backend requirements

The repository currently uses CGRTools-style equivalence for the primary exact
metric.

Critical settings:

- `LOCALMAPPER_CGRTOOLS_IGNORE=1`
- `LOCALMAPPER_CGR_MP_CONTEXT=fork`
- `LOCALMAPPER_CGR_TIMEOUT_SECONDS=2`

These are exported in
[scripts/experiments/run_active_learning.sh](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/experiments/run_active_learning.sh:27).

Why this matters:

- parser behavior changes whether noisy reactions are counted or dropped
- timeout behavior changes whether difficult reactions become failures or hangs
- multiprocessing mode changes practical evaluation stability

If another agent swaps in another equivalence backend, it must report:

- backend name
- parser policy
- timeout policy
- failure behavior
- whether results are directly comparable to CGRTools-based runs

Do not compare results across backends without making that explicit.

## Recommended comparison modes

### Mode 1: compare LocalMapper variants

Use the existing sweep and keep the equivalence backend fixed.

Questions answered:

- scratch vs pretrained
- low vs standard budget
- easier vs harder datasets

### Mode 2: compare another mapper under the same evaluation

Use the same datasets, same splits, same seed, and same exact-equivalence
metric. The new system should produce mapped reactions on the same evaluation
rows, then be scored through the same metric code or a byte-for-byte equivalent
reimplementation.

This is the cleanest way to compare AAM systems.

### Mode 3: compare another equivalence backend

Keep mapped reactions fixed and recompute `aam.equiv_exact_match_accuracy`
under the new backend. This isolates the effect of the evaluator itself.

Do not mix this with retraining unless the actual question is about the whole
system, not just equivalence checking.

## Minimal protocol for comparing another AAM system

Another agent should follow this protocol.

1. Choose the comparison target.
   `mapper`, `training mode`, or `equivalence backend`
2. Freeze all other axes.
3. Record the run metadata needed for reproducibility.
4. Score predictions with the same metric definitions.
5. Report the primary metric first:
   `aam.equiv_exact_match_accuracy`
6. Report secondary context:
   `aam.atom_accuracy`, `invalid_mapping_count`, calibration metrics, template-confidence metrics
7. Plot learning curves against cumulative annotation budget, not only final bars.
8. State explicitly if confidence outputs are not comparable across systems.

## Files another agent should use

- Experiment launcher:
  [scripts/experiments/comparison_sweep.sh](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/experiments/comparison_sweep.sh:1)
- Per-run loop:
  [scripts/experiments/run_active_learning.sh](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/experiments/run_active_learning.sh:1)
- Metric computation:
  [scripts/Test.py](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/Test.py:60)
- Metadata writing:
  [scripts/RunMetadata.py](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/RunMetadata.py:80)
- Aggregation / plotting helper:
  [outputs/analyze_metrics_summaries.py](/homes/biertank/lukas/Documents/repos/LocalMapper/outputs/analyze_metrics_summaries.py:1)

## Output artifacts to expect

For each run:

- `predictions/metrics_<split>_<iteration>.json`
- `metrics_summary.json`
- `metrics_summary.csv`
- `run_metadata.json`

For cross-run analysis:

- `<run_id>_metrics_all_iterations.csv`
- `<run_id>_metrics_final_iterations.csv`
- `<run_id>_metrics_deltas.csv`
- dashboard and trajectory PNGs

## Common failure modes

- Comparing runs with different equivalence backends as if they were the same
- Comparing systems with different sampled reactions
- Reporting raw exact match as the main AAM metric
- Ignoring invalid mapping counts
- Comparing confidence-driven metrics when one system lacks a comparable score
- Changing epoch budget and active-learning budget at the same time
- Mixing pilot runs and final runs in one table
- Forgetting that `ringreactions` uses a different budget grid than other datasets

## Reporting order

For papers, notes, or agent handoff summaries, report in this order:

1. `aam.equiv_exact_match_accuracy`
2. `aam.atom_accuracy`
3. `aam.invalid_mapping_count`
4. `score_calibration.ap` and `score_calibration.f1`
5. `template_confidence.coverage` and `template_confidence.confident_accuracy`
6. provenance:
   backend, timeout settings, run id, dataset, seed, budget, init mode

That ordering keeps the chemistry-valid exact metric primary while still
showing whether the model's confidence signals are useful and whether the run
was stable.
