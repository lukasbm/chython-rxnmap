# LocalMapper comparison experiment design

## Changes from `paper.md`

The original LocalMapper paper used a human-in-the-loop workflow: the model selected reactions for review, chemists corrected the atom-to-atom mapping (AAM), and the corrected mappings were added to the next training round. This repository keeps the active-learning structure but replaces the manual review step with dataset ground truth. For each sampled reaction, `scripts.Sample` stores the dataset mapping as the emulated correction and records the model prediction that would have been shown to a chemist in `model_mapped_rxn` and `model_template`. The annotation rows are marked with `annotation_source=dataset_ground_truth`.

The evaluation path was changed back toward the paper for exact AAM comparison. The paper evaluates whether predicted and reference mappings induce the same condensed graph of reaction (CGR). The current implementation therefore uses CGRTools as the only exact-equivalence backend: `mapping_comparison_backend()` reports `cgrtools`, `cgr_signature()` computes a CGRTools signature, and `mappings_are_equivalent()` compares the predicted and reference CGR signatures. Raw string equality is still saved separately as `raw_exact_match_accuracy`, but it is not used as the paper-aligned exact AAM metric.

## CGRTools stabilization

CGRTools was failing in practice because individual reaction parsing or CGR composition can hang or take much longer than typical reactions. A simple direct call made the full evaluation vulnerable to one difficult reaction. The implementation now computes CGR signatures through a small isolated worker process with a per-reaction timeout controlled by `LOCALMAPPER_CGR_TIMEOUT_SECONDS`.

The multiprocessing context defaults to `fork` through `LOCALMAPPER_CGR_MP_CONTEXT=fork`. This matters because `spawn` added enough startup overhead that the first CGR request often exceeded the short timeout even for valid reactions. With `fork`, the worker starts quickly enough for a 2 second timeout on normal rows, while slow rows return `None` and are counted as non-equivalent instead of blocking the run. Parser errors are ignored by default through `LOCALMAPPER_CGRTOOLS_IGNORE=1`, matching the practical need to continue evaluation across noisy datasets.

## Metrics and metadata

Per-iteration evaluation metrics are saved in `predictions/metrics_<split>_<iteration>.json`. The run-level summaries `metrics_summary.json` and `metrics_summary.csv` flatten those metrics across iterations and are the main files needed for downstream tables and plots. The exact AAM metric to report is `aam.equiv_exact_match_accuracy`, with `aam.equivalence_backend=cgrtools`.

Each run now also writes `run_metadata.json` in the run directory. This records dataset, model name, seed, split, initialization mode, active-learning budget, training parameters, pretrained checkpoint, split fractions, CGRTools settings, software versions, and git state. This file is intended for provenance, while `metrics_summary.csv` remains the primary analysis table.

## Experimental design

The comparison should vary the factors that define the LocalMapper use case:

- Dataset: at minimum `USPTO_50K`, `Golden`, `ringreactions`, and `metAMDB`; additional datasets such as `NatComm` and `schneider` can be added when runtime permits.
- Initialization: train from scratch and fine-tune from the released LocalMapper checkpoint.
- Annotation budget: compare a low-budget setting against the paper-style budget. The default sweep uses `50 x 3` and `200 x 5` annotations for most datasets, and `5 x 5` and `10 x 10` for `ringreactions`.
- Seed: use seed 0 for pilot experiments; add more seeds only after the pipeline is stable and the expected runtime is known.

The provided launcher is `scripts/experiments/comparison_sweep.sh`. It runs all selected dataset, mode, budget, and seed combinations through the same `Sample -> Train -> Test` loop and stores each run under `outputs/<dataset>/<model>_seed<seed>/`.

The argument for running the full default matrix is that each axis answers a different comparison question:

- Multiple datasets test whether the method generalizes beyond a single chemistry domain and whether the CGRTools evaluation remains stable on both standard and difficult datasets.
- Scratch versus pretrained initialization tests whether gains come from the active-learning procedure itself or from starting from the released checkpoint.
- Low versus standard annotation budgets test sample efficiency, which is the central claim of the LocalMapper active-learning setup.

With the default settings and `SEEDS=0`, the launcher performs 16 top-level runs:

- `USPTO_50K`, `Golden`, and `metAMDB`: 2 modes x 2 budgets = 4 runs each.
- `ringreactions`: 2 modes x 2 budgets = 4 runs.

Because each run contains several active-learning iterations, the full default matrix expands to 78 `Sample -> Train -> Test` iterations in total:

- `USPTO_50K`, `Golden`, and `metAMDB`: `(3 + 5)` iterations x 2 modes x 3 datasets = 48 iterations.
- `ringreactions`: `(5 + 10)` iterations x 2 modes = 30 iterations.

This is a reasonable comparison matrix, but it is already expensive. A common workflow is:

- Pilot: `PLAN_ONLY=1` first, then `COMPARISON_NUM_EPOCHS=10` with the full matrix or a subset.
- Final comparison: keep the same matrix, restore the main epoch budget, and add more seeds only after runtime is understood.

Epochs should not be a primary sweep factor for the comparison experiment. They affect optimization quality and runtime, but they do not directly test the active-learning method. Fixing `NUM_EPOCHS` keeps the comparison interpretable. For pilot runs, use a smaller value such as `COMPARISON_NUM_EPOCHS=10`; for final runs, use the paper-like default of 100 unless validation shows that early stopping consistently ends earlier.

K-fold cross-validation is also not recommended as a default factor. The experiment already has a repeated active-learning procedure over several datasets, two initialization modes, and multiple annotation budgets. K-folds would multiply cost substantially and are less aligned with the paper, which uses defined dataset splits and out-of-distribution datasets. Prefer fixed splits plus a small number of random seeds if uncertainty estimates are needed.

## Suggested figures

- Learning curves: exact CGR-equivalent AAM accuracy versus cumulative annotation budget, faceted by dataset and initialization mode.
- Budget efficiency: final exact AAM accuracy at each budget, with paired bars for scratch and pretrained runs.
- Calibration plots: confidence or mapping-score correctness curves using `score_calibration` metrics.
- Coverage versus accuracy: confident-template coverage against exact AAM accuracy, especially for the pretrained comparison.
- Runtime or failure diagnostics: CGR invalid/timeout counts and evaluation wall time per dataset, useful because metAMDB has a visible long tail.
