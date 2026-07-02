# LocalMapper Mapping Pipeline Report

This report documents the current LocalMapper active-learning implementation, the mapping-equivalence fixes added for metAMDB-style data, the remaining training risks, and the recommended long-term path toward multi-target supervision.

## Executive Summary

The original pipeline used raw mapped reaction strings as both the training target and the evaluation target. This is brittle for datasets such as `metAMDB`, where the same atom-to-atom correspondence can be written with different atom-map numbers, SMILES atom order, or kekulized/canonical forms. A run on `scripts/experiments/metamdb_pretrained_200x5.sh` after equivalence-aware evaluation reached `EquivExact: 0.6973` while `RawExact: 0.0120`, demonstrating that raw string equality severely underestimates mapping quality.

The implemented short-term fix keeps training differentiable by normalizing each dataset row to a single canonical mapped reaction before labels are extracted. Raw alternatives are retained for equivalence-aware evaluation. This removes representation noise without changing the model or loss. It does not solve genuine multi-target ambiguity when one row contains multiple distinct valid atom correspondences.

## Current Data Flow

Datasets are defined in `localmapper/dataset.py` and loaded through `DATASETS`. Raw dataset-class items preserve the first source alternative in `rxn`; items returned by `load_reactions(...)` / `select_split(...)` normalize that field before training or evaluation. A selected item contains:

- `id`: stable dataset row identifier.
- `rxn`: canonicalized mapped reaction used for training labels, annotation emulation, and template extraction.
- `mapped_rxns`: original mapped alternatives from the source row, retained for evaluation.
- `split`: assigned by `select_split`.
- `original_split`: source split metadata where relevant.
- `num_mappings`: number of original alternatives in the source row.

All datasets are now treated as unsplit pools unless they are intrinsically unsplit already. For `metAMDB`, `train_metamdb_filtered.csv` and `test_metamdb_filtered.csv` are combined into one pool, and `train` / `val` / `test` are controlled by `seed`, `val_fraction`, and `test_fraction`. The same combined-split approach applies to `ringreactions`; `NatComm` is no longer hard-coded to test-only.

Active learning proceeds as follows:

1. `scripts.Sample` calls `load_reactions(...)` for the requested split.
2. The sampled rows are saved as emulated annotations using each item’s normalized `rxn`.
3. `scripts.Train` loads all annotations through `training_items_from_annotations(...)`.
4. For iteration `> 1`, `scripts.Train` also loads confident pseudo-labels from the previous prediction CSV.
5. Training uses per-product-atom cross entropy over model logits.
6. `scripts.Test` decodes model scores into mapped reactions and evaluates them with equivalence-aware correctness.

## Template Extraction and Pseudo-Labeling

Templates are extracted from mapped reactions through `localmapper.LocalTemplate.template_extractor.extract_from_reaction`. The extractor identifies changed mapped atoms and emits a local reaction SMARTS template. The template library is built from human/emulated annotations.

Pseudo-labeling currently uses the LocalMapper paper’s knowledge-based confidence heuristic:

- A prediction is considered confident if its predicted template appears in the verified template library.
- Up to `CONFIDENT_PER_TEMPLATE` predictions are sampled per template.
- The predicted `mapped_rxn` becomes a pseudo-labeled training reaction with the same weight as real annotations.

This was problematic for scratch metAMDB training because template membership did not imply correct mapping. Earlier runs showed pseudo-labels with roughly chance-level correctness swamping the real annotations. Pretrained metAMDB runs are much better, but pseudo-label quality should still be monitored with `EquivExact`, not raw exact match.

## Evaluation: Raw Exact vs EquivExact

The original evaluation computed:

```text
result["mapped_rxn"] == item["rxn"]
```

That fails whenever an equivalent mapping is written differently. The current evaluation compares mapping signatures instead:

- canonicalize reactant and product atom order,
- remove atom-map labels to compare the demapped reaction pattern,
- compare the product-atom to reactant-atom correspondence,
- accept a prediction if it matches any original alternative in `mapped_rxns`.

`scripts.Test` now writes both:

- `is_correct`: equivalence-aware correctness.
- `raw_is_correct`: raw-string or raw-alternative match.

It prints:

```text
EquivExact: equivalence-aware exact mapping accuracy
RawExact: raw string exact accuracy
```

The printed `Accuracy`, `F1`, `MCC`, and `AP` are not mapping accuracy. They measure how well the model’s internal `mapping_score` separates correct from incorrect decoded mappings after thresholding. In highly imbalanced settings these can be misleading. `EquivExact` is the primary mapping metric.

## Training Target Normalization

The implemented short-term training fix is in `localmapper/dataset.py`:

- Source alternatives are parsed and retained in `mapped_rxns`.
- During split selection, the first valid alternative is canonicalized by `canonicalize_map_rxn(...)`.
- The canonicalized mapped reaction becomes the selected item’s `rxn`.
- All original alternatives remain in `item["mapped_rxns"]`.

This means `get_mapping_label(item["rxn"])` now sees a stable representation. It reduces inconsistent labels from arbitrary SMILES ordering, atom-map numbering, and equivalent mapped forms while keeping the loss differentiable and unchanged.

This fix is intentionally conservative:

- It does not decode predictions during training.
- It does not introduce non-differentiable sequence-level loss.
- It does not change the model architecture.
- It does not attempt to choose among distinct valid correspondences beyond keeping the first canonical valid target.

## Remaining Mismatch: Training Objective vs Decoder

Training optimizes per-product-atom cross entropy on raw logits:

```text
for each product atom, predict the reactant atom index
```

Inference uses `localmapper.atom_mapper.AtomMapper.generate_atom_mapping(...)`, which performs greedy constrained decoding:

- softmax over logits,
- symbol masking,
- row/column normalization,
- sequential product/reactant assignment,
- neighbor reweighting after each assignment,
- masking of already mapped atoms.

Canonicalizing labels fixes representation noise, but it does not make the training objective identical to the inference decoder. This mismatch already existed in the original implementation and mirrors the paper’s high-level setup: learn `p(atom_r | atom_p)`, then decode a full AAM greedily. It is acceptable as a first-order objective, but errors can still arise when locally plausible atom choices cause globally inconsistent decoded mappings.

## Remaining Ambiguity: Distinct Valid Signatures

Some metAMDB rows contain many alternatives that do not collapse to a single mapping signature. Example sampled rows from `test_metamdb_filtered.csv` showed:

- `333` alternatives collapsing to `37` distinct signatures.
- `76` alternatives collapsing to `76` distinct signatures.
- `144` alternatives collapsing to `16` distinct signatures.

For these rows, a single canonical target still penalizes other valid atom correspondences. The current normalization solves representation-level false negatives but not genuine multi-solution ambiguity.

## Long-Term Remedy: Multi-Target Training

A principled long-term solution is to train against all distinct valid correspondence signatures for a row.

Recommended direction:

1. At dataset load time, canonicalize all alternatives and group them by distinct mapping signature.
2. Store per-row valid label vectors, one per distinct signature.
3. Replace single-target cross entropy with a multi-target objective.
4. Use a minimum-loss or marginal-likelihood formulation:
   - `min CE` over valid targets is simple and directly rewards any valid mapping.
   - `-logsumexp(-CE(valid_targets))` is smoother and uses all valid targets.
5. Keep the current single-target path for rows with exactly one distinct signature.

The implementation should avoid generating all equivalent SMILES forms. It should operate on canonical correspondence signatures and label vectors. This is cheaper, deterministic, and aligned with the evaluation logic.

Open design questions for multi-target training:

- Whether to cap the number of distinct targets per row for very large ambiguous rows.
- Whether to weight targets equally or by frequency among alternatives.
- Whether pseudo-labels should remain single-target predictions or be stored as one decoded signature.
- Whether validation loss should use the same multi-target objective or report both single-target and multi-target loss.

## Recommended Experiment Sequence

Use this order to isolate effects:

1. Re-run `metAMDB` pretrained with canonical single-target normalization enabled and existing equivalence evaluation.
2. Compare `EquivExact`, `RawExact`, and pseudo-label counts by iteration.
3. Run with `CONFIDENT_PER_TEMPLATE=0` to measure annotation-only behavior.
4. Run with smaller pseudo-label budgets such as `CONFIDENT_PER_TEMPLATE=5`, `10`, and `25`.
5. Use `scripts.DatasetStats` to quantify diversity and alternative counts.
6. Add a separate ambiguity-inspection utility or extend `DatasetStats` to report distinct mapping-signature counts across alternatives.
7. Only implement multi-target loss if a material fraction of training rows has multiple distinct signatures and performance indicates ambiguity is hurting training.

Suggested commands:

```bash
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl \
  uv run python -m scripts.DatasetStats --datasets=metamdb --top_n=10

CONFIDENT_PER_TEMPLATE=0 \
  scripts/experiments/metamdb_pretrained_200x5.sh

CONFIDENT_PER_TEMPLATE=10 MODEL=LocalMapper_pretrained_pseudo10 \
  scripts/experiments/metamdb_pretrained_200x5.sh
```

Use a fresh `MODEL` name after dataset semantics change. Old output directories contain annotations and predictions generated under old ID, split, and target-normalization behavior and should not be mixed with new runs.

## Known Gotchas

- Existing outputs under `outputs/metAMDB/...` are not comparable across loader changes.
- `test_fraction` and `val_fraction` now matter for datasets that previously had explicit split files.
- `RawExact` can remain low even when mapping quality is high.
- `Accuracy` can look high when exact mapping quality is poor due to class imbalance.
- Pseudo-labels are still trusted by template membership, not by an independently calibrated confidence score.
- Template extraction depends on the chosen normalized target and may differ from templates extracted from a different valid correspondence.
- The current training loss remains single-target and per-atom, not sequence-level and not decoder-aware.

## Current Status

Implemented:

- combined unsplit dataset loading for metAMDB/ringreactions/NatComm,
- seed/fraction-controlled train/val/test splits,
- equivalence-aware evaluation with `EquivExact` and `RawExact`,
- raw alternative retention via `mapped_rxns`,
- canonical single-target normalization before label extraction,
- `scripts.DatasetStats` for diversity and template inspection,
- experiment shell scripts using `uv run python` by default.

Not implemented:

- multi-target training loss,
- decoder-aware training,
- calibrated pseudo-label confidence,
- exhaustive ambiguity statistics across all alternatives,
- automatic invalidation or migration of old output directories.
