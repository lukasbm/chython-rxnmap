# Nora AAM evaluation

This folder contains the standalone evaluation harness for this repository's
GraphormerMapper AAM experiments.

## Data layout

Primary datasets are expected in the copied LocalMapper-style root `data/data/`:

- `Golden/raw_data.csv`
- `Golden/test_data.csv`
- `metAMDB/train_metamdb_filtered.csv`
- `metAMDB/test_metamdb_filtered.csv`

Local sanity datasets already follow the same convention:

- `schneider/schneider50k.tsv`
- `ringreactions/train_ringreactions.csv`
- `ringreactions/test_ringreactions.csv`

## Training

Scratch training:

```bash
uv run python -m nora.train --dataset golden --mode scratch --max_epochs 1
```

Fine-tuning from the packaged pretrained checkpoint:

```bash
uv run python -m nora.train --dataset golden --mode fine_tuned --max_epochs 1
```

Both modes save a final checkpoint under `experiment_results/checkpoints/` by
default and write a JSON summary with dataset stats and checkpoint path.

## Evaluation

Evaluate the configured chython mapper:

```bash
uv run python -m nora.evaluate --dataset golden --split test
```

Run the default Golden + metAMDB evaluation wrapper:

```bash
scripts/evaluate_all.sh
```

Useful wrapper overrides:

```bash
EVAL_LIMIT=10 scripts/evaluate_all.sh
RUN_TRAINING=1 MAX_EPOCHS=1 EVAL_LIMIT=10 scripts/evaluate_all.sh
DATASETS=golden,metamdb,ringreactions TIMEOUT_SECONDS=30 scripts/evaluate_all.sh
```

Run a parser/metric smoke test without invoking a mapper:

```bash
uv run python -m nora.evaluate --dataset ringreactions --backend reference_copy --limit 5
```

Outputs are written under `experiment_results/evaluation/`:

- `predictions_<dataset>_<split>_<mode>_<backend>.jsonl`
- `metrics_<dataset>_<split>_<mode>_<backend>.json`
- `run_metadata_<dataset>_<split>_<mode>_<backend>.json`
- `metrics_summary.csv`

## Current limitation

The default `chython_reset_mapping` backend uses `Reaction.reset_mapping()`.
This repository does not expose a hook that makes that method consume an
arbitrary local checkpoint. Training checkpoints are saved for scratch and
fine-tuned runs, but checkpoint-backed prediction needs a mapper adapter before
those checkpoints can be scored through `nora.evaluate`.
