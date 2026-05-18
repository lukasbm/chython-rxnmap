nora package — usage and notes

Overview

- nora/ contains an experiment harness to train/evaluate the RXNMap model (MLM + mapping). The supervised mapping loss was removed: training now uses MLM-only objectives; mapping evaluation uses the model's original mapping path (model(batch, mapping_task=True)).

Quick examples

- Evaluate pretrained baseline (Python):
    python -m nora.nora --dataset=ringreactions --batch_size=64

- Run main training (scratch + finetune flows):
    python -m nora.nora --dataset=ringreactions --max_epochs=10 --batch_size=32

- Run the ring comparison suite (creates JSON & text outputs):
    python nora/run_ring_comparison.py

- Run optuna tuning:
    python nora/nora_optuna.py

Datasets

- Use nora.datasets.get_dataset(name, split, root=...) to load datasets.
- Provide `data_root` to point to your dataset directory when using CLI scripts.

Exporting results

- All experiment scripts write JSON artifacts (experiment_results/ or lightning_logs/*/test_metrics.json). Use those JSON files (or convert to CSV) to compare mappers in your paper.
- If you need per-reaction mapping assignments, modify evaluate_mapping_metrics to collect assignments and write them with write_json; this keeps comparisons reproducible.

Notes / gotchas

- Mapping evaluation calls the original mapping forward: model(batch, mapping_task=True). Confirm the model checkpoint you load is compatible with this API.
- No supervised mapping loss is used in nora/ by default — this prevents biasing mapping metrics.
- If you prefer a single exporter, run the evaluations and collect the returned dicts; they contain keys like `mapping_atom_accuracy`, `mapping_top1`, `mapping_exact_match` and `mlm_*` metrics.

Running training/finetuning on any dataset

- Use the helper script `nora/run.py` to perform a single experiment (scratch or finetune) on any dataset that nora.datasets can load. Examples:

    # Finetune pretrained model on ringreactions
    python -m nora.run --dataset=ringreactions --mode=finetune --max_epochs=10 --batch_size=32 --data_root=/path/to/data

    # Train from scratch on USPTO-50k
    python -m nora.run --dataset=uspto50k --mode=scratch --max_epochs=50 --batch_size=16 --data_root=/path/to/data

    # Use custom CSVs (pass csv paths to override dataset splits)
    python -m nora.run --dataset=ringreactions --train_csv=/path/to/train.csv --test_csv=/path/to/test.csv --mode=scratch

- Output: The script writes a JSON summary to `experiment_results/nora_run/nora_run_results.json` (or the folder set with `--output_dir`). It also produces Lightning logs and checkpoints under `lightning_logs/`.

Notes:
- The script uses `nora.datasets.get_dataset(...)` to load data; ensure your CSV or dataset files conform to the dataset loader you use (the common format is the repository's packed chython reaction format or the CSV layouts already used by `ringreactions`/`schneider50k`).
- For per-reaction mapping exports (assignments), extend `evaluate_mapping_metrics` to collect assignments and write them out; this keeps your comparison canonical when comparing to external mappers.

Contact

- For questions about running experiments or to request an adapter scaffold for an external mapper, open an issue or message me with the mapper input/output format you can produce.
