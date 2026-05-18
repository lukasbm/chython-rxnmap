from __future__ import annotations

import fire
from pathlib import Path
from typing import Optional

from nora.datasets import get_dataset, print_dataset_stats, CombinedReactionDataset
from nora.nora import (
    run_scratch_experiment,
    run_finetune_experiment,
    write_json,
)


def main(
    dataset: str = "ringreactions",
    train_split: str = "train",
    test_split: str = "test",
    train_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    data_root: Optional[str] = None,
    mode: str = "finetune",  # 'finetune' or 'scratch'
    batch_size: int = 16,
    max_epochs: int = 10,
    seed: int = 42,
    masking_rate: float = 0.15,
    learning_rate: float = 1e-4,
    finetune_learning_rate: float = 1e-5,
    dropout: float = 0.1,
    accumulate_grad_batches: int = 1,
    num_workers: int = 4,
    use_aim: bool = False,
    aim_experiment: Optional[str] = None,
    output_dir: str = "experiment_results/nora_run",
    combined_train: Optional[str] = None,
) -> dict:
    """Run a single experiment (scratch or finetune) on a dataset.

    dataset: either a registered dataset name (e.g. 'ringreactions', 'schneider50k', 'uspto50k')
             or a dataset key that nora.datasets knows how to load.
    train_csv / test_csv: optional — if provided, passed as csv_path to get_dataset for flexible CSV-based inputs.
    combined_train: optional name of another dataset to combine with the main train set (used for finetuning).

    Returns the metrics dict and writes a JSON summary to output_dir/nora_run_results.json
    """

    # Resolve datasets
    if train_csv or test_csv:
        train_ds = get_dataset(dataset, split="train", csv_path=train_csv, root=data_root)
        test_ds = get_dataset(dataset, split="test", csv_path=test_csv, root=data_root)
    else:
        train_ds = get_dataset(dataset, split=train_split, root=data_root)
        test_ds = get_dataset(dataset, split=test_split, root=data_root)

    print_dataset_stats(train_ds)
    print_dataset_stats(test_ds)

    if len(train_ds.packed) == 0 or len(test_ds.packed) == 0:
        raise RuntimeError("No valid reactions available after parsing train/test datasets.")

    # Allow combining another dataset for finetuning
    train_input = train_ds
    if combined_train:
        other = get_dataset(combined_train, split="train", root=data_root)
        train_input = CombinedReactionDataset(train_ds, other, name=f"{dataset}+{combined_train}-train")
        print(f"Using combined train dataset: {train_input}")

    if mode == "scratch":
        results = run_scratch_experiment(
            train_input,
            test_ds,
            batch_size=batch_size,
            max_epochs=max_epochs,
            seed=seed,
            masking_rate=masking_rate,
            learning_rate=learning_rate,
            dropout=dropout,
            run_name=f"{dataset}_scratch",
            use_aim=use_aim,
            aim_experiment=aim_experiment,
            accumulate_grad_batches=accumulate_grad_batches,
            num_workers=num_workers,
        )
    elif mode == "finetune":
        results = run_finetune_experiment(
            train_input,
            test_ds,
            batch_size=batch_size,
            max_epochs=max_epochs,
            seed=seed,
            masking_rate=masking_rate,
            finetune_learning_rate=finetune_learning_rate,
            run_name=f"{dataset}_finetune",
            use_aim=use_aim,
            aim_experiment=aim_experiment,
            accumulate_grad_batches=accumulate_grad_batches,
            num_workers=num_workers,
        )
    else:
        raise ValueError("mode must be 'scratch' or 'finetune'")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(write_json(out_dir / "nora_run_results.json", results))
    print(f"Wrote summary to: {results_path}")
    return results


if __name__ == "__main__":
    fire.Fire(main)
