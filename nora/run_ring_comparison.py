from __future__ import annotations

from pathlib import Path
from typing import Any

import fire

from nora.datasets import CombinedReactionDataset, get_dataset, print_dataset_stats
from nora.nora import (
    Model,
    evaluate_model,
    run_finetune_experiment,
    run_scratch_experiment,
    write_json,
)


def _evaluate_splits(
        model: Model,
        train_dataset: Any,
        test_dataset: Any,
        batch_size: int,
        seed: int,
) -> dict[str, dict[str, float]]:
    return {
        "train": evaluate_model(model, train_dataset, batch_size=batch_size, mask_seed=seed),
        "test": evaluate_model(model, test_dataset, batch_size=batch_size, mask_seed=seed),
    }


def _mapping_accuracy_table(
        evaluations: dict[str, dict[str, dict[str, float]]]
) -> dict[str, dict[str, float]]:
    return {
        split: {
            model_name: float(metrics["mapping_atom_accuracy"])
            for model_name, split_metrics in evaluations.items()
            for current_split, metrics in split_metrics.items()
            if current_split == split
        }
        for split in ("train", "test")
    }


def main(
        batch_size: int = 16,
        seed: int = 42,
        scratch_epochs: int = 50,
        finetune_epochs: int = 10,
        masking_rate: float = 0.15,
        learning_rate: float = 1e-4,
        finetune_learning_rate: float = 1e-5,
        dropout: float = 0.1,
        accumulate_grad_batches: int = 1,
        num_workers: int = 4,
        use_combined_train: bool = True,
        use_aim: bool = True,
        aim_experiment: str = "ring_comparison",
        output_dir: str = "experiment_results/ring_comparison",
) -> dict[str, Any]:
    """Run all ring-reactions comparison scenarios.

    Args:
        accumulate_grad_batches: Accumulate gradients over N batches (effective batch = batch_size * N).
            Use to simulate larger batch sizes without extra VRAM.
        num_workers: DataLoader worker processes. Set 0 to disable.
        use_combined_train: If True, combine ring train + Schneider50k train as finetuning
            source (recommended). If False, use ring train only.
    """
    ring_train = get_dataset("ringreactions", split="train")
    ring_test = get_dataset("ringreactions", split="test")

    print_dataset_stats(ring_train)
    print_dataset_stats(ring_test)

    if len(ring_train.packed) == 0 or len(ring_test.packed) == 0:
        raise RuntimeError("No valid reactions available after parsing train/test datasets.")

    if use_combined_train:
        schneider_train = get_dataset("schneider50k", split="train")
        print_dataset_stats(schneider_train)
        finetune_train = CombinedReactionDataset(ring_train, schneider_train, name="ring+schneider-train")
        print(f"Combined train dataset: {finetune_train}")
    else:
        finetune_train = ring_train

    finetune_mlm = run_finetune_experiment(
        train_dataset=finetune_train,
        test_dataset=ring_test,
        batch_size=batch_size,
        max_epochs=finetune_epochs,
        seed=seed,
        masking_rate=masking_rate,
        finetune_learning_rate=finetune_learning_rate,
        run_name="ring_finetune_mlm",
        use_aim=use_aim,
        aim_experiment=aim_experiment,
        accumulate_grad_batches=accumulate_grad_batches,
        num_workers=num_workers,
    )
    scratch_mlm = run_scratch_experiment(
        train_dataset=ring_train,
        test_dataset=ring_test,
        batch_size=batch_size,
        max_epochs=scratch_epochs,
        seed=seed + 2,
        masking_rate=masking_rate,
        learning_rate=learning_rate,
        dropout=dropout,
        run_name="ring_scratch_mlm",
        use_aim=use_aim,
        aim_experiment=aim_experiment,
        accumulate_grad_batches=accumulate_grad_batches,
        num_workers=num_workers,
    )

    checkpoints = {
        "finetune_mlm": str(finetune_mlm.get("last_checkpoint", "")),
        "scratch_mlm": str(scratch_mlm.get("last_checkpoint", "")),
    }
    missing = [name for name, ckpt in checkpoints.items() if not ckpt or not Path(ckpt).exists()]
    if missing:
        raise RuntimeError(f"Missing checkpoint files for: {', '.join(missing)}")

    models = {
        "pretrained": Model.pretrained(),
        "finetune_mlm": Model.load_from_checkpoint(checkpoints["finetune_mlm"]),
        "scratch_mlm": Model.load_from_checkpoint(checkpoints["scratch_mlm"]),
    }
    for model in models.values():
        model.eval()

    evaluations = {
        name: _evaluate_splits(model, ring_train, ring_test, batch_size=batch_size, seed=seed)
        for name, model in models.items()
    }
    accuracies = _mapping_accuracy_table(evaluations)

    results = {
        "dataset": "ringreactions",
        "use_combined_train": use_combined_train,
        "batch_size": batch_size,
        "seed": seed,
        "scratch_epochs": scratch_epochs,
        "finetune_epochs": finetune_epochs,
        "masking_rate": masking_rate,
        "learning_rate": learning_rate,
        "finetune_learning_rate": finetune_learning_rate,
        "dropout": dropout,
        "use_aim": use_aim,
        "checkpoints": checkpoints,
        "evaluations": evaluations,
        "mapping_atom_accuracy": accuracies,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results["results_json"] = write_json(out_dir / "comparison_results.json", results)

    lines = []
    for split in ("train", "test"):
        for model_name in ("pretrained", "finetune_mlm", "scratch_mlm"):
            lines.append(f"{split.upper()} {model_name} {accuracies[split][model_name]:.6f}")
    accuracies_txt = out_dir / "mapping_atom_accuracy.txt"
    accuracies_txt.write_text("\n".join(lines) + "\n")
    print(f"Wrote artifacts to: {accuracies_txt}")
    results["accuracy_table_txt"] = str(accuracies_txt)

    return results


if __name__ == "__main__":
    fire.Fire(main)
