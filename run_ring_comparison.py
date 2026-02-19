from __future__ import annotations

from pathlib import Path
from typing import Any

import fire

from datasets import get_dataset, print_dataset_stats
from nora import (
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
        mlm_weight: float = 0.1,
        use_aim: bool = True,
        aim_experiment: str = "ring_comparison",
        output_dir: str = "experiment_results/ring_comparison",
) -> dict[str, Any]:
    train_dataset = get_dataset("ringreactions", split="train")
    test_dataset = get_dataset("ringreactions", split="test")

    print_dataset_stats(train_dataset)
    print_dataset_stats(test_dataset)

    if len(train_dataset.packed) == 0 or len(test_dataset.packed) == 0:
        raise RuntimeError("No valid reactions available after parsing train/test datasets.")

    finetune_mlm = run_finetune_experiment(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        max_epochs=finetune_epochs,
        seed=seed,
        masking_rate=masking_rate,
        finetune_learning_rate=finetune_learning_rate,
        run_name="ring_finetune_mlm",
        use_aim=use_aim,
        aim_experiment=aim_experiment,
        use_supervised_loss=False,
        mlm_weight=mlm_weight,
    )
    finetune_supervised = run_finetune_experiment(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        max_epochs=finetune_epochs,
        seed=seed + 1,
        masking_rate=masking_rate,
        finetune_learning_rate=finetune_learning_rate,
        run_name="ring_finetune_supervised",
        use_aim=use_aim,
        aim_experiment=aim_experiment,
        use_supervised_loss=True,
        mlm_weight=mlm_weight,
    )
    scratch_mlm = run_scratch_experiment(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        max_epochs=scratch_epochs,
        seed=seed + 2,
        masking_rate=masking_rate,
        learning_rate=learning_rate,
        dropout=dropout,
        run_name="ring_scratch_mlm",
        use_aim=use_aim,
        aim_experiment=aim_experiment,
        use_supervised_loss=False,
        mlm_weight=mlm_weight,
    )
    scratch_supervised = run_scratch_experiment(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        max_epochs=scratch_epochs,
        seed=seed + 3,
        masking_rate=masking_rate,
        learning_rate=learning_rate,
        dropout=dropout,
        run_name="ring_scratch_supervised",
        use_aim=use_aim,
        aim_experiment=aim_experiment,
        use_supervised_loss=True,
        mlm_weight=mlm_weight,
    )

    checkpoints = {
        "finetune_mlm": str(finetune_mlm.get("last_checkpoint", "")),
        "finetune_supervised": str(finetune_supervised.get("last_checkpoint", "")),
        "scratch_mlm": str(scratch_mlm.get("last_checkpoint", "")),
        "scratch_supervised": str(scratch_supervised.get("last_checkpoint", "")),
    }
    missing = [name for name, ckpt in checkpoints.items() if not ckpt or not Path(ckpt).exists()]
    if missing:
        raise RuntimeError(f"Missing checkpoint files for: {', '.join(missing)}")

    models = {
        "pretrained": Model.pretrained(),
        "finetune_mlm": Model.load_from_checkpoint(checkpoints["finetune_mlm"]),
        "finetune_supervised": Model.load_from_checkpoint(checkpoints["finetune_supervised"]),
        "scratch_mlm": Model.load_from_checkpoint(checkpoints["scratch_mlm"]),
        "scratch_supervised": Model.load_from_checkpoint(checkpoints["scratch_supervised"]),
    }
    for model in models.values():
        model.eval()

    evaluations = {
        name: _evaluate_splits(model, train_dataset, test_dataset, batch_size=batch_size, seed=seed)
        for name, model in models.items()
    }
    accuracies = _mapping_accuracy_table(evaluations)
    train_source = str(train_dataset.stats.get("source", "train_ringreactions.csv"))
    test_source = str(test_dataset.stats.get("source", "test_ringreactions.csv"))

    results = {
        "dataset": "ringreactions",
        "train_source": train_source,
        "test_source": test_source,
        "batch_size": batch_size,
        "seed": seed,
        "scratch_epochs": scratch_epochs,
        "finetune_epochs": finetune_epochs,
        "masking_rate": masking_rate,
        "learning_rate": learning_rate,
        "finetune_learning_rate": finetune_learning_rate,
        "dropout": dropout,
        "mlm_weight": mlm_weight,
        "use_aim": use_aim,
        "checkpoints": checkpoints,
        "evaluations": evaluations,
        "mapping_atom_accuracy": accuracies,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_json = out_dir / "comparison_results.json"
    results["results_json"] = write_json(results_json, results)

    lines = []
    for split in ("train", "test"):
        for model_name in ("pretrained", "finetune_mlm", "finetune_supervised", "scratch_mlm", "scratch_supervised"):
            lines.append(f"{split.upper()} {model_name} {accuracies[split][model_name]:.6f}")
    accuracies_txt = out_dir / "mapping_atom_accuracy.txt"
    accuracies_txt.write_text("\n".join(lines) + "\n")
    print(f"Wrote artifacts to: {accuracies_txt}")
    results["accuracy_table_txt"] = str(accuracies_txt)

    return results


if __name__ == "__main__":
    fire.Fire(main)
