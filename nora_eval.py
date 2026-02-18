from __future__ import annotations

from pathlib import Path

import fire
import torch
from pytorch_lightning import seed_everything

from nora import (
    Model,
    evaluate_model,
    print_metrics,
)
from datasets import get_dataset, print_dataset_stats


def load_model_from_checkpoint(checkpoint_path: str | None, model_type: str = "pretrained") -> Model:
    """
    Load a model from checkpoint or create a pretrained baseline.
    
    Args:
        checkpoint_path: Path to Lightning checkpoint file (.ckpt), or None for pretrained baseline
        model_type: Type of model - "pretrained", "baseline", "finetuned", or "trained"
    
    Returns:
        Loaded Model instance
    """
    if checkpoint_path is None or checkpoint_path == "" or checkpoint_path.lower() == "none":
        print(f"Loading pretrained baseline model (no checkpoint provided)")
        return Model.pretrained()
    
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading {model_type} model from checkpoint: {checkpoint_path}")
    model = Model.load_from_checkpoint(checkpoint_path)
    return model


def evaluate_checkpoint(
        checkpoint_path: str | None,
        dataset_name: str = "ringreactions",
        split: str = "test",
        csv_path: str | None = None,
        data_root: str | None = None,
        batch_size: int = 16,
        seed: int = 42,
        model_type: str = "pretrained",
) -> dict[str, float]:
    """
    Evaluate a single checkpoint on a dataset.
    
    Args:
        checkpoint_path: Path to checkpoint file or None for pretrained
        dataset_name: Dataset name ("ringreactions", "uspto50k")
        split: Dataset split ("train", "test", "val")
        csv_path: Override CSV path for ringreactions
        data_root: Root directory for datasets
        batch_size: Batch size for evaluation
        seed: Random seed for reproducibility
        model_type: Descriptive name for the model type
    
    Returns:
        Dictionary of evaluation metrics
    """
    seed_everything(seed, workers=True)
    
    if dataset_name.lower() == "ringreactions":
        path = csv_path or f"{split}_ringreactions.csv"
        eval_dataset = get_dataset("ringreactions", split=split, csv_path=path, root=data_root)
    else:
        eval_dataset = get_dataset(dataset_name, split=split, root=data_root)
    
    print_dataset_stats(eval_dataset)
    
    if len(eval_dataset.packed) == 0:
        raise RuntimeError("No valid reactions available in evaluation dataset.")
    
    model = load_model_from_checkpoint(checkpoint_path, model_type)
    model.eval()
    
    metrics = evaluate_model(model, eval_dataset, batch_size=batch_size, mask_seed=seed)
    print_metrics(f"{model_type}_on_{dataset_name}_{split}", metrics)
    
    return metrics


def main(
        dataset: str = "ringreactions",
        split: str = "test",
        csv_path: str | None = None,
        data_root: str | None = None,
        baseline_checkpoint: str | None = None,
        finetuned_checkpoint: str | None = None,
        trained_checkpoint: str | None = None,
        batch_size: int = 16,
        seed: int = 42,
):
    """
    Evaluate multiple model checkpoints on a given dataset and compare results.
    
    Args:
        dataset: Path to evaluation dataset CSV file
        baseline_checkpoint: Path to baseline model checkpoint (or None for pretrained)
        finetuned_checkpoint: Path to finetuned model checkpoint (optional)
        trained_checkpoint: Path to trained-from-scratch model checkpoint (optional)
        batch_size: Batch size for evaluation
        seed: Random seed for reproducibility
    
    Usage:
        # Evaluate pretrained baseline only
        python nora_eval.py --dataset=test.csv
        
        # Evaluate baseline and finetuned
        python nora_eval.py --dataset=test.csv --baseline_checkpoint=ckpt/baseline.ckpt --finetuned_checkpoint=ckpt/finetuned.ckpt
        
        # Evaluate all three models
        python nora_eval.py --dataset=test.csv --baseline_checkpoint=ckpt/baseline.ckpt --finetuned_checkpoint=ckpt/finetuned.ckpt --trained_checkpoint=ckpt/trained.ckpt
    """
    results = {}
    
    print("=" * 80)
    print("BASELINE MODEL EVALUATION")
    print("=" * 80)
    baseline_metrics = evaluate_checkpoint(
        baseline_checkpoint, dataset, batch_size, seed, model_type="baseline"
    )
    results["baseline"] = baseline_metrics
    
    if finetuned_checkpoint:
        print("\n" + "=" * 80)
        print("FINETUNED MODEL EVALUATION")
        print("=" * 80)
        finetuned_metrics = evaluate_checkpoint(
            finetuned_checkpoint, dataset, split, csv_path, data_root, batch_size, seed, model_type="finetuned"
        )
        results["finetuned"] = finetuned_metrics
    
    if trained_checkpoint:
        print("\n" + "=" * 80)
        print("TRAINED-FROM-SCRATCH MODEL EVALUATION")
        print("=" * 80)
        trained_metrics = evaluate_checkpoint(
            trained_checkpoint, dataset, split, csv_path, data_root, batch_size, seed, model_type="trained"
        )
        results["trained"] = trained_metrics
    
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    metric_keys = [
        "mlm_loss_total",
        "mlm_atom_accuracy",
        "mlm_neighbor_accuracy",
        "mapping_atom_accuracy",
        "mapping_exact_match",
        "mapping_top1",
        "mapping_topk",
    ]
    
    for key in metric_keys:
        print(f"\n{key}:")
        for model_name, metrics in results.items():
            if key in metrics:
                value = metrics[key]
                print(f"  {model_name:15s}: {value:.6f}")
        
        if len(results) > 1 and key in results["baseline"]:
            baseline_value = results["baseline"][key]
            for model_name, metrics in results.items():
                if model_name != "baseline" and key in metrics:
                    delta = metrics[key] - baseline_value
                    print(f"  {model_name:15s} delta: {delta:+.6f}")


if __name__ == "__main__":
    fire.Fire(main)
