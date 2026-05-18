from __future__ import annotations

from typing import Literal

import fire
import optuna

from nora.datasets import get_dataset, print_dataset_stats
from nora.nora import (
    Model,
    evaluate_model,
    print_metrics,
    run_scratch_experiment,
    run_finetune_experiment,
    write_json,
)


def main(
        dataset: str = "ringreactions",
        train: str | None = None,
        test: str | None = None,
        data_root: str | None = None,
        n_trials: int = 5,
        max_epochs: int | None = None,
        batch_size: int = 16,
        seed: int = 42,
        study_name: str = "rxnmap_optuna",
        use_aim: bool = True,
        mode: Literal["finetune", "scratch"] = "finetune",
        output_json: str = "experiment_results/nora_optuna_summary.json",
) -> dict[str, object]:
    """
    Hyperparameter tuning with Optuna.
    
    Args:
        dataset: Dataset name ("ringreactions", "uspto50k")
        train: Training split name or CSV path for ringreactions
        test: Test split name or CSV path for ringreactions
        data_root: Root directory for datasets
        n_trials: Number of Optuna trials
        max_epochs: Max training epochs (default: 10 for finetune, 100 for scratch)
        batch_size: Batch size
        seed: Random seed
        study_name: Optuna study name
        use_aim: Enable Aim logging
        mode: Training mode ("finetune" or "scratch")
    
    Examples:
        # Ring reactions (default)
        uv run nora_optuna.py --dataset=ringreactions --mode=finetune
        
        # USPTO-50k
        uv run nora_optuna.py --dataset=uspto50k --data_root=data/
        
        # Custom CSV files
        uv run nora_optuna.py --dataset=ringreactions --train=my_train.csv --test=my_test.csv
        
    """
    if dataset.lower() == "ringreactions":
        train_path = train or "train_ringreactions.csv"
        test_path = test or "test_ringreactions.csv"
        train_dataset = get_dataset("ringreactions", split="train", csv_path=train_path, root=data_root)
        test_dataset = get_dataset("ringreactions", split="test", csv_path=test_path, root=data_root)
    else:
        train_split = train or "train"
        test_split = test or "test"
        train_dataset = get_dataset(dataset, split=train_split, root=data_root)
        test_dataset = get_dataset(dataset, split=test_split, root=data_root)

    print_dataset_stats(train_dataset)
    print_dataset_stats(test_dataset)

    if len(train_dataset.packed) == 0 or len(test_dataset.packed) == 0:
        raise RuntimeError("No valid reactions available after parsing train/test datasets.")

    print(f"\n{'=' * 70}")
    print(f"TRAINING MODE: {mode.upper()}")
    print(f"{'=' * 70}\n")

    # Set default epochs based on mode if not specified
    epochs = max_epochs if max_epochs is not None else (10 if mode == "finetune" else 100)
    print(f"Using max_epochs={epochs} for {mode} mode\n")

    baseline_metrics = evaluate_model(
        Model.pretrained(), test_dataset, batch_size=batch_size, mask_seed=seed
    )
    print_metrics("pretrained_baseline_on_test", baseline_metrics)

    def objective(trial: optuna.Trial) -> float:
        masking_rate = trial.suggest_float("masking_rate", 0.05, 0.35)

        if mode == "finetune":
            # Lower learning rates for finetuning
            learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)

            metrics = run_finetune_experiment(
                train_dataset,
                test_dataset,
                batch_size=batch_size,
                max_epochs=epochs,
                seed=seed + trial.number,
                masking_rate=masking_rate,
                finetune_learning_rate=learning_rate,
                run_name=f"{study_name}_trial_{trial.number}",
                use_aim=use_aim,
                aim_experiment=study_name,
            )
        else:  # mode == "scratch"
            # Higher learning rates and tune dropout for scratch training
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
            dropout = trial.suggest_float("dropout", 0.0, 0.3)

            metrics = run_scratch_experiment(
                train_dataset,
                test_dataset,
                batch_size=batch_size,
                max_epochs=epochs,
                seed=seed + trial.number,
                masking_rate=masking_rate,
                learning_rate=learning_rate,
                dropout=dropout,
                run_name=f"{study_name}_trial_{trial.number}",
                use_aim=use_aim,
                aim_experiment=study_name,
            )

        trial.set_user_attr("mlm_loss_total", metrics["mlm_loss_total"])
        trial.set_user_attr("mlm_loss_atom", metrics["mlm_loss_atom"])
        trial.set_user_attr("mlm_loss_neighbor", metrics["mlm_loss_neighbor"])
        trial.set_user_attr("mlm_atom_accuracy", metrics["mlm_atom_accuracy"])
        trial.set_user_attr("mlm_neighbor_accuracy", metrics["mlm_neighbor_accuracy"])
        trial.set_user_attr("mlm_perplexity", metrics["mlm_perplexity"])
        trial.set_user_attr("mapping_atom_accuracy", metrics["mapping_atom_accuracy"])
        trial.set_user_attr("mapping_exact_match", metrics["mapping_exact_match"])
        trial.set_user_attr("mapping_top1", metrics["mapping_top1"])
        trial.set_user_attr("mapping_topk", metrics["mapping_topk"])
        trial.set_user_attr("mapping_assignment_coverage", metrics["mapping_assignment_coverage"])
        trial.set_user_attr("mapping_mean_similarity", metrics["mapping_mean_similarity"])

        # Print trial results
        print(f"\n{'=' * 70}")
        print(f"TRIAL {trial.number} RESULTS ({mode.upper()})")
        print(f"{'=' * 70}")
        print(f"Hyperparameters:")
        print(f"  masking_rate: {masking_rate:.6f}")
        print(f"  learning_rate: {learning_rate:.6e}")
        if mode == "scratch":
            print(f"  dropout: {dropout:.6f}")
        print(f"Metrics:")
        print(f"  mlm_loss_total: {metrics['mlm_loss_total']:.6f}")
        print(f"  mlm_atom_accuracy: {metrics['mlm_atom_accuracy']:.6f}")
        print(f"  mlm_neighbor_accuracy: {metrics['mlm_neighbor_accuracy']:.6f}")
        print(f"  mapping_atom_accuracy: {metrics['mapping_atom_accuracy']:.6f}")
        print(f"  mapping_exact_match: {metrics['mapping_exact_match']:.6f}")
        print(f"  mapping_top1: {metrics['mapping_top1']:.6f}")
        print(f"  mapping_topk: {metrics['mapping_topk']:.6f}")
        print(f"{'=' * 70}\n")

        return float(metrics["mapping_atom_accuracy"])

    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(objective, n_trials=n_trials)

    print("=" * 60)
    print("OPTUNA BEST TRIAL")
    print("=" * 60)
    print(f"best_value(mapping_atom_accuracy): {study.best_value:.6f}")
    print(f"best_params: {study.best_trial.params}")
    print(f"best_trial_mlm_loss_total: {study.best_trial.user_attrs.get('mlm_loss_total'):.6f}")
    print(f"best_trial_mlm_atom_accuracy: {study.best_trial.user_attrs.get('mlm_atom_accuracy'):.6f}")
    print(f"best_trial_mlm_neighbor_accuracy: {study.best_trial.user_attrs.get('mlm_neighbor_accuracy'):.6f}")
    print(
        f"best_trial_mapping_exact_match: {study.best_trial.user_attrs.get('mapping_exact_match'):.6f}"
    )
    print(f"best_trial_mapping_topk: {study.best_trial.user_attrs.get('mapping_topk'):.6f}")

    trials_summary = []
    for trial in study.trials:
        trials_summary.append(
            {
                "number": trial.number,
                "state": str(trial.state),
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
        )

    payload = {
        "study_name": study_name,
        "mode": mode,
        "n_trials": n_trials,
        "max_epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "best_value": study.best_value,
        "best_params": study.best_trial.params,
        "best_user_attrs": study.best_trial.user_attrs,
        "baseline_metrics_test": baseline_metrics,
        "trials": trials_summary,
    }
    payload["summary_json"] = write_json(output_json, payload)
    return payload


if __name__ == "__main__":
    fire.Fire(main)
