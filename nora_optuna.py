from __future__ import annotations

from pathlib import Path

import fire
import optuna

from nora import (
    ExperimentConfig,
    Model,
    evaluate_model,
    load_reaction_dataset,
    print_dataset_summary,
    print_metrics,
    run_scratch_experiment,
)


def main(
    train: str = "train_ringreactions.csv",
    test: str = "test_ringreactions.csv",
    n_trials: int = 5,
    max_epochs: int = 1,
    batch_size: int = 16,
    seed: int = 42,
    study_name: str = "rxnmap_optuna",
):
    train_dataset = load_reaction_dataset(Path(train), name="train")
    test_dataset = load_reaction_dataset(Path(test), name="test")
    print_dataset_summary(train_dataset)
    print_dataset_summary(test_dataset)

    if not train_dataset.packed or not test_dataset.packed:
        raise RuntimeError("No valid reactions available after parsing train/test datasets.")

    baseline_metrics = evaluate_model(
        Model.pretrained(), test_dataset, batch_size=batch_size, mask_seed=seed
    )
    print_metrics("pretrained_baseline_on_test", baseline_metrics)

    def objective(trial: optuna.Trial) -> float:
        config = ExperimentConfig(
            batch_size=batch_size,
            max_epochs=max_epochs,
            seed=seed + trial.number,
            masking_rate=trial.suggest_float("masking_rate", 0.05, 0.35),
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            dropout=trial.suggest_float("dropout", 0.0, 0.3),
        )
        metrics = run_scratch_experiment(
            train_dataset,
            test_dataset,
            config,
            run_name=study_name,
        )
        trial.set_user_attr("mlm_loss_total", metrics["mlm_loss_total"])
        trial.set_user_attr("mapping_exact_match", metrics["mapping_exact_match"])
        trial.set_user_attr("mapping_topk", metrics["mapping_topk"])
        return float(metrics["mapping_atom_accuracy"])

    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(objective, n_trials=n_trials)

    print("=" * 60)
    print("OPTUNA BEST TRIAL")
    print("=" * 60)
    print(f"best_value(mapping_atom_accuracy): {study.best_value:.6f}")
    print(f"best_params: {study.best_trial.params}")
    print(f"best_trial_mlm_loss_total: {study.best_trial.user_attrs.get('mlm_loss_total'):.6f}")
    print(
        f"best_trial_mapping_exact_match: {study.best_trial.user_attrs.get('mapping_exact_match'):.6f}"
    )
    print(f"best_trial_mapping_topk: {study.best_trial.user_attrs.get('mapping_topk'):.6f}")


if __name__ == "__main__":
    fire.Fire(main)
