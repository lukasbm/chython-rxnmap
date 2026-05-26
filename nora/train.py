from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import fire
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from torch.optim import AdamW

from chytorch.zoo.rxnmap import Model
from nora.datasets import (
    GoldenDataset,
    MetamdbDataset,
    RingReactionsDataset,
    Schneider50kDataset,
    print_dataset_stats,
)


def write_json(path: Path | str, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return str(output)


def build_model(mode: str, masking_rate: float, learning_rate: float, dropout: float) -> Model:
    model_kwargs = {
        "masking_rate": masking_rate,
        "optimizer": partial(AdamW, lr=learning_rate),
        "dropout": dropout,
    }
    if mode == "pretrained":
        return Model.pretrained(**model_kwargs)
    if mode == "scratch":
        return Model(**model_kwargs)
    raise ValueError("mode must be 'scratch' or 'pretrained'")


def run_training(
    *,
    dataset: str,
    train_split: str,
    test_split: str,
    train_csv: str | None,
    test_csv: str | None,
    data_root: str | None,
    batch_size: int,
    max_epochs: int,
    seed: int,
    mode: str,
    masking_rate: float,
    learning_rate: float,
    dropout: float,
    output_json: str,
) -> dict[str, Any]:
    if dataset.lower() == "ringreactions":
        train_dataset = RingReactionsDataset(
            split=train_split,
            csv_path=train_csv or "train_ringreactions.csv",
            root=data_root,
        )
        test_dataset = RingReactionsDataset(
            split=test_split,
            csv_path=test_csv or "test_ringreactions.csv",
            root=data_root,
        )
    else:
        dataset_key = dataset.lower()
        if dataset_key in {"schneider50k", "uspto50k"}:
            train_dataset = Schneider50kDataset(split=train_split, root=data_root)
            test_dataset = Schneider50kDataset(split=test_split, root=data_root)
        elif dataset_key == "metamdb":
            train_dataset = MetamdbDataset(split=train_split, root=data_root)
            test_dataset = MetamdbDataset(split=test_split, root=data_root)
        elif dataset_key == "golden":
            train_dataset = GoldenDataset(split=train_split, root=data_root)
            test_dataset = GoldenDataset(split=test_split, root=data_root)
        else:
            raise ValueError(
                "Unknown dataset. Use ringreactions, schneider50k/uspto50k, metamdb, or golden."
            )

    print_dataset_stats(train_dataset)
    print_dataset_stats(test_dataset)
    if len(train_dataset.packed) == 0:
        raise RuntimeError("No valid reactions available in the training dataset.")

    seed_everything(seed, workers=True)
    model = build_model(mode, masking_rate, learning_rate, dropout)
    train_loader = model.prepare_dataloader(
        train_dataset.packed,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        persistent_workers=False,
    )

    logger = CSVLogger("lightning_logs", name=f"rxnmap_{mode}")
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=model.configure_callbacks(),
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_dataloaders=train_loader)

    log_dir = Path(logger.log_dir)
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = checkpoint_dir / "last.ckpt"
    trainer.save_checkpoint(str(last_checkpoint))

    results: dict[str, Any] = {
        "dataset": dataset,
        "train_split": train_split,
        "test_split": test_split,
        "seed": seed,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "mode": mode,
        "masking_rate": masking_rate,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "log_dir": str(log_dir),
        "last_checkpoint": str(last_checkpoint),
        "train_dataset": train_dataset.stats,
        "test_dataset": test_dataset.stats,
    }
    results["summary_json"] = write_json(output_json, results)
    return results


def main(
    dataset: str = "ringreactions",
    train_split: str = "train",
    test_split: str = "test",
    train_csv: str | None = None,
    test_csv: str | None = None,
    data_root: str | None = None,
    batch_size: int = 16,
    max_epochs: int = 1,
    seed: int = 42,
    mode: str = "scratch",
    masking_rate: float = 0.15,
    learning_rate: float = 1e-4,
    dropout: float = 0.1,
    output_json: str = "experiment_results/nora_main_summary.json",
) -> dict[str, Any]:
    """Train the rxnmap model with the same minimal flow shown in the README."""
    return run_training(
        dataset=dataset,
        train_split=train_split,
        test_split=test_split,
        train_csv=train_csv,
        test_csv=test_csv,
        data_root=data_root,
        batch_size=batch_size,
        max_epochs=max_epochs,
        seed=seed,
        mode=mode,
        masking_rate=masking_rate,
        learning_rate=learning_rate,
        dropout=dropout,
        output_json=output_json,
    )


if __name__ == "__main__":
    fire.Fire(main)
