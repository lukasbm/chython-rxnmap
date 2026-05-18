from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Iterable

from chython import smiles

RING_TRAIN_PATH = Path("train_ringreactions.csv")
RING_TEST_PATH = Path("test_ringreactions.csv")
SCHNEIDER50K_PATH = Path("schneider50k.tsv")
SCHNEIDER_COLUMN = "clean_rxn"

DatasetFactory = Callable[..., "ReactionDatasetBase"]
_DATASET_REGISTRY: dict[str, DatasetFactory] = {}


def _resolve_path(*, root: str | Path | None, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or root is None:
        return candidate
    return Path(root) / candidate


class ReactionDatasetBase:
    def __init__(self, dataset_name: str, source: Path, split: str):
        self.dataset_name = dataset_name
        self.source = source
        self.split = split
        self._reactions: list[Any] = []
        self._packed: list[bytes] = []
        self._total = 0
        self._failed = 0

    @property
    def reactions(self) -> list[Any]:
        return self._reactions

    @property
    def packed(self) -> list[bytes]:
        return self._packed

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total": self._total,
            "valid": len(self._packed),
            "failed": self._failed,
            "split": self.split,
            "source": str(self.source),
        }

    def __len__(self) -> int:
        return len(self._packed)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}('{self.dataset_name}', "
            f"split='{self.split}', total={self._total}, valid={len(self._packed)}, failed={self._failed})"
        )


def _parse_reactions(rows: Iterable[str], source: Path) -> tuple[list[Any], list[bytes], int, int]:
    reactions: list[Any] = []
    packed: list[bytes] = []
    total = 0
    failed = 0
    warning_count = 0

    for row in rows:
        reaction_smiles = row.strip()
        if not reaction_smiles:
            continue
        total += 1
        try:
            reaction = smiles(reaction_smiles)
            reaction.canonicalize()
            reactions.append(reaction)
            packed.append(reaction.pack())
        except Exception as exc:
            failed += 1
            if warning_count < 10:
                print(f"Warning: failed to parse reaction #{total} from {source}: {exc}")
                warning_count += 1

    return reactions, packed, total, failed


class RingReactionsDataset(ReactionDatasetBase):
    def __init__(
        self,
        *,
        split: str = "train",
        csv_path: str | Path | None = None,
        root: str | Path | None = None,
    ):
        if split not in {"train", "test"}:
            raise ValueError(f"ringreactions split must be 'train' or 'test', got '{split}'")
        default_path = RING_TRAIN_PATH if split == "train" else RING_TEST_PATH
        resolved_path = _resolve_path(root=root, path=csv_path or default_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Ring reactions file not found: {resolved_path}")
        super().__init__(dataset_name=f"ringreactions-{split}", source=resolved_path, split=split)
        with resolved_path.open("r", newline="") as handle:
            rows = (row[0] for row in csv.reader(handle) if row)
            self._reactions, self._packed, self._total, self._failed = _parse_reactions(rows, resolved_path)


class Schneider50kDataset(ReactionDatasetBase):
    def __init__(
        self,
        *,
        split: str = "train",
        tsv_path: str | Path | None = None,
        root: str | Path | None = None,
    ):
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"schneider50k split must be one of train/val/test/all, got '{split}'")
        resolved_path = _resolve_path(root=root, path=tsv_path or SCHNEIDER50K_PATH)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Schneider file not found: {resolved_path}")
        super().__init__(dataset_name="schneider50k", source=resolved_path, split=split)

        with resolved_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            all_rows = [row.get(SCHNEIDER_COLUMN, "").strip() for row in reader]

        n_rows = len(all_rows)
        if n_rows == 0:
            self._reactions, self._packed, self._total, self._failed = [], [], 0, 0
            return

        train_end = int(n_rows * 0.8)
        val_end = int(n_rows * 0.9)
        if split == "train":
            rows = all_rows[:train_end]
        elif split == "val":
            rows = all_rows[train_end:val_end]
        elif split == "test":
            rows = all_rows[val_end:]
        else:
            rows = all_rows

        self._reactions, self._packed, self._total, self._failed = _parse_reactions(rows, resolved_path)


def register_dataset(name: str, factory: DatasetFactory, aliases: tuple[str, ...] = ()) -> None:
    canonical = name.lower()
    _DATASET_REGISTRY[canonical] = factory
    for alias in aliases:
        _DATASET_REGISTRY[alias.lower()] = factory


def get_dataset(
    name: str,
    split: str = "train",
    root: str | Path | None = None,
    **kwargs,
) -> ReactionDatasetBase:
    factory = _DATASET_REGISTRY.get(name.lower())
    if factory is None:
        known = ", ".join(sorted(set(_DATASET_REGISTRY)))
        raise ValueError(f"Unknown dataset '{name}'. Registered datasets: {known}")
    return factory(split=split, root=root, **kwargs)


def print_dataset_stats(dataset: ReactionDatasetBase) -> None:
    stats = dataset.stats
    print(f"{dataset}: {', '.join(f'{key}={value}' for key, value in stats.items())}")


class CombinedReactionDataset:
    """Combines multiple datasets into one in-memory dataset."""

    def __init__(self, *datasets: ReactionDatasetBase, name: str | None = None):
        self.dataset_name = name or "+".join(dataset.dataset_name for dataset in datasets)
        self.split = "combined"
        self._reactions: list[Any] = [reaction for dataset in datasets for reaction in dataset.reactions]
        self._packed: list[bytes] = [packed for dataset in datasets for packed in dataset.packed]
        self._total: int = sum(dataset._total for dataset in datasets)
        self._failed: int = sum(dataset._failed for dataset in datasets)

    @property
    def reactions(self) -> list[Any]:
        return self._reactions

    @property
    def packed(self) -> list[bytes]:
        return self._packed

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total": self._total,
            "valid": len(self._packed),
            "failed": self._failed,
            "split": self.split,
        }

    def __len__(self) -> int:
        return len(self._packed)

    def __repr__(self) -> str:
        return (
            f"CombinedReactionDataset('{self.dataset_name}', "
            f"n={len(self._packed)}, total={self._total}, failed={self._failed})"
        )


register_dataset("ringreactions", RingReactionsDataset)
register_dataset("schneider50k", Schneider50kDataset, aliases=("uspto50k",))
