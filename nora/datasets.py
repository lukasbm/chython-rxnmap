from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from chython import smiles


def _resolve_path(*, root: str | Path | None, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or root is None:
        return candidate
    return Path(root) / candidate


def _read_lines(path: Path) -> list[str]:
    with path.open("r") as handle:
        return [line.strip() for line in handle if line.strip()]


def _read_csv_column(
    path: Path,
    *,
    delimiter: str = ",",
    column_index: int = 0,
) -> list[str]:
    with path.open("r", newline="") as handle:
        rows = []
        for row in csv.reader(handle, delimiter=delimiter):
            if not row or len(row) <= column_index:
                continue
            value = row[column_index].strip()
            if value:
                rows.append(value)
        return rows


def _read_tsv_column(path: Path, column: str) -> list[str]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = []
        for row in reader:
            value = row.get(column, "").strip()
            if value:
                rows.append(value)
        return rows


def _split_rows(
    rows: list[str],
    *,
    split: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> list[str]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
    if not rows:
        return []
    train_end = int(len(rows) * train_ratio)
    val_end = int(len(rows) * (train_ratio + val_ratio))
    if split == "train":
        return rows[:train_end]
    if split == "val":
        return rows[train_end:val_end]
    return rows[val_end:]


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


class ReactionDatasetBase:
    def __init__(self, dataset_name: str, source: Path, split: str, rows: Iterable[str]):
        self.dataset_name = dataset_name
        self.source = source
        self.split = split
        self._reactions: list[Any] = []
        self._packed: list[bytes] = []
        self._total = 0
        self._failed = 0
        self._reactions, self._packed, self._total, self._failed = _parse_reactions(rows, source)

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


class GoldenDataset(ReactionDatasetBase):
    DEFAULT_SMILES_PATH = Path("golden.smiles")

    def __init__(
        self,
        *,
        split: str = "train",
        smiles_path: str | Path | None = None,
        root: str | Path | None = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"golden split must be 'train', 'val', or 'test', got '{split}'")
        resolved_path = _resolve_path(root=root, path=smiles_path or self.DEFAULT_SMILES_PATH)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Golden file not found: {resolved_path}")
        rows = _split_rows(_read_lines(resolved_path), split=split)
        super().__init__(dataset_name="golden", source=resolved_path, split=split, rows=rows)


class Schneider50kDataset(ReactionDatasetBase):
    DEFAULT_TSV_PATH = Path("schneider50k.tsv")
    COLUMN = "clean_rxn"

    def __init__(
        self,
        *,
        split: str = "train",
        tsv_path: str | Path | None = None,
        root: str | Path | None = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"schneider50k split must be 'train', 'val', or 'test', got '{split}'")
        resolved_path = _resolve_path(root=root, path=tsv_path or self.DEFAULT_TSV_PATH)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Schneider file not found: {resolved_path}")
        rows = _split_rows(_read_tsv_column(resolved_path, self.COLUMN), split=split)
        super().__init__(dataset_name="schneider50k", source=resolved_path, split=split, rows=rows)


class RingReactionsDataset(ReactionDatasetBase):
    TRAIN_PATH = Path("train_ringreactions.csv")
    TEST_PATH = Path("test_ringreactions.csv")

    def __init__(
        self,
        *,
        split: str = "train",
        csv_path: str | Path | None = None,
        root: str | Path | None = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"ringreactions split must be 'train', 'val', or 'test', got '{split}'")
        if split == "test":
            resolved_path = _resolve_path(root=root, path=csv_path or self.TEST_PATH)
            if not resolved_path.exists():
                raise FileNotFoundError(f"Ring reactions file not found: {resolved_path}")
            rows = _read_csv_column(resolved_path)
        else:
            resolved_path = _resolve_path(root=root, path=csv_path or self.TRAIN_PATH)
            if not resolved_path.exists():
                raise FileNotFoundError(f"Ring reactions file not found: {resolved_path}")
            rows = _split_rows(_read_csv_column(resolved_path), split=split, train_ratio=0.9, val_ratio=0.1)
        super().__init__(dataset_name="ringreactions", source=resolved_path, split=split, rows=rows)


class MetamdbDataset(ReactionDatasetBase):
    TRAIN_PATH = Path("train_metamdb_filtered.csv")
    TEST_PATH = Path("test_metamdb_filtered.csv")
    DELIMITER = ";"

    def __init__(
        self,
        *,
        split: str = "train",
        csv_path: str | Path | None = None,
        root: str | Path | None = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"metamdb split must be 'train', 'val', or 'test', got '{split}'")
        if split == "test":
            resolved_path = _resolve_path(root=root, path=csv_path or self.TEST_PATH)
            if not resolved_path.exists():
                raise FileNotFoundError(f"MetaDB file not found: {resolved_path}")
            rows = _read_csv_column(resolved_path, delimiter=self.DELIMITER, column_index=1)
        else:
            resolved_path = _resolve_path(root=root, path=csv_path or self.TRAIN_PATH)
            if not resolved_path.exists():
                raise FileNotFoundError(f"MetaDB file not found: {resolved_path}")
            rows = _split_rows(
                _read_csv_column(resolved_path, delimiter=self.DELIMITER, column_index=1),
                split=split,
                train_ratio=0.9,
                val_ratio=0.1,
            )
        super().__init__(dataset_name="metamdb", source=resolved_path, split=split, rows=rows)


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
