from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Iterable

from chython import smiles

csv.field_size_limit(sys.maxsize)


def _resolve_path(*, root: str | Path | None, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or root is None:
        return candidate
    return Path(root) / candidate


def _read_lines(path: Path) -> list[str]:
    with path.open("r") as handle:
        return [line.strip() for line in handle if line.strip()]


def _read_csv_rows(path: Path, *, delimiter: str = ",") -> list[list[str]]:
    with path.open("r", newline="") as handle:
        return [
            [value.strip() for value in row if value.strip()]
            for row in csv.reader(handle, delimiter=delimiter)
            if row
        ]


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


def _iter_csv_column(
    path: Path,
    *,
    delimiter: str = ",",
    column_index: int = 0,
) -> Iterable[str]:
    with path.open("r", newline="") as handle:
        for row in csv.reader(handle, delimiter=delimiter):
            if not row or len(row) <= column_index:
                continue
            value = row[column_index].strip()
            if value:
                yield value


def _read_tsv_column(path: Path, column: str) -> list[str]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = []
        for row in reader:
            value = row.get(column, "").strip()
            if value:
                rows.append(value)
        return rows


def _read_csv_named_column(path: Path, column: str) -> list[str]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            value = row.get(column, "").strip()
            if value:
                rows.append(value)
        return rows


def _first_alternatives(rows: Iterable[str]) -> list[str]:
    values = []
    for row in rows:
        parts = [part.strip() for part in row.split(",") if part.strip()]
        if parts:
            values.append(parts[0])
    return values


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
        self.rows = [row.strip() for row in rows if row.strip()]
        self._reactions: list[Any] = []
        self._packed: list[bytes] = []
        self._total = 0
        self._failed = 0
        self._reactions, self._packed, self._total, self._failed = _parse_reactions(self.rows, source)

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
    TRAIN_PATH = Path("golden_train.smiles")
    VAL_PATH = Path("golden_val.smiles")
    TEST_PATH = Path("golden_test.smiles")
    LOCALMAPPER_RAW_PATH = Path("Golden/raw_data.csv")
    LOCALMAPPER_TEST_PATH = Path("Golden/test_data.csv")

    def __init__(
        self,
        *,
        split: str = "train",
        smiles_path: str | Path | None = None,
        root: str | Path | None = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"golden split must be 'train', 'val', or 'test', got '{split}'")

        split_paths = {
            "train": self.TRAIN_PATH,
            "val": self.VAL_PATH,
            "test": self.TEST_PATH,
        }
        if smiles_path is None:
            split_path = _resolve_path(root=root, path=split_paths[split])
            if split_path.exists():
                resolved_path = split_path
                rows = _read_lines(resolved_path)
                super().__init__(dataset_name="golden", source=resolved_path, split=split, rows=rows)
                return

            localmapper_path = _resolve_path(
                root=root,
                path=self.LOCALMAPPER_TEST_PATH if split == "test" else self.LOCALMAPPER_RAW_PATH,
            )
            if localmapper_path.exists():
                rows = _read_csv_named_column(localmapper_path, "mapped_rxn")
                if split == "val":
                    rows = _split_rows(rows, split=split)
                super().__init__(dataset_name="golden", source=localmapper_path, split=split, rows=rows)
                return

        resolved_path = _resolve_path(root=root, path=smiles_path or self.DEFAULT_SMILES_PATH)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Golden file not found: {resolved_path}")
        rows = _split_rows(_read_lines(resolved_path), split=split)
        super().__init__(dataset_name="golden", source=resolved_path, split=split, rows=rows)


class Schneider50kDataset(ReactionDatasetBase):
    DEFAULT_TSV_PATH = Path("schneider50k.tsv")
    LOCALMAPPER_TSV_PATH = Path("schneider/schneider50k.tsv")
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
        if not resolved_path.exists() and tsv_path is None:
            resolved_path = _resolve_path(root=root, path=self.LOCALMAPPER_TSV_PATH)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Schneider file not found: {resolved_path}")
        rows = _split_rows(_read_tsv_column(resolved_path, self.COLUMN), split=split)
        super().__init__(dataset_name="schneider50k", source=resolved_path, split=split, rows=rows)


class RingReactionsDataset(ReactionDatasetBase):
    TRAIN_PATH = Path("train_ringreactions.csv")
    TEST_PATH = Path("test_ringreactions.csv")
    LOCALMAPPER_TRAIN_PATH = Path("ringreactions/train_ringreactions.csv")
    LOCALMAPPER_TEST_PATH = Path("ringreactions/test_ringreactions.csv")

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
            if not resolved_path.exists() and csv_path is None:
                resolved_path = _resolve_path(root=root, path=self.LOCALMAPPER_TEST_PATH)
            if not resolved_path.exists():
                raise FileNotFoundError(f"Ring reactions file not found: {resolved_path}")
            rows = _first_alternatives(",".join(row) for row in _read_csv_rows(resolved_path))
        else:
            resolved_path = _resolve_path(root=root, path=csv_path or self.TRAIN_PATH)
            if not resolved_path.exists() and csv_path is None:
                resolved_path = _resolve_path(root=root, path=self.LOCALMAPPER_TRAIN_PATH)
            if not resolved_path.exists():
                raise FileNotFoundError(f"Ring reactions file not found: {resolved_path}")
            all_rows = _first_alternatives(",".join(row) for row in _read_csv_rows(resolved_path))
            rows = all_rows if split == "train" else _split_rows(all_rows, split=split, train_ratio=0.9, val_ratio=0.1)
        super().__init__(dataset_name="ringreactions", source=resolved_path, split=split, rows=rows)


class MetamdbDataset(ReactionDatasetBase):
    TRAIN_PATH = Path("train_metamdb_filtered.csv")
    TEST_PATH = Path("test_metamdb_filtered.csv")
    LOCALMAPPER_TRAIN_PATH = Path("metAMDB/train_metamdb_filtered.csv")
    LOCALMAPPER_TEST_PATH = Path("metAMDB/test_metamdb_filtered.csv")
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
            if not resolved_path.exists() and csv_path is None:
                resolved_path = _resolve_path(root=root, path=self.LOCALMAPPER_TEST_PATH)
            if not resolved_path.exists():
                raise FileNotFoundError(f"MetaDB file not found: {resolved_path}")
            rows = _first_alternatives(_read_csv_column(resolved_path, delimiter=self.DELIMITER, column_index=1))
        else:
            resolved_path = _resolve_path(root=root, path=csv_path or self.TRAIN_PATH)
            if not resolved_path.exists() and csv_path is None:
                resolved_path = _resolve_path(root=root, path=self.LOCALMAPPER_TRAIN_PATH)
            if not resolved_path.exists():
                raise FileNotFoundError(f"MetaDB file not found: {resolved_path}")
            all_rows = _first_alternatives(_read_csv_column(resolved_path, delimiter=self.DELIMITER, column_index=1))
            rows = all_rows if split == "train" else _split_rows(all_rows, split=split, train_ratio=0.9, val_ratio=0.1)
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


def load_reference_rows(
    *,
    dataset: str,
    split: str = "test",
    root: str | Path = "data/data",
    limit: int | None = None,
) -> list[list[str]]:
    """Load mapped reference reactions for evaluation.

    Each returned item is a list of acceptable references for one evaluation
    input. Most datasets have one reference per row; ringreactions stores
    alternatives as comma-separated mapped reactions.
    """
    dataset_key = dataset.lower()
    root_path = Path(root)
    if dataset_key == "golden":
        path = _resolve_path(root=root_path, path=GoldenDataset.LOCALMAPPER_TEST_PATH if split == "test" else GoldenDataset.LOCALMAPPER_RAW_PATH)
        if path.exists():
            rows = [[row] for row in _read_csv_named_column(path, "mapped_rxn")]
        else:
            ds = GoldenDataset(split=split, root=root_path)
            rows = [[row] for row in ds.rows]
    elif dataset_key in {"schneider50k", "uspto50k"}:
        ds = Schneider50kDataset(split=split, root=root_path)
        rows = [[row] for row in ds.rows]
    elif dataset_key == "metamdb":
        path = _resolve_path(
            root=root_path,
            path=MetamdbDataset.LOCALMAPPER_TEST_PATH if split == "test" else MetamdbDataset.LOCALMAPPER_TRAIN_PATH,
        )
        if path.exists():
            rows = []
            for row in _iter_csv_column(path, delimiter=MetamdbDataset.DELIMITER, column_index=1):
                rows.append([part.strip() for part in row.split(",") if part.strip()])
                if limit is not None and len(rows) >= limit:
                    break
        else:
            ds = MetamdbDataset(split=split, root=root_path)
            rows = [[row] for row in ds.rows]
    elif dataset_key == "ringreactions":
        if split == "test":
            path = _resolve_path(root=root_path, path=RingReactionsDataset.TEST_PATH)
            if not path.exists():
                path = _resolve_path(root=root_path, path=RingReactionsDataset.LOCALMAPPER_TEST_PATH)
            rows = _read_csv_rows(path)
        else:
            path = _resolve_path(root=root_path, path=RingReactionsDataset.TRAIN_PATH)
            if not path.exists():
                path = _resolve_path(root=root_path, path=RingReactionsDataset.LOCALMAPPER_TRAIN_PATH)
            all_rows = [row[0] for row in _read_csv_rows(path) if row]
            selected_rows = (
                all_rows
                if split == "train"
                else _split_rows(all_rows, split=split, train_ratio=0.9, val_ratio=0.1)
            )
            rows = [[row] for row in selected_rows]
    else:
        raise ValueError(
            "Unknown dataset. Use ringreactions, schneider50k/uspto50k, metamdb, or golden."
        )
    return rows[:limit] if limit is not None else rows
