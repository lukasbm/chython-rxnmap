from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from chython import smiles
from torch_geometric.data import InMemoryDataset

RING_TRAIN_PATH = Path("train_ringreactions.csv")
RING_TEST_PATH = Path("test_ringreactions.csv")
SCHNEIDER50K_PATH = Path("schneider50k.tsv")
SCHNEIDER_COLUMN = "clean_rxn"


class ReactionDatasetBase(InMemoryDataset):
    def __init__(self, dataset_name: str, source: Path, split: str):
        self.dataset_name = dataset_name
        self.source = source
        self.split = split
        self._reactions: list[Any] = []
        self._packed: list[bytes] = []
        self._total = 0
        self._failed = 0
        super().__init__(root=".")

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return []

    def download(self):
        return

    def process(self):
        return

    @property
    def reactions(self) -> list[Any]:
        return self._reactions

    @property
    def packed(self) -> list[bytes]:
        return self._packed

    def len(self) -> int:
        return len(self._packed)

    def get(self, idx: int) -> bytes:
        return self._packed[idx]

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total": self._total,
            "valid": len(self._packed),
            "failed": self._failed,
            "split": self.split,
            "source": str(self.source),
        }

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
    def __init__(self, split: str = "train", csv_path: str | None = None):
        if split not in {"train", "test"}:
            raise ValueError(f"ringreactions split must be 'train' or 'test', got '{split}'")
        path = Path(csv_path) if csv_path else (RING_TRAIN_PATH if split == "train" else RING_TEST_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Ring reactions file not found: {path}")
        super().__init__(dataset_name=f"ringreactions-{split}", source=path, split=split)
        with path.open("r", newline="") as f:
            rows = (row[0] for row in csv.reader(f) if row)
            self._reactions, self._packed, self._total, self._failed = _parse_reactions(rows, path)


class Schneider50kDataset(ReactionDatasetBase):
    def __init__(self, split: str = "train", tsv_path: str | None = None):
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"schneider50k split must be one of train/val/test/all, got '{split}'")
        path = Path(tsv_path) if tsv_path else SCHNEIDER50K_PATH
        if not path.exists():
            raise FileNotFoundError(f"Schneider file not found: {path}")
        super().__init__(dataset_name="schneider50k", source=path, split=split)

        with path.open("r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            all_rows = [row.get(SCHNEIDER_COLUMN, "").strip() for row in reader]

        n = len(all_rows)
        if n == 0:
            self._reactions, self._packed, self._total, self._failed = [], [], 0, 0
            return

        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        if split == "train":
            rows = all_rows[:train_end]
        elif split == "val":
            rows = all_rows[train_end:val_end]
        elif split == "test":
            rows = all_rows[val_end:]
        else:
            rows = all_rows

        self._reactions, self._packed, self._total, self._failed = _parse_reactions(rows, path)


def get_dataset(
    name: str,
    split: str = "train",
    root: str | Path | None = None,
    **kwargs,
) -> ReactionDatasetBase:
    del root  # kept for API compatibility; dataset paths are intentionally fixed/simple
    name_lower = name.lower()
    if name_lower == "ringreactions":
        return RingReactionsDataset(split=split, csv_path=kwargs.get("csv_path"))
    if name_lower in {"schneider50k", "uspto50k"}:
        return Schneider50kDataset(split=split, tsv_path=kwargs.get("tsv_path"))
    raise ValueError("Unknown dataset. Use 'ringreactions' or 'schneider50k'.")


def print_dataset_stats(dataset: ReactionDatasetBase):
    stats = dataset.stats
    print(f"{dataset}: {', '.join(f'{k}={v}' for k, v in stats.items())}")
