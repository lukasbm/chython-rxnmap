"""
Dataset abstractions for reaction data using PyTorch Geometric.

Provides standardized interface for loading different reaction datasets.
"""
from __future__ import annotations

import csv
from abc import abstractmethod
from pathlib import Path
from typing import Any, Iterator

from chython import smiles
from torch_geometric.data import Dataset as PyGDataset


class ReactionDatasetBase(PyGDataset):
    """
    Base class for reaction datasets.
    
    All reaction datasets must provide:
    - reactions: list of chython reaction objects with atom mappings
    - packed: list of packed (serialized) reactions for model input
    """
    
    def __init__(self, root: str | Path | None = None, transform=None, pre_transform=None, pre_filter=None):
        root = Path(root) if root else Path.cwd() / "data"
        super().__init__(str(root), transform, pre_transform, pre_filter)
        self._reactions: list[Any] = []
        self._packed: list[bytes] = []
        self._load_data()
    
    @abstractmethod
    def _load_data(self):
        """Load reactions from source. Must populate self._reactions and self._packed."""
        pass
    
    @property
    def reactions(self) -> list[Any]:
        """Chython reaction objects with atom mappings."""
        return self._reactions
    
    @property
    def packed(self) -> list[bytes]:
        """Packed (serialized) reactions for model input."""
        return self._packed
    
    def len(self) -> int:
        return len(self._reactions)
    
    def get(self, idx: int):
        """Return packed reaction at index."""
        return self._packed[idx]
    
    @property
    def stats(self) -> dict[str, int]:
        """Dataset statistics."""
        return {
            "total": len(self._reactions),
            "valid": len(self._packed),
            "failed": 0,
        }


class CSVReactionDataset(ReactionDatasetBase):
    """
    Dataset that loads reactions from CSV file.
    
    Expects CSV with reaction SMILES in first column, with atom mappings.
    """
    
    def __init__(
        self, 
        csv_path: str | Path,
        name: str | None = None,
        root: str | Path | None = None,
        transform=None,
        pre_transform=None,
        pre_filter=None
    ):
        self.csv_path = Path(csv_path)
        self.dataset_name = name or self.csv_path.stem
        self._total = 0
        self._failed = 0
        super().__init__(root, transform, pre_transform, pre_filter)
    
    def _load_data(self):
        """Load reactions from CSV file."""
        self._reactions = []
        self._packed = []
        self._total = 0
        self._failed = 0
        
        with self.csv_path.open("r", newline="") as f:
            for row in csv.reader(f):
                if not row or not row[0].strip():
                    continue
                
                self._total += 1
                reaction_smiles = row[0].strip()
                
                try:
                    reaction = smiles(reaction_smiles)
                    reaction.canonicalize()
                    self._reactions.append(reaction)
                    self._packed.append(reaction.pack())
                except Exception as exc:
                    self._failed += 1
                    print(f"Warning: Could not parse reaction #{self._total} from {self.dataset_name}: {exc}")
    
    @property
    def stats(self) -> dict[str, int]:
        return {
            "total": self._total,
            "valid": len(self._packed),
            "failed": self._failed,
            "source": str(self.csv_path),
        }
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}('{self.dataset_name}', "
            f"total={self._total}, valid={len(self._packed)}, failed={self._failed})"
        )


class RingReactionsDataset(CSVReactionDataset):
    """Ring reactions dataset (train_ringreactions.csv / test_ringreactions.csv)."""
    pass


class USPTO50kDataset(ReactionDatasetBase):
    """
    USPTO-50k dataset.
    
    Standard benchmark dataset for reaction prediction and atom mapping.
    Expected format: CSV with reaction SMILES in first column.
    """
    
    def __init__(
        self,
        split: str = "train",
        root: str | Path | None = None,
        transform=None,
        pre_transform=None,
        pre_filter=None
    ):
        self.split = split
        self.dataset_name = f"USPTO50k-{split}"
        self._total = 0
        self._failed = 0
        super().__init__(root, transform, pre_transform, pre_filter)
    
    def _load_data(self):
        """Load USPTO-50k data from root directory."""
        # Look for USPTO50k CSV files in root directory
        possible_paths = [
            self.root / f"uspto50k_{self.split}.csv",
            self.root / f"USPTO50k_{self.split}.csv",
            self.root / f"USPTO_50k_{self.split}.csv",
            self.root / "USPTO_50K" / f"{self.split}.csv",
            self.root / "uspto50k" / f"{self.split}.csv",
        ]
        
        csv_path = None
        for path in possible_paths:
            if Path(path).exists():
                csv_path = Path(path)
                break
        
        if csv_path is None:
            raise FileNotFoundError(
                f"USPTO-50k {self.split} split not found. Tried: {[str(p) for p in possible_paths]}\n"
                f"Please download USPTO-50k and place in {self.root}/"
            )
        
        self._reactions = []
        self._packed = []
        self._total = 0
        self._failed = 0
        
        with csv_path.open("r", newline="") as f:
            for row in csv.reader(f):
                if not row or not row[0].strip():
                    continue
                
                self._total += 1
                reaction_smiles = row[0].strip()
                
                try:
                    reaction = smiles(reaction_smiles)
                    reaction.canonicalize()
                    self._reactions.append(reaction)
                    self._packed.append(reaction.pack())
                except Exception as exc:
                    self._failed += 1
                    if self._failed < 10:  # Limit warnings
                        print(f"Warning: Could not parse USPTO50k reaction #{self._total}: {exc}")
    
    @property
    def stats(self) -> dict[str, int]:
        return {
            "total": self._total,
            "valid": len(self._packed),
            "failed": self._failed,
            "split": self.split,
        }
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(split='{self.split}', "
            f"total={self._total}, valid={len(self._packed)}, failed={self._failed})"
        )


# Dataset registry for easy access
DATASETS = {
    "ringreactions": RingReactionsDataset,
    "uspto50k": USPTO50kDataset,
}


def get_dataset(
    name: str,
    split: str = "train",
    root: str | Path | None = None,
    **kwargs
) -> ReactionDatasetBase:
    """
    Get dataset by name.
    
    Args:
        name: Dataset name ("ringreactions", "uspto50k")
        split: Data split ("train", "test", "val")
        root: Root directory for data files
        **kwargs: Additional dataset-specific arguments
    
    Returns:
        Dataset instance
    
    Examples:
        # Ring reactions from CSV
        train_ds = get_dataset("ringreactions", split="train", 
                              csv_path="train_ringreactions.csv")
        
        # USPTO-50k
        train_ds = get_dataset("uspto50k", split="train", root="data/")
    """
    name_lower = name.lower()
    
    if name_lower == "ringreactions":
        # Special handling for ringreactions - need csv_path
        csv_path = kwargs.get("csv_path")
        if csv_path is None:
            # Default paths
            if split == "train":
                csv_path = "train_ringreactions.csv"
            elif split == "test":
                csv_path = "test_ringreactions.csv"
            else:
                raise ValueError(f"Unknown split '{split}' for ringreactions. Use 'train' or 'test'.")
        
        return RingReactionsDataset(csv_path=csv_path, name=f"ringreactions-{split}", root=root)
    
    elif name_lower == "uspto50k":
        return USPTO50kDataset(split=split, root=root)
    
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}"
        )


def print_dataset_stats(dataset: ReactionDatasetBase):
    """Print dataset statistics."""
    stats = dataset.stats
    print(f"{dataset}: {', '.join(f'{k}={v}' for k, v in stats.items())}")
