# Dataset Usage Guide

This repository now uses PyTorch Geometric's Dataset abstraction for loading reaction data.

## Available Datasets

### 1. Ring Reactions
Your custom ring reaction dataset from CSV files.

```bash
# Default (uses train_ringreactions.csv and test_ringreactions.csv)
uv run nora_optuna.py --dataset=ringreactions --mode=finetune

# Custom CSV paths
uv run nora_optuna.py --dataset=ringreactions --train=my_train.csv --test=my_test.csv
```

### 2. USPTO-50k
Standard benchmark dataset for reaction prediction and atom mapping.

```bash
# Assumes USPTO-50k files are in data/ directory
uv run nora_optuna.py --dataset=uspto50k --data_root=data/ --mode=finetune

# The dataset loader will look for files like:
#   data/uspto50k_train.csv
#   data/uspto50k_test.csv
#   data/USPTO50k_train.csv
#   etc.
```

## Usage Examples

### Option A: MLM-only finetuning (default)
```bash
# Ring reactions with 10 epochs (default for finetune)
uv run nora_optuna.py --dataset=ringreactions --n_trials=15 --batch_size=32

# USPTO-50k
uv run nora_optuna.py --dataset=uspto50k --data_root=data/ --n_trials=10 --batch_size=32
```

### Option B: Supervised mapping loss finetuning
```bash
# Ring reactions with supervised loss
uv run nora_optuna.py --dataset=ringreactions --n_trials=15 --batch_size=32 \
    --use_supervised_loss=True

# USPTO-50k with supervised loss
uv run nora_optuna.py --dataset=uspto50k --data_root=data/ --n_trials=10 \
    --batch_size=32 --use_supervised_loss=True
```

### Training from scratch
```bash
# Ring reactions (100 epochs default for scratch)
uv run nora_optuna.py --dataset=ringreactions --mode=scratch --n_trials=10

# USPTO-50k
uv run nora_optuna.py --dataset=uspto50k --data_root=data/ --mode=scratch
```

### Custom epochs and parameters
```bash
# Finetune with custom epochs
uv run nora_optuna.py --dataset=ringreactions --max_epochs=20 --mode=finetune

# Adjust MLM weight in supervised mode
uv run nora_optuna.py --dataset=ringreactions --use_supervised_loss=True --mlm_weight=0.05
```

## Adding New Datasets

To add a new dataset, edit `datasets.py`:

1. Create a new class inheriting from `ReactionDatasetBase`
2. Implement the `_load_data()` method
3. Add to the `DATASETS` registry
4. Update `get_dataset()` function

Example:
```python
class MyCustomDataset(ReactionDatasetBase):
    def __init__(self, split: str = "train", root: str | Path | None = None):
        self.split = split
        self.dataset_name = f"mycustom-{split}"
        super().__init__(root)
    
    def _load_data(self):
        # Load reactions from your source
        # Must populate self._reactions and self._packed
        pass
```

Then use:
```bash
uv run nora_optuna.py --dataset=mycustom --data_root=path/to/data/
```

## Dataset Format

All datasets must provide reactions as SMILES strings with atom mappings in the format:
```
[CH3:1][OH:2].[CH2:3]=[O:4]>>[CH3:1][O:2][CH2:3][OH:4]
```

CSV files should have reaction SMILES in the first column (other columns ignored).

## Python API

You can also use the datasets directly in Python:

```python
from datasets import get_dataset, print_dataset_stats

# Load ring reactions
train_ds = get_dataset("ringreactions", split="train", csv_path="train_ringreactions.csv")
test_ds = get_dataset("ringreactions", split="test", csv_path="test_ringreactions.csv")

# Load USPTO-50k
train_ds = get_dataset("uspto50k", split="train", root="data/")
test_ds = get_dataset("uspto50k", split="test", root="data/")

# Access data
reactions = train_ds.reactions  # List of chython reaction objects
packed = train_ds.packed        # List of packed (serialized) reactions

# Print statistics
print_dataset_stats(train_ds)
```
