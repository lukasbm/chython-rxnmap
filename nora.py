from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import fire
import torch
from chython import smiles
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor, rand
from torch.nn.functional import cross_entropy, normalize
from torch.optim import AdamW
from torch.utils.data import DataLoader

from chytorch.utils.data import ReactionDataset
from chytorch.zoo.rxnmap import Model


@dataclass
class DatasetBundle:
    name: str
    path: Path
    reactions: list[Any]
    packed: list[bytes]
    total: int
    failed: int


def iter_first_column_reactions(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        for row in csv.reader(f):
            if row and row[0].strip():
                yield row[0].strip()


def load_reaction_dataset(csv_path: Path, name: str) -> DatasetBundle:
    reactions: list[Any] = []
    packed: list[bytes] = []
    total = 0
    failed = 0
    for reaction_smiles in iter_first_column_reactions(csv_path):
        total += 1
        try:
            reaction = smiles(reaction_smiles)
            reaction.canonicalize()
            reactions.append(reaction)
            packed.append(reaction.pack())
        except Exception as exc:
            failed += 1
            print(f"Warning: Could not parse {name} reaction #{total}: {exc}")
    return DatasetBundle(name=name, path=csv_path, reactions=reactions, packed=packed, total=total, failed=failed)


def build_training_model(masking_rate: float, learning_rate: float, dropout: float) -> Model:
    return Model(
        masking_rate=masking_rate,
        optimizer=partial(AdamW, lr=learning_rate),
        dropout=dropout,
    )


def build_finetune_model(masking_rate: float, finetune_learning_rate: float) -> Model:
    model = Model.pretrained()
    model.masking_rate = masking_rate
    model.optimizer = partial(AdamW, lr=finetune_learning_rate)
    return model


class SupervisedMappingModel(Model):
    """Model with supervised mapping loss in addition to MLM loss."""
    
    def __init__(self, *args, mlm_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlm_weight = mlm_weight
        self._reactions = None
    
    def set_reactions(self, reactions: list[Any]):
        """Store reactions for supervised mapping loss computation."""
        self._reactions = reactions
    
    def prepare_dataloader_supervised(self, reactions: list[Any], packed: list[bytes], **kwargs):
        """Prepare dataloader that includes ground truth mappings."""
        self._reactions = reactions
        
        def collate_with_mappings(batch_items):
            from chytorch.utils.data import collate_reactions
            # Get standard batch tensors
            atoms, neighbors, distances, roles = collate_reactions(batch_items)
            
            # Extract ground truth mappings for this batch
            batch_mappings = []
            for item_idx in range(len(batch_items)):
                reaction = reactions[item_idx] if item_idx < len(reactions) else None
                if reaction:
                    # Extract reactant and product atom mappings
                    reactant_maps = [n for m in reaction.reactants for n in m]
                    product_maps = [n for m in reaction.products for n in m]
                    batch_mappings.append((reactant_maps, product_maps))
                else:
                    batch_mappings.append(([], []))
            
            return atoms, neighbors, distances, roles, batch_mappings
        
        ds = ReactionDataset(packed, distance_cutoff=self.encoder.max_distance, unpack=True)
        return DataLoader(ds, collate_fn=collate_with_mappings, **kwargs)
    
    def compute_mapping_loss(self, embeddings: Tensor, atoms: Tensor, roles: Tensor, 
                            batch_mappings: list[tuple[list[int], list[int]]]) -> Tensor:
        """Compute supervised mapping loss from ground truth."""
        total_loss = torch.tensor(0.0, device=embeddings.device)
        num_valid = 0
        
        batch_size = embeddings.shape[0]
        for i in range(batch_size):
            reactant_maps, product_maps = batch_mappings[i]
            if not reactant_maps or not product_maps:
                continue
            
            role_tokens = roles[i]
            embedding = embeddings[i]
            atom_tokens = atoms[i]
            
            # Extract reactant and product indices
            reactant_idx = torch.where(role_tokens == 2)[0]
            product_idx = torch.where(role_tokens == 3)[0]
            
            if len(reactant_maps) != len(reactant_idx) or len(product_maps) != len(product_idx):
                continue
            
            # Build ground truth mapping matrix
            # gt_mappings[i, j] = 1 if reactant i maps to product j
            gt_mappings = torch.zeros(len(reactant_idx), len(product_idx), device=embeddings.device)
            for r_idx, r_map in enumerate(reactant_maps):
                if r_map > 0:  # 0 means unmapped
                    for p_idx, p_map in enumerate(product_maps):
                        if p_map == r_map:
                            # Also check atom types match
                            r_atom = atom_tokens[reactant_idx[r_idx]]
                            p_atom = atom_tokens[product_idx[p_idx]]
                            if r_atom == p_atom:
                                gt_mappings[r_idx, p_idx] = 1.0
            
            # Skip if no valid mappings
            if gt_mappings.sum() == 0:
                continue
            
            # Compute similarity matrix (cosine similarity)
            r_emb = normalize(embedding[reactant_idx], dim=-1)
            p_emb = normalize(embedding[product_idx], dim=-1)
            similarity = torch.mm(r_emb, p_emb.t())  # [n_reactants, n_products]
            
            # Compute cross-entropy loss for each reactant atom
            # We want high similarity for correct mappings
            for r_idx in range(len(reactant_idx)):
                gt_product = gt_mappings[r_idx]
                if gt_product.sum() > 0:
                    # Normalize to get probability distribution
                    gt_product = gt_product / gt_product.sum()
                    # Use log_softmax and compute negative log likelihood
                    log_probs = torch.log_softmax(similarity[r_idx], dim=0)
                    loss = -(gt_product * log_probs).sum()
                    total_loss = total_loss + loss
                    num_valid += 1
        
        if num_valid > 0:
            return total_loss / num_valid
        return torch.tensor(0.0, device=embeddings.device)
    
    def training_step(self, batch, batch_idx):
        if len(batch) == 5:
            # Supervised mode with mappings
            a, n, d, r, batch_mappings = batch
        else:
            # Fallback to MLM-only mode
            a, n, d, r = batch
            batch_mappings = None
        
        m = r > 1  # atoms only
        
        # MLM loss (same as parent)
        ma = a.masked_fill((rand(a.shape, device=a.device) < self.masking_rate) & m, 2)
        mn = n.masked_fill((rand(n.shape, device=n.device) < self.masking_rate) & m, 1)
        
        x = self.encoder((ma, mn, d, r))[m]
        atoms = self.mlma(x)
        neighbors = self.mlmn(x)
        
        mlm_loss_a = cross_entropy(atoms, a[m].long() - 3)
        mlm_loss_n = cross_entropy(neighbors, n[m].long() - 2)
        mlm_loss = mlm_loss_a + mlm_loss_n
        
        self.log("trn_loss_mlm_a", mlm_loss_a.item(), sync_dist=True)
        self.log("trn_loss_mlm_n", mlm_loss_n.item(), sync_dist=True)
        self.log("trn_loss_mlm", mlm_loss.item(), sync_dist=True)
        
        # Supervised mapping loss
        if batch_mappings is not None:
            # Get clean embeddings (no masking for mapping)
            clean_embeddings = self.encoder((a, n, d, r))
            mapping_loss = self.compute_mapping_loss(clean_embeddings, a, r, batch_mappings)
            
            self.log("trn_loss_mapping", mapping_loss.item(), sync_dist=True)
            
            # Weighted combination
            total_loss = mapping_loss + self.mlm_weight * mlm_loss
            self.log("trn_loss_tot", total_loss.item(), sync_dist=True)
            return total_loss
        else:
            # MLM only
            self.log("trn_loss_tot", mlm_loss.item(), sync_dist=True)
            return mlm_loss


def build_supervised_finetune_model(
    masking_rate: float, 
    finetune_learning_rate: float,
    mlm_weight: float = 0.1
) -> SupervisedMappingModel:
    """Build a model with supervised mapping loss."""
    # Load pretrained weights into supervised model
    base_model = Model.pretrained()
    
    # Create supervised model with same architecture
    supervised_model = SupervisedMappingModel(
        masking_rate=masking_rate,
        mlm_weight=mlm_weight,
    )
    
    # Copy pretrained weights
    supervised_model.load_state_dict(base_model.state_dict(), strict=False)
    
    # Set optimizer
    supervised_model.optimizer = partial(AdamW, lr=finetune_learning_rate)
    
    return supervised_model


def evaluate_mlm_metrics(
        model: Model, packed_reactions: list[bytes], batch_size: int, mask_seed: int = 0
) -> dict[str, float]:
    dataloader = model.prepare_dataloader(packed_reactions, batch_size=batch_size, shuffle=False)
    model.eval()
    rng = torch.Generator()
    rng.manual_seed(mask_seed)

    atom_total = 0
    atom_correct = 0
    neighbor_total = 0
    neighbor_correct = 0
    atom_loss_sum = 0.0
    neighbor_loss_sum = 0.0

    with torch.no_grad():
        for a, n, d, r in dataloader:
            atom_mask = r > 1
            random_atoms = torch.rand(a.shape, generator=rng, device=a.device)
            random_neighbors = torch.rand(n.shape, generator=rng, device=n.device)
            masked_atoms = a.masked_fill((random_atoms < model.masking_rate) & atom_mask, 2)
            masked_neighbors = n.masked_fill((random_neighbors < model.masking_rate) & atom_mask, 1)

            embedding = model.encoder((masked_atoms, masked_neighbors, d, r))[atom_mask]
            atom_logits = model.mlma(embedding)
            neighbor_logits = model.mlmn(embedding)
            atom_target = a[atom_mask].long() - 3
            neighbor_target = n[atom_mask].long() - 2

            atom_loss = cross_entropy(atom_logits, atom_target)
            neighbor_loss = cross_entropy(neighbor_logits, neighbor_target)

            atom_count = atom_target.numel()
            atom_total += atom_count
            neighbor_total += atom_count
            atom_correct += (atom_logits.argmax(dim=-1) == atom_target).sum().item()
            neighbor_correct += (neighbor_logits.argmax(dim=-1) == neighbor_target).sum().item()
            atom_loss_sum += atom_loss.item() * atom_count
            neighbor_loss_sum += neighbor_loss.item() * atom_count

    atom_loss_value = atom_loss_sum / atom_total
    neighbor_loss_value = neighbor_loss_sum / neighbor_total
    total_loss_value = atom_loss_value + neighbor_loss_value
    return {
        "mlm_loss_total": total_loss_value,
        "mlm_loss_atom": atom_loss_value,
        "mlm_loss_neighbor": neighbor_loss_value,
        "mlm_atom_accuracy": atom_correct / atom_total,
        "mlm_neighbor_accuracy": neighbor_correct / neighbor_total,
        "mlm_perplexity": float(torch.exp(torch.tensor(total_loss_value)).item()),
    }


def _greedy_assignment(similarity: torch.Tensor) -> tuple[list[int], list[float]]:
    n_product, n_reactant = similarity.shape
    work = similarity.clone()
    assigned = [-1] * n_product
    scores = [float("nan")] * n_product

    for _ in range(min(n_product, n_reactant)):
        best_score, flat_index = torch.max(work.reshape(-1), dim=0)
        if not torch.isfinite(best_score) or best_score.item() <= -1e8:
            break
        product_idx = int(flat_index.item() // n_reactant)
        reactant_idx = int(flat_index.item() % n_reactant)
        assigned[product_idx] = reactant_idx
        scores[product_idx] = float(best_score.item())
        work[product_idx, :] = -1e9
        work[:, reactant_idx] = -1e9
    return assigned, scores


def evaluate_mapping_metrics(model: Model, dataset: DatasetBundle, top_k: int = 3, batch_size: int = 64) -> dict[str, float]:
    """
    Evaluate mapping metrics on dataset using PyTorch dataloader for batched inference.
    
    Uses standard PyTorch batching for efficient GPU utilization.
    """
    model.eval()

    mapped_reactions = 0
    exact_matches = 0
    mappable_atoms = 0
    correct_atoms = 0
    top1_hits = 0
    topk_hits = 0
    assigned_atoms = 0
    score_sum = 0.0
    score_count = 0

    with torch.no_grad():
        # Use PyTorch dataloader for efficient batching
        dataloader = model.prepare_dataloader(dataset.packed, batch_size=batch_size, shuffle=False)
        
        # Track which reaction we're on (since dataloader batches are independent of dataset.reactions)
        reaction_idx = 0
        
        for batch in dataloader:
            atoms, neighbors, distances, roles = batch
            embeddings = model(batch)
            batch_size_actual = atoms.shape[0]

            # Process each reaction in the batch
            for i in range(batch_size_actual):
                if reaction_idx >= len(dataset.reactions):
                    break
                
                reaction = dataset.reactions[reaction_idx]
                reaction_idx += 1
                
                atom_tokens = atoms[i]
                role_tokens = roles[i]
                embedding = embeddings[i]

                # Extract reactant and product token indices
                reactant_token_idx = torch.where(role_tokens == 2)[0]
                product_token_idx = torch.where(role_tokens == 3)[0]
                if reactant_token_idx.numel() == 0 or product_token_idx.numel() == 0:
                    continue

                # Get ground truth mappings from reaction
                reactant_maps = [n for m in reaction.reactants for n in m]
                product_maps = [n for m in reaction.products for n in m]
                if len(reactant_maps) != reactant_token_idx.numel() or len(product_maps) != product_token_idx.numel():
                    continue

                # Compute similarity matrix
                reactant_map_to_local = {map_num: idx for idx, map_num in enumerate(reactant_maps)}
                reactant_embeddings = normalize(embedding[reactant_token_idx], dim=-1)
                product_embeddings = normalize(embedding[product_token_idx], dim=-1)
                similarity = product_embeddings @ reactant_embeddings.T

                # Mask out different atom types
                reactant_types = atom_tokens[reactant_token_idx]
                product_types = atom_tokens[product_token_idx]
                same_atom_type = product_types[:, None] == reactant_types[None, :]
                similarity = similarity.masked_fill(~same_atom_type, -1e9)

                # Greedy assignment
                assigned, assigned_scores = _greedy_assignment(similarity)

                # Evaluate metrics for this reaction
                has_mappable_atom = False
                reaction_is_exact = True
                for product_local_idx, product_map_num in enumerate(product_maps):
                    reactant_local_idx = reactant_map_to_local.get(product_map_num)
                    if reactant_local_idx is None:
                        continue
                    has_mappable_atom = True
                    mappable_atoms += 1

                    # Top-k accuracy
                    ranked_indices = torch.topk(
                        similarity[product_local_idx],
                        k=min(top_k, similarity.shape[1]),
                    ).indices.tolist()
                    if ranked_indices and ranked_indices[0] == reactant_local_idx:
                        top1_hits += 1
                    if reactant_local_idx in ranked_indices:
                        topk_hits += 1

                    # Assignment accuracy
                    predicted_local_idx = assigned[product_local_idx]
                    if predicted_local_idx != -1:
                        assigned_atoms += 1
                        score_sum += assigned_scores[product_local_idx]
                        score_count += 1
                    if predicted_local_idx == reactant_local_idx:
                        correct_atoms += 1
                    else:
                        reaction_is_exact = False

                if has_mappable_atom:
                    mapped_reactions += 1
                    if reaction_is_exact:
                        exact_matches += 1

    if mappable_atoms == 0:
        return {
            "mapping_atom_accuracy": 0.0,
            "mapping_exact_match": 0.0,
            "mapping_top1": 0.0,
            "mapping_topk": 0.0,
            "mapping_assignment_coverage": 0.0,
            "mapping_mean_similarity": 0.0,
        }

    return {
        "mapping_atom_accuracy": correct_atoms / mappable_atoms,
        "mapping_exact_match": exact_matches / mapped_reactions if mapped_reactions else 0.0,
        "mapping_top1": top1_hits / mappable_atoms,
        "mapping_topk": topk_hits / mappable_atoms,
        "mapping_assignment_coverage": assigned_atoms / mappable_atoms,
        "mapping_mean_similarity": score_sum / score_count if score_count else 0.0,
    }


def evaluate_model(
        model: Model, dataset: DatasetBundle, batch_size: int, mask_seed: int = 0
) -> dict[str, float]:
    mlm_metrics = evaluate_mlm_metrics(model, dataset.packed, batch_size=batch_size, mask_seed=mask_seed)
    mapping_metrics = evaluate_mapping_metrics(model, dataset)
    return {**mlm_metrics, **mapping_metrics}


def run_training_experiment(
        model: Model,
        train_dataset: DatasetBundle,
        test_dataset: DatasetBundle,
        batch_size: int,
        max_epochs: int,
        seed: int,
        run_name: str = "rxnmap_training",
        use_aim: bool = False,
        aim_experiment: str | None = None,
        use_supervised_loss: bool = False,
) -> dict[str, float | str]:
    seed_everything(seed, workers=True)
    
    # Use supervised dataloader if model supports it
    if use_supervised_loss and isinstance(model, SupervisedMappingModel):
        train_loader = model.prepare_dataloader_supervised(
            train_dataset.reactions, train_dataset.packed, 
            batch_size=batch_size, shuffle=True
        )
    else:
        train_loader = model.prepare_dataloader(
            train_dataset.packed, batch_size=batch_size, shuffle=True
        )

    loggers = [CSVLogger("lightning_logs", name=run_name)]
    if use_aim:
        try:
            from aim.pytorch_lightning import AimLogger
            aim_logger = AimLogger(
                experiment=aim_experiment or run_name,
                train_metric_prefix='train/',
                val_metric_prefix='val/',
                test_metric_prefix='test/',
            )
            loggers.append(aim_logger)
        except Exception as e:
            print(f"Warning: aim logging unavailable ({type(e).__name__}: {e}), using CSV logger only")

    progress_bar = TQDMProgressBar(refresh_rate=10)
    
    # Auto-detect GPU if available, fallback to CPU
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        precision="32",
        max_epochs=max_epochs,
        logger=loggers,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        callbacks=[progress_bar],
    )
    model.train()
    trainer.fit(model, train_dataloaders=train_loader)

    test_metrics = evaluate_model(model, test_dataset, batch_size=batch_size, mask_seed=seed)

    csv_logger = loggers[0]
    last_checkpoint = Path(csv_logger.log_dir) / "checkpoints" / "last.ckpt"
    return {
        **test_metrics,
        "log_dir": csv_logger.log_dir,
        "last_checkpoint": str(last_checkpoint) if last_checkpoint.exists() else "",
    }


def run_scratch_experiment(
        train_dataset: DatasetBundle,
        test_dataset: DatasetBundle,
        batch_size: int,
        max_epochs: int,
        seed: int,
        masking_rate: float,
        learning_rate: float,
        dropout: float,
        run_name: str = "rxnmap_training",
        use_aim: bool = False,
        aim_experiment: str | None = None,
) -> dict[str, float | str]:
    model = build_training_model(masking_rate, learning_rate, dropout)
    return run_training_experiment(
        model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        max_epochs=max_epochs,
        seed=seed,
        run_name=run_name,
        use_aim=use_aim,
        aim_experiment=aim_experiment,
    )


def run_finetune_experiment(
        train_dataset: DatasetBundle,
        test_dataset: DatasetBundle,
        batch_size: int,
        max_epochs: int,
        seed: int,
        masking_rate: float,
        finetune_learning_rate: float,
        run_name: str = "rxnmap_finetune",
        use_aim: bool = False,
        aim_experiment: str | None = None,
        use_supervised_loss: bool = False,
        mlm_weight: float = 0.1,
) -> dict[str, float | str]:
    if use_supervised_loss:
        model = build_supervised_finetune_model(
            masking_rate, finetune_learning_rate, mlm_weight=mlm_weight
        )
    else:
        model = build_finetune_model(masking_rate, finetune_learning_rate)
    
    return run_training_experiment(
        model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        max_epochs=max_epochs,
        seed=seed,
        run_name=run_name,
        use_aim=use_aim,
        aim_experiment=aim_experiment,
        use_supervised_loss=use_supervised_loss,
    )


def print_dataset_summary(dataset: DatasetBundle):
    print(
        f"{dataset.name}: total={dataset.total}, packed={len(dataset.packed)}, failed={dataset.failed}, source={dataset.path}"
    )


def print_metrics(label: str, metrics: dict[str, float | str]):
    print(f"{label}:")
    for key in (
            "mlm_loss_total",
            "mlm_loss_atom",
            "mlm_loss_neighbor",
            "mlm_atom_accuracy",
            "mlm_neighbor_accuracy",
            "mlm_perplexity",
            "mapping_atom_accuracy",
            "mapping_exact_match",
            "mapping_top1",
            "mapping_topk",
            "mapping_assignment_coverage",
            "mapping_mean_similarity",
    ):
        if key in metrics:
            print(f"  {key}: {metrics[key]:.6f}")
    if "last_checkpoint" in metrics:
        print(f"  last_checkpoint: {metrics['last_checkpoint'] or 'not found'}")
    if "log_dir" in metrics:
        print(f"  log_dir: {metrics['log_dir']}")


def main(
        train: str = "train_ringreactions.csv",
        test: str = "test_ringreactions.csv",
        batch_size: int = 16,
        max_epochs: int = 1,
        seed: int = 42,
        masking_rate: float = 0.15,
        learning_rate: float = 1e-4,
        finetune_learning_rate: float = 1e-5,
        dropout: float = 0.1,
        use_aim: bool = False,
        aim_experiment: str | None = None,
):
    train_dataset = load_reaction_dataset(Path(train), name="train")
    test_dataset = load_reaction_dataset(Path(test), name="test")
    print_dataset_summary(train_dataset)
    print_dataset_summary(test_dataset)

    if not train_dataset.packed or not test_dataset.packed:
        raise RuntimeError("No valid reactions available after parsing train/test datasets.")

    seed_everything(seed, workers=True)
    baseline_model = Model.pretrained()
    baseline_metrics = evaluate_model(
        baseline_model, test_dataset, batch_size=batch_size, mask_seed=seed
    )
    print_metrics("pretrained_baseline_on_test", baseline_metrics)

    print(
        f"Training config: epochs={max_epochs}, batch_size={batch_size}, masking_rate={masking_rate}, lr={learning_rate}, finetune_lr={finetune_learning_rate}, dropout={dropout}, use_aim={use_aim}"
    )
    scratch_metrics = run_scratch_experiment(
        train_dataset,
        test_dataset,
        batch_size=batch_size,
        max_epochs=max_epochs,
        seed=seed,
        masking_rate=masking_rate,
        learning_rate=learning_rate,
        dropout=dropout,
        use_aim=use_aim,
        aim_experiment=aim_experiment,
    )
    print_metrics("scratch_trained_model_on_test", scratch_metrics)

    finetuned_metrics = run_finetune_experiment(
        train_dataset,
        test_dataset,
        batch_size=batch_size,
        max_epochs=max_epochs,
        seed=seed,
        masking_rate=masking_rate,
        finetune_learning_rate=finetune_learning_rate,
        use_aim=use_aim,
        aim_experiment=aim_experiment,
    )
    print_metrics("finetuned_pretrained_model_on_test", finetuned_metrics)

    print("=" * 60)
    print("DELTA (scratch - pretrained)")
    print("=" * 60)
    print(
        f"mlm_loss_total: {scratch_metrics['mlm_loss_total'] - baseline_metrics['mlm_loss_total']:+.6f}"
    )
    print(
        f"mapping_atom_accuracy: {scratch_metrics['mapping_atom_accuracy'] - baseline_metrics['mapping_atom_accuracy']:+.6f}"
    )
    print(
        f"mapping_exact_match: {scratch_metrics['mapping_exact_match'] - baseline_metrics['mapping_exact_match']:+.6f}"
    )
    print("=" * 60)
    print("DELTA (finetuned - pretrained)")
    print("=" * 60)
    print(
        f"mlm_loss_total: {finetuned_metrics['mlm_loss_total'] - baseline_metrics['mlm_loss_total']:+.6f}"
    )
    print(
        f"mapping_atom_accuracy: {finetuned_metrics['mapping_atom_accuracy'] - baseline_metrics['mapping_atom_accuracy']:+.6f}"
    )
    print(
        f"mapping_exact_match: {finetuned_metrics['mapping_exact_match'] - baseline_metrics['mapping_exact_match']:+.6f}"
    )


if __name__ == "__main__":
    fire.Fire(main)
