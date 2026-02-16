from chython import smiles
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from chytorch.zoo.rxnmap import Model

# Create a more realistic dataset with more examples
data = []
reactions = [
    "CCO.CC(=O)O>>CCOC(=O)C.O",
    "C.O>>CO",
    "CC(C)O.CCO>>CCOCC",
    "C1=CC=CC=C1.CCBr>>C1=CC=CC=C1CC",
    "CC(C)=CCBr.CC(C)>>CC(C)C(C)C",
    "c1ccccc1O.CCBr>>c1ccccc1OCC",
    "CC(=O)Cl.c1ccccc1>>CC(=O)c1ccccc1",
    "c1ccccc1Cl.c1ccccc1>>c1ccccc1c2ccccc2",
    "CC(C)Br.CC(=O)O>>CC(C)OC(=O)C",
    "c1ccccc1Br.c1ccccc1>>c1ccccc1c2ccccc2",
] * 5  # Repeat to get 50 examples

print(f"Loading {len(reactions)} reactions...")
for smiles_str in reactions:
    try:
        r = smiles(smiles_str)
        r.canonicalize()
        data.append(r.pack())
    except Exception as e:
        print(f"Warning: Could not parse {smiles_str}: {e}")

print(f"Loaded {len(data)} reactions successfully\n")

# Load pretrained model for inference demo
model = Model.pretrained()
model.eval()

# Create dataloader for inference demo
dl_inference = model.prepare_dataloader(data, batch_size=10)

print("=" * 60)
print("INFERENCE DEMO - Using Pretrained Model")
print("=" * 60)
for i, batch in enumerate(dl_inference):
    embeddings = model(batch)
    print(
        f"Batch {i}: shape={embeddings.shape}, min={embeddings.min():.4f}, max={embeddings.max():.4f}, mean={embeddings.mean():.4f}"
    )
    if i >= 2:  # Just show first 3 batches
        break

print("\n" + "=" * 60)
print("TRAINING - 50 Epochs on Synthetic Data")
print("=" * 60 + "\n")

# Setup training callbacks
checkpoint_callback = ModelCheckpoint(
    monitor="trn_loss_tot",
    dirpath="checkpoints",
    filename="{epoch:02d}-{trn_loss_tot:.2f}",
    save_weights_only=True,
    save_top_k=3,
    save_last=True,
    verbose=True,
)

# Use CSVLogger for metrics tracking
logger = CSVLogger("lightning_logs", name="rxnmap_training")

trainer = Trainer(
    accelerator="cpu",
    devices=1,
    precision="32",
    max_epochs=50,
    callbacks=[checkpoint_callback],
    logger=logger,
    log_every_n_steps=1,
    enable_progress_bar=True,
    num_sanity_val_steps=0,  # Skip validation checks for faster training
)

# Create a fresh model for training
training_model = Model(masking_rate=0.15)

print(f"Starting training on {len(data)} reactions with batch size 10...")
print(f"Training steps per epoch: {len(data) // 10}\n")

trainer.fit(training_model, dl_inference)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
if checkpoint_callback.best_model_score is not None:
    print(f"Best loss: {checkpoint_callback.best_model_score:.4f}")
print(f"Total epochs completed: {trainer.current_epoch + 1}")
print(f"\nMetrics logged to: lightning_logs/rxnmap_training/")
print("To view metrics, check: lightning_logs/rxnmap_training/version_*/metrics.csv")
