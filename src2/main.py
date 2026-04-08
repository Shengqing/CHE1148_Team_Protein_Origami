import argparse
import torch
from pathlib import Path
from src2.train_eval import train_and_validate, run_unified_evaluation
from src2.constants import NUM_EPOCHS, PATIENCE

def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate Protein VAE")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training .pt file")
    parser.add_argument("--train_mmap", type=str, required=True, help="Path to training .dat memmap file")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation .pt file")
    parser.add_argument("--val_mmap", type=str, required=True, help="Path to validation .dat memmap file")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    
    args = parser.parse_args()

    # Define minimal config
    config = {
        "BATCH_SIZE": 128,
        "LATENT_DIM": 64,
        "HIDDEN_DIM": 256,
        "LEARNING_RATE": 1e-3,
        "WEIGHT_DECAY": 1e-4
    }

    print("Starting training pipeline...")
    model, train_ds, val_ds = train_and_validate(
        config=config,
        train_path=args.train_path,
        train_mmap=args.train_mmap,
        val_path=args.val_path,
        val_mmap=args.val_mmap,
        epochs=args.epochs,
        patience=PATIENCE
    )

    print("\nStarting evaluation pipeline...")
    # NOTE: run_unified_evaluation expects a val_loader and a master_lookup dictionary.
    # We will create a dummy master_lookup or you should load your actual sequence:deltaG dict here.
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_ds, batch_size=config["BATCH_SIZE"], shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Placeholder for database lookup used in notebook
    master_lookup = {} 
    
    run_unified_evaluation(
        model=model,
        val_loader=val_loader,
        val_ds=val_ds,
        train_ds_len=len(train_ds),
        val_path=args.val_path,
        device=device,
        master_lookup=master_lookup,
        num_samples=10, # Keep small for quick testing
        n_bootstraps=3
    )

if __name__ == "__main__":
    main()
