import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

from src.constants import NUM_EPOCHS, PATIENCE
from src.vae_train import run_unified_evaluation, train_and_validate


def main():
    parser = argparse.ArgumentParser(
        description="Train and Evaluate Protein VAE"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/processed/train_sampled_esm2_650m.pt",
        help="Path to training .pt file",
    )
    parser.add_argument(
        "--train_mmap",
        type=str,
        default="data/processed/train_sampled_esm2_650m.dat",
        help="Path to training .dat memmap file",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/processed/val_full_esm2_650m.pt",
        help="Path to validation .pt file",
    )
    parser.add_argument(
        "--val_mmap",
        type=str,
        default="data/processed/val_full_esm2_650m.dat",
        help="Path to validation .dat memmap file",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/processed/all_sequences.csv",
        help="Path to all_sequences.csv file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of training epochs",
    )

    args = parser.parse_args()

    # Define minimal config
    config = {
        "BATCH_SIZE": 128,
        "LATENT_DIM": 256,
        "HIDDEN_DIM": 512,
        "LEARNING_RATE": 8e-5,
        "WEIGHT_DECAY": 2e-6,
        "SEQ_WEIGHT": 1.0,
        "KL_WEIGHT": 0.2,
    }

    import gdown
    import pandas as pd

    from src.utils import convert_to_memmap

    Path(args.train_path).parent.mkdir(parents=True, exist_ok=True)

    if not Path(args.train_path).exists():
        print(f"Downloading training data to {args.train_path}...")
        gdown.download(
            id="1hT51HoLUSZohAl4CRIRopnh0gqnj8Pz4",
            output=args.train_path,
            quiet=False,
            fuzzy=True,
        )

    if not Path(args.val_path).exists():
        print(f"Downloading validation data to {args.val_path}...")
        gdown.download(
            id="1H0lC_7G0cwr8hB0dMyrWf7tcEUB8oB4e",
            output=args.val_path,
            quiet=False,
            fuzzy=True,
        )

    if not Path(args.csv_path).exists():
        print(f"Downloading sequences csv to {args.csv_path}...")
        gdown.download(
            id="13j3-jBOsLweSQhU-7qcApWfcLyUNo30FjrMSZUxVMMo",
            output=args.csv_path,
            quiet=False,
            fuzzy=True,
        )

    if not Path(args.train_mmap).exists():
        print(f"Creating memory map for training data at {args.train_mmap}...")
        convert_to_memmap(args.train_path, args.train_mmap)

    if not Path(args.val_mmap).exists():
        print(f"Creating memory map for validation data at {args.val_mmap}...")
        convert_to_memmap(args.val_path, args.val_mmap)

    print("Starting training pipeline...")
    model, train_ds, val_ds = train_and_validate(
        config=config,
        train_path=args.train_path,
        train_mmap=args.train_mmap,
        val_path=args.val_path,
        val_mmap=args.val_mmap,
        epochs=args.epochs,
        patience=PATIENCE,
    )

    print("\nStarting evaluation pipeline...")
    from torch.utils.data import DataLoader

    val_loader = DataLoader(
        val_ds, batch_size=config["BATCH_SIZE"], shuffle=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load database lookup used in notebook
    try:
        master_df = pd.read_csv(args.csv_path)
        master_lookup = dict(zip(master_df["aa_seq"], master_df["deltaG"]))
    except Exception as e:
        print(f"Could not load sequences CSV due to {e}. Using empty lookup.")
        master_lookup = {}

    run_unified_evaluation(
        model=model,
        val_loader=val_loader,
        val_ds=val_ds,
        train_ds_len=len(train_ds),
        val_path=args.val_path,
        device=device,
        master_lookup=master_lookup,
        num_samples=1000,
        n_bootstraps=10,
    )


if __name__ == "__main__":
    main()
