"""Train the U-Net garimpo detector."""

import argparse
import csv
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from garimpo.augmentation import JointAugmentation
from garimpo.datasets import GarimpoDataset
from garimpo.losses import WeightedBCEDiceLoss, compute_pos_weight
from garimpo.model import build_unet
from garimpo.splits import make_scene_split

REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)

CSV_FIELDS = [
    "epoch",
    "steps",
    "train_bce",
    "train_dice_loss",
    "train_combined",
    "train_dice",
    "train_iou",
    "val_bce",
    "val_dice_loss",
    "val_combined",
    "val_dice",
    "val_iou",
    "epoch_seconds",
]


def build_train_sampler(train_filenames: list[str], labels_csv: Path, generator: torch.Generator) -> WeightedRandomSampler:
    labels = pd.read_csv(labels_csv)
    label_by_filename = dict(zip(labels["filename"], labels["label"]))
    is_positive = np.array([label_by_filename[f] == "positive" for f in train_filenames])

    n_positive = int(is_positive.sum())
    n_negative = len(is_positive) - n_positive
    weights = np.where(is_positive, 1.0 / n_positive, 1.0 / n_negative)

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_filenames),
        replacement=True,
        generator=generator,
    )


class EpochStats:
    """Accumulates loss sums and global TP/FP/FN over an epoch."""

    def __init__(self) -> None:
        self.n_samples = 0
        self.sum_bce = 0.0
        self.sum_dice_loss = 0.0
        self.sum_combined = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, combined: torch.Tensor, bce: torch.Tensor, dice: torch.Tensor, logits: torch.Tensor, target: torch.Tensor) -> None:
        batch_size = target.shape[0]
        self.n_samples += batch_size
        self.sum_bce += bce.item() * batch_size
        self.sum_dice_loss += dice.item() * batch_size
        self.sum_combined += combined.item() * batch_size

        pred = torch.sigmoid(logits) > 0.5
        truth = target > 0.5
        self.tp += int((pred & truth).sum().item())
        self.fp += int((pred & ~truth).sum().item())
        self.fn += int((~pred & truth).sum().item())

    def summary(self, eps: float = 1e-7) -> dict[str, float]:
        dice = (2 * self.tp + eps) / (2 * self.tp + self.fp + self.fn + eps)
        iou = (self.tp + eps) / (self.tp + self.fp + self.fn + eps)
        return {
            "bce": self.sum_bce / self.n_samples,
            "dice_loss": self.sum_dice_loss / self.n_samples,
            "combined": self.sum_combined / self.n_samples,
            "dice": dice,
            "iou": iou,
        }


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    stats = EpochStats()

    with torch.set_grad_enabled(is_train):
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            combined, bce, dice = loss_fn(logits, masks)

            if is_train:
                optimizer.zero_grad()
                combined.backward()
                optimizer.step()

            stats.update(combined, bce, dice, logits, masks)

    return stats.summary()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles-dir", type=Path, default=REPO_ROOT / "data" / "tiles")
    parser.add_argument("--masks-dir", type=Path, default=REPO_ROOT / "data" / "masks")
    parser.add_argument("--labels-csv", type=Path, default=REPO_ROOT / "data" / "labels.csv")
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)

    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--max-bce-pos-weight", type=float, default=50.0)

    parser.add_argument("--augment", dest="augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--balanced-sampler", dest="balanced_sampler", action="store_true", default=True)
    parser.add_argument("--no-balanced-sampler", dest="balanced_sampler", action="store_false")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = args.output_dir or (REPO_ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_filenames, val_filenames = make_scene_split(args.labels_csv, args.val_fraction)

    train_transform = JointAugmentation(args.seed) if args.augment else None
    train_ds = GarimpoDataset(train_filenames, args.tiles_dir, args.masks_dir, transform=train_transform)
    val_ds = GarimpoDataset(val_filenames, args.tiles_dir, args.masks_dir, transform=None)

    generator = torch.Generator().manual_seed(args.seed)
    if args.balanced_sampler:
        sampler = build_train_sampler(train_filenames, args.labels_csv, generator)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=args.num_workers
        )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    steps_per_epoch = len(train_loader)
    logger.info(
        "Train: %d tiles, batch size %d -> %d steps/epoch. Val: %d tiles.",
        len(train_filenames), args.batch_size, steps_per_epoch, len(val_filenames),
    )
    logger.info("Augmentation: %s | Balanced sampler: %s", args.augment, args.balanced_sampler)

    pos_weight = compute_pos_weight(train_filenames, args.masks_dir, args.max_bce_pos_weight)
    logger.info("BCE pos_weight (capped at %.1f): %.2f", args.max_bce_pos_weight, pos_weight)

    model = build_unet().to(device)
    loss_fn = WeightedBCEDiceLoss(pos_weight, args.bce_weight, args.dice_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    serializable_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    best_val_combined = float("inf")
    epochs_without_improvement = 0
    best_checkpoint_path = output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_metrics = run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        epoch_seconds = time.time() - t0

        logger.info(
            "Epoch %d/%d [%.1fs] | train: bce=%.4f dice_loss=%.4f combined=%.4f dice=%.4f iou=%.4f "
            "| val: bce=%.4f dice_loss=%.4f combined=%.4f dice=%.4f iou=%.4f",
            epoch, args.epochs, epoch_seconds,
            train_metrics["bce"], train_metrics["dice_loss"], train_metrics["combined"], train_metrics["dice"], train_metrics["iou"],
            val_metrics["bce"], val_metrics["dice_loss"], val_metrics["combined"], val_metrics["dice"], val_metrics["iou"],
        )

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writerow({
                "epoch": epoch,
                "steps": steps_per_epoch,
                "train_bce": train_metrics["bce"],
                "train_dice_loss": train_metrics["dice_loss"],
                "train_combined": train_metrics["combined"],
                "train_dice": train_metrics["dice"],
                "train_iou": train_metrics["iou"],
                "val_bce": val_metrics["bce"],
                "val_dice_loss": val_metrics["dice_loss"],
                "val_combined": val_metrics["combined"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "epoch_seconds": epoch_seconds,
            })

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_combined": val_metrics["combined"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
            },
            checkpoints_dir / f"epoch_{epoch:03d}.pt",
        )

        if val_metrics["combined"] < best_val_combined - args.min_delta:
            best_val_combined = val_metrics["combined"]
            epochs_without_improvement = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_combined": best_val_combined, "args": serializable_args},
                best_checkpoint_path,
            )
            logger.info("  -> new best checkpoint (val_combined=%.4f)", best_val_combined)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                logger.info("Early stopping: no improvement in %d epochs.", args.early_stopping_patience)
                break

    logger.info("Done. Best val_combined=%.4f. Checkpoint: %s. Metrics: %s", best_val_combined, best_checkpoint_path, csv_path)


if __name__ == "__main__":
    main()
