#!/usr/bin/env python3
"""Train a small binary CNN for synthetic nanowire crops."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Pure (torch-free) crop preprocessing lives in nanowire_ml.crops so the rest of
# the pipeline can use it without installing torch. Re-exported here for the
# many call sites that import these from train_classifier.
from nanowire_ml.crops import (  # noqa: F401
    LABEL_TO_TARGET,
    pca_align_mask,
    pca_align_soft_gray,
    preprocess_image,
    robust_normalize_gray,
)


class NanowireDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str, augment: bool = False, preprocess_mode: str = "raw"):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.augment = augment
        self.preprocess_mode = preprocess_mode
        self.samples = []
        for label in ["bad", "single"]:
            for path in sorted((self.dataset_dir / split / label).glob("*.png")):
                self.samples.append((path, LABEL_TO_TARGET[label], label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read {path}")
        img = preprocess_image(img, self.preprocess_mode)
        if self.augment:
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 1)
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 0)
            k = np.random.randint(0, 4)
            if k:
                img = np.rot90(img, k).copy()
            img = np.clip(img.astype(np.float32) + np.random.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
        x = torch.from_numpy(img.astype(np.float32)[None, :, :] / 255.0)
        y = torch.tensor([target], dtype=torch.float32)
        return x, y, str(path), label


class SmallNanowireCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def metrics_from_logits(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
    y = targets.detach().cpu().numpy().ravel().astype(int)
    pred = (probs >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {"accuracy": accuracy, "precision_single": precision, "recall_single": recall,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def run_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    logits_all = []
    targets_all = []
    for x, y, _, _ in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        logits_all.append(logits.detach())
        targets_all.append(y.detach())
    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    return total_loss / len(loader.dataset), metrics_from_logits(logits, targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="experimental_sem_results/nanowire_ml_dataset")
    parser.add_argument("--output-dir", default="experimental_sem_results/nanowire_ml_model")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--preprocess-mode", choices=("raw", "pca_mask", "soft_gray_pca"), default="raw")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = NanowireDataset(Path(args.dataset_dir), "train", augment=True, preprocess_mode=args.preprocess_mode)
    val_ds = NanowireDataset(Path(args.dataset_dir), "val", augment=False, preprocess_mode=args.preprocess_mode)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SmallNanowireCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, None, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        score = val_metrics["precision_single"] + val_metrics["recall_single"]
        if score > best_score:
            best_score = score
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "args": vars(args)}, out_dir / "best_model.pt")
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.3f} val_precision={val_metrics['precision_single']:.3f} "
            f"val_recall={val_metrics['recall_single']:.3f}"
        )

    with (out_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(out_dir / "best_model.pt")


if __name__ == "__main__":
    main()
