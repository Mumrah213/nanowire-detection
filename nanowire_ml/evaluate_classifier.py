#!/usr/bin/env python3
"""Evaluate nanowire CNN and write threshold/mistake sheets."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from nanowire_ml.train_classifier import NanowireDataset, SmallNanowireCNN, preprocess_image
from nanowire_ml.topology import topology_features, topology_veto_reasons


def load_model(checkpoint: Path, device):
    model = SmallNanowireCNN().to(device)
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def collect_predictions(model, loader, device, topology_veto=False, veto_args=None):
    rows = []
    with torch.no_grad():
        for x, y, paths, labels in loader:
            probs = torch.sigmoid(model(x.to(device))).cpu().numpy().ravel()
            targets = y.numpy().ravel().astype(int)
            for prob, target, path, label in zip(probs, targets, paths, labels):
                row = {"path": path, "label": label, "target": int(target), "prob_single": float(prob)}
                if topology_veto:
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    # Veto uses the original mask-like sample, not the model tensor after augmentation.
                    img = preprocess_image(img, "pca_mask") if veto_args.get("veto_aligned", False) else img
                    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    topo = topology_features(img)
                    reasons = topology_veto_reasons(
                        topo,
                        max_secondary_orientation=veto_args["max_secondary_orientation"],
                        max_off_line_fraction=veto_args["max_off_line_fraction"],
                        max_width_px=veto_args["max_width_px"],
                        max_branchpoints=veto_args["max_branchpoints"],
                    )
                    row.update({
                        **topo,
                        "topology_pass": not reasons,
                        "topology_veto_reasons": ",".join(reasons),
                    })
                rows.append(row)
    return rows


def metrics(rows, threshold, topology_veto=False):
    pred = np.array([
        row["prob_single"] >= threshold and (row.get("topology_pass", True) or not topology_veto)
        for row in rows
    ], dtype=bool)
    y = np.array([row["target"] == 1 for row in rows], dtype=bool)
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    tn = int((~pred & ~y).sum())
    fn = int((~pred & y).sum())
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / max(1, len(rows)),
        "precision_single": tp / max(1, tp + fp),
        "recall_single": tp / max(1, tp + fn),
        "false_positive_rate": fp / max(1, fp + tn),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def make_sheet(rows, out_path: Path, title_key="prob_single", limit=120):
    tile = 128
    label_h = 28
    cols = 8
    thumbs = []
    for row in rows[:limit]:
        img = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        color = (0, 255, 0) if row["target"] == 1 else (0, 0, 255)
        cv2.rectangle(rgb, (0, 0), (rgb.shape[1] - 1, rgb.shape[0] - 1), color, 2)
        canvas = np.full((tile + label_h, tile, 3), 24, dtype=np.uint8)
        canvas[:tile] = cv2.resize(rgb, (tile, tile), interpolation=cv2.INTER_AREA)
        cv2.putText(canvas, f"p={row[title_key]:.2f}", (3, tile + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, Path(row["path"]).name[:18], (3, tile + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)
        thumbs.append(canvas)
    rows_needed = max(1, int(np.ceil(len(thumbs) / cols)))
    sheet = np.full((rows_needed * (tile + label_h), cols * tile, 3), 20, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        sheet[r * (tile + label_h):(r + 1) * (tile + label_h), c * tile:(c + 1) * tile] = thumb
    cv2.imwrite(str(out_path), sheet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="experimental_sem_results/nanowire_ml_dataset")
    parser.add_argument("--checkpoint", default="experimental_sem_results/nanowire_ml_model/best_model.pt")
    parser.add_argument("--output-dir", default="experimental_sem_results/nanowire_ml_eval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--preprocess-mode", choices=("raw", "pca_mask", "soft_gray_pca"), default="raw")
    parser.add_argument("--topology-veto", action="store_true")
    parser.add_argument("--max-secondary-orientation", type=float, default=0.24)
    parser.add_argument("--max-off-line-fraction", type=float, default=0.22)
    parser.add_argument("--max-width-px", type=float, default=5.5)
    parser.add_argument("--max-branchpoints", type=int, default=18)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(Path(args.checkpoint), device)
    ds = NanowireDataset(Path(args.dataset_dir), args.split, augment=False, preprocess_mode=args.preprocess_mode)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    veto_args = {
        "max_secondary_orientation": args.max_secondary_orientation,
        "max_off_line_fraction": args.max_off_line_fraction,
        "max_width_px": args.max_width_px,
        "max_branchpoints": args.max_branchpoints,
        "veto_aligned": False,
    }
    rows = collect_predictions(model, loader, device, topology_veto=args.topology_veto, veto_args=veto_args)

    thresholds = [round(v, 2) for v in np.linspace(0.1, 0.95, 18)]
    metric_rows = [metrics(rows, threshold, topology_veto=args.topology_veto) for threshold in thresholds]
    with (out_dir / f"{args.split}_threshold_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)
    (out_dir / f"{args.split}_predictions.json").write_text(json.dumps(rows, indent=2))

    sorted_single = sorted([row for row in rows if row["target"] == 1], key=lambda row: row["prob_single"])
    sorted_bad = sorted([row for row in rows if row["target"] == 0], key=lambda row: -row["prob_single"])
    make_sheet(sorted_single, out_dir / f"{args.split}_lowest_scoring_singles.png")
    make_sheet(sorted_bad, out_dir / f"{args.split}_highest_scoring_bad.png")
    print(out_dir / f"{args.split}_threshold_metrics.csv")


if __name__ == "__main__":
    main()
