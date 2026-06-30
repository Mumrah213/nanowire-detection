#!/usr/bin/env python3
"""Run the synthetic nanowire CNN on real SEM connected-component crops."""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sem_preprocess import preprocess  # noqa: E402
from nanowire_ml.crops import preprocess_image  # noqa: E402
from nanowire_ml.topology import topology_features, topology_veto_reasons  # noqa: E402


def load_model(checkpoint: Path, device):
    # torch (and the model class) are imported lazily so this module can be
    # imported and the non-CNN candidate helpers used without torch installed.
    import torch

    from nanowire_ml.train_classifier import SmallNanowireCNN

    model = SmallNanowireCNN().to(device)
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def candidate_components(gray, binary, cutoff_y):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    rows = []
    for label in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if y >= cutoff_y or area < 20:
            continue
        if not (w >= 6 or h >= 6):
            continue
        if 8 <= area <= 90 and 3 <= w <= 14 and 3 <= h <= 14:
            continue
        rows.append({"component": int(label), "bbox": [x, y, w, h], "area": int(area)})
    return rows, labels


def crop_component(gray, bbox, size=128):
    x, y, w, h = bbox
    pad = max(18, int(max(w, h) * 0.7))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad)
    y1 = min(gray.shape[0], y + h + pad)
    crop = gray[y0:y1, x0:x1]
    canvas = np.full((size, size), int(np.median(gray)), dtype=np.uint8)
    scale = min(size / crop.shape[1], size / crop.shape[0])
    resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    yy = (size - resized.shape[0]) // 2
    xx = (size - resized.shape[1]) // 2
    canvas[yy:yy + resized.shape[0], xx:xx + resized.shape[1]] = resized
    return canvas


def crop_component_mask(labels, component, bbox, size=128):
    x, y, w, h = bbox
    pad = max(18, int(max(w, h) * 0.7))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(labels.shape[1], x + w + pad)
    y1 = min(labels.shape[0], y + h + pad)
    crop = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    crop[labels[y0:y1, x0:x1] == component] = 255
    canvas = np.zeros((size, size), dtype=np.uint8)
    scale = min(size / crop.shape[1], size / crop.shape[0])
    resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    yy = (size - resized.shape[0]) // 2
    xx = (size - resized.shape[1]) // 2
    canvas[yy:yy + resized.shape[0], xx:xx + resized.shape[1]] = resized
    return canvas


def component_mask(labels, component, bbox):
    x, y, w, h = bbox
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[labels[y:y + h, x:x + w] == component] = 255
    return mask


def make_sheet(rows, crop_dir, out_path, threshold):
    first = cv2.imread(str(crop_dir / rows[0]["crop"]), cv2.IMREAD_GRAYSCALE) if rows else None
    tile = int(first.shape[0]) if first is not None else 128
    label_h = 32
    cols = 8
    thumbs = []
    for row in rows:
        img = cv2.imread(str(crop_dir / row["crop"]), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        keep = bool(row.get("accepted_single", row["prob_single"] >= threshold))
        color = (0, 255, 0) if keep else (0, 0, 255)
        cv2.rectangle(rgb, (0, 0), (rgb.shape[1] - 1, rgb.shape[0] - 1), color, 2)
        canvas = np.full((tile + label_h, tile, 3), 24, dtype=np.uint8)
        canvas[:tile] = rgb
        cv2.putText(canvas, f"p={row['prob_single']:.2f}", (3, tile + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"c{row['component']} a{row['area']}", (3, tile + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(canvas)
    rows_needed = max(1, int(np.ceil(len(thumbs) / cols)))
    sheet = np.full((rows_needed * (tile + label_h), cols * tile, 3), 20, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        sheet[r * (tile + label_h):(r + 1) * (tile + label_h), c * tile:(c + 1) * tile] = thumb
    cv2.imwrite(str(out_path), sheet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default="experimental_sem/13.tif")
    parser.add_argument("--checkpoint", default="experimental_sem_results/nanowire_ml_model/best_model.pt")
    parser.add_argument("--output-dir", default="experimental_sem_results/nanowire_ml_real")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--crop-mode", choices=("raw", "mask"), default="raw")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--preprocess-mode", choices=("raw", "pca_mask", "soft_gray_pca"), default="raw")
    parser.add_argument("--topology-veto", action="store_true")
    parser.add_argument("--max-secondary-orientation", type=float, default=0.24)
    parser.add_argument("--max-off-line-fraction", type=float, default=0.22)
    parser.add_argument("--max-width-px", type=float, default=5.5)
    parser.add_argument("--max-branchpoints", type=int, default=18)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read {args.image}")
    _, binary, report = preprocess(gray)
    rows, labels = candidate_components(gray, binary, report["annotation_band"]["cutoff_y"])

    import torch  # CNN inference path; this entry point requires the [cnn] extra.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), device)
    predictions = []
    with torch.no_grad():
        for row in rows:
            if args.crop_mode == "mask":
                crop = crop_component_mask(labels, row["component"], row["bbox"], size=args.crop_size)
            else:
                crop = crop_component(gray, row["bbox"], size=args.crop_size)
            crop = preprocess_image(crop, args.preprocess_mode)
            crop_name = f"component_{row['component']:04d}.png"
            cv2.imwrite(str(crop_dir / crop_name), crop)
            x = torch.from_numpy(crop.astype(np.float32)[None, None, :, :] / 255.0).to(device)
            prob = float(torch.sigmoid(model(x)).cpu().numpy().ravel()[0])
            topo = topology_features(component_mask(labels, row["component"], row["bbox"]))
            veto_reasons = topology_veto_reasons(
                topo,
                max_secondary_orientation=args.max_secondary_orientation,
                max_off_line_fraction=args.max_off_line_fraction,
                max_width_px=args.max_width_px,
                max_branchpoints=args.max_branchpoints,
            )
            topology_pass = not veto_reasons
            accepted = prob >= args.threshold and (topology_pass or not args.topology_veto)
            predictions.append({
                **row,
                "crop": crop_name,
                "prob_single": prob,
                "topology_pass": topology_pass,
                "topology_veto_reasons": ",".join(veto_reasons),
                "accepted_single": accepted,
                **topo,
            })

    predictions.sort(key=lambda row: -row["prob_single"])
    with (out_dir / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0].keys()) if predictions else ["component"])
        writer.writeheader()
        writer.writerows(predictions)
    (out_dir / "predictions.json").write_text(json.dumps(predictions, indent=2))
    make_sheet(predictions, crop_dir, out_dir / "real_component_predictions.png", args.threshold)
    print(f"{args.image}: {sum(row['accepted_single'] for row in predictions)}/{len(predictions)} accepted at threshold {args.threshold}")
    print(out_dir / "real_component_predictions.png")


if __name__ == "__main__":
    main()
