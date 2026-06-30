#!/usr/bin/env python3
"""Visualize PCA-alignment preprocessing steps for real SEM components."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sem_preprocess import preprocess  # noqa: E402
from nanowire_ml.predict_real_components import candidate_components, crop_component, crop_component_mask  # noqa: E402
from nanowire_ml.topology import topology_features, topology_veto_reasons  # noqa: E402
from nanowire_ml.train_classifier import SmallNanowireCNN, pca_align_mask  # noqa: E402


def load_model(checkpoint: Path, device):
    model = SmallNanowireCNN().to(device)
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def pca_alignment_debug(mask: np.ndarray, output_size: int) -> dict:
    blur = cv2.GaussianBlur(mask, (0, 0), 0.6)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(binary > 0)
    if len(xs) < 3:
        return {
            "binary": binary,
            "axis_overlay": cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
            "rotated": binary,
            "tight": binary,
            "final": cv2.resize(binary, (output_size, output_size), interpolation=cv2.INTER_NEAREST),
            "angle_deg": 0.0,
            "rotate_deg": 0.0,
            "center": [0.0, 0.0],
        }

    points = np.column_stack([xs, ys]).astype(np.float32)
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    angle_deg = float(np.degrees(np.arctan2(major[1], major[0])))
    rotate_deg = 90.0 - angle_deg

    overlay = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    length = max(binary.shape) * 0.45
    p0 = (int(round(center[0] - major[0] * length)), int(round(center[1] - major[1] * length)))
    p1 = (int(round(center[0] + major[0] * length)), int(round(center[1] + major[1] * length)))
    cv2.line(overlay, p0, p1, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(overlay, (int(round(center[0])), int(round(center[1]))), 3, (0, 0, 255), -1, cv2.LINE_AA)

    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), rotate_deg, 1.0)
    rotated = cv2.warpAffine(binary, matrix, (binary.shape[1], binary.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)

    ys2, xs2 = np.nonzero(rotated > 0)
    if len(xs2) >= 3:
        x0, x1 = int(xs2.min()), int(xs2.max()) + 1
        y0, y1 = int(ys2.min()), int(ys2.max()) + 1
        pad = max(4, int(0.18 * max(x1 - x0, y1 - y0)))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(rotated.shape[1], x1 + pad)
        y1 = min(rotated.shape[0], y1 + pad)
        tight = rotated[y0:y1, x0:x1]
    else:
        tight = rotated

    final = pca_align_mask(mask, output_size=output_size)
    return {
        "binary": binary,
        "axis_overlay": overlay,
        "rotated": rotated,
        "tight": tight,
        "final": final,
        "angle_deg": angle_deg,
        "rotate_deg": float(rotate_deg),
        "center": [float(center[0]), float(center[1])],
    }


def panel_image(title: str, image: np.ndarray, tile: int = 180) -> np.ndarray:
    label_h = 32
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = min(tile / image.shape[1], tile / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.full((tile + label_h, tile, 3), 24, dtype=np.uint8)
    yy = (tile - resized.shape[0]) // 2
    xx = (tile - resized.shape[1]) // 2
    canvas[yy:yy + resized.shape[0], xx:xx + resized.shape[1]] = resized
    cv2.putText(canvas, title[:28], (4, tile + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def make_component_debug(row: dict, gray: np.ndarray, labels: np.ndarray, model, device, args, out_dir: Path) -> dict:
    component = int(row["component"])
    bbox = row["bbox"]
    raw_crop = crop_component(gray, bbox, size=args.crop_size)
    mask_crop = crop_component_mask(labels, component, bbox, size=args.crop_size)
    debug = pca_alignment_debug(mask_crop, output_size=args.crop_size)
    final = debug["final"]

    with torch.no_grad():
        x = torch.from_numpy(final.astype(np.float32)[None, None, :, :] / 255.0).to(device)
        prob = float(torch.sigmoid(model(x)).cpu().numpy().ravel()[0]) if model else float("nan")

    topo = topology_features((labels[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] == component).astype(np.uint8) * 255)
    veto = topology_veto_reasons(topo)
    accepted = prob >= args.threshold and not veto

    panels = [
        panel_image("raw SEM crop", raw_crop),
        panel_image("component mask", mask_crop),
        panel_image(f"PCA axis {debug['angle_deg']:.1f} deg", debug["axis_overlay"]),
        panel_image(f"rotated {debug['rotate_deg']:.1f} deg", debug["rotated"]),
        panel_image("tight rotated crop", debug["tight"]),
        panel_image(f"CNN input p={prob:.2f}", final),
    ]
    sheet = np.concatenate(panels, axis=1)
    footer = np.full((64, sheet.shape[1], 3), 24, dtype=np.uint8)
    color = (0, 255, 0) if accepted else (0, 0, 255)
    text = (
        f"component={component} bbox={bbox} area={row['area']} accepted={accepted} "
        f"veto={','.join(veto) if veto else 'none'} width={topo['topology_estimated_width_px']:.2f} "
        f"branch={topo['topology_branchpoints']} off={topo['pca_off_line_fraction']:.2f} "
        f"hough={topo['hough_orientation_clusters']}/{topo['hough_secondary_weight_fraction']:.2f}"
    )
    cv2.putText(footer, text[:180], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    cv2.putText(footer, text[180:360], (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    sheet = np.concatenate([sheet, footer], axis=0)

    out_path = out_dir / f"component_{component:04d}_alignment_steps.png"
    cv2.imwrite(str(out_path), sheet)
    return {
        **row,
        "prob_single": prob,
        "accepted_single": accepted,
        "veto_reasons": ",".join(veto),
        "angle_deg": debug["angle_deg"],
        "rotate_deg": debug["rotate_deg"],
        "debug_image": str(out_path),
        **topo,
    }


def make_montage(debug_rows: list[dict], out_path: Path) -> None:
    thumbs = []
    tile_w, tile_h = 360, 140
    for row in debug_rows:
        img = cv2.imread(row["debug_image"])
        if img is None:
            continue
        thumb = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        color = (0, 255, 0) if row["accepted_single"] else (0, 0, 255)
        cv2.rectangle(thumb, (0, 0), (tile_w - 1, tile_h - 1), color, 2)
        cv2.putText(thumb, f"c{row['component']} p={row['prob_single']:.2f} rot={row['rotate_deg']:.1f}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        thumbs.append(thumb)
    cols = 2
    rows_needed = max(1, math.ceil(len(thumbs) / cols))
    canvas = np.full((rows_needed * tile_h, cols * tile_w, 3), 20, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        canvas[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = thumb
    cv2.imwrite(str(out_path), canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default="experimental_sem/13.tif")
    parser.add_argument("--checkpoint", default="experimental_sem_results/nanowire_ml_pca_jagged_smoke_model/best_model.pt")
    parser.add_argument("--output-dir", default="experimental_sem_results/nanowire_alignment_steps_13")
    parser.add_argument("--components", default="209,138,85,187,16,229,168")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--crop-size", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read {args.image}")
    _, binary, report = preprocess(gray)
    rows, labels = candidate_components(gray, binary, report["annotation_band"]["cutoff_y"])
    wanted = {int(item) for item in args.components.split(",") if item.strip()}
    selected = [row for row in rows if int(row["component"]) in wanted]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), device) if Path(args.checkpoint).exists() else None

    debug_rows = [make_component_debug(row, gray, labels, model, device, args, out_dir) for row in selected]
    debug_rows.sort(key=lambda row: int(row["component"]))
    with (out_dir / "alignment_steps.csv").open("w", newline="") as f:
        if debug_rows:
            writer = csv.DictWriter(f, fieldnames=list(debug_rows[0].keys()))
            writer.writeheader()
            writer.writerows(debug_rows)
    (out_dir / "alignment_steps.json").write_text(json.dumps(debug_rows, indent=2))
    make_montage(debug_rows, out_dir / "alignment_steps_montage.png")
    print(out_dir / "alignment_steps_montage.png")


if __name__ == "__main__":
    main()
