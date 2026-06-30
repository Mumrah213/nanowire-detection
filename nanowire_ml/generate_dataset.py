#!/usr/bin/env python3
"""Generate synthetic nanowire train/val/test splits."""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np


DEFAULT_NM_PER_PX = 24.5
CATEGORIES_SINGLE = ["single", "long_single"]
CATEGORIES_BAD = ["crossed", "parallel_cluster", "messy_cluster", "fragment", "artifact"]


def seed_all(seed: int) -> random.Random:
    random.seed(seed)
    np.random.seed(seed)
    return random.Random(seed)


def noisy_background(size: int, rng: random.Random) -> np.ndarray:
    base = rng.uniform(86, 112)
    noise = np.random.normal(0, rng.uniform(4.0, 9.0), (size, size)).astype(np.float32)
    y = np.linspace(-1, 1, size, dtype=np.float32)[:, None]
    x = np.linspace(-1, 1, size, dtype=np.float32)[None, :]
    gradient = rng.uniform(-6, 6) * x + rng.uniform(-6, 6) * y
    bg = cv2.GaussianBlur(base + noise + gradient, (0, 0), rng.uniform(0.3, 1.0))
    return np.clip(bg, 0, 255).astype(np.float32)


def endpoints(center, length_px, angle):
    dx = math.cos(angle) * length_px / 2.0
    dy = math.sin(angle) * length_px / 2.0
    return (center[0] - dx, center[1] - dy), (center[0] + dx, center[1] + dy)


def draw_wire(canvas, center, length_px, diameter_px, angle, rng, brightness=None, curvature=0.0):
    if brightness is None:
        brightness = rng.uniform(80, 145)
    overlay = np.zeros_like(canvas, dtype=np.float32)
    thickness = max(1, int(round(diameter_px)))
    if abs(curvature) < 0.05:
        p0, p1 = endpoints(center, length_px, angle)
        cv2.line(overlay, tuple(map(int, map(round, p0))), tuple(map(int, map(round, p1))),
                 brightness, thickness, cv2.LINE_AA)
    else:
        normal = np.array([-math.sin(angle), math.cos(angle)], dtype=np.float32)
        pts = []
        for t in np.linspace(-1, 1, 18):
            x = center[0] + math.cos(angle) * length_px * t / 2.0
            y = center[1] + math.sin(angle) * length_px * t / 2.0
            bend = curvature * (1 - t * t) * length_px * 0.22
            pts.append([x + normal[0] * bend, y + normal[1] * bend])
        cv2.polylines(overlay, [np.array(pts, dtype=np.int32).reshape((-1, 1, 2))], False,
                      brightness, thickness, cv2.LINE_AA)
    canvas += cv2.GaussianBlur(overlay, (0, 0), rng.uniform(0.35, 0.9))


def wire_params(rng, nm_per_px, length_range=(800, 2000)):
    diameter_nm = rng.uniform(40, 80)
    length_nm = rng.uniform(*length_range)
    return length_nm / nm_per_px, diameter_nm / nm_per_px, length_nm, diameter_nm


def safe_center(size, length_px, rng, extra=10.0):
    margin = min(size / 2.0 - 8.0, length_px / 2.0 + extra)
    return np.array([rng.uniform(margin, size - margin), rng.uniform(margin, size - margin)])


def add_dot_markers(canvas, rng, nm_per_px):
    if rng.random() > 0.4:
        return
    spacing_px = 2500.0 / nm_per_px
    origin = (rng.uniform(-spacing_px, spacing_px), rng.uniform(-spacing_px, spacing_px))
    radius = max(1, int(round((25.0 / nm_per_px) / 2.0)))
    for i in range(-3, 5):
        for j in range(-3, 5):
            x = origin[0] + i * spacing_px
            y = origin[1] + j * spacing_px
            if 3 <= x < canvas.shape[1] - 3 and 3 <= y < canvas.shape[0] - 3 and rng.random() < 0.35:
                val = rng.choice([rng.uniform(-30, -10), rng.uniform(35, 75)])
                cv2.circle(canvas, (int(round(x)), int(round(y))), radius, val, -1, cv2.LINE_AA)


def degrade(img, rng):
    out = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.25, 1.2))
    out += np.random.normal(0, rng.uniform(2.0, 7.0), out.shape).astype(np.float32)
    lo, hi = np.percentile(out, [0.2, 99.8])
    if hi > lo:
        out = (out - lo) / (hi - lo) * 255.0
        out = out * rng.uniform(0.55, 0.95) + rng.uniform(12, 35)
    return np.clip(out, 0, 255).astype(np.uint8)


def render(category, size, nm_per_px, rng):
    img = noisy_background(size, rng)
    center = np.array([rng.uniform(28, size - 28), rng.uniform(28, size - 28)])
    angle = rng.uniform(0, math.pi)
    meta = {"category": category}

    if category == "single":
        length_px, diameter_px, length_nm, diameter_nm = wire_params(rng, nm_per_px, (800, 2000))
        center = safe_center(size, length_px, rng)
        draw_wire(img, center, length_px, diameter_px, angle, rng, curvature=rng.uniform(-0.12, 0.12))
        meta.update({"label": "single", "length_nm": length_nm, "diameter_nm": diameter_nm})
    elif category == "long_single":
        length_px, diameter_px, length_nm, diameter_nm = wire_params(rng, nm_per_px, (2000, 4200))
        draw_wire(img, center, length_px, diameter_px, angle, rng, curvature=rng.uniform(-0.08, 0.08))
        meta.update({"label": "single", "length_nm": length_nm, "diameter_nm": diameter_nm})
    elif category == "crossed":
        base_len, base_diam, _, _ = wire_params(rng, nm_per_px, (700, 2000))
        for k in range(rng.choice([2, 2, 3])):
            draw_wire(img, center + np.array([rng.uniform(-8, 8), rng.uniform(-8, 8)]),
                      base_len * rng.uniform(0.65, 1.15), base_diam * rng.uniform(0.9, 1.2),
                      angle + k * rng.uniform(0.6, 2.1), rng, curvature=rng.uniform(-0.08, 0.08))
        meta["label"] = "bad"
    elif category == "parallel_cluster":
        count = rng.randint(2, 7)
        length_px, diameter_px, _, _ = wire_params(rng, nm_per_px, (650, 2100))
        normal = np.array([-math.sin(angle), math.cos(angle)])
        for k in range(count):
            offset = (k - (count - 1) / 2.0) * rng.uniform(diameter_px * 0.7, diameter_px * 1.9)
            draw_wire(img, center + normal * offset + np.array([rng.uniform(-8, 8), rng.uniform(-8, 8)]),
                      length_px * rng.uniform(0.45, 1.05), diameter_px * rng.uniform(0.9, 1.25),
                      angle + rng.uniform(-0.12, 0.12), rng, curvature=rng.uniform(-0.07, 0.07))
        meta["label"] = "bad"
    elif category == "messy_cluster":
        for _ in range(rng.randint(4, 10)):
            length_px, diameter_px, _, _ = wire_params(rng, nm_per_px, (450, 1700))
            draw_wire(img, center + np.array([rng.uniform(-24, 24), rng.uniform(-24, 24)]),
                      length_px * rng.uniform(0.25, 0.85), diameter_px * rng.uniform(0.8, 1.3),
                      rng.uniform(0, math.pi), rng, curvature=rng.uniform(-0.15, 0.15))
        meta["label"] = "bad"
    elif category == "fragment":
        length_px, diameter_px, length_nm, diameter_nm = wire_params(rng, nm_per_px, (160, 760))
        center = safe_center(size, length_px, rng, extra=8)
        draw_wire(img, center, length_px, diameter_px, angle, rng, curvature=rng.uniform(-0.08, 0.08))
        meta.update({"label": "bad", "length_nm": length_nm, "diameter_nm": diameter_nm})
    elif category == "artifact":
        if rng.random() < 0.65:
            arm = rng.uniform(8, 24)
            diameter = rng.uniform(1.5, 3.2)
            draw_wire(img, center + np.array([arm / 2, 0]), arm, diameter, 0, rng, brightness=rng.uniform(80, 150))
            draw_wire(img, center + np.array([0, -arm / 2]), arm, diameter, math.pi / 2, rng, brightness=rng.uniform(80, 150))
        else:
            cv2.circle(img, tuple(map(int, center)), rng.randint(2, 7), rng.uniform(70, 140), -1, cv2.LINE_AA)
        meta["label"] = "bad"
    else:
        raise ValueError(category)

    add_dot_markers(img, rng, nm_per_px)
    return degrade(img, rng), meta


def make_contact_sheet(rows, image_root, out_path, cols=8):
    tile = 128
    label_h = 24
    thumbs = []
    for row in rows:
        img = cv2.imread(str(image_root / row["split"] / row["label"] / row["filename"]), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        color = (0, 255, 0) if row["label"] == "single" else (0, 0, 255)
        cv2.rectangle(rgb, (0, 0), (rgb.shape[1] - 1, rgb.shape[0] - 1), color, 2)
        canvas = np.full((tile + label_h, tile, 3), 24, dtype=np.uint8)
        canvas[:tile] = cv2.resize(rgb, (tile, tile), interpolation=cv2.INTER_AREA)
        cv2.putText(canvas, row["category"], (3, tile + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, row["label"], (3, tile + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(canvas)
    rows_needed = max(1, math.ceil(len(thumbs) / cols))
    sheet = np.full((rows_needed * (tile + label_h), cols * tile, 3), 20, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        sheet[r * (tile + label_h):(r + 1) * (tile + label_h), c * tile:(c + 1) * tile] = thumb
    cv2.imwrite(str(out_path), sheet)


def generate_split(split, per_binary, out_dir, size, nm_per_px, rng):
    rows = []
    category_counts = {}
    single_each = math.ceil(per_binary / len(CATEGORIES_SINGLE))
    bad_each = math.ceil(per_binary / len(CATEGORIES_BAD))
    for label, categories, each in [("single", CATEGORIES_SINGLE, single_each), ("bad", CATEGORIES_BAD, bad_each)]:
        for category in categories:
            for idx in range(each):
                if category_counts.get(label, 0) >= per_binary:
                    break
                img, meta = render(category, size, nm_per_px, rng)
                label_dir = out_dir / split / meta["label"]
                label_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{category}_{category_counts.get(label, 0):05d}.png"
                cv2.imwrite(str(label_dir / filename), img)
                row = {"split": split, "filename": filename, "category": category, "label": meta["label"],
                       "nm_per_px": nm_per_px, "size_px": size, **{k: v for k, v in meta.items() if k not in {"label", "category"}}}
                rows.append(row)
                category_counts[label] = category_counts.get(label, 0) + 1
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experimental_sem_results/nanowire_ml_dataset")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--nm-per-px", type=float, default=DEFAULT_NM_PER_PX)
    parser.add_argument("--train-per-binary", type=int, default=800)
    parser.add_argument("--val-per-binary", type=int, default=160)
    parser.add_argument("--test-per-binary", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = seed_all(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, n in [("train", args.train_per_binary), ("val", args.val_per_binary), ("test", args.test_per_binary)]:
        rows.extend(generate_split(split, n, out_dir, args.size, args.nm_per_px, rng))

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (out_dir / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "metadata.json").write_text(json.dumps(rows, indent=2))

    for split in ["train", "val", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        make_contact_sheet(split_rows[: min(160, len(split_rows))], out_dir, out_dir / f"{split}_contact_sheet.png")
    print(f"Wrote {len(rows)} images to {out_dir}")


if __name__ == "__main__":
    main()
