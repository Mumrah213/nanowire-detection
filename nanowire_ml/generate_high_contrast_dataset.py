#!/usr/bin/env python3
"""Generate a high-contrast synthetic nanowire dataset.

This mirrors the simple synthetic-L philosophy: clean geometric features,
large foreground/background contrast, nuisance variation from rotation,
translation, linewidth, noise, and blur.
"""

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


SPLITS = ("train", "val", "test")
SINGLE_CATEGORIES = ("single", "long_single")
BAD_CATEGORIES = ("crossed", "parallel_cluster", "messy_cluster", "fragment", "artifact")
DEFAULT_NM_PER_PX = 24.5


def _draw_lines_array(
    lines: list[dict],
    canvas_size: int,
    noise_std: float,
    blur_sigma: float,
    rng: np.random.Generator,
    polarity: str,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(1, 1), dpi=canvas_size)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    if polarity == "bright_on_dark":
        bg = "k"
        fg = "w"
    elif polarity == "dark_on_bright":
        bg = "w"
        fg = "k"
    else:
        raise ValueError(f"Unknown polarity: {polarity}")

    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)

    for line in lines:
        ax.plot(
            line["x"],
            line["y"],
            linewidth=line["linewidth"],
            color=fg,
            solid_capstyle="round",
            alpha=line.get("alpha", 1.0),
        )

    fig.canvas.draw()
    data = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(np.float64) / 255.0
    plt.close(fig)

    if noise_std > 0:
        data = np.clip(data + rng.normal(0, noise_std / 255.0, data.shape), 0, 1)
    if blur_sigma > 0:
        data = np.array([
            ndimage.gaussian_filter(data[:, :, c], sigma=blur_sigma)
            for c in range(3)
        ]).transpose(1, 2, 0)
    return np.clip(data * 255.0, 0, 255).astype(np.uint8)


def _draw_mask_lines_array(
    lines: list[dict],
    canvas_size: int,
    noise_std: float,
    blur_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw connected-component-like white masks on black background."""
    import cv2

    img = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    for line in lines:
        pts = np.column_stack([
            np.interp(line["x"], [-5, 5], [0, canvas_size - 1]),
            np.interp(line["y"], [-5, 5], [canvas_size - 1, 0]),
        ]).astype(np.int32)
        thickness = max(1, int(round(line["linewidth"])))
        keep_prob = float(line.get("keep_prob", 1.0))
        if keep_prob >= 0.98 or len(pts) < 4:
            cv2.polylines(img, [pts.reshape((-1, 1, 2))], False, 255, thickness, cv2.LINE_AA)
        else:
            # Draw short segments with random gaps to mimic thresholded SEM masks.
            for idx in range(len(pts) - 1):
                if rng.random() <= keep_prob:
                    cv2.line(img, tuple(pts[idx]), tuple(pts[idx + 1]), 255, thickness, cv2.LINE_AA)

        if rng.random() < 0.45:
            # Add a small attached ridge/rough spot.
            anchor = pts[int(rng.integers(0, len(pts)))]
            length = int(rng.integers(3, 12))
            angle = float(rng.uniform(0, 2 * math.pi))
            end = (int(anchor[0] + math.cos(angle) * length), int(anchor[1] + math.sin(angle) * length))
            cv2.line(img, tuple(anchor), end, 255, max(1, thickness - 1), cv2.LINE_AA)

    if rng.random() < 0.75:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if rng.random() < 0.5:
            img = cv2.dilate(img, kernel, iterations=1)
        if rng.random() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)

    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), min(0.8, blur_sigma))
    out = img.astype(np.float32)
    if noise_std > 0:
        out += rng.normal(0, min(10.0, noise_std), out.shape).astype(np.float32)
    out = np.clip(out, 0, 255)
    # Re-threshold after blur/noise so the result looks like a connected-component mask.
    out = (out > float(rng.uniform(65, 135))).astype(np.uint8) * 255
    return np.repeat(out[:, :, None], 3, axis=2)


def line_points(center_x, center_y, length_units, angle, curvature, n=18):
    xs = []
    ys = []
    normal = (-math.sin(angle), math.cos(angle))
    for t in np.linspace(-1, 1, n):
        x = center_x + math.cos(angle) * length_units * t / 2.0
        y = center_y + math.sin(angle) * length_units * t / 2.0
        bend = curvature * (1 - t * t) * length_units * 0.22
        xs.append(x + normal[0] * bend)
        ys.append(y + normal[1] * bend)
    return xs, ys


def physical_to_plot_units(px: float, canvas_size: int) -> float:
    return px / canvas_size * 10.0


def target_wire(rng, nm_per_px, length_nm_range=(800, 2000), diameter_nm_range=(40, 80)):
    length_nm = float(rng.uniform(*length_nm_range))
    diameter_nm = float(rng.uniform(*diameter_nm_range))
    return {
        "length_nm": length_nm,
        "diameter_nm": diameter_nm,
        "length_px": length_nm / nm_per_px,
        "diameter_px": diameter_nm / nm_per_px,
    }


def add_wire_line(lines, center, params, angle, canvas_size, rng, curvature_range=(-0.10, 0.10), linewidth_scale_range=(0.8, 1.9), jagged=False):
    length_units = physical_to_plot_units(params["length_px"], canvas_size)
    linewidth = max(0.8, params["diameter_px"] * rng.uniform(*linewidth_scale_range))
    x, y = line_points(center[0], center[1], length_units, angle, float(rng.uniform(*curvature_range)))
    line = {"x": x, "y": y, "linewidth": linewidth, "alpha": float(rng.uniform(0.75, 1.0))}
    if jagged:
        line["keep_prob"] = float(rng.uniform(0.72, 1.0))
    lines.append(line)


def render_sample(category: str, canvas_size: int, nm_per_px: float, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    rotation = float(rng.uniform(0, math.pi))
    scale = float(rng.uniform(0.9, 1.1))
    tx = float(rng.uniform(-0.45, 0.45))
    ty = float(rng.uniform(-0.45, 0.45))
    noise_std = float(rng.uniform(0.0, 18.0))
    blur_sigma = float(rng.uniform(0.0, 1.6))
    center = (tx, ty)
    lines = []
    metadata = {
        "category": category,
        "rotation": round(math.degrees(rotation), 3),
        "scale": round(scale, 4),
        "tx": round(tx, 4),
        "ty": round(ty, 4),
        "noise_std": round(noise_std, 3),
        "blur_sigma": round(blur_sigma, 4),
    }
    render_mode = "mask" if float(rng.random()) < 0.65 else "smooth"
    metadata["render_mode"] = render_mode

    if category == "single":
        params = target_wire(rng, nm_per_px, (800, 2000))
        params["length_px"] *= scale
        add_wire_line(lines, center, params, rotation, canvas_size, rng, jagged=render_mode == "mask")
        metadata.update({"label": "single", **{k: round(v, 3) for k, v in params.items() if k.endswith("_nm")}})
    elif category == "long_single":
        params = target_wire(rng, nm_per_px, (2000, 4300))
        params["length_px"] *= scale
        add_wire_line(lines, center, params, rotation, canvas_size, rng, curvature_range=(-0.04, 0.04), jagged=render_mode == "mask")
        metadata.update({"label": "single", **{k: round(v, 3) for k, v in params.items() if k.endswith("_nm")}})
    elif category == "crossed":
        base = target_wire(rng, nm_per_px, (750, 2100))
        count = int(rng.choice([2, 2, 3]))
        for idx in range(count):
            local = (center[0] + float(rng.uniform(-0.25, 0.25)), center[1] + float(rng.uniform(-0.25, 0.25)))
            params = dict(base)
            params["length_px"] *= float(rng.uniform(0.55, 1.1)) * scale
            params["diameter_px"] *= float(rng.uniform(0.85, 1.2))
            add_wire_line(lines, local, params, rotation + idx * float(rng.uniform(0.65, 2.15)), canvas_size, rng, jagged=render_mode == "mask")
        metadata["label"] = "bad"
    elif category == "parallel_cluster":
        params = target_wire(rng, nm_per_px, (650, 2200))
        count = int(rng.integers(2, 8))
        normal = np.array([-math.sin(rotation), math.cos(rotation)])
        spacing_units = physical_to_plot_units(params["diameter_px"], canvas_size) * float(rng.uniform(0.8, 2.4))
        for idx in range(count):
            offset = (idx - (count - 1) / 2.0) * spacing_units
            local = np.array(center) + normal * offset + rng.normal(0, 0.12, 2)
            local_params = dict(params)
            local_params["length_px"] *= float(rng.uniform(0.45, 1.05)) * scale
            local_params["diameter_px"] *= float(rng.uniform(0.9, 1.25))
            add_wire_line(lines, tuple(local), local_params, rotation + float(rng.uniform(-0.12, 0.12)), canvas_size, rng, jagged=render_mode == "mask")
        metadata["label"] = "bad"
    elif category == "messy_cluster":
        for _ in range(int(rng.integers(4, 11))):
            params = target_wire(rng, nm_per_px, (350, 1700))
            params["length_px"] *= float(rng.uniform(0.25, 0.9)) * scale
            local = (center[0] + float(rng.uniform(-1.3, 1.3)), center[1] + float(rng.uniform(-1.3, 1.3)))
            add_wire_line(lines, local, params, float(rng.uniform(0, math.pi)), canvas_size, rng, jagged=render_mode == "mask")
        metadata["label"] = "bad"
    elif category == "fragment":
        params = target_wire(rng, nm_per_px, (160, 760))
        params["length_px"] *= scale
        add_wire_line(lines, center, params, rotation, canvas_size, rng, jagged=render_mode == "mask")
        metadata.update({"label": "bad", **{k: round(v, 3) for k, v in params.items() if k.endswith("_nm")}})
    elif category == "artifact":
        if float(rng.random()) < 0.75:
            arm = float(rng.uniform(0.45, 1.4))
            lw = float(rng.uniform(7.0, 16.0))
            x0, y0 = center
            lines.append({"x": [x0, x0 + arm], "y": [y0, y0], "linewidth": lw, "alpha": 1.0})
            lines.append({"x": [x0, x0], "y": [y0, y0 + arm], "linewidth": lw, "alpha": 1.0})
        else:
            # compact bright block/blob, approximated by several short strokes
            for _ in range(int(rng.integers(3, 7))):
                local = (center[0] + float(rng.uniform(-0.25, 0.25)), center[1] + float(rng.uniform(-0.25, 0.25)))
                lines.append({
                    "x": [local[0] - 0.12, local[0] + 0.12],
                    "y": [local[1], local[1] + float(rng.uniform(-0.08, 0.08))],
                    "linewidth": float(rng.uniform(7, 16)),
                    "alpha": 1.0,
                })
        metadata["label"] = "bad"
    else:
        raise ValueError(category)

    if render_mode == "mask":
        img = _draw_mask_lines_array(lines, canvas_size, noise_std, blur_sigma, rng)
    else:
        img = _draw_lines_array(lines, canvas_size, noise_std, blur_sigma, rng, polarity="bright_on_dark")
    return img, metadata


def write_contact_sheet(rows: list[dict], output_dir: Path, split: str, out_path: Path, max_images=160, seed=0):
    tile = 96
    label_h = 26
    cols = 8
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    subset = [rows[int(idx)] for idx in order[:max_images]]
    sheet = np.full((math.ceil(len(subset) / cols) * (tile + label_h), cols * tile, 3), 20, dtype=np.uint8)
    for idx, row in enumerate(subset):
        img = plt.imread(output_dir / split / row["label"] / row["filename"])
        img = (img[:, :, :3] * 255).astype(np.uint8) if img.max() <= 1 else img[:, :, :3].astype(np.uint8)
        img = ndimage.zoom(img, (tile / img.shape[0], tile / img.shape[1], 1), order=1).astype(np.uint8)
        color = (0, 255, 0) if row["label"] == "single" else (255, 0, 0)
        canvas = np.full((tile + label_h, tile, 3), 20, dtype=np.uint8)
        canvas[:tile] = img
        # OpenCV uses BGR, but this sheet is written via cv2 only after color labels.
        import cv2
        cv2.rectangle(canvas, (0, 0), (tile - 1, tile - 1), color[::-1], 2)
        cv2.putText(canvas, row["category"], (3, tile + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, color[::-1], 1, cv2.LINE_AA)
        cv2.putText(canvas, row["label"], (3, tile + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
        r, c = divmod(idx, cols)
        sheet[r * (tile + label_h):(r + 1) * (tile + label_h), c * tile:(c + 1) * tile] = canvas
    import cv2
    cv2.imwrite(str(out_path), sheet)


def generate_split(split: str, count_per_binary: int, output_dir: Path, canvas_size: int, nm_per_px: float, rng: np.random.Generator) -> list[dict]:
    rows = []
    counters = {"single": 0, "bad": 0}
    plan = [("single", SINGLE_CATEGORIES), ("bad", BAD_CATEGORIES)]
    for label, categories in plan:
        each = math.ceil(count_per_binary / len(categories))
        for category in categories:
            for _ in range(each):
                if counters[label] >= count_per_binary:
                    break
                img, meta = render_sample(category, canvas_size, nm_per_px, rng)
                label_dir = output_dir / split / meta["label"]
                label_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{category}_{counters[label]:05d}.png"
                plt.imsave(label_dir / filename, img)
                row = {
                    "split": split,
                    "filename": filename,
                    "category": category,
                    "label": meta["label"],
                    "canvas_size": canvas_size,
                    "nm_per_px": nm_per_px,
                    **{k: v for k, v in meta.items() if k not in {"label", "category"}},
                }
                rows.append(row)
                counters[label] += 1
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experimental_sem_results/nanowire_ml_high_contrast_dataset")
    parser.add_argument("--train-per-binary", type=int, default=800)
    parser.add_argument("--val-per-binary", type=int, default=160)
    parser.add_argument("--test-per-binary", type=int, default=160)
    parser.add_argument("--canvas-size", type=int, default=64)
    parser.add_argument("--nm-per-px", type=float, default=DEFAULT_NM_PER_PX)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    for split, n in (("train", args.train_per_binary), ("val", args.val_per_binary), ("test", args.test_per_binary)):
        rows.extend(generate_split(split, n, output_dir, args.canvas_size, args.nm_per_px, rng))

    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "metadata.json").write_text(json.dumps(rows, indent=2))
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        write_contact_sheet(split_rows, output_dir, split, output_dir / f"{split}_contact_sheet.png", seed=args.seed)
    print(f"Wrote {len(rows)} high-contrast samples to {output_dir}")


if __name__ == "__main__":
    main()
