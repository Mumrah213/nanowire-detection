#!/usr/bin/env python3
"""RANSAC-style lattice fitting to find the largest equidistant marker set.

After initial blob detection, this step finds the largest subset of blobs
that forms a regular equidistant grid pattern. Uses the dominant inter-blob
distance as the target grid spacing and estimates the lattice orientation
from pair-vector angle clusters.

The approach:
1. Find dominant inter-blob distances
2. Estimate continuous lattice angles from pair vectors near those spacings
3. For each spacing/angle hypothesis, try blob origins
4. Return the largest subset that forms a regular grid

Usage:
    python grid_ransac.py <image_path> [--output-dir DIR]
    python grid_ransac.py <directory> --output-dir DIR
"""

import argparse
import glob
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from utils.blob_utils import detect_blob_components


def find_candidate_spacings(blob_positions: list[tuple],
                             min_dist: float = 20.0,
                             max_dist: float = 300.0) -> list[float]:
    """Find candidate grid spacings from inter-blob distance histogram.

    Returns list of dominant grid spacings (distances).
    """
    n = len(blob_positions)
    distances = []

    for i in range(n):
        for j in range(i + 1, n):
            dx = blob_positions[j][0] - blob_positions[i][0]
            dy = blob_positions[j][1] - blob_positions[i][1]
            dist = math.sqrt(dx ** 2 + dy ** 2)
            if min_dist <= dist <= max_dist:
                distances.append(dist)

    if not distances:
        return []

    # Histogram with 2px bins
    dist_buckets = Counter([round(d / 2) * 2 for d in distances])

    # Find peaks in the histogram
    peaks = []
    sorted_dists = sorted(dist_buckets.items(), key=lambda x: -x[1])

    for dist, count in sorted_dists:
        if count < len(distances) * 0.02:
            break

        # Check if it's a local maximum
        is_peak = True
        for delta in [-2, -1, 1, 2]:
            neighbor_count = dist_buckets.get(dist + delta, 0)
            if neighbor_count >= count * 0.8:
                is_peak = False
                break

        if is_peak and count > 5:
            peaks.append(float(dist))

        if len(peaks) >= 3:
            break

    return peaks


def fit_grid_to_blobs_2d(
    blob_positions: list[tuple],
    spacing_x: float,
    spacing_y: float,
    origin: tuple[float, float],
    angle: float,
    tolerance: float = 8.0,
) -> list[int]:
    """Fit a 2D regular grid to blobs.

    Returns list of blob indices that fit the pattern.
    """
    n = len(blob_positions)
    fitted = []

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    for idx, (cx, cy) in enumerate(blob_positions):
        # Transform to grid coordinates
        dx = cx - origin[0]
        dy = cy - origin[1]

        # Rotate to align with grid
        gx = dx * cos_a + dy * sin_a
        gy = -dx * sin_a + dy * cos_a

        # Round to nearest grid index
        gi = round(gx / spacing_x)
        gj = round(gy / spacing_y)

        # Check if blob is close to the grid position
        expected_x = origin[0] + gi * spacing_x * cos_a - gj * spacing_y * sin_a
        expected_y = origin[1] + gi * spacing_x * sin_a + gj * spacing_y * cos_a

        dist = math.sqrt((cx - expected_x) ** 2 + (cy - expected_y) ** 2)

        if dist < tolerance:
            fitted.append(idx)

    return fitted


def estimate_spacing_angles(
    blob_positions: list[tuple],
    candidate_spacings: list[float],
    spacing_tolerance: float = 6.0,
    angle_bin_deg: float = 0.5,
) -> list[tuple[float, float, int]]:
    """Estimate continuous lattice angles from pair vectors.

    A square lattice has two equivalent axes 90 degrees apart, so pair-vector
    angles are folded into a 90-degree range. This avoids the main weakness of
    the earlier RANSAC loop, which only tried a few hard-coded orientations.
    """
    pair_vectors = []
    n = len(blob_positions)

    for i in range(n):
        for j in range(i + 1, n):
            dx = blob_positions[j][0] - blob_positions[i][0]
            dy = blob_positions[j][1] - blob_positions[i][1]
            dist = math.hypot(dx, dy)
            pair_vectors.append((dist, dx, dy))

    hypotheses = []
    for spacing in candidate_spacings:
        angle_counts = Counter()
        for dist, dx, dy in pair_vectors:
            if abs(dist - spacing) > spacing_tolerance:
                continue

            angle = math.degrees(math.atan2(dy, dx)) % 90.0
            if angle > 45.0:
                angle -= 90.0
            angle = round(angle / angle_bin_deg) * angle_bin_deg
            angle_counts[angle] += 1

        for angle, support in angle_counts.most_common(4):
            if support > 2:
                hypotheses.append((float(spacing), math.radians(angle), int(support)))

    hypotheses.sort(key=lambda item: -item[2])
    return hypotheses


def ransac_2d_grid_fitting(
    blob_positions: list[tuple],
    candidate_spacings: list[float],
    n_iterations: int = 200,
    tolerance: float = 8.0,
) -> dict:
    """RANSAC-style 2D grid fitting to find the largest equidistant marker set.

    Estimates likely spacing/orientation pairs from inter-blob vectors, then
    tries blob origins against each candidate lattice.
    Returns dict with best fit parameters and inlier indices.
    """
    best_inliers = []
    best_origin = None
    best_spacing_x = 0
    best_spacing_y = 0
    best_angle = 0
    best_support = 0

    spacing_angles = estimate_spacing_angles(blob_positions, candidate_spacings)
    max_hypotheses = max(8, n_iterations // 10)

    for spacing, angle, support in spacing_angles[:max_hypotheses]:
        for origin in blob_positions:
            inliers = fit_grid_to_blobs_2d(
                blob_positions, spacing, spacing, origin, angle, tolerance
            )

            if (
                len(inliers) > len(best_inliers)
                or (len(inliers) == len(best_inliers) and support > best_support)
            ):
                best_inliers = inliers
                best_origin = origin
                best_spacing_x = spacing
                best_spacing_y = spacing
                best_angle = angle
                best_support = support

    return {
        "origin": best_origin,
        "inlier_indices": set(best_inliers),
        "n_inliers": len(best_inliers),
        "n_total": len(blob_positions),
        "spacing": best_spacing_x,
        "angle": best_angle,
        "orientation_support": best_support,
    }


def process_image(
    image_path: str,
    min_area: int = 5,
    max_area: int = 5000,
    n_ransac_iters: int = 200,
) -> dict:
    """Process a single image: detect blobs, find grid spacing, RANSAC fitting."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    detected_blobs, _ = detect_blob_components(img, min_area, max_area)

    blob_positions = [(float(b["x"]), float(b["y"])) for b in detected_blobs]

    if not blob_positions:
        return {
            "image": str(image_path),
            "total_blobs": 0,
            "grid_blobs": [],
            "dirt_blobs": [],
            "grid_spacing": 0,
            "grid_angle": 0,
            "ransac_result": None,
        }

    # Step 1: Find candidate grid spacings
    candidate_spacings = find_candidate_spacings(blob_positions)

    if not candidate_spacings:
        return {
            "image": str(image_path),
            "total_blobs": len(blob_positions),
            "grid_blobs": [],
            "dirt_blobs": [{**b, "classification": "dirt"} for b in detected_blobs],
            "grid_spacing": 0,
            "grid_angle": 0,
            "ransac_result": None,
        }

    # Step 2: RANSAC 2D grid fitting
    ransac_result = ransac_2d_grid_fitting(
        blob_positions, candidate_spacings,
        n_iterations=n_ransac_iters, tolerance=8.0
    )

    # Classify blobs
    grid_blobs = []
    dirt_blobs = []

    for i, blob_info in enumerate(detected_blobs):
        blob_info = dict(blob_info)

        if i in ransac_result["inlier_indices"]:
            blob_info["classification"] = "grid"
            grid_blobs.append(blob_info)
        else:
            blob_info["classification"] = "dirt"
            dirt_blobs.append(blob_info)

    origin = ransac_result["origin"]
    angle_deg = math.degrees(ransac_result["angle"])

    return {
        "image": str(image_path),
        "total_blobs": len(blob_positions),
        "grid_blobs": grid_blobs,
        "dirt_blobs": dirt_blobs,
        "grid_spacing": round(ransac_result["spacing"], 2),
        "grid_angle": round(ransac_result["angle"], 4),
        "grid_angle_deg": round(angle_deg, 3),
        "lattice": {
            "spacing_px": round(ransac_result["spacing"], 2),
            "angle_rad": round(ransac_result["angle"], 6),
            "angle_deg": round(angle_deg, 3),
            "origin": [round(float(origin[0]), 2), round(float(origin[1]), 2)] if origin else None,
            "orientation_support": ransac_result["orientation_support"],
        },
        "ransac_result": {
            "origin": list(ransac_result["origin"]) if ransac_result["origin"] else None,
            "n_inliers": ransac_result["n_inliers"],
            "n_total": ransac_result["n_total"],
            "orientation_support": ransac_result["orientation_support"],
        },
    }


def draw_classification(image_path: str, result: dict,
                        out_path_dirt: str, out_path_grid: str,
                        out_path_combined: str = None):
    """Draw versions: dirt marked, grid only, and combined."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    h, w, _ = img.shape

    # Version 1: Mark dirt blobs in red
    img_dirt = img.copy()
    for b in result["dirt_blobs"]:
        x, y = int(b["x"]), int(b["y"])
        cv2.circle(img_dirt, (x, y), 8, (0, 0, 255), -1)
        cv2.circle(img_dirt, (x, y), 12, (0, 0, 255), 2)

    # Version 2: Only show grid blobs
    img_grid = img.copy()
    for b in result["grid_blobs"]:
        x, y = int(b["x"]), int(b["y"])
        cv2.circle(img_grid, (x, y), 3, (0, 255, 0), -1)
        cv2.rectangle(img_grid, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 0), 1)

    # Version 3: Combined
    img_combined = img.copy()
    for b in result["grid_blobs"]:
        x, y = int(b["x"]), int(b["y"])
        cv2.circle(img_combined, (x, y), 3, (0, 255, 0), -1)
        cv2.rectangle(img_combined, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 0), 1)
    for b in result["dirt_blobs"]:
        x, y = int(b["x"]), int(b["y"])
        cv2.circle(img_combined, (x, y), 8, (0, 0, 255), -1)
        cv2.circle(img_combined, (x, y), 12, (0, 0, 255), 2)

    # Add labels
    cv2.putText(img_dirt, f"DIRT: {len(result['dirt_blobs'])}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img_grid, f"GRID: {len(result['grid_blobs'])}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img_combined, f"GRID: {len(result['grid_blobs'])} | DIRT: {len(result['dirt_blobs'])}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Save
    cv2.imwrite(str(out_path_dirt), img_dirt)
    cv2.imwrite(str(out_path_grid), img_grid)
    if out_path_combined:
        cv2.imwrite(str(out_path_combined), img_combined)

    return img_dirt, img_grid, img_combined


def draw_diagnostic_panel(image_path: str, result: dict, out_path: str):
    """Draw before, accepted lattice, and rejected dirt side by side."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None

    before = img.copy()
    accepted = img.copy()
    rejected = img.copy()

    all_blobs = result["grid_blobs"] + result["dirt_blobs"]
    for b in all_blobs:
        x, y = int(b["x"]), int(b["y"])
        cv2.circle(before, (x, y), 4, (255, 255, 0), -1)
        cv2.circle(before, (x, y), 8, (255, 255, 0), 1)

    for b in result["grid_blobs"]:
        x, y = int(b["x"]), int(b["y"])
        cv2.circle(accepted, (x, y), 4, (0, 255, 0), -1)
        cv2.rectangle(accepted, (x - 6, y - 6), (x + 6, y + 6), (0, 255, 0), 1)

    for b in result["dirt_blobs"]:
        x, y = int(b["x"]), int(b["y"])
        cv2.circle(rejected, (x, y), 5, (0, 0, 255), -1)
        cv2.circle(rejected, (x, y), 11, (0, 0, 255), 2)

    lattice = result.get("lattice") or {}
    spacing = lattice.get("spacing_px", result.get("grid_spacing", 0))
    angle_deg = lattice.get("angle_deg", result.get("grid_angle_deg", 0))
    origin = lattice.get("origin")
    origin_text = f"origin=({origin[0]:.1f},{origin[1]:.1f})" if origin else "origin=None"

    panels = [
        (before, f"BEFORE: {result['total_blobs']} blobs"),
        (accepted, f"ACCEPTED LATTICE: {len(result['grid_blobs'])}"),
        (rejected, f"REJECTED DIRT: {len(result['dirt_blobs'])}"),
    ]

    header_h = 66
    rendered = []
    for panel, title in panels:
        canvas = cv2.copyMakeBorder(panel, header_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(canvas, f"spacing={spacing:.2f}px angle={angle_deg:.2f}deg",
                    (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
        cv2.putText(canvas, origin_text, (12, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
        rendered.append(canvas)

    panel = np.hstack(rendered)
    cv2.imwrite(str(out_path), panel)
    return panel


def main():
    parser = argparse.ArgumentParser(description="RANSAC 2D grid fitting: find largest equidistant marker set.")
    parser.add_argument("path", nargs="+", help="Image file(s) or directory")
    parser.add_argument("--min-area", type=int, default=5, help="Min blob area (default: 5)")
    parser.add_argument("--max-area", type=int, default=5000, help="Max blob area (default: 5000)")
    parser.add_argument("--ransac-iters", type=int, default=200,
                        help="RANSAC iterations (default: 200)")
    parser.add_argument("--output-dir", type=str, default="grid_ransac_results",
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    paths = []
    for p in args.path:
        if Path(p).is_file():
            paths.append(Path(p))
        else:
            paths.extend(sorted(Path(p).glob("*.png")) + sorted(Path(p).glob("*.tif")) + sorted(Path(p).glob("*.tiff")) + sorted(Path(p).glob("*.jpg")) + sorted(Path(p).glob("*.jpeg")))

    all_results = []

    for img_path in paths:
        print(f"Processing {img_path.name}...")
        result = process_image(str(img_path), args.min_area, args.max_area, args.ransac_iters)

        # Draw results
        img_dirt, img_grid, img_combined = draw_classification(
            str(img_path), result,
            out_dir / f"dirt_{img_path.name}",
            out_dir / f"grid_{img_path.name}",
            out_dir / f"combined_{img_path.name}",
        )
        draw_diagnostic_panel(
            str(img_path), result,
            out_dir / f"panel_{img_path.name}",
        )

        ransac = result.get("ransac_result")
        print(f"  Total: {result['total_blobs']}, Grid: {len(result['grid_blobs'])}, "
              f"Dirt: {len(result['dirt_blobs'])}", end="")
        if ransac and ransac.get("n_inliers"):
            lattice = result.get("lattice", {})
            origin = lattice.get("origin")
            origin_text = f", origin=({origin[0]:.1f},{origin[1]:.1f})" if origin else ""
            print(f", RANSAC: {ransac['n_inliers']}/{ransac['n_total']} "
                  f"(spacing={lattice.get('spacing_px', 0):.2f}px, "
                  f"angle={lattice.get('angle_deg', 0):.2f}deg"
                  f"{origin_text}, support={lattice.get('orientation_support', 0)})")
        else:
            print()

        all_results.append(result)

    # Save JSON results
    json_path = out_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
