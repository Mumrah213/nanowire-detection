#!/usr/bin/env python3
"""Tag suitable nanowire candidates with grid (um) coordinates.

Fuses two existing flows:

1. The grid/L coordinate pipeline (``grid_pipeline``) fits the dot
   lattice and one classified L marker, which fixes a physical (um) coordinate
   frame for the field.
2. The nanowire pipeline (``fused_nanowire_pipeline`` or
   ``segmentation_baseline``) proposes nanowire candidates and tiers them
   single / review / bad.

This wrapper runs both, then assigns each *suitable* candidate the physical
coordinate of its location, so a good nanowire can be found again on the device.
It writes a combined overlay and a CSV/JSON table.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grid_pipeline.lattice import pixel_to_um  # noqa: E402
from grid_pipeline.draw import draw_legend_bar, draw_ring, COL_DETECTED  # noqa: E402

# Display categories derived from the fused label + topology reject reasons (BGR).
CATEGORY_COLOR = {
    "single": (95, 185, 95),       # green  - suitable isolated nanowire
    "review": (0, 210, 255),       # amber  - borderline nanowire
    "bunched": (255, 130, 255),    # pink/magenta - crossed / branched / too thick
    "too short": (180, 180, 100),  # cyan-ish - nanowire below length threshold
    "other": (150, 150, 150),      # gray   - markers / electrodes / noise / blobs
}
BUNCHED_REASONS = {"multi_orientation", "too_branched", "too_many_endpoints",
                   "off_axis_mass", "too_broad_pca", "too_wide", "too_thick"}


def detect_diameter_outliers(rows: list[dict], max_diameter_nm: float | None = None) -> float:
    """Compute diameter threshold for outlier detection.

    If max_diameter_nm is set, use that. Otherwise, compute from the population
    of single candidates using median + 3*MAD (robust outlier detection).

    Returns the threshold in pixels (full diameter). Candidates exceeding this
    get marked as too_thick.
    """
    singles = [r for r in rows if r.get("final_label") == "single"]
    if not singles:
        return float("inf")

    # Get diameters: 2 * pca_perp_p90_px (full diameter in pixels)
    diameters_px = [2.0 * r.get("pca_perp_p90_px", 0.0) for r in singles]

    if max_diameter_nm is not None:
        # User specified a hard limit - return sentinel, caller converts
        return max_diameter_nm

    # Need enough samples for robust statistics
    if len(diameters_px) < 6:
        return float("inf")

    # Robust outlier detection: median + 3 * MAD
    median = float(np.median(diameters_px))
    mad = float(np.median(np.abs(np.array(diameters_px) - median)))

    # Guard against very tight distributions (MAD near zero)
    # Use at least 20% of median as minimum spread
    min_spread_px = 0.20 * median
    effective_mad = max(mad, min_spread_px / 1.4826)

    threshold_px = median + 3.0 * 1.4826 * effective_mad

    return threshold_px


def apply_diameter_filter(rows: list[dict], threshold_px: float) -> int:
    """Mark candidates exceeding diameter threshold as too_thick.

    Returns the number of candidates rejected.
    """
    rejected = 0
    for row in rows:
        if row.get("final_label") != "single":
            continue
        diameter_px = 2.0 * row.get("pca_perp_p90_px", 0.0)
        if diameter_px > threshold_px:
            row["final_label"] = "bad"
            reasons = row.get("reject_reasons", "")
            row["reject_reasons"] = f"{reasons},too_thick" if reasons else "too_thick"
            rejected += 1
    return rejected


def display_category(row: dict) -> str:
    label = row.get("final_label")
    if label == "single":
        return "single"
    if label == "review":
        return "review"
    reasons = {r for r in str(row.get("reject_reasons", "")).split(",") if r}
    # Check categories in priority order
    if reasons & BUNCHED_REASONS:
        return "bunched"
    if reasons & {"too_short", "proposal_too_short"}:
        return "too short"
    return "other"


def run_grid_pipeline(image: Path, out_dir: Path) -> dict:
    """Run the grid coordinate pipeline (single variant) and return its detail JSON."""
    cmd = [sys.executable, "-m", "grid_pipeline.pipeline", str(image),
           "--no-contrast-sweep", "--output-dir", str(out_dir)]
    subprocess.run(cmd, cwd=str(ROOT), check=True, capture_output=True, text=True)
    detail = out_dir / "details" / f"{image.stem}_pipeline.json"
    return json.loads(detail.read_text())


def run_nanowire_pipeline(image: Path, out_dir: Path, policy: str) -> list[dict]:
    """Run the fused nanowire pipeline and return its candidate rows."""
    cmd = [sys.executable, "nanowire_ml/fused_nanowire_pipeline.py", str(image),
           "--output-dir", str(out_dir), "--policy", policy]
    subprocess.run(cmd, cwd=str(ROOT), check=True, capture_output=True, text=True)
    rows_path = out_dir / policy / f"{image.stem}_fused_rows.json"
    return json.loads(rows_path.read_text())


def coordinate_frame(grid_detail: dict):
    """Build a pixel->um mapper from the grid detail, or None if no L anchor."""
    lattice = grid_detail.get("lattice")
    anchor_px = grid_detail.get("selected_anchor_px")
    anchor_nm = grid_detail.get("selected_anchor_nm")
    if not lattice or not anchor_px or not anchor_nm:
        return None
    lat = {"spacing": lattice["spacing_px"], "angle": lattice["angle_rad"], "origin": lattice["origin"]}
    pitch = float(grid_detail.get("scale", {}).get("grid_pitch_nm") or 2500.0)
    return lambda px, py: pixel_to_um(px, py, lat, anchor_px, anchor_nm, pitch)


def candidate_center(row: dict) -> tuple[float, float]:
    x, y, w, h = row["bbox"]
    return x + w / 2.0, y + h / 2.0


def label_text(img, x, y, text, color):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_overlay(gray: np.ndarray, grid_detail: dict, tagged: list[dict], suitable: set, out_path: Path) -> None:
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # faint grid context: detected dot markers + selected-L anchor
    for dot in grid_detail.get("dot_candidates_detail", []):
        cv2.circle(img, (int(round(dot["x"])), int(round(dot["y"]))), 2, (110, 110, 110), -1, cv2.LINE_AA)
    anchor = grid_detail.get("selected_anchor_px")
    if anchor:
        ax, ay = int(round(anchor[0])), int(round(anchor[1]))
        cv2.drawMarker(img, (ax, ay), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)

    present = []
    # draw 'other' first (faint), then the rest on top
    order = {"other": 0, "too short": 1, "bunched": 2, "review": 3, "single": 4}
    for row in sorted(tagged, key=lambda r: order.get(r["category"], 0)):
        cat = row["category"]
        if cat not in present:
            present.append(cat)
        x, y, w, h = [int(v) for v in row["bbox"]]
        color = CATEGORY_COLOR[cat]
        thickness = 2 if cat in ("single", "review") else 1
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        um = row.get("um")
        coord = f"({um[0]:.1f},{um[1]:.1f})" if um else ""
        # suitable singles show coordinate only (green box implies the category);
        # everything else is labelled by category + coordinate.
        tag = coord if cat == "single" else f"{cat} {coord}".strip()
        if tag:
            label_text(img, x, max(12, y - 6), tag, color)

    n = sum(1 for r in tagged if r["final_label"] in suitable)
    title = f"suitable nanowire candidates: {n}  |  frame: {grid_detail.get('selected_l') or 'no L'}"
    label_text(img, 12, 26, title, (255, 255, 0))

    legend_labels = {
        "single": "single (suitable)",
        "review": "review",
        "bunched": "bunched/thick",
        "too short": "too short",
        "other": "other",
    }
    legend = [(CATEGORY_COLOR[c], legend_labels[c])
              for c in ("single", "review", "bunched", "too short", "other") if c in present]
    img = draw_legend_bar(img, legend)
    cv2.imwrite(str(out_path), img)


def gentle_contrast(gray: np.ndarray, cutoff_y: int | None = None) -> np.ndarray:
    """Apply gentle contrast normalization (less extreme than robust_contrast)."""
    region = gray[:cutoff_y] if cutoff_y else gray
    # Use p1-p99.5 for a darker, less overexposed look
    p_lo, p_hi = float(np.percentile(region, 1)), float(np.percentile(region, 99.5))
    # Map to 0-220 instead of 0-255 to keep it slightly darker
    stretched = np.clip((gray.astype(np.float32) - p_lo) * 220.0 / max(1, p_hi - p_lo), 0, 220)
    return stretched.astype(np.uint8)


def generate_candidate_crops(
    gray: np.ndarray,
    grid_detail: dict,
    tagged: list[dict],
    suitable: set,
    mapper,
    out_dir: Path,
    stem: str,
    crop_size_um: float = 5.0,
) -> list[dict]:
    """Generate a zoomed crop figure and JSON for each suitable candidate.

    Each crop shows a fixed µm region (default 5x5 µm) centered on the candidate,
    with µm coordinate boundary labels.
    """
    lattice = grid_detail.get("lattice")
    scale = grid_detail.get("scale")
    if not lattice or not scale or mapper is None:
        return []

    nm_per_px = float(scale.get("nm_per_px", 24.6))
    px_per_um = 1000.0 / nm_per_px
    crop_half_px = int(crop_size_um / 2.0 * px_per_um)

    cutoff_y = grid_detail.get("annotation_band", {}).get("cutoff_y")
    dots = grid_detail.get("dot_candidates_detail", [])

    crops_dir = out_dir / "candidates"
    crops_dir.mkdir(parents=True, exist_ok=True)

    gray_norm = gentle_contrast(gray, cutoff_y)
    h_img, w_img = gray.shape
    candidate_records = []

    for row in tagged:
        if row.get("final_label") not in suitable:
            continue

        comp_id = row["component"]
        bbox = row["bbox"]
        cx, cy = candidate_center(row)
        um_center = row.get("um")
        if um_center is None:
            continue

        # Crop region: fixed size centered on candidate
        x0 = max(0, int(cx - crop_half_px))
        y0 = max(0, int(cy - crop_half_px))
        x1 = min(w_img, int(cx + crop_half_px))
        y1 = min(h_img, int(cy + crop_half_px))

        crop = cv2.cvtColor(gray_norm[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
        crop_h, crop_w = crop.shape[:2]

        # Compute µm boundaries of the crop
        um_left, um_top = mapper(x0, y0)
        um_right, um_bottom = mapper(x1, y1)
        # Ensure consistent ordering (left < right, coordinates may be negative)
        um_x_min, um_x_max = min(um_left, um_right), max(um_left, um_right)
        um_y_min, um_y_max = min(um_top, um_bottom), max(um_top, um_bottom)

        # Draw dot markers that fall within the crop
        dot_r = max(4, int(0.02 * crop_half_px))
        for dot in dots:
            dx, dy = dot["x"], dot["y"]
            if x0 <= dx < x1 and y0 <= dy < y1:
                lx, ly = int(round(dx - x0)), int(round(dy - y0))
                draw_ring(crop, lx, ly, COL_DETECTED, dot_r, 1)

        # Compute length and diameter in nm
        # Length: skeleton pixels * nm_per_px (approximates arc length)
        # Diameter: 2 * pca_perp_p90_px * nm_per_px (90th percentile width from PCA axis)
        skeleton_px = float(row.get("pruned_skeleton_pixels") or row.get("topology_skeleton_pixels") or 0)
        width_px = float(row.get("pca_perp_p90_px") or row.get("topology_estimated_width_px") or 0)
        length_nm = skeleton_px * nm_per_px
        diameter_nm = 2.0 * width_px * nm_per_px

        # Draw the candidate bounding box
        bx, by, bw, bh = bbox
        cv2.rectangle(crop, (int(bx - x0), int(by - y0)),
                      (int(bx - x0 + bw), int(by - y0 + bh)),
                      CATEGORY_COLOR["single"], 2)

        # Add boundary frame and µm labels
        font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.35
        border = 2

        # Draw thin border to indicate crop boundary
        cv2.rectangle(crop, (border, border), (crop_w - border - 1, crop_h - border - 1),
                      (180, 180, 180), 1)

        # Corner labels showing µm coordinates
        def put_label(text, pos, anchor="tl"):
            (tw, th), _ = cv2.getTextSize(text, font, fs, 1)
            if anchor == "tl":
                org = (pos[0] + 4, pos[1] + th + 4)
            elif anchor == "tr":
                org = (pos[0] - tw - 4, pos[1] + th + 4)
            elif anchor == "bl":
                org = (pos[0] + 4, pos[1] - 4)
            else:  # br
                org = (pos[0] - tw - 4, pos[1] - 4)
            cv2.putText(crop, text, org, font, fs, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(crop, text, org, font, fs, (220, 220, 220), 1, cv2.LINE_AA)

        # Show x,y bounds at corners
        put_label(f"({um_x_min:.1f}, {um_y_min:.1f})", (0, 0), "tl")
        put_label(f"({um_x_max:.1f}, {um_y_max:.1f})", (crop_w, crop_h), "br")

        # Dimensions label (top right)
        dim_label = f"L={length_nm:.0f}nm d={diameter_nm:.0f}nm"
        put_label(dim_label, (crop_w, 0), "tr")

        # Center label with candidate center (placed above bottom-right corner label)
        center_label = f"center: ({um_center[0]:.2f}, {um_center[1]:.2f}) um"
        (tw, _), _ = cv2.getTextSize(center_label, font, 0.4, 1)
        label_x = (crop_w - tw) // 2
        cv2.putText(crop, center_label, (label_x, crop_h - 18), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(crop, center_label, (label_x, crop_h - 18), font, 0.4, (95, 185, 95), 1, cv2.LINE_AA)

        # Save crop image
        crop_filename = f"{stem}_candidate_{comp_id:04d}.png"
        cv2.imwrite(str(crops_dir / crop_filename), crop)

        # Build candidate record with full coordinate info
        record = {
            "component": comp_id,
            "category": row["category"],
            "final_label": row["final_label"],
            "confidence": round(float(row.get("final_confidence", 0.0)), 3),
            "center_um": [round(um_center[0], 3), round(um_center[1], 3)],
            "bounds_um": {
                "x_min": round(um_x_min, 3),
                "x_max": round(um_x_max, 3),
                "y_min": round(um_y_min, 3),
                "y_max": round(um_y_max, 3),
            },
            "length_nm": round(length_nm, 1),
            "length_um": round(length_nm / 1000.0, 3),
            "diameter_nm": round(diameter_nm, 1),
            "diameter_um": round(diameter_nm / 1000.0, 3),
            "bbox_px": bbox,
            "crop_region_px": [x0, y0, x1 - x0, y1 - y0],
            "crop_file": crop_filename,
        }
        candidate_records.append(record)

        # Write individual JSON for this candidate
        (crops_dir / f"{stem}_candidate_{comp_id:04d}.json").write_text(
            json.dumps(record, indent=2)
        )

    # Write summary JSON with all candidates
    (crops_dir / f"{stem}_candidates_summary.json").write_text(
        json.dumps(candidate_records, indent=2)
    )

    return candidate_records


def write_table(tagged: list[dict], suitable: set, out_dir: Path, stem: str) -> None:
    cols = ["component", "category", "final_label", "final_confidence", "candidate_tier",
            "single_score", "flank_edge_like_fraction", "reject_reasons",
            "bbox", "center_px", "um_x", "um_y", "suitable"]
    records = []
    for row in tagged:
        um = row.get("um")
        records.append({
            "component": row.get("component"),
            "category": row.get("category"),
            "final_label": row.get("final_label"),
            "final_confidence": round(float(row.get("final_confidence", 0.0)), 3),
            "candidate_tier": row.get("candidate_tier"),
            "single_score": round(float(row.get("single_score", 0.0)), 3),
            "flank_edge_like_fraction": round(float(row.get("flank_edge_like_fraction", 0.0)), 3),
            "reject_reasons": row.get("reject_reasons", ""),
            "bbox": row.get("bbox"),
            "center_px": [round(v, 1) for v in candidate_center(row)],
            "um_x": round(um[0], 2) if um else None,
            "um_y": round(um[1], 2) if um else None,
            "suitable": row.get("final_label") in suitable,
        })
    (out_dir / f"{stem}_candidates.json").write_text(json.dumps(records, indent=2))
    with (out_dir / f"{stem}_candidates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(records)


def process(args) -> None:
    image = Path(args.image)
    stem = image.stem
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_detail = run_grid_pipeline(image, out_dir / "grid")
    rows = run_nanowire_pipeline(image, out_dir / "nanowire", args.policy)

    # Apply diameter filter to reject outliers among singles
    # At this point, "single" candidates have passed all other checks (bunched, too_short, etc.)
    scale = grid_detail.get("scale", {})
    nm_per_px = float(scale.get("nm_per_px", 24.6))

    if args.max_diameter_nm is not None:
        # User specified a hard limit - convert nm to px (full diameter)
        threshold_px = args.max_diameter_nm / nm_per_px
        n_rejected = apply_diameter_filter(rows, threshold_px)
        if n_rejected > 0:
            print(f"  -> Rejected {n_rejected} candidate(s) exceeding max diameter {args.max_diameter_nm:.0f} nm")
    else:
        # Auto-detect outliers using robust statistics
        threshold_px = detect_diameter_outliers(rows)
        if threshold_px < float("inf"):
            threshold_nm = threshold_px * nm_per_px  # threshold_px is already full diameter
            n_rejected = apply_diameter_filter(rows, threshold_px)
            if n_rejected > 0:
                print(f"  -> Rejected {n_rejected} diameter outlier(s) exceeding auto-threshold {threshold_nm:.0f} nm")

    mapper = coordinate_frame(grid_detail)
    suitable = set(args.suitable_tiers.split(","))

    for row in rows:
        if mapper is not None:
            cx, cy = candidate_center(row)
            ux, uy = mapper(cx, cy)
            row["um"] = [ux, uy]
        else:
            row["um"] = None
        row["category"] = display_category(row)

    gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    draw_overlay(gray, grid_detail, rows, suitable, out_dir / f"{stem}_candidates_overlay.png")
    write_table(rows, suitable, out_dir, stem)

    # Generate per-candidate crops with µm boundaries
    candidate_crops = generate_candidate_crops(gray, grid_detail, rows, suitable, mapper, out_dir, stem)

    n_suitable = sum(1 for r in rows if r["final_label"] in suitable)
    frame = grid_detail.get("selected_l") or "NO L (coords unavailable)"
    print(f"{image}: {n_suitable} suitable candidate(s) [{args.suitable_tiers}], coordinate frame = {frame}")
    print(out_dir / f"{stem}_candidates_overlay.png")
    if candidate_crops:
        print(f"  -> {len(candidate_crops)} candidate crops in {out_dir / 'candidates'}")

    # Print summary table
    if candidate_crops:
        print("\nSuitable nanowire candidates:")
        print("-" * 85)
        print(f"{'#':>4}  {'x (µm)':>8}  {'y (µm)':>8}  {'length (nm)':>11}  {'diam (nm)':>10}  {'conf':>5}  crop")
        print("-" * 85)
        for c in candidate_crops:
            print(f"{c['component']:>4}  {c['center_um'][0]:>8.2f}  {c['center_um'][1]:>8.2f}  "
                  f"{c['length_nm']:>11.0f}  {c['diameter_nm']:>10.0f}  {c['confidence']:>5.2f}  {c['crop_file']}")
        print("-" * 85)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default="experimental_sem/13.tif")
    parser.add_argument("--output-dir", default="experimental_sem_results/nanowire_coordinate_candidates_13")
    parser.add_argument("--policy", choices=("high_recall", "high_precision"), default="high_precision")
    parser.add_argument("--suitable-tiers", default="single",
                        help="comma-separated final labels treated as suitable (e.g. 'single' or 'single,review')")
    parser.add_argument("--max-diameter-nm", type=float, default=None,
                        help="Maximum nanowire diameter in nm; exceeding candidates are rejected. "
                             "If not set, uses robust outlier detection (median + 3*MAD).")
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
