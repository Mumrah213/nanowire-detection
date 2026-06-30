#!/usr/bin/env python3
"""Visualize dimension measurements for candidates to understand discrepancies."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_debug_panel(debug_dir: Path, stem: str, component: int) -> np.ndarray | None:
    """Load the segmentation debug panel for a component."""
    panel_path = debug_dir / f"{stem}_component_{component:04d}_segmentation.png"
    if panel_path.exists():
        return cv2.imread(str(panel_path))
    return None


def create_dimension_comparison(
    records: list[dict],
    debug_dir: Path,
    stem: str,
    output_path: Path,
    max_candidates: int = 12,
) -> None:
    """Create a visualization comparing dimension measurements across candidates."""

    # Sort by discrepancy between skeleton and bbox length
    def length_discrepancy(r):
        if r["length_bbox_nm"] > 0:
            return abs(r["length_skeleton_nm"] - r["length_bbox_nm"]) / r["length_bbox_nm"]
        return 0

    sorted_records = sorted(records, key=length_discrepancy, reverse=True)[:max_candidates]

    panels = []
    for r in sorted_records:
        panel = load_debug_panel(debug_dir, stem, r["component"])
        if panel is None:
            continue

        # Extract just the first two panels (raw and proposal) from the debug image
        h, w = panel.shape[:2]
        panel_w = w // 4
        panel_h = h // 2

        raw = panel[0:panel_h, 0:panel_w]
        proposal = panel[0:panel_h, panel_w:2*panel_w]
        skeleton = panel[panel_h:, 0:panel_w]
        pca = panel[panel_h:, 2*panel_w:3*panel_w]

        # Combine into a 2x2 grid
        top = np.hstack([raw, proposal])
        bottom = np.hstack([skeleton, pca])
        combined = np.vstack([top, bottom])

        # Add measurement annotations
        info_h = 60
        info_bar = np.full((info_h, combined.shape[1], 3), 32, np.uint8)

        font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.4
        tier = r["category"]
        comp = r["component"]

        cv2.putText(info_bar, f"#{comp} ({tier})", (4, 14), font, fs, (200, 200, 200), 1, cv2.LINE_AA)

        # Length comparison
        l_skel = r["length_skeleton_nm"]
        l_bbox = r["length_bbox_nm"]
        l_diff = abs(l_skel - l_bbox) / max(l_bbox, 1) * 100
        color = (100, 200, 100) if l_diff < 20 else (100, 180, 255) if l_diff < 50 else (100, 100, 255)
        cv2.putText(info_bar, f"L: skel={l_skel:.0f} bbox={l_bbox:.0f} ({l_diff:.0f}% diff)",
                    (4, 30), font, fs, color, 1, cv2.LINE_AA)

        # Diameter comparison
        d_pca = r["diameter_pca_nm"]
        d_topo = r["diameter_topo_nm"]
        d_area = r["diameter_area_nm"]
        cv2.putText(info_bar, f"d: pca={d_pca:.0f} topo={d_topo:.0f} area={d_area:.0f}",
                    (4, 46), font, fs, (200, 200, 200), 1, cv2.LINE_AA)

        panels.append(np.vstack([combined, info_bar]))

    if not panels:
        print("No panels to visualize")
        return

    # Arrange in a grid
    n = len(panels)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    # Pad panels to same size
    max_h = max(p.shape[0] for p in panels)
    max_w = max(p.shape[1] for p in panels)

    padded = []
    for p in panels:
        if p.shape[0] < max_h or p.shape[1] < max_w:
            new_p = np.full((max_h, max_w, 3), 24, np.uint8)
            new_p[:p.shape[0], :p.shape[1]] = p
            padded.append(new_p)
        else:
            padded.append(p)

    # Fill remaining slots
    while len(padded) < rows * cols:
        padded.append(np.full((max_h, max_w, 3), 24, np.uint8))

    # Assemble grid
    grid_rows = []
    for row in range(rows):
        row_panels = padded[row * cols:(row + 1) * cols]
        grid_rows.append(np.hstack(row_panels))

    grid = np.vstack(grid_rows)

    # Add title
    title_h = 30
    title_bar = np.full((title_h, grid.shape[1], 3), 24, np.uint8)
    cv2.putText(title_bar, f"Dimension measurement comparison (sorted by length discrepancy)",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    final = np.vstack([title_bar, grid])
    cv2.imwrite(str(output_path), final)
    print(f"Wrote {output_path}")


def main():
    import argparse
    from dimension_analysis import extract_dimensions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",
                        default="experimental_sem_results/nanowire_coordinate_candidates/13",
                        help="Directory containing pipeline results")
    parser.add_argument("--output", default="docs/images/dimension_comparison.png")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    stem = results_dir.name

    fused_rows = list((results_dir / "nanowire" / "high_precision").glob("*_fused_rows.json"))[0]
    grid_detail = list((results_dir / "grid" / "details").glob("*_pipeline.json"))[0]
    debug_dir = results_dir / "nanowire" / "segmentation" / "debug_panels"

    records = extract_dimensions(fused_rows, grid_detail)

    output_path = ROOT / args.output
    create_dimension_comparison(records, debug_dir, stem, output_path)


if __name__ == "__main__":
    main()
