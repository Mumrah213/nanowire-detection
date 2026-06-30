#!/usr/bin/env python3
"""Extract nanowire dimensions from all candidates and compare measurement approaches.

Gathers geometric features (skeleton length, PCA width, bbox, area, etc.) for all
candidates regardless of tier, enabling statistical analysis and comparison of
different measurement methods.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def extract_dimensions(fused_rows_path: Path, grid_detail_path: Path) -> list[dict]:
    """Extract dimension features from all candidates."""
    rows = json.loads(fused_rows_path.read_text())
    grid_detail = json.loads(grid_detail_path.read_text())

    scale = grid_detail.get("scale", {})
    nm_per_px = float(scale.get("nm_per_px", 24.6))

    records = []
    for row in rows:
        # Skip non-elongated objects (markers, noise)
        if row.get("topology_aspect", 0) < 2.0:
            continue

        # Geometric measurements (current approach)
        skeleton_px = float(row.get("pruned_skeleton_pixels") or row.get("topology_skeleton_pixels") or 0)
        width_pca_px = float(row.get("pca_perp_p90_px") or 0)
        width_topo_px = float(row.get("topology_estimated_width_px") or 0)

        # Alternative measurements for comparison
        bbox = row.get("bbox", [0, 0, 0, 0])
        bbox_major = max(bbox[2], bbox[3])
        bbox_minor = min(bbox[2], bbox[3])
        area_px = float(row.get("topology_area") or row.get("proposal_area") or 0)

        # PCA-based length estimate (major axis)
        pca_aspect = float(row.get("pca_aspect") or row.get("line_aspect") or 1.0)
        pca_minor_major = float(row.get("pca_minor_major_ratio") or 0.1)

        # Compute dimensions in nm using different methods
        length_skeleton_nm = skeleton_px * nm_per_px
        length_bbox_nm = bbox_major * nm_per_px

        # Width: compare PCA 90th percentile vs topology estimate
        diameter_pca_nm = 2.0 * width_pca_px * nm_per_px
        diameter_topo_nm = width_topo_px * nm_per_px
        diameter_bbox_nm = bbox_minor * nm_per_px

        # Area-based diameter estimate (assuming roughly cylindrical)
        if skeleton_px > 0:
            diameter_area_nm = (area_px / skeleton_px) * nm_per_px
        else:
            diameter_area_nm = 0.0

        records.append({
            "component": row.get("component"),
            "category": row.get("candidate_tier"),
            "final_label": row.get("final_label"),
            "confidence": round(float(row.get("final_confidence", 0.0)), 3),

            # Raw pixel measurements
            "skeleton_px": skeleton_px,
            "width_pca_px": width_pca_px,
            "width_topo_px": width_topo_px,
            "bbox_major_px": bbox_major,
            "bbox_minor_px": bbox_minor,
            "area_px": area_px,
            "pca_aspect": round(pca_aspect, 2),

            # Length estimates (nm)
            "length_skeleton_nm": round(length_skeleton_nm, 1),
            "length_bbox_nm": round(length_bbox_nm, 1),

            # Diameter estimates (nm)
            "diameter_pca_nm": round(diameter_pca_nm, 1),
            "diameter_topo_nm": round(diameter_topo_nm, 1),
            "diameter_bbox_nm": round(diameter_bbox_nm, 1),
            "diameter_area_nm": round(diameter_area_nm, 1),

            # Reject reasons for analysis
            "reject_reasons": row.get("reject_reasons", ""),
        })

    return records


def print_statistics(records: list[dict]) -> None:
    """Print summary statistics comparing measurement methods."""
    import statistics

    singles = [r for r in records if r["final_label"] == "single"]
    reviews = [r for r in records if r["final_label"] == "review"]
    bads = [r for r in records if r["final_label"] == "bad"]

    print(f"\nDimension analysis: {len(records)} elongated candidates")
    print(f"  single: {len(singles)}, review: {len(reviews)}, bad: {len(bads)}")

    if not singles:
        print("  No single candidates to analyze")
        return

    # Length comparison
    len_skel = [r["length_skeleton_nm"] for r in singles]
    len_bbox = [r["length_bbox_nm"] for r in singles]

    print(f"\nLength estimates (single candidates, n={len(singles)}):")
    print(f"  skeleton:  mean={statistics.mean(len_skel):.0f} nm, "
          f"std={statistics.stdev(len_skel) if len(len_skel) > 1 else 0:.0f} nm, "
          f"range=[{min(len_skel):.0f}, {max(len_skel):.0f}]")
    print(f"  bbox:      mean={statistics.mean(len_bbox):.0f} nm, "
          f"std={statistics.stdev(len_bbox) if len(len_bbox) > 1 else 0:.0f} nm, "
          f"range=[{min(len_bbox):.0f}, {max(len_bbox):.0f}]")

    # Diameter comparison
    dia_pca = [r["diameter_pca_nm"] for r in singles]
    dia_topo = [r["diameter_topo_nm"] for r in singles]
    dia_area = [r["diameter_area_nm"] for r in singles]

    print(f"\nDiameter estimates (single candidates, n={len(singles)}):")
    print(f"  pca_p90:   mean={statistics.mean(dia_pca):.0f} nm, "
          f"std={statistics.stdev(dia_pca) if len(dia_pca) > 1 else 0:.0f} nm, "
          f"range=[{min(dia_pca):.0f}, {max(dia_pca):.0f}]")
    print(f"  topology:  mean={statistics.mean(dia_topo):.0f} nm, "
          f"std={statistics.stdev(dia_topo) if len(dia_topo) > 1 else 0:.0f} nm, "
          f"range=[{min(dia_topo):.0f}, {max(dia_topo):.0f}]")
    print(f"  area/len:  mean={statistics.mean(dia_area):.0f} nm, "
          f"std={statistics.stdev(dia_area) if len(dia_area) > 1 else 0:.0f} nm, "
          f"range=[{min(dia_area):.0f}, {max(dia_area):.0f}]")

    # Correlation between methods
    if len(singles) > 2:
        def correlation(x, y):
            n = len(x)
            mean_x, mean_y = sum(x)/n, sum(y)/n
            cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
            std_x = (sum((xi - mean_x)**2 for xi in x) / n) ** 0.5
            std_y = (sum((yi - mean_y)**2 for yi in y) / n) ** 0.5
            return cov / (std_x * std_y) if std_x > 0 and std_y > 0 else 0

        print(f"\nCorrelations (single candidates):")
        print(f"  length:   skeleton vs bbox = {correlation(len_skel, len_bbox):.3f}")
        print(f"  diameter: pca vs topology  = {correlation(dia_pca, dia_topo):.3f}")
        print(f"  diameter: pca vs area/len  = {correlation(dia_pca, dia_area):.3f}")


def print_table(records: list[dict], max_rows: int = 30) -> None:
    """Print a pandas-style table of dimensions."""
    print(f"\n{'comp':>5} {'tier':>7} {'L_skel':>7} {'L_bbox':>7} {'d_pca':>6} {'d_topo':>6} {'d_area':>6} {'aspect':>6}")
    print("-" * 62)

    for r in records[:max_rows]:
        print(f"{r['component']:>5} {r['category']:>7} "
              f"{r['length_skeleton_nm']:>7.0f} {r['length_bbox_nm']:>7.0f} "
              f"{r['diameter_pca_nm']:>6.0f} {r['diameter_topo_nm']:>6.0f} "
              f"{r['diameter_area_nm']:>6.0f} {r['pca_aspect']:>6.1f}")

    if len(records) > max_rows:
        print(f"  ... ({len(records) - max_rows} more rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",
                        default="experimental_sem_results/nanowire_coordinate_candidates/13",
                        help="Directory containing pipeline results")
    parser.add_argument("--output", help="Write records to JSON file")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    fused_rows = results_dir / "nanowire" / "high_precision" / f"{results_dir.name}_fused_rows.json"
    grid_detail = results_dir / "grid" / "details" / f"{results_dir.name}_pipeline.json"

    if not fused_rows.exists():
        # Try to find the file with different naming
        fused_candidates = list((results_dir / "nanowire" / "high_precision").glob("*_fused_rows.json"))
        if fused_candidates:
            fused_rows = fused_candidates[0]
        else:
            print(f"Could not find fused_rows.json in {results_dir}")
            sys.exit(1)

    if not grid_detail.exists():
        grid_candidates = list((results_dir / "grid" / "details").glob("*_pipeline.json"))
        if grid_candidates:
            grid_detail = grid_candidates[0]
        else:
            print(f"Could not find pipeline.json in {results_dir}")
            sys.exit(1)

    records = extract_dimensions(fused_rows, grid_detail)

    print_table(records)
    print_statistics(records)

    if args.output:
        Path(args.output).write_text(json.dumps(records, indent=2))
        print(f"\nWrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
