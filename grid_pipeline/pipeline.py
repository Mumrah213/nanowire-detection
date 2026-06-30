#!/usr/bin/env python3
"""Grid-first SEM pipeline with geometry/template L-orientation inference."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Each step lives in its own module; this file just wires them together.
from utils.sem_preprocess import preprocess  # noqa: E402
from grid_pipeline.config import GridConfig  # noqa: E402
from grid_pipeline.detect import (  # noqa: E402
    dark_dot_binary,
    l_candidate_binary,
    compact_dot_candidates,
    adaptive_bright_binary,
)
from grid_pipeline.lattice import (  # noqa: E402
    choose_dot_grid,
    classify_grid_markers,
    split_removed,
    serializable_lattice,
    predict_big_l_positions,
)
from grid_pipeline.l_markers import (  # noqa: E402
    find_l_candidates,
    enforce_l_scale_consistency,
    select_l_candidate,
    candidate_summary,
)
from grid_pipeline.draw import draw_overlay  # noqa: E402


def process_image_variant(path: Path, out_dir: Path, args, variant_name: str | None = None) -> dict:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {path}")
    contrast, bright_binary_strict, report = preprocess(gray)
    bright_cut = report["contrast"]["threshold"]
    bright_binary = bright_binary_strict
    if args.adaptive_bright:
        bright_binary, bright_cut = adaptive_bright_binary(gray, report, bright_binary_strict, args)
    dark_binary, dark_threshold = dark_dot_binary(gray, report, args.dark_dot_sigma)
    l_binary, l_threshold = l_candidate_binary(gray, report, args.l_sigma)
    cutoff_y = report["annotation_band"]["cutoff_y"]
    stem = path.stem

    bright_dots_strict = compact_dot_candidates(bright_binary_strict, cutoff_y)
    bright_dots = compact_dot_candidates(bright_binary, cutoff_y)
    dark_dots = compact_dot_candidates(dark_binary, cutoff_y)
    # Fit the lattice on the confident (global-threshold) dots so flooding from
    # adaptive detection cannot corrupt lattice finding; then classify the full
    # (adaptive) candidate set against that lattice.
    dot_polarity, _, lattice = choose_dot_grid(bright_dots_strict, dark_dots)
    detected_dots = bright_dots if dot_polarity == "bright" else dark_dots
    dots, rejected_dots, lattice = classify_grid_markers(detected_dots, lattice, args)
    if lattice and len(dots) < 6:
        lattice = None
    dot_tol_px = (
        max(args.dot_min_lattice_tolerance_px, args.dot_lattice_tolerance_fraction * float(lattice["spacing"]))
        if lattice and lattice.get("spacing") else 0.0
    )
    removed_near_node, removed_noise = split_removed(rejected_dots, lattice, band_px=1.3 * dot_tol_px)
    n_detected = len(dots)
    scale = {
        "grid_pitch_nm": float(args.grid_pitch_nm),
        "spacing_px": float(lattice["spacing"]) if lattice else None,
        "px_per_nm": float(lattice["spacing"]) / float(args.grid_pitch_nm) if lattice else None,
        "nm_per_px": float(args.grid_pitch_nm) / float(lattice["spacing"]) if lattice else None,
    }
    candidates = find_l_candidates(gray, l_binary, cutoff_y, lattice, args, out_dir, stem)
    l_scale_rejections, l_scale_reference = enforce_l_scale_consistency(candidates, lattice, args)
    selected, selection_notes = select_l_candidate(candidates, lattice, args)
    accepted_candidates = [
        candidate for candidate in candidates
        if candidate.get("accepted_l_candidate", False)
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{stem}_contrast.png"), contrast)
    cv2.imwrite(str(out_dir / f"{stem}_bright_binary.png"), bright_binary)
    cv2.imwrite(str(out_dir / f"{stem}_dark_binary.png"), dark_binary)
    cv2.imwrite(str(out_dir / f"{stem}_l_binary.png"), l_binary)
    predicted_big_ls = predict_big_l_positions(selected, lattice, args) if args.predict_big_l else []
    overlay_path = out_dir / f"{stem}_overlay.png"
    panel_path = (out_dir / f"{stem}_overlay_panel.png") if args.l_panel else None
    expected_missing = draw_overlay(gray, dots, lattice, candidates, selected, report, scale, overlay_path, removed_near_node, panel_path, predicted_big_ls, args.marker_coords)

    lattice_found = bool(lattice)
    inliers = int(lattice["n_inliers"]) if lattice else 0
    total = int(lattice["n_total"]) if lattice else len(dots)
    inlier_ratio = inliers / total if total else 0.0
    quality = "coordinate_ready" if lattice_found and selected and inlier_ratio >= 0.50 else "review"
    if not lattice_found:
        quality = "no_grid"
    elif not selected:
        quality = "grid_only"

    result = {
        "image": str(path),
        "overlay": str(overlay_path),
        "overlay_panel": str(panel_path) if panel_path else None,
        "variant_name": variant_name or "single",
        "dark_dot_sigma": float(args.dark_dot_sigma),
        "l_sigma": float(args.l_sigma),
        "quality": quality,
        "dot_polarity": dot_polarity,
        "bright_dot_candidates": len(bright_dots),
        "dark_dot_candidates": len(dark_dots),
        "dot_candidates": len(dots),
        "detected_dots": n_detected,
        "dot_like_components": n_detected + len(rejected_dots),
        "rejected_dot_like_components": len(rejected_dots),
        "removed_near_node": len(removed_near_node),
        "removed_noise": int(removed_noise),
        "expected_missing_markers": int(expected_missing or 0),
        "lattice_found": lattice_found,
        "lattice_inliers": inliers,
        "lattice_total": total,
        "inlier_ratio": inlier_ratio,
        "scale": scale,
        "l_candidates": len(accepted_candidates),
        "l_like_components": len(candidates),
        "rejected_l_like_components": len(candidates) - len(accepted_candidates),
        "l_scale_rejections": int(l_scale_rejections),
        "l_scale_reference": l_scale_reference,
        "l_candidate_counts": {
            "size": dict(Counter(candidate["size"] for candidate in accepted_candidates)),
            "orientation": dict(Counter(candidate["orientation"] for candidate in accepted_candidates)),
            "label": dict(Counter(candidate["label_size_orientation"] for candidate in accepted_candidates)),
            "rejection_reasons": dict(Counter(
                reason
                for candidate in candidates
                for reason in candidate.get("rejection_reasons", [])
            )),
        },
        "selected_l": selected["label_size_orientation"] if selected else None,
        "selected_orientation": selected["orientation"] if selected else None,
        "selected_size": selected["size"] if selected else None,
        "selected_confidence": selected["confidence"] if selected else None,
        "selected_anchor_px": selected["anchor_px"] if selected else None,
        "selected_anchor_nm": selected["anchor_nm"] if selected else None,
        "most_likely_candidate": candidate_summary(selected),
        "selection_notes": selection_notes,
    }
    detail = {
        **result,
        "thresholds": {
            "bright_threshold": report["contrast"]["threshold"],
            "dark_threshold": dark_threshold,
            "l_threshold": l_threshold,
        },
        "preprocess": report,
        "scale": scale,
        "lattice": serializable_lattice(lattice),
        "dot_candidates_detail": dots,
        "rejected_dot_like_components_detail": removed_near_node,
        "thresholds_bright_cut": float(bright_cut),
        "expected_missing_markers_detail": int(expected_missing or 0),
        "l_candidates_detail": accepted_candidates,
        "rejected_l_like_components_detail": [
            candidate for candidate in candidates
            if not candidate.get("accepted_l_candidate", False)
        ],
    }
    (out_dir / f"{stem}_pipeline.json").write_text(json.dumps(detail, indent=2))
    return result


def parse_float_list(value: str) -> list[float]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def outcome_score(result: dict) -> float:
    quality_bonus = {
        "coordinate_ready": 100.0,
        "grid_only": 45.0,
        "review": 25.0,
        "no_grid": 0.0,
    }.get(str(result.get("quality")), 0.0)
    selected_bonus = 20.0 if result.get("selected_l") else 0.0
    confidence = float(result.get("selected_confidence") or 0.0)
    inlier_ratio = float(result.get("inlier_ratio") or 0.0)
    inliers = min(30, int(result.get("lattice_inliers") or 0))
    candidates = int(result.get("l_candidates") or 0)
    rejected = int(result.get("rejected_l_like_components") or 0)
    candidate_penalty = max(0, candidates - 3) * 4.0 + min(12, rejected) * 0.5
    return quality_bonus + selected_bonus + 18.0 * confidence + 25.0 * inlier_ratio + inliers - candidate_penalty


def publish_winning_outputs(result: dict, out_dir: Path, stem: str) -> None:
    overlays_dir = out_dir / "overlays"
    details_dir = out_dir / "details"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)

    source_overlay = Path(result["overlay"])
    published_overlay = overlays_dir / f"{stem}_overlay.png"
    if source_overlay.exists():
        shutil.copy2(source_overlay, published_overlay)
        result["diagnostic_overlay"] = str(source_overlay)
        result["overlay"] = str(published_overlay)

    if result.get("overlay_panel"):
        source_panel = Path(result["overlay_panel"])
        if source_panel.exists():
            published_panel = overlays_dir / f"{stem}_overlay_panel.png"
            shutil.copy2(source_panel, published_panel)
            result["overlay_panel"] = str(published_panel)

    source_detail = source_overlay.with_name(f"{stem}_pipeline.json")
    published_detail = details_dir / f"{stem}_pipeline.json"
    if source_detail.exists():
        shutil.copy2(source_detail, published_detail)
        result["diagnostic_detail"] = str(source_detail)
        result["detail"] = str(published_detail)


def process_image(path: Path, out_dir: Path, args) -> dict:
    if not args.contrast_sweep:
        variant_dir = out_dir / "_diagnostics" / path.stem / "single"
        result = process_image_variant(path, variant_dir, args)
        result["contrast_sweep_enabled"] = False
        result["contrast_sweep_score"] = outcome_score(result)
        publish_winning_outputs(result, out_dir, path.stem)
        return result

    dark_values = parse_float_list(args.dark_dot_sigma_values)
    l_values = parse_float_list(args.l_sigma_values)
    sweep_root = out_dir / "_contrast_sweep" / path.stem
    sweep_rows = []

    for dark_sigma in dark_values:
        for l_sigma in l_values:
            variant_args = copy.copy(args)
            variant_args.dark_dot_sigma = float(dark_sigma)
            variant_args.l_sigma = float(l_sigma)
            variant_name = f"dark{dark_sigma:g}_l{l_sigma:g}".replace(".", "p")
            variant_dir = sweep_root / variant_name
            result = process_image_variant(path, variant_dir, variant_args, variant_name=variant_name)
            result["contrast_sweep_enabled"] = True
            result["contrast_sweep_score"] = outcome_score(result)
            sweep_rows.append(result)

    sweep_rows.sort(key=lambda row: row["contrast_sweep_score"], reverse=True)
    best = dict(sweep_rows[0])
    best["contrast_sweep_variants"] = [
        {
            "variant_name": row["variant_name"],
            "dark_dot_sigma": row["dark_dot_sigma"],
            "l_sigma": row["l_sigma"],
            "quality": row["quality"],
            "lattice_inliers": row["lattice_inliers"],
            "lattice_total": row["lattice_total"],
            "inlier_ratio": row["inlier_ratio"],
            "l_candidates": row["l_candidates"],
            "l_like_components": row.get("l_like_components"),
            "rejected_l_like_components": row.get("rejected_l_like_components"),
            "selected_l": row["selected_l"],
            "selected_confidence": row["selected_confidence"],
            "most_likely_candidate": row.get("most_likely_candidate"),
            "score": row["contrast_sweep_score"],
            "overlay": row["overlay"],
        }
        for row in sweep_rows
    ]
    (sweep_root / "sweep_summary.json").write_text(json.dumps(best["contrast_sweep_variants"], indent=2))
    publish_winning_outputs(best, out_dir, path.stem)
    return best


def iter_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    suffixes = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    return sorted(path for path in input_path.iterdir() if path.suffix.lower() in suffixes)


def write_summary(rows: list[dict], out_dir: Path) -> None:
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    fieldnames = [
        "image", "quality", "dot_polarity", "dot_candidates",
        "detected_dots", "lattice_found",
        "lattice_inliers", "lattice_total", "inlier_ratio", "l_candidates",
        "dot_like_components", "rejected_dot_like_components", "removed_near_node", "removed_noise", "expected_missing_markers",
        "l_like_components", "rejected_l_like_components",
        "selected_l", "selected_orientation", "selected_size", "selected_confidence",
        "contrast_sweep_score", "variant_name", "dark_dot_sigma", "l_sigma",
        "overlay", "detail", "diagnostic_overlay", "diagnostic_detail",
    ]
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def parse_args() -> GridConfig:
    """Build the config from defaults, overriding only the runtime essentials.

    Everything else is tuned by editing grid_pipeline/config.py.
    """
    cfg = GridConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=cfg.input)
    parser.add_argument("--output-dir", default=cfg.output_dir)
    parser.add_argument("--no-contrast-sweep", dest="contrast_sweep",
                        action="store_false", default=cfg.contrast_sweep)
    args = parser.parse_args()
    cfg.input = args.input
    cfg.output_dir = args.output_dir
    cfg.contrast_sweep = args.contrast_sweep
    return cfg


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in iter_images(Path(args.input)):
        result = process_image(image_path, out_dir, args)
        rows.append(result)
        selected = result["selected_l"] or "no_L"
        variant = result.get("variant_name", "single")
        print(
            f"{image_path}: {result['quality']} grid={result['lattice_inliers']}/{result['lattice_total']} "
            f"most_likely={selected} candidates={result['l_candidates']} "
            f"variant={variant} score={result.get('contrast_sweep_score', 0.0):.2f}"
        )
    write_summary(rows, out_dir)
    print(out_dir / "summary.csv")


if __name__ == "__main__":
    main()
