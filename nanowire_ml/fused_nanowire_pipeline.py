#!/usr/bin/env python3
"""Fused nanowire candidate pipeline: blob proposals + topology + CNN.

The design is deliberately staged:

1. Blob/connected-component proposals from SEM preprocessing.
2. Local segmentation and topology from ``segmentation_baseline``.
3. Optional CNN probability on the same component proposals.
4. Fusion policies for high-recall and high-precision operating points.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sem_preprocess import preprocess  # noqa: E402
from nanowire_ml.predict_real_components import candidate_components, crop_component, crop_component_mask, load_model  # noqa: E402
from nanowire_ml.segmentation_baseline import make_contact_sheet, process_image, robust_normalize  # noqa: E402
from nanowire_ml.crops import preprocess_image  # noqa: E402


DEFAULT_CHECKPOINT = Path("experimental_sem_results/nanowire_ml_pca_jagged_smoke_model/best_model.pt")


def load_cnn_if_available(checkpoint: Path, disabled: bool):
    if disabled:
        return None, None, "disabled"
    if not checkpoint.exists():
        return None, None, f"missing:{checkpoint}"
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device)
    return model, device, str(checkpoint)


def cnn_predictions(gray: np.ndarray, labels: np.ndarray, rows: list[dict], args, out_dir: Path) -> dict[int, dict]:
    model, device, source = load_cnn_if_available(Path(args.checkpoint), args.no_cnn)
    crop_dir = out_dir / "cnn_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    predictions = {}
    if model is None:
        for row in rows:
            predictions[int(row["component"])] = {
                "cnn_available": False,
                "cnn_source": source,
                "cnn_single_prob": None,
                "cnn_crop": "",
            }
        return predictions

    import torch

    model.eval()
    with torch.no_grad():
        for row in rows:
            component = int(row["component"])
            if args.cnn_crop_mode == "mask":
                crop = crop_component_mask(labels, component, row["bbox"], size=args.cnn_crop_size)
            else:
                crop = crop_component(gray, row["bbox"], size=args.cnn_crop_size)
            crop = preprocess_image(crop, args.cnn_preprocess_mode)
            crop_name = f"component_{component:04d}_{args.cnn_crop_mode}_{args.cnn_preprocess_mode}.png"
            cv2.imwrite(str(crop_dir / crop_name), crop)
            x = torch.from_numpy(crop.astype(np.float32)[None, None, :, :] / 255.0).to(device)
            prob = float(torch.sigmoid(model(x)).cpu().numpy().ravel()[0])
            predictions[component] = {
                "cnn_available": True,
                "cnn_source": source,
                "cnn_single_prob": prob,
                "cnn_crop": str(crop_dir / crop_name),
            }
    return predictions


def has_severe_topology_reason(row: dict) -> bool:
    severe = {"too_small", "too_short", "too_wide", "too_thick", "too_broad_pca", "off_axis_mass", "multi_orientation", "contact_edge"}
    reasons = {item for item in str(row.get("reject_reasons", "")).split(",") if item}
    return bool(reasons & severe)


def has_proposal_filter_reason(row: dict) -> bool:
    return any(
        reason.startswith("proposal_")
        for reason in str(row.get("reject_reasons", "")).split(",")
        if reason
    )


def fuse_high_recall(row: dict, args) -> tuple[str, str, float]:
    seg_tier = row["candidate_tier"]
    seg_score = float(row["single_score"])
    cnn_prob = row.get("cnn_single_prob")
    cnn_prob = None if cnn_prob in ("", None) else float(cnn_prob)

    if has_proposal_filter_reason(row):
        return "bad", "proposal_geometry_reject", 1.0

    if seg_tier == "single":
        if cnn_prob is not None and cnn_prob < args.cnn_conflict_low:
            return "single", "topology_single_cnn_low_warning", max(seg_score, 0.80)
        return "single", "topology_single", max(seg_score, cnn_prob or 0.0)

    if seg_tier == "review":
        if cnn_prob is not None and cnn_prob >= args.recall_cnn_promote:
            return "single", "review_promoted_by_cnn", max(seg_score, cnn_prob)
        if seg_score >= args.recall_score_promote:
            return "single", "review_promoted_by_topology_score", seg_score
        return "review", "topology_review", max(seg_score, cnn_prob or 0.0)

    if cnn_prob is not None and cnn_prob >= args.recall_cnn_rescue and not has_severe_topology_reason(row):
        return "review", "bad_rescued_to_review_by_cnn", max(seg_score, cnn_prob)
    return "bad", "topology_bad", max(1.0 - seg_score, 1.0 - (cnn_prob or 0.0))


def fuse_high_precision(row: dict, args) -> tuple[str, str, float]:
    seg_tier = row["candidate_tier"]
    seg_score = float(row["single_score"])
    cnn_prob = row.get("cnn_single_prob")
    cnn_prob = None if cnn_prob in ("", None) else float(cnn_prob)
    context_warning = bool(row.get("context_warning", False))

    if has_proposal_filter_reason(row):
        return "bad", "proposal_geometry_reject", 1.0

    if seg_tier == "single":
        if context_warning:
            return "review", "single_with_context_warning", seg_score
        if cnn_prob is None:
            return "single", "topology_single_no_cnn", seg_score
        if cnn_prob >= args.precision_cnn_confirm:
            return "single", "topology_single_cnn_confirmed", min(seg_score, cnn_prob)
        return "review", "topology_single_cnn_not_confirmed", max(seg_score, cnn_prob)

    if seg_tier == "review":
        if cnn_prob is not None and cnn_prob >= args.precision_cnn_confirm and seg_score >= args.precision_review_score:
            return "review", "review_cnn_supported", min(seg_score, cnn_prob)
        # Promote very high-scoring reviews to single (fallback when CNN unavailable)
        if cnn_prob is None and seg_score >= args.precision_review_promote_score:
            return "single", "review_promoted_high_score", seg_score
        # Keep moderate-scoring reviews as review
        if cnn_prob is None and seg_score >= args.precision_review_score_no_cnn:
            return "review", "review_high_topology_score", seg_score
        return "bad", "review_not_precision_clean", 1.0 - seg_score

    if cnn_prob is not None and cnn_prob >= args.precision_cnn_rescue and not has_severe_topology_reason(row):
        return "review", "bad_cnn_conflict_review", cnn_prob
    return "bad", "topology_bad", max(1.0 - seg_score, 1.0 - (cnn_prob or 0.0))


def fuse_rows(rows: list[dict], policy: str, args) -> list[dict]:
    fused = []
    for row in rows:
        if policy == "high_recall":
            label, reason, confidence = fuse_high_recall(row, args)
        elif policy == "high_precision":
            label, reason, confidence = fuse_high_precision(row, args)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        fused.append({
            **row,
            "fusion_policy": policy,
            "final_label": label,
            "final_confidence": float(confidence),
            "fusion_reason": reason,
        })
    order = {"single": 0, "review": 1, "bad": 2}
    fused.sort(key=lambda item: (order[item["final_label"]], -float(item["final_confidence"]), int(item["component"])))
    return fused


def final_color(row: dict) -> tuple[int, int, int]:
    if row["final_label"] == "single":
        return (0, 255, 0)
    if row["final_label"] == "review":
        return (0, 220, 255)
    return (0, 0, 255)


def draw_fused_overlay(gray: np.ndarray, rows: list[dict], out_path: Path) -> None:
    overlay = cv2.cvtColor(robust_normalize(gray), cv2.COLOR_GRAY2BGR)
    for row in rows:
        x, y, w, h = [int(v) for v in row["bbox"]]
        color = final_color(row)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
        label = f"c{row['component']} {row['final_label']}"
        cv2.putText(overlay, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), overlay)


def make_fused_sheet(rows: list[dict], debug_dir: Path, out_path: Path, label: str | None = None, cols: int = 2) -> None:
    selected = [row for row in rows if label is None or row["final_label"] == label]
    tile_h = 382
    tile_w = 600
    thumbs = []
    for row in selected:
        img = cv2.imread(str(debug_dir / row["debug_panel"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        color = final_color(row)
        panel = np.full((tile_h, tile_w, 3), 24, dtype=np.uint8)
        scale = min(tile_w / img.shape[1], 302 / img.shape[0])
        resized = cv2.resize(img, (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        panel[:resized.shape[0], :resized.shape[1]] = resized
        cv2.rectangle(panel, (0, 0), (tile_w - 1, tile_h - 1), color, 2)
        cnn = row.get("cnn_single_prob")
        cnn_text = "cnn=n/a" if cnn in ("", None) else f"cnn={float(cnn):.2f}"
        y0 = 320
        cv2.putText(panel, f"c{row['component']} final={row['final_label']} conf={float(row['final_confidence']):.2f}", (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 1, cv2.LINE_AA)
        cv2.putText(panel, f"seg={row['candidate_tier']} seg_score={float(row['single_score']):.2f} {cnn_text}", (8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, str(row["fusion_reason"])[:62], (8, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1, cv2.LINE_AA)
        thumbs.append(panel)

    rows_needed = max(1, math.ceil(len(thumbs) / cols))
    sheet = np.full((rows_needed * tile_h, cols * tile_w, 3), 20, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        sheet[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = thumb
    cv2.imwrite(str(out_path), sheet)


def write_rows(rows: list[dict], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = []
    for row in rows:
        clean = {}
        for key, value in row.items():
            if isinstance(value, (np.bool_, bool)):
                clean[key] = bool(value)
            elif isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating,)):
                clean[key] = float(value)
            else:
                clean[key] = value
        serializable.append(clean)
    (out_dir / f"{stem}_fused_rows.json").write_text(json.dumps(serializable, indent=2))
    with (out_dir / f"{stem}_fused_rows.csv").open("w", newline="") as f:
        fieldnames = list(serializable[0].keys()) if serializable else ["image"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serializable)


def segmentation_args(args, seg_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        image=args.image,
        output_dir=str(seg_dir),
        segmentation_mode=args.segmentation_mode,
        pad_fraction=args.pad_fraction,
        min_pad=args.min_pad,
        min_area=args.min_area,
        min_skeleton_px=args.min_skeleton_px,
        min_proposal_major_px=args.min_proposal_major_px,
        min_proposal_area=args.min_proposal_area,
        max_proposal_major_px=args.max_proposal_major_px,
        max_proposal_bbox_area=args.max_proposal_bbox_area,
        max_proposal_area=args.max_proposal_area,
        max_large_proposal_minor_px=args.max_large_proposal_minor_px,
        max_marker_area=args.max_marker_area,
        max_marker_major_px=args.max_marker_major_px,
        max_marker_minor_px=args.max_marker_minor_px,
        min_aspect=args.min_aspect,
        max_width_px=args.max_width_px,
        max_diameter_px=args.max_diameter_px,
        max_pca_ratio=args.max_pca_ratio,
        max_off_line_fraction=args.max_off_line_fraction,
        max_secondary_orientation=args.max_secondary_orientation,
        max_flank_edge_fraction=args.max_flank_edge_fraction,
        max_branchpoints=args.max_branchpoints,
        max_endpoints=args.max_endpoints,
        prune_iterations=args.prune_iterations,
        score_threshold=args.score_threshold,
        max_score_width_px=args.max_score_width_px,
        max_score_off_line_fraction=args.max_score_off_line_fraction,
        max_score_secondary_orientation=args.max_score_secondary_orientation,
        max_context_extra_wires=args.max_context_extra_wires,
        max_context_extra_distance_px=args.max_context_extra_distance_px,
        min_context_extra_area_fraction=args.min_context_extra_area_fraction,
        context_dilation_px=args.context_dilation_px,
    )


def process(args) -> None:
    image_path = Path(args.image)
    stem = image_path.stem
    out_dir = Path(args.output_dir)
    seg_dir = out_dir / "segmentation"
    rows, _ = process_image(segmentation_args(args, seg_dir))
    debug_dir = seg_dir / "debug_panels"

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {image_path}")
    _, binary, report = preprocess(gray)
    _, labels = candidate_components(gray, binary, report["annotation_band"]["cutoff_y"])
    cnn = cnn_predictions(gray, labels, rows, args, out_dir)
    merged = [{**row, **cnn.get(int(row["component"]), {})} for row in rows]

    policies = ["high_recall", "high_precision"] if args.policy == "both" else [args.policy]
    summary = {}
    for policy in policies:
        policy_dir = out_dir / policy
        fused = fuse_rows(merged, policy, args)
        write_rows(fused, policy_dir, stem)
        draw_fused_overlay(gray, fused, policy_dir / f"{stem}_fused_overlay.png")
        make_fused_sheet(fused, debug_dir, policy_dir / f"{stem}_fused_contact_sheet.png")
        make_fused_sheet(fused, debug_dir, policy_dir / f"{stem}_fused_single_sheet.png", label="single")
        make_fused_sheet(fused, debug_dir, policy_dir / f"{stem}_fused_review_sheet.png", label="review")
        make_fused_sheet(fused, debug_dir, policy_dir / f"{stem}_fused_bad_sheet.png", label="bad")
        counts = {label: sum(row["final_label"] == label for row in fused) for label in ("single", "review", "bad")}
        summary[policy] = counts

    (out_dir / f"{stem}_fused_summary.json").write_text(json.dumps(summary, indent=2))
    for policy, counts in summary.items():
        print(f"{image_path} {policy}: single={counts['single']} review={counts['review']} bad={counts['bad']}")
        print(out_dir / policy / f"{stem}_fused_contact_sheet.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fused blob + segmentation + CNN nanowire pipeline.")
    parser.add_argument("image", nargs="?", default="experimental_sem/13.tif")
    parser.add_argument("--output-dir", default="experimental_sem_results/nanowire_fused_13")
    parser.add_argument("--policy", choices=("high_recall", "high_precision", "both"), default="both")

    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--no-cnn", action="store_true")
    parser.add_argument("--cnn-crop-mode", choices=("mask", "raw"), default="mask")
    parser.add_argument("--cnn-crop-size", type=int, default=64)
    parser.add_argument("--cnn-preprocess-mode", choices=("raw", "pca_mask", "soft_gray_pca"), default="pca_mask")

    parser.add_argument("--segmentation-mode", choices=("proposal", "bright", "dark", "auto"), default="auto")
    parser.add_argument("--pad-fraction", type=float, default=0.9)
    parser.add_argument("--min-pad", type=int, default=20)
    parser.add_argument("--min-area", type=int, default=35)
    parser.add_argument("--min-skeleton-px", type=int, default=30,
                        help="reject as too_short if skeleton length is below this (750nm at ~25nm/px)")
    parser.add_argument("--min-proposal-major-px", type=int, default=30)
    parser.add_argument("--min-proposal-area", type=int, default=35)
    parser.add_argument("--max-proposal-major-px", type=int, default=260)
    parser.add_argument("--max-proposal-bbox-area", type=int, default=45000)
    parser.add_argument("--max-proposal-area", type=int, default=2500)
    parser.add_argument("--max-large-proposal-minor-px", type=int, default=80)
    parser.add_argument("--max-marker-area", type=int, default=120)
    parser.add_argument("--max-marker-major-px", type=int, default=28)
    parser.add_argument("--max-marker-minor-px", type=int, default=12)
    parser.add_argument("--min-aspect", type=float, default=4.0)
    parser.add_argument("--max-width-px", type=float, default=5.5)
    parser.add_argument("--max-diameter-px", type=float, default=4.0,
                        help="reject as too_thick if PCA diameter (2*pca_perp_p90) exceeds this (~100nm at 25nm/px)")
    parser.add_argument("--max-pca-ratio", type=float, default=0.18)
    parser.add_argument("--max-off-line-fraction", type=float, default=0.24)
    parser.add_argument("--max-secondary-orientation", type=float, default=0.28)
    parser.add_argument("--max-flank-edge-fraction", type=float, default=0.50)
    parser.add_argument("--max-branchpoints", type=int, default=7)
    parser.add_argument("--max-endpoints", type=int, default=8)
    parser.add_argument("--prune-iterations", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.72)
    parser.add_argument("--max-score-width-px", type=float, default=4.2)
    parser.add_argument("--max-score-off-line-fraction", type=float, default=0.16)
    parser.add_argument("--max-score-secondary-orientation", type=float, default=0.24)
    parser.add_argument("--max-context-extra-wires", type=int, default=0)
    parser.add_argument("--max-context-extra-distance-px", type=float, default=8.0)
    parser.add_argument("--min-context-extra-area-fraction", type=float, default=0.18)
    parser.add_argument("--context-dilation-px", type=int, default=61)

    parser.add_argument("--cnn-conflict-low", type=float, default=0.20)
    parser.add_argument("--recall-cnn-promote", type=float, default=0.55)
    parser.add_argument("--recall-score-promote", type=float, default=0.88)
    parser.add_argument("--recall-cnn-rescue", type=float, default=0.90)
    parser.add_argument("--precision-cnn-confirm", type=float, default=0.55)
    parser.add_argument("--precision-review-score", type=float, default=0.90)
    parser.add_argument("--precision-review-score-no-cnn", type=float, default=0.85,
                        help="Keep review candidates with this score when CNN unavailable")
    parser.add_argument("--precision-review-promote-score", type=float, default=0.88,
                        help="Promote review to single with this score when CNN unavailable")
    parser.add_argument("--precision-cnn-rescue", type=float, default=0.95)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
