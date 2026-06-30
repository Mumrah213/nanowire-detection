"""Step 3: find L markers, score their orientation, and pick the anchor L.

Each L candidate is a connected component scored by edge-fill + template match
into one of UL/UR/LR/LL, sized small/big from its area, and checked for lattice
and big/small-scale consistency before one is selected as the coordinate anchor.
"""

import math
from pathlib import Path

import cv2
import numpy as np

from grid_pipeline.config import ORIENTATION_TO_SIGN_NM, ORIENTATION_EDGES, OPPOSITE_EDGES
from grid_pipeline.lattice import lattice_cell, lattice_residual


def fixed_crop_window(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    lattice: dict | None,
    crop_lattice_steps: float,
) -> tuple[int, int, int, int, int]:
    x, y, w, h = bbox
    if lattice and lattice.get("spacing"):
        side = int(round(float(lattice["spacing"]) * crop_lattice_steps))
    else:
        side = int(round(max(w, h) + 2 * max(18, int(max(w, h) * 0.8))))
    side = max(side, max(w, h) + 4)
    cx, cy = x + w / 2.0, y + h / 2.0
    left = min(max(0, int(round(cx - side / 2.0))), max(0, image_shape[1] - side))
    top = min(max(0, int(round(cy - side / 2.0))), max(0, image_shape[0] - side))
    right = min(image_shape[1], left + side)
    bottom = min(image_shape[0], top + side)
    return left, top, right, bottom, side


def component_anchor(comp: np.ndarray, x: int, y: int, orientation: str) -> tuple[float, float]:
    ys, xs = np.nonzero(comp)
    if len(xs) < 8:
        fallback = {
            "UL": (x, y),
            "UR": (x + comp.shape[1] - 1, y),
            "LR": (x + comp.shape[1] - 1, y + comp.shape[0] - 1),
            "LL": (x, y + comp.shape[0] - 1),
        }
        return tuple(map(float, fallback[orientation]))

    row_counts = comp.sum(axis=1).astype(np.float32)
    col_counts = comp.sum(axis=0).astype(np.float32)
    dense_rows = np.where(row_counts >= max(3.0, 0.22 * float(row_counts.max())))[0]
    dense_cols = np.where(col_counts >= max(3.0, 0.22 * float(col_counts.max())))[0]
    if len(dense_rows) == 0:
        dense_rows = ys
    if len(dense_cols) == 0:
        dense_cols = xs
    ay = float(np.percentile(dense_rows, 8 if "U" in orientation else 92))
    ax = float(np.percentile(dense_cols, 8 if orientation.endswith("L") else 92))
    return float(x + ax), float(y + ay)


def size_from_l_area(area: int, lattice: dict | None, small_max: float, big_min: float) -> dict:
    if not lattice or not lattice.get("spacing"):
        return {
            "size": "unknown",
            "area_norm": None,
            "size_reason": "no_grid_spacing",
        }
    spacing = float(lattice["spacing"])
    area_norm = float(area) / (spacing * spacing)
    if area_norm <= small_max:
        size = "small"
    elif area_norm >= big_min:
        size = "big"
    else:
        size = "ambiguous"
    return {
        "size": size,
        "area_norm": float(area_norm),
        "size_reason": f"area_norm<={small_max:g} small, >={big_min:g} big",
    }


def anchor_nm(orientation: str, size: str, small_l_nm: float, big_l_nm: float) -> list[float] | None:
    if orientation not in ORIENTATION_TO_SIGN_NM or size not in {"small", "big"}:
        return None
    magnitude = small_l_nm if size == "small" else big_l_nm
    sx, sy = ORIENTATION_TO_SIGN_NM[orientation]
    return [float(sx * magnitude), float(sy * magnitude)]


def l_template(orientation: str, size: int, thickness_fraction: float) -> np.ndarray:
    thickness = max(2, int(round(size * thickness_fraction)))
    template = np.zeros((size, size), dtype=np.uint8)
    if "U" in orientation:
        template[:thickness, :] = 1
    else:
        template[size - thickness:, :] = 1
    if orientation.endswith("L"):
        template[:, :thickness] = 1
    else:
        template[:, size - thickness:] = 1
    return template


def template_orientation_scores(comp: np.ndarray, template_size: int = 64) -> dict:
    mask = cv2.resize(
        comp.astype(np.uint8),
        (template_size, template_size),
        interpolation=cv2.INTER_NEAREST,
    )
    mask = (mask > 0).astype(np.uint8)
    scores = {}
    for orientation in ORIENTATION_TO_SIGN_NM:
        best = {
            "iou": 0.0,
            "coverage": 0.0,
            "extra_fraction": 1.0,
            "thickness_fraction": None,
        }
        for thickness_fraction in (0.10, 0.14, 0.18, 0.23, 0.29, 0.36):
            tmpl = l_template(orientation, template_size, thickness_fraction)
            intersection = int(np.logical_and(mask, tmpl).sum())
            union = int(np.logical_or(mask, tmpl).sum())
            mask_area = max(1, int(mask.sum()))
            template_area = max(1, int(tmpl.sum()))
            iou = intersection / max(1, union)
            coverage = intersection / mask_area
            template_coverage = intersection / template_area
            extra_fraction = 1.0 - coverage
            # IoU alone over-penalizes thick or ragged real SEM components; keep
            # foreground coverage in the score so partial Ls can still rank.
            score = 0.55 * iou + 0.30 * coverage + 0.15 * template_coverage
            if score > float(best.get("score", -1.0)):
                best = {
                    "score": float(score),
                    "iou": float(iou),
                    "coverage": float(coverage),
                    "template_coverage": float(template_coverage),
                    "extra_fraction": float(extra_fraction),
                    "thickness_fraction": float(thickness_fraction),
                }
        scores[orientation] = best
    return scores


def edge_orientation_scores(edge_fill: dict[str, float]) -> dict[str, float]:
    scores = {}
    for orientation, edges in ORIENTATION_EDGES.items():
        arm = float(edge_fill[edges[0]] + edge_fill[edges[1]]) / 2.0
        opposites = (OPPOSITE_EDGES[edges[0]], OPPOSITE_EDGES[edges[1]])
        opposite = float(edge_fill[opposites[0]] + edge_fill[opposites[1]]) / 2.0
        balance = 1.0 - min(1.0, abs(edge_fill[edges[0]] - edge_fill[edges[1]]))
        scores[orientation] = float(np.clip(0.75 * arm - 0.35 * opposite + 0.15 * balance, 0.0, 1.0))
    return scores


def choose_orientation(comp: np.ndarray, edge_fill: dict[str, float], args) -> dict:
    edge_scores = edge_orientation_scores(edge_fill)
    template_scores = template_orientation_scores(comp, args.template_size)
    combined = {}
    for orientation in ORIENTATION_TO_SIGN_NM:
        combined[orientation] = (
            args.edge_score_weight * edge_scores[orientation]
            + args.template_score_weight * template_scores[orientation]["score"]
        )
    ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
    orientation, confidence = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = float(confidence - runner_up)
    geom_orientation = max(edge_scores.items(), key=lambda item: item[1])[0]
    template_orientation = max(
        template_scores.items(),
        key=lambda item: item[1]["score"],
    )[0]
    return {
        "orientation": orientation,
        "confidence": float(confidence),
        "orientation_margin": margin,
        "geometry_orientation": geom_orientation,
        "template_orientation": template_orientation,
        "geometry_template_agree": bool(geom_orientation == template_orientation),
        "edge_scores": edge_scores,
        "template_scores": template_scores,
        "combined_scores": {key: float(value) for key, value in combined.items()},
    }


def find_l_candidates(gray: np.ndarray, binary: np.ndarray, cutoff_y: int, lattice: dict | None, args, out_dir: Path, stem: str) -> list[dict]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    crop_dir = out_dir / "l_crops" / stem
    crop_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for label in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if y >= cutoff_y:
            continue
        if not (args.min_l_area <= area <= args.max_l_area and args.min_l_side <= w <= args.max_l_side and args.min_l_side <= h <= args.max_l_side):
            continue
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > args.max_l_aspect:
            continue

        comp = labels[y:y + h, x:x + w] == label
        patch = max(3, min(w, h) // 4)
        edge_fill = {
            "top": float(comp[:patch, :].sum()) / float(patch * w),
            "bottom": float(comp[h - patch:, :].sum()) / float(patch * w),
            "left": float(comp[:, :patch].sum()) / float(patch * h),
            "right": float(comp[:, w - patch:].sum()) / float(patch * h),
        }
        strong_edges = {edge for edge, fill in edge_fill.items() if fill >= args.edge_fill_min}
        if len(strong_edges) < 2:
            continue

        size_info = size_from_l_area(area, lattice, args.small_area_norm_max, args.big_area_norm_min)
        if args.reject_ambiguous_size and size_info["size"] not in {"small", "big"}:
            continue

        left, top, right, bottom, side = fixed_crop_window((x, y, w, h), gray.shape, lattice, args.crop_lattice_steps)
        crop = np.zeros((bottom - top, right - left), dtype=np.uint8)
        crop[labels[top:bottom, left:right] == label] = 255
        crop_path = crop_dir / f"{stem}_L_candidate_{len(rows):03d}.png"
        cv2.imwrite(str(crop_path), crop)

        orientation_info = choose_orientation(comp, edge_fill, args)
        orientation = orientation_info["orientation"]
        if (
            orientation_info["confidence"] < args.orientation_score_min
            or orientation_info["orientation_margin"] < args.orientation_margin_min
        ):
            continue

        anchor = component_anchor(comp, x, y, orientation)
        lattice_residual_px = lattice_residual(anchor, lattice) if lattice else None
        size = size_info["size"]
        tolerance_fraction = (
            args.big_lattice_tolerance_fraction
            if size == "big"
            else args.lattice_tolerance_fraction
        )
        lattice_tolerance_px = max(
            args.min_lattice_tolerance_px,
            tolerance_fraction * float(lattice["spacing"]),
        ) if lattice and lattice.get("spacing") else None
        lattice_consistent = (
            lattice_residual_px is not None
            and lattice_tolerance_px is not None
            and lattice_residual_px <= lattice_tolerance_px
        )
        rejection_reasons = []
        if size_info["size"] not in {"small", "big"}:
            rejection_reasons.append(f"size_{size_info['size']}")
        if orientation_info["confidence"] < args.orientation_score_min:
            rejection_reasons.append("orientation_score_low")
        if orientation_info["orientation_margin"] < args.orientation_margin_min:
            rejection_reasons.append("orientation_margin_low")
        if lattice and args.require_lattice_consistency and not lattice_consistent:
            rejection_reasons.append("off_lattice")

        row = {
            "label": int(label),
            "bbox": [int(x), int(y), int(w), int(h)],
            "area_px": int(area),
            "major_px": int(max(w, h)),
            "minor_px": int(min(w, h)),
            "aspect": float(aspect),
            "edge_fill": edge_fill,
            "strong_edges": sorted(strong_edges),
            "geometry_orientation": orientation_info["geometry_orientation"],
            "template_orientation": orientation_info["template_orientation"],
            "orientation": orientation,
            "geometry_template_agree": orientation_info["geometry_template_agree"],
            "confidence": float(orientation_info["confidence"]),
            "orientation_margin": float(orientation_info["orientation_margin"]),
            "edge_scores": orientation_info["edge_scores"],
            "template_scores": orientation_info["template_scores"],
            "combined_scores": orientation_info["combined_scores"],
            "size": size,
            "label_size_orientation": f"{size}_{orientation}" if size in {"small", "big"} else f"{size}_L",
            "area_norm": size_info["area_norm"],
            "size_reason": size_info["size_reason"],
            "anchor_px": [float(anchor[0]), float(anchor[1])],
            "anchor_nm": anchor_nm(orientation, size, args.small_l_nm, args.big_l_nm),
            "crop_path": str(crop_path),
            "crop_window": [int(left), int(top), int(right), int(bottom)],
            "crop_side_px": int(side),
            "lattice_residual_px": lattice_residual_px,
            "lattice_tolerance_px": lattice_tolerance_px,
            "lattice_consistent": bool(lattice_consistent),
            "accepted_l_candidate": not rejection_reasons,
            "rejection_reasons": rejection_reasons,
        }
        rows.append(row)

    rows.sort(key=lambda item: (
        item["size"] not in {"small", "big"},
        item["lattice_residual_px"] if item["lattice_residual_px"] is not None else 1e9,
        -item["confidence"],
    ))
    return rows


def enforce_l_scale_consistency(candidates: list[dict], lattice: dict | None, args) -> tuple[int, str | None]:
    """Big and small Ls are the same corner marker at two scales.

    A big L at physical (sx*big_nm, sy*big_nm) implies the small L sits at
    (sx*small_nm, sy*small_nm) -- same orientation, at a fixed step offset
    along the same diagonal: |(big_nm-small_nm)/pitch| cells on each lattice
    axis. Applies for all four rotations via ORIENTATION_TO_SIGN_NM.

    Conflicts are resolved by confidence, but with a thumb on the scale for big
    Ls (they are larger and more distinctive): the reference is whichever of the
    best big / best small wins on `confidence`, after adding `big_l_confidence_bonus`
    to the big. The opposite-size accepted candidates inconsistent with that
    reference (orientation or position) are rejected.
    Returns (number rejected, reference size or None).
    """
    if not args.enforce_l_scale or not lattice or not lattice.get("spacing"):
        return 0, None
    accepted = [c for c in candidates if c.get("accepted_l_candidate")]
    bigs = [c for c in accepted if c["size"] == "big"]
    smalls = [c for c in accepted if c["size"] == "small"]
    if not bigs or not smalls:
        return 0, None

    best_big = max(bigs, key=lambda c: float(c.get("confidence", 0.0)))
    best_small = max(smalls, key=lambda c: float(c.get("confidence", 0.0)))
    big_eff = float(best_big.get("confidence", 0.0)) + float(args.big_l_confidence_bonus)
    small_eff = float(best_small.get("confidence", 0.0))

    if big_eff >= small_eff:
        ref, others, ref_size = best_big, smalls, "big"
    else:
        ref, others, ref_size = best_small, bigs, "small"

    ri, rj = lattice_cell((ref["anchor_px"][0], ref["anchor_px"][1]), lattice)
    step = abs((float(args.big_l_nm) - float(args.small_l_nm)) / float(args.grid_pitch_nm))
    tol = float(args.l_scale_cell_tolerance)

    rejected = 0
    for c in others:
        ci, cj = lattice_cell((c["anchor_px"][0], c["anchor_px"][1]), lattice)
        orientation_ok = c["orientation"] == ref["orientation"]
        position_ok = abs(abs(ri - ci) - step) <= tol and abs(abs(rj - cj) - step) <= tol
        if not (orientation_ok and position_ok):
            c["accepted_l_candidate"] = False
            kind = "orientation" if not orientation_ok else "position"
            c.setdefault("rejection_reasons", []).append(f"inconsistent_with_{ref_size}_L_{kind}")
            rejected += 1
    return rejected, ref_size


def select_l_candidate(candidates: list[dict], lattice: dict | None, args) -> tuple[dict | None, list[str]]:
    accepted = [
        candidate for candidate in candidates
        if candidate["size"] in {"small", "big"}
        and candidate["confidence"] >= args.orientation_score_min
        and candidate["orientation_margin"] >= args.orientation_margin_min
        and candidate.get("accepted_l_candidate", False)
    ]
    if not accepted:
        off_lattice = sum("off_lattice" in candidate.get("rejection_reasons", []) for candidate in candidates)
        if off_lattice:
            return None, [f"{off_lattice}_candidate(s)_rejected_off_lattice"]
        return None, ["no_candidate_passed_size_orientation_and_lattice_checks"]
    if lattice and lattice.get("spacing") and not args.require_lattice_consistency:
        inconsistent = [
            candidate for candidate in accepted
            if not candidate.get("lattice_consistent", False)
        ]
        if inconsistent:
            return accepted[0], ["off_lattice_selection_allowed_by_flag"]
    accepted.sort(key=lambda candidate: (
        -float(candidate["confidence"]),
        -float(candidate["orientation_margin"]),
        float(candidate["lattice_residual_px"] if candidate["lattice_residual_px"] is not None else 1e9),
    ))
    if lattice and lattice.get("spacing"):
        return accepted[0], ["selected_lattice_consistent_by_confidence"]
    return accepted[0], ["selected_without_grid_consistency_check"]


def candidate_summary(candidate: dict | None) -> dict | None:
    if candidate is None:
        return None
    keys = [
        "label_size_orientation",
        "orientation",
        "size",
        "confidence",
        "orientation_margin",
        "anchor_px",
        "anchor_nm",
        "bbox",
        "area_px",
        "area_norm",
        "lattice_residual_px",
        "lattice_tolerance_px",
        "lattice_consistent",
        "crop_path",
    ]
    return {key: candidate.get(key) for key in keys}


# Overlay color scheme (BGR). Hollow rings so the marker stays visible inside.
