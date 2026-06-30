#!/usr/bin/env python3
"""Classical segmentation + skeleton topology baseline for SEM nanowires.

This pipeline is intentionally model-light:

1. Reuse the SEM preprocessing connected components as object proposals.
2. Segment each proposal locally from the raw crop.
3. Skeletonize the segmented object and compute topology/line features.
4. Accept only clean single-wire candidates by transparent geometry rules.

The output is meant for visual inspection: CSV/JSON metrics, an image-wide
overlay, per-component debug panels, and contact sheets.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sem_preprocess import preprocess  # noqa: E402
from nanowire_ml.predict_real_components import candidate_components  # noqa: E402
from nanowire_ml.topology import morphological_skeleton, skeleton_topology, topology_features  # noqa: E402


def robust_normalize(gray: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.5) -> np.ndarray:
    lo = float(np.percentile(gray, lo_p))
    hi = float(np.percentile(gray, hi_p))
    if hi <= lo:
        hi = float(gray.max()) if gray.size else lo + 1.0
    out = (gray.astype(np.float32) - lo) / max(1.0, hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def padded_crop(array: np.ndarray, bbox: list[int], pad_fraction: float = 0.9, min_pad: int = 20) -> tuple[np.ndarray, tuple[int, int]]:
    x, y, w, h = [int(v) for v in bbox]
    pad = max(min_pad, int(max(w, h) * pad_fraction))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(array.shape[1], x + w + pad)
    y1 = min(array.shape[0], y + h + pad)
    return array[y0:y1, x0:x1], (x0, y0)


def proposal_filter_reason(row: dict, args) -> str:
    """Reject obvious non-nanowire proposals before local segmentation.

    These are proposal-level gates. They are intentionally about the original
    connected component, not the locally selected segment, because large metal
    electrodes can otherwise contribute one clean-looking edge that passes
    nanowire topology.
    """
    x, y, w, h = [int(v) for v in row["bbox"]]
    area = int(row["area"])
    major = max(w, h)
    minor = min(w, h)
    bbox_area = w * h

    if major < args.min_proposal_major_px:
        return "proposal_too_short"
    if area < args.min_proposal_area:
        return "proposal_area_too_small"
    if major > args.max_proposal_major_px and bbox_area > args.max_proposal_bbox_area:
        return "proposal_too_large"
    if area > args.max_proposal_area and minor > args.max_large_proposal_minor_px:
        return "proposal_too_massive"
    if (
        area <= args.max_marker_area
        and major <= args.max_marker_major_px
        and minor <= args.max_marker_minor_px
    ):
        return "proposal_marker_like"
    return ""


def select_component_near_center(binary: np.ndarray, preferred_mask: np.ndarray | None = None) -> np.ndarray:
    n, labels, stats, centroids = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    if n <= 1:
        return np.zeros_like(binary, dtype=np.uint8)

    center = np.array([binary.shape[1] / 2.0, binary.shape[0] / 2.0])
    best_label = 0
    best_score = -1e18
    preferred = preferred_mask > 0 if preferred_mask is not None else None
    for label in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if area < 8:
            continue
        centroid = np.array(centroids[label])
        distance_penalty = 0.20 * float(np.linalg.norm(centroid - center))
        overlap_bonus = 0.0
        if preferred is not None:
            overlap = int(((labels == label) & preferred).sum())
            overlap_bonus = 4.0 * overlap
        elongation_bonus = 2.0 * max(w, h)
        score = float(area) + elongation_bonus + overlap_bonus - distance_penalty
        if score > best_score:
            best_score = score
            best_label = label

    selected = np.zeros_like(binary, dtype=np.uint8)
    if best_label:
        selected[labels == best_label] = 255
    return selected


def directional_close(mask: np.ndarray, length: int = 7) -> np.ndarray:
    """Close gaps along the principal axis of the mask.

    Uses PCA to find the wire direction, then applies a line-shaped closing
    kernel to bridge small gaps without thickening the wire perpendicular
    to its axis.
    """
    if not mask.any():
        return mask

    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 5:
        return mask

    # Find principal axis via PCA
    points = np.column_stack([xs, ys]).astype(np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    axis = evecs[:, order[0]]

    # Create line kernel along axis
    angle_rad = np.arctan2(axis[1], axis[0])
    angle_deg = np.degrees(angle_rad)

    # Build rotated line kernel
    ksize = length | 1  # ensure odd
    kernel = np.zeros((ksize, ksize), dtype=np.uint8)
    center = ksize // 2
    for i in range(ksize):
        t = i - center
        x = int(round(center + t * np.cos(angle_rad)))
        y = int(round(center + t * np.sin(angle_rad)))
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1

    if kernel.sum() < 3:
        kernel = np.zeros((ksize, ksize), dtype=np.uint8)
        kernel[center, :] = 1

    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed


def local_threshold_mask(raw_crop: np.ndarray, seed_mask: np.ndarray, mode: str, context_dilation_px: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return a local object mask and reproducible threshold diagnostics."""
    norm = robust_normalize(raw_crop)
    seed = seed_mask > 0
    dilated_seed = cv2.dilate(seed.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    context_kernel_size = max(3, int(context_dilation_px) | 1)
    context_region = cv2.dilate(
        seed.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (context_kernel_size, context_kernel_size)),
    ) > 0

    if mode == "proposal":
        mask = seed.astype(np.uint8) * 255
        context_mask = mask.copy()
        threshold = None
    else:
        pixels = norm[dilated_seed] if dilated_seed.any() else norm.reshape(-1)
        median = float(np.median(pixels))
        mad = float(np.median(np.abs(pixels.astype(np.float32) - median)))
        sigma = 1.4826 * mad

        if mode == "bright":
            threshold = max(float(np.percentile(pixels, 90)), median + 2.0 * sigma)
            raw_mask = ((norm >= threshold) & dilated_seed).astype(np.uint8) * 255
            context_mask = ((norm >= threshold) & context_region).astype(np.uint8) * 255
        elif mode == "dark":
            threshold = min(float(np.percentile(pixels, 10)), median - 2.0 * sigma)
            raw_mask = ((norm <= threshold) & dilated_seed).astype(np.uint8) * 255
            context_mask = ((norm <= threshold) & context_region).astype(np.uint8) * 255
        elif mode == "auto":
            bright_thr = max(float(np.percentile(pixels, 90)), median + 2.0 * sigma)
            dark_thr = min(float(np.percentile(pixels, 10)), median - 2.0 * sigma)
            bright_mask = ((norm >= bright_thr) & dilated_seed).astype(np.uint8) * 255
            dark_mask = ((norm <= dark_thr) & dilated_seed).astype(np.uint8) * 255
            bright_overlap = int(((bright_mask > 0) & seed).sum())
            dark_overlap = int(((dark_mask > 0) & seed).sum())
            if dark_overlap > bright_overlap * 1.25:
                raw_mask = dark_mask
                context_mask = ((norm <= dark_thr) & context_region).astype(np.uint8) * 255
                threshold = dark_thr
                mode = "dark"
            else:
                raw_mask = bright_mask
                context_mask = ((norm >= bright_thr) & context_region).astype(np.uint8) * 255
                threshold = bright_thr
                mode = "bright"
        else:
            raise ValueError(f"Unknown segmentation mode: {mode}")

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # Directional close to bridge gaps along wire axis
        raw_mask = directional_close(raw_mask, length=9)
        context_mask = cv2.morphologyEx(context_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        context_mask = cv2.morphologyEx(context_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = select_component_near_center(raw_mask, seed_mask)

        if int((mask > 0).sum()) < max(8, int(0.35 * seed.sum())):
            mask = seed.astype(np.uint8) * 255
            context_mask = mask.copy()
            mode = f"{mode}_fallback_proposal"

    return mask, context_mask, {
        "segmentation_mode_used": mode,
        "local_threshold": None if threshold is None else float(threshold),
        "local_norm_p1": float(np.percentile(norm, 1)),
        "local_norm_p99": float(np.percentile(norm, 99)),
    }


def context_contamination_features(selected_mask: np.ndarray, context_mask: np.ndarray) -> dict:
    selected = selected_mask > 0
    context = context_mask > 0
    selected_area = max(1, int(selected.sum()))
    expanded = cv2.dilate(selected.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    extra = context & ~expanded
    n, labels, stats, _ = cv2.connectedComponentsWithStats(extra.astype(np.uint8), 8)
    extra_components = 0
    elongated_extra_components = 0
    extra_area = 0
    largest_extra_area = 0
    nearest_extra_distance = 999.0
    sy, sx = np.nonzero(selected)
    selected_points = np.column_stack([sx, sy]).astype(np.float32) if len(sx) else np.zeros((0, 2), dtype=np.float32)

    for label in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if area < 8:
            continue
        comp = labels == label
        topo = topology_features(comp.astype(np.uint8) * 255)
        extra_components += 1
        extra_area += area
        largest_extra_area = max(largest_extra_area, area)
        pca_ratio = max(float(topo["pca_minor_major_ratio"]), 1e-4)
        line_aspect = max(float(topo["topology_aspect"]), min(50.0, 1.0 / pca_ratio))
        if line_aspect >= 4.0 and topo["topology_estimated_width_px"] <= 5.5:
            elongated_extra_components += 1
        if len(selected_points):
            ey, ex = np.nonzero(comp)
            extra_points = np.column_stack([ex, ey]).astype(np.float32)
            # The components are small enough that a dense distance calculation is acceptable here.
            if len(extra_points):
                d2 = ((extra_points[:, None, :] - selected_points[None, :, :]) ** 2).sum(axis=2)
                nearest_extra_distance = min(nearest_extra_distance, float(np.sqrt(d2.min())))

    return {
        "context_extra_components": int(extra_components),
        "context_elongated_extra_components": int(elongated_extra_components),
        "context_extra_area": int(extra_area),
        "context_largest_extra_area": int(largest_extra_area),
        "context_extra_area_fraction": float(extra_area / selected_area),
        "context_nearest_extra_distance_px": float(nearest_extra_distance),
    }


def pca_axis(mask: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 3:
        return None
    points = np.column_stack([xs, ys]).astype(np.float32)
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    axis = evecs[:, order[0]]
    half_length = max(8.0, 2.2 * math.sqrt(float(evals[0])))
    return (float(center[0]), float(center[1])), (float(axis[0]), float(axis[1])), float(half_length)


def flank_contrast_features(raw_crop: np.ndarray, mask: np.ndarray) -> dict:
    """Compare the wire intensity with the substrate on both perpendicular sides.

    A real nanowire is bright with dark substrate on BOTH flanks; the edge of a
    bright metal contact has metal (bright) on one side and substrate (dark) on
    the other. We sample intensity along the wire centerline and at a small
    perpendicular offset on each side, then report ``flank_edge_like_fraction``:
    the share of the wire's contrast that the *brighter* flank reproduces
    (~0 for a clean wire, ~1 for a contact edge).
    """
    default = {
        "flank_wire_intensity": 0.0,
        "flank_dark_intensity": 0.0,
        "flank_bright_intensity": 0.0,
        "flank_min_contrast_px": 0.0,
        "flank_edge_like_fraction": 0.0,
    }
    axis = pca_axis(mask)
    wire_vals = raw_crop[mask > 0]
    if axis is None or wire_vals.size < 5:
        return default
    (cx, cy), (vx, vy), half = axis
    perp = (-vy, vx)
    i_wire = float(np.median(wire_vals))

    ys, xs = np.nonzero(mask > 0)
    pts = np.column_stack([xs, ys]).astype(np.float32) - np.array([cx, cy], np.float32)
    perp_coord = np.abs(pts @ np.array(perp, np.float32))
    half_width = float(np.percentile(perp_coord, 90)) if perp_coord.size else 2.0
    offset = max(4.0, half_width + 4.0)

    h, w = raw_crop.shape
    left, right = [], []
    for t in np.linspace(-0.8 * half, 0.8 * half, 21):
        bx, by = cx + t * vx, cy + t * vy
        for sign, store in ((1.0, left), (-1.0, right)):
            ix = int(round(bx + sign * offset * perp[0]))
            iy = int(round(by + sign * offset * perp[1]))
            if 0 <= ix < w and 0 <= iy < h:
                store.append(float(raw_crop[iy, ix]))
    if len(left) < 3 or len(right) < 3:
        return default

    i_left, i_right = float(np.median(left)), float(np.median(right))
    dark, bright = min(i_left, i_right), max(i_left, i_right)
    denom = max(i_wire - dark, 1.0)
    edge_like = float(np.clip((bright - dark) / denom, 0.0, 2.0))
    # Too little contrast to judge (faint object) -> do not flag as an edge.
    if (i_wire - dark) < 8.0:
        edge_like = 0.0
    return {
        "flank_wire_intensity": i_wire,
        "flank_dark_intensity": dark,
        "flank_bright_intensity": bright,
        "flank_min_contrast_px": float(i_wire - bright),
        "flank_edge_like_fraction": edge_like,
    }


def skeleton_neighbor_count(skeleton: np.ndarray) -> np.ndarray:
    sk = skeleton.astype(np.uint8)
    padded = np.pad(sk, 1)
    count = np.zeros_like(sk, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            count += padded[1 + dy:1 + dy + sk.shape[0], 1 + dx:1 + dx + sk.shape[1]]
    return count


def prune_skeleton_spurs(skeleton: np.ndarray, iterations: int) -> np.ndarray:
    """Trim short endpoint spurs before graph topology is counted."""
    pruned = skeleton.astype(bool).copy()
    for _ in range(max(0, iterations)):
        if not pruned.any():
            break
        neighbors = skeleton_neighbor_count(pruned)
        endpoints = pruned & (neighbors <= 1)
        if not endpoints.any():
            break
        pruned[endpoints] = False
    return pruned


def pruned_skeleton_features(mask: np.ndarray, prune_iterations: int) -> dict:
    skeleton = morphological_skeleton(mask)
    pruned = prune_skeleton_spurs(skeleton, prune_iterations)
    pixels, endpoints, branchpoints = skeleton_topology(pruned)
    neighbors = skeleton_neighbor_count(pruned) if pruned.any() else np.zeros_like(pruned, dtype=np.uint8)
    branch_mask = pruned & (neighbors >= 3)
    endpoint_mask = pruned & (neighbors == 1)
    branch_groups = max(0, cv2.connectedComponents(branch_mask.astype(np.uint8), 8)[0] - 1)
    endpoint_groups = max(0, cv2.connectedComponents(endpoint_mask.astype(np.uint8), 8)[0] - 1)
    return {
        "pruned_skeleton_pixels": int(pixels),
        "pruned_endpoints": int(endpoints),
        "pruned_branchpoints": int(branchpoints),
        "pruned_endpoint_groups": int(endpoint_groups),
        "pruned_branchpoint_groups": int(branch_groups),
    }


def score_single_wire(features: dict) -> tuple[float, dict]:
    """Continuous score for ranking single-wire likelihood.

    This deliberately leans on rotation-invariant PCA features. Bbox aspect is
    still useful as context, but it is a poor primary elongation metric for
    diagonal wires because a clean 45-degree line has a nearly square bbox.
    """
    pca_ratio = max(float(features["pca_minor_major_ratio"]), 1e-4)
    pca_aspect = min(50.0, 1.0 / pca_ratio)
    line_aspect = max(float(features["topology_aspect"]), pca_aspect)
    width = float(features["topology_estimated_width_px"])
    off_axis = float(features["pca_off_line_fraction"])
    secondary = float(features["hough_secondary_weight_fraction"])
    branch_groups = float(features.get("pruned_branchpoint_groups", features["topology_branchpoints"]))
    endpoint_groups = float(features.get("pruned_endpoint_groups", features["topology_endpoints"]))
    area = float(features["topology_area"])

    aspect_score = min(1.0, line_aspect / 12.0)
    width_score = np.clip((6.5 - width) / 4.5, 0.0, 1.0)
    off_axis_score = np.clip((0.35 - off_axis) / 0.35, 0.0, 1.0)
    secondary_score = np.clip((0.35 - secondary) / 0.35, 0.0, 1.0)
    branch_score = np.clip((10.0 - branch_groups) / 10.0, 0.0, 1.0)
    endpoint_score = np.clip((10.0 - endpoint_groups) / 10.0, 0.0, 1.0)
    area_score = np.clip((area - 25.0) / 90.0, 0.0, 1.0)

    score = (
        0.28 * aspect_score
        + 0.20 * width_score
        + 0.18 * off_axis_score
        + 0.14 * secondary_score
        + 0.10 * branch_score
        + 0.05 * endpoint_score
        + 0.05 * area_score
    )
    diagnostics = {
        "single_score": float(score),
        "pca_aspect": float(pca_aspect),
        "line_aspect": float(line_aspect),
        "score_aspect": float(aspect_score),
        "score_width": float(width_score),
        "score_off_axis": float(off_axis_score),
        "score_secondary": float(secondary_score),
        "score_branch": float(branch_score),
        "score_endpoint": float(endpoint_score),
        "score_area": float(area_score),
    }
    return float(score), diagnostics


def panel_image(title: str, img: np.ndarray, panel_size: int = 220) -> np.ndarray:
    panel = np.full((panel_size, panel_size, 3), 24, dtype=np.uint8)
    title_h = 24
    scale = min(panel_size / img.shape[1], (panel_size - title_h) / img.shape[0])
    resized = cv2.resize(
        img,
        (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    yy = title_h + (panel_size - title_h - resized.shape[0]) // 2
    xx = (panel_size - resized.shape[1]) // 2
    panel[yy:yy + resized.shape[0], xx:xx + resized.shape[1]] = resized
    cv2.putText(panel, title, (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def draw_crop_debug(raw_crop: np.ndarray, seed_mask: np.ndarray, seg_mask: np.ndarray, context_mask: np.ndarray) -> np.ndarray:
    norm = robust_normalize(raw_crop)
    skeleton = morphological_skeleton(seg_mask)
    pruned_skeleton = prune_skeleton_spurs(skeleton, 4)
    panels = []

    raw_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    seed_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    seed_rgb[seed_mask > 0] = (0, 255, 255)
    seg_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    seg_rgb[seg_mask > 0] = (0, 255, 0)
    context_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    context_rgb[context_mask > 0] = (0, 180, 255)
    context_rgb[seg_mask > 0] = (0, 255, 0)
    skel_rgb = seg_rgb.copy()
    skel_rgb[skeleton] = (0, 0, 255)
    pruned_rgb = seg_rgb.copy()
    pruned_rgb[pruned_skeleton] = (255, 120, 0)

    axis_rgb = skel_rgb.copy()
    axis = pca_axis(seg_mask)
    if axis is not None:
        center, vec, half = axis
        x0 = int(round(center[0] - vec[0] * half))
        y0 = int(round(center[1] - vec[1] * half))
        x1 = int(round(center[0] + vec[0] * half))
        y1 = int(round(center[1] + vec[1] * half))
        cv2.line(axis_rgb, (x0, y0), (x1, y1), (255, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(axis_rgb, (int(round(center[0])), int(round(center[1]))), 2, (255, 255, 255), -1, cv2.LINE_AA)

    panels = [
        panel_image(title, img)
        for title, img in (
        ("raw", raw_rgb),
        ("proposal", seed_rgb),
        ("all local", context_rgb),
        ("local seg", seg_rgb),
        ("skeleton", skel_rgb),
        ("pruned skel", pruned_rgb),
        ("pca axis", axis_rgb),
        )
    ]
    blank = np.full_like(panels[0], 24)
    top = np.hstack(panels[:4])
    bottom = np.hstack(panels[4:] + [blank])
    return np.vstack([top, bottom])


def classify_single(features: dict, args) -> tuple[bool, bool, str, list[str]]:
    reasons = []
    score, score_features = score_single_wire(features)
    features.update(score_features)
    if features["topology_area"] < args.min_area:
        reasons.append("too_small")
    skeleton_px = features.get("pruned_skeleton_pixels") or features.get("topology_skeleton_pixels") or 0
    if skeleton_px < args.min_skeleton_px:
        reasons.append("too_short")
    if features["line_aspect"] < args.min_aspect:
        reasons.append("not_elongated")
    if features["topology_estimated_width_px"] > args.max_width_px:
        reasons.append("too_wide")
    # Note: too_thick check moved to post-processing for outlier detection
    if features["pca_minor_major_ratio"] > args.max_pca_ratio:
        reasons.append("too_broad_pca")
    if features["pca_off_line_fraction"] > args.max_off_line_fraction:
        reasons.append("off_axis_mass")
    if (
        features["hough_orientation_clusters"] >= 2
        and features["hough_secondary_weight_fraction"] > args.max_secondary_orientation
    ):
        reasons.append("multi_orientation")
    branchpoints = features.get("pruned_branchpoint_groups", features["topology_branchpoints"])
    endpoints = features.get("pruned_endpoint_groups", features["topology_endpoints"])
    if branchpoints > args.max_branchpoints:
        reasons.append("too_branched")
    if endpoints > args.max_endpoints:
        reasons.append("too_many_endpoints")
    if features.get("flank_edge_like_fraction", 0.0) > args.max_flank_edge_fraction:
        reasons.append("contact_edge")
    context_warning = (
        features.get("context_elongated_extra_components", 0) > args.max_context_extra_wires
        and features.get("context_nearest_extra_distance_px", 999.0) <= args.max_context_extra_distance_px
        and features.get("context_extra_area_fraction", 0.0) >= args.min_context_extra_area_fraction
    )
    features["context_warning"] = bool(context_warning)
    strict_pass = not reasons
    score_pass = (
        score >= args.score_threshold
        and features["topology_area"] >= args.min_area
        and features["topology_estimated_width_px"] <= args.max_score_width_px
        and features["pca_off_line_fraction"] <= args.max_score_off_line_fraction
        and not (
            features["hough_orientation_clusters"] >= 2
            and features["hough_secondary_weight_fraction"] > args.max_score_secondary_orientation
        )
    )
    severe_reasons = {"too_small", "too_short", "too_wide", "too_broad_pca", "off_axis_mass", "multi_orientation", "contact_edge"}
    review_candidate = bool(score_pass and not (set(reasons) & severe_reasons))
    if strict_pass:
        return True, False, "single", reasons
    if review_candidate:
        review_reasons = [reason for reason in reasons if reason not in {"not_elongated", "too_branched", "too_many_endpoints"}]
        return False, True, "review", review_reasons
    return False, False, "bad", reasons


def tier_color(row: dict) -> tuple[int, int, int]:
    if row["candidate_tier"] == "single":
        return (0, 255, 0)
    if row["candidate_tier"] == "review":
        return (0, 220, 255)
    return (0, 0, 255)


def draw_full_overlay(gray: np.ndarray, rows: list[dict], out_path: Path) -> None:
    overlay = cv2.cvtColor(robust_normalize(gray), cv2.COLOR_GRAY2BGR)
    for row in rows:
        x, y, w, h = row["bbox"]
        keep = row["candidate_tier"] != "bad"
        color = tier_color(row)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
        label = f"c{row['component']}"
        if keep:
            label += f" {row['candidate_tier']}"
        cv2.putText(overlay, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), overlay)


def make_contact_sheet(
    rows: list[dict],
    debug_dir: Path,
    out_path: Path,
    accepted_only: bool = False,
    tier: str | None = None,
    cols: int = 2,
) -> None:
    if tier is not None:
        selected = [row for row in rows if row["candidate_tier"] == tier]
    elif accepted_only:
        selected = [row for row in rows if row["candidate_tier"] != "bad"]
    else:
        selected = rows
    tile_h = 360
    tile_w = 560
    thumbs = []
    for row in selected:
        img = cv2.imread(str(debug_dir / row["debug_panel"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        color = tier_color(row)
        panel = np.full((tile_h, tile_w, 3), 24, dtype=np.uint8)
        scale = min(tile_w / img.shape[1], 302 / img.shape[0])
        resized = cv2.resize(img, (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        panel[:resized.shape[0], :resized.shape[1]] = resized
        cv2.rectangle(panel, (0, 0), (tile_w - 1, tile_h - 1), color, 2)
        y0 = 320
        cv2.putText(panel, f"c{row['component']} {row['candidate_tier']} score={row['single_score']:.2f}", (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
        cv2.putText(panel, f"line_aspect={row['line_aspect']:.1f} width={row['topology_estimated_width_px']:.1f}px branches={row['pruned_branchpoint_groups']}", (8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        if row["reject_reasons"]:
            reason = row["reject_reasons"][:48]
        elif row.get("context_warning"):
            reason = "pass; nearby context"
        else:
            reason = "pass"
        cv2.putText(panel, reason, (8, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
        thumbs.append(panel)

    rows_needed = max(1, math.ceil(len(thumbs) / cols))
    sheet = np.full((rows_needed * tile_h, cols * tile_w, 3), 20, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        sheet[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = thumb
    cv2.imwrite(str(out_path), sheet)


def process_image(args) -> tuple[list[dict], Path]:
    image_path = Path(args.image)
    out_dir = Path(args.output_dir)
    debug_dir = out_dir / "debug_panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {image_path}")

    _, binary, report = preprocess(gray)
    proposals, labels = candidate_components(gray, binary, report["annotation_band"]["cutoff_y"])
    rows = []
    for row in proposals:
        proposal_reason = proposal_filter_reason(row, args)
        if proposal_reason:
            bbox = row["bbox"]
            x, y, w, h = [int(v) for v in bbox]
            raw_crop, (x0, y0) = padded_crop(gray, bbox, args.pad_fraction, args.min_pad)
            label_crop = labels[y0:y0 + raw_crop.shape[0], x0:x0 + raw_crop.shape[1]]
            seed_mask = np.zeros(raw_crop.shape, dtype=np.uint8)
            seed_mask[label_crop == row["component"]] = 255
            debug_name = f"{image_path.stem}_component_{row['component']:04d}_segmentation.png"
            cv2.imwrite(str(debug_dir / debug_name), draw_crop_debug(raw_crop, seed_mask, seed_mask, seed_mask))
            rows.append({
                "image": str(image_path),
                "component": int(row["component"]),
                "bbox": bbox,
                "crop_origin": [int(x0), int(y0)],
                "proposal_area": int(row["area"]),
                "accepted_single": False,
                "review_candidate": False,
                "candidate_tier": "bad",
                "reject_reasons": proposal_reason,
                "debug_panel": debug_name,
                "segmentation_mode_used": "proposal_filter",
                "local_threshold": None,
                "local_norm_p1": 0.0,
                "local_norm_p99": 0.0,
                "proposal_major_px": int(max(w, h)),
                "proposal_minor_px": int(min(w, h)),
                "proposal_bbox_area": int(w * h),
                "context_warning": False,
                "single_score": 0.0,
                "pca_aspect": 0.0,
                "line_aspect": 0.0,
                "score_aspect": 0.0,
                "score_width": 0.0,
                "score_off_axis": 0.0,
                "score_secondary": 0.0,
                "score_branch": 0.0,
                "score_endpoint": 0.0,
                "score_area": 0.0,
                **{key: 0 for key in (
                    "topology_area",
                    "topology_skeleton_pixels",
                    "topology_endpoints",
                    "topology_branchpoints",
                    "hough_segments",
                    "hough_orientation_clusters",
                    "pruned_skeleton_pixels",
                    "pruned_endpoints",
                    "pruned_branchpoints",
                    "pruned_endpoint_groups",
                    "pruned_branchpoint_groups",
                    "context_extra_components",
                    "context_elongated_extra_components",
                    "context_extra_area",
                    "context_largest_extra_area",
                )},
                **{key: 0.0 for key in (
                    "topology_aspect",
                    "topology_estimated_width_px",
                    "pca_minor_major_ratio",
                    "pca_off_line_fraction",
                    "pca_perp_p90_px",
                    "hough_secondary_weight_fraction",
                    "context_extra_area_fraction",
                    "context_nearest_extra_distance_px",
                    "flank_wire_intensity",
                    "flank_dark_intensity",
                    "flank_bright_intensity",
                    "flank_min_contrast_px",
                    "flank_edge_like_fraction",
                )},
            })
            continue
        bbox = row["bbox"]
        raw_crop, (x0, y0) = padded_crop(gray, bbox, args.pad_fraction, args.min_pad)
        label_crop = labels[y0:y0 + raw_crop.shape[0], x0:x0 + raw_crop.shape[1]]
        seed_mask = np.zeros(raw_crop.shape, dtype=np.uint8)
        seed_mask[label_crop == row["component"]] = 255
        local_mask, context_mask, seg_report = local_threshold_mask(raw_crop, seed_mask, args.segmentation_mode, args.context_dilation_px)

        # Compute features from local_mask first
        local_features = topology_features(local_mask)
        local_pruned = pruned_skeleton_features(local_mask, args.prune_iterations)

        # Check if skeleton is too short compared to bbox (fragmented segmentation)
        # If so, use the proposal mask for length measurement
        bbox_major = max(bbox[2], bbox[3])
        skeleton_px = local_pruned.get("pruned_skeleton_pixels", 0) or local_features.get("topology_skeleton_pixels", 0)
        skeleton_coverage = skeleton_px / max(1, bbox_major)

        if skeleton_coverage < 0.4 and bbox_major > 20:
            # Segmentation fragmented - use proposal mask for skeleton
            proposal_features = topology_features(seed_mask)
            proposal_pruned = pruned_skeleton_features(seed_mask, args.prune_iterations)
            # Use proposal skeleton length but keep local mask for width/other features
            local_pruned["pruned_skeleton_pixels"] = proposal_pruned["pruned_skeleton_pixels"]
            local_features["topology_skeleton_pixels"] = proposal_features["topology_skeleton_pixels"]
            seg_report["segmentation_mode_used"] += "_proposal_skeleton"

        features = {
            **local_features,
            **local_pruned,
            **context_contamination_features(local_mask, context_mask),
            **flank_contrast_features(raw_crop, local_mask),
        }
        accepted, review_candidate, tier, reasons = classify_single(features, args)
        debug_name = f"{image_path.stem}_component_{row['component']:04d}_segmentation.png"
        cv2.imwrite(str(debug_dir / debug_name), draw_crop_debug(raw_crop, seed_mask, local_mask, context_mask))

        rows.append({
            "image": str(image_path),
            "component": int(row["component"]),
            "bbox": bbox,
            "crop_origin": [int(x0), int(y0)],
            "proposal_area": int(row["area"]),
            "proposal_major_px": int(max(bbox[2], bbox[3])),
            "proposal_minor_px": int(min(bbox[2], bbox[3])),
            "proposal_bbox_area": int(bbox[2] * bbox[3]),
            "accepted_single": bool(accepted),
            "review_candidate": bool(review_candidate),
            "candidate_tier": tier,
            "reject_reasons": ",".join(reasons),
            "debug_panel": debug_name,
            **seg_report,
            **features,
        })

    tier_order = {"single": 0, "review": 1, "bad": 2}
    rows.sort(key=lambda item: (tier_order[item["candidate_tier"]], -item["single_score"], item["bbox"][1], item["bbox"][0]))
    draw_full_overlay(gray, rows, out_dir / f"{image_path.stem}_segmentation_overlay.png")
    make_contact_sheet(rows, debug_dir, out_dir / f"{image_path.stem}_segmentation_contact_sheet.png")
    make_contact_sheet(rows, debug_dir, out_dir / f"{image_path.stem}_segmentation_accepts.png", accepted_only=True)
    make_contact_sheet(rows, debug_dir, out_dir / f"{image_path.stem}_segmentation_single_sheet.png", tier="single")
    make_contact_sheet(rows, debug_dir, out_dir / f"{image_path.stem}_segmentation_review_sheet.png", tier="review")
    make_contact_sheet(rows, debug_dir, out_dir / f"{image_path.stem}_segmentation_bad_sheet.png", tier="bad")
    cv2.imwrite(str(out_dir / f"{image_path.stem}_proposal_binary.png"), binary)
    (out_dir / f"{image_path.stem}_preprocess_report.json").write_text(json.dumps(report, indent=2))

    serializable = [{**item, "accepted_single": bool(item["accepted_single"])} for item in rows]
    (out_dir / f"{image_path.stem}_segmentation_rows.json").write_text(json.dumps(serializable, indent=2))
    with (out_dir / f"{image_path.stem}_segmentation_rows.csv").open("w", newline="") as f:
        fieldnames = list(serializable[0].keys()) if serializable else ["image"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serializable)

    return rows, out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical segmentation + topology baseline for nanowire candidates.")
    parser.add_argument("image", nargs="?", default="experimental_sem/13.tif")
    parser.add_argument("--output-dir", default="experimental_sem_results/nanowire_segmentation_baseline_13")
    parser.add_argument("--segmentation-mode", choices=("proposal", "bright", "dark", "auto"), default="proposal")
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
    parser.add_argument("--max-flank-edge-fraction", type=float, default=0.50,
                        help="reject as contact_edge if the brighter perpendicular flank fills "
                             "more than this fraction of the wire's contrast (0=clean wire, 1=metal edge)")
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
    args = parser.parse_args()

    rows, out_dir = process_image(args)
    accepted = sum(1 for row in rows if row["accepted_single"])
    print(f"{args.image}: {accepted}/{len(rows)} accepted as single by segmentation/topology baseline")
    print(out_dir / f"{Path(args.image).stem}_segmentation_contact_sheet.png")
    print(out_dir / f"{Path(args.image).stem}_segmentation_overlay.png")


if __name__ == "__main__":
    main()
