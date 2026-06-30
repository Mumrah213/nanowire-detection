"""Topology features and veto rules for nanowire component masks."""

import math

import cv2
import numpy as np


def morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    work = (mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(work)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work) > 0:
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        residue = cv2.subtract(work, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        work = cv2.erode(work, element)
    return skeleton > 0


def skeleton_topology(skeleton: np.ndarray) -> tuple[int, int, int]:
    if not skeleton.any():
        return 0, 0, 0
    sk = skeleton.astype(np.uint8)
    padded = np.pad(sk, 1)
    neighbor_count = np.zeros_like(sk, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbor_count += padded[1 + dy:1 + dy + sk.shape[0], 1 + dx:1 + dx + sk.shape[1]]
    endpoints = int(((sk == 1) & (neighbor_count == 1)).sum())
    branchpoints = int(((sk == 1) & (neighbor_count >= 3)).sum())
    return int(sk.sum()), endpoints, branchpoints


def pca_line_features(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 3:
        return {
            "pca_minor_major_ratio": 1.0,
            "pca_off_line_fraction": 1.0,
            "pca_perp_p90_px": 999.0,
        }
    points = np.column_stack([xs, ys]).astype(np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    evecs = evecs[:, order]
    major = math.sqrt(float(evals[0])) if evals[0] > 0 else 0.0
    minor = math.sqrt(float(evals[1])) if len(evals) > 1 and evals[1] > 0 else 0.0
    ratio = minor / major if major > 0 else 1.0
    axis = evecs[:, 0]
    perpendicular = np.array([-axis[1], axis[0]])
    distances = np.abs(centered @ perpendicular)
    return {
        "pca_minor_major_ratio": float(ratio),
        "pca_off_line_fraction": float((distances > 3.0).mean()),
        "pca_perp_p90_px": float(np.percentile(distances, 90)),
    }


def hough_orientation_features(mask: np.ndarray) -> dict:
    image = (mask > 0).astype(np.uint8) * 255
    max_dim = max(image.shape)
    min_line_length = max(6, int(0.22 * max_dim))
    threshold = max(5, int(0.12 * max_dim))
    lines = cv2.HoughLinesP(
        image,
        1,
        np.pi / 180.0,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=4,
    )
    if lines is None:
        return {
            "hough_segments": 0,
            "hough_orientation_clusters": 0,
            "hough_secondary_weight_fraction": 0.0,
        }

    bins: dict[int, float] = {}
    segment_count = 0
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in line]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_line_length:
            continue
        segment_count += 1
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        bucket = int(round(angle / 15.0) * 15) % 180
        bins[bucket] = bins.get(bucket, 0.0) + length
    if not bins:
        return {
            "hough_segments": 0,
            "hough_orientation_clusters": 0,
            "hough_secondary_weight_fraction": 0.0,
        }

    weights = sorted(bins.values(), reverse=True)
    total = sum(weights)
    significant = [weight for weight in weights if weight / total >= 0.20]
    secondary = weights[1] / total if len(weights) > 1 else 0.0
    return {
        "hough_segments": int(segment_count),
        "hough_orientation_clusters": int(len(significant)),
        "hough_secondary_weight_fraction": float(secondary),
    }


def topology_features(mask: np.ndarray) -> dict:
    mask = (mask > 0).astype(np.uint8)
    area = int(mask.sum())
    ys, xs = np.nonzero(mask)
    if area == 0:
        return {
            "topology_area": 0,
            "topology_aspect": 0.0,
            "topology_estimated_width_px": 999.0,
            "topology_skeleton_pixels": 0,
            "topology_endpoints": 0,
            "topology_branchpoints": 0,
            **pca_line_features(mask),
            **hough_orientation_features(mask),
        }

    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    skeleton = morphological_skeleton(mask * 255)
    skeleton_pixels, endpoints, branchpoints = skeleton_topology(skeleton)
    estimated_width = area / max(1, skeleton_pixels)
    return {
        "topology_area": area,
        "topology_aspect": float(max(width, height) / max(1, min(width, height))),
        "topology_estimated_width_px": float(estimated_width),
        "topology_skeleton_pixels": int(skeleton_pixels),
        "topology_endpoints": int(endpoints),
        "topology_branchpoints": int(branchpoints),
        **pca_line_features(mask),
        **hough_orientation_features(mask),
    }


def topology_veto_reasons(
    features: dict,
    max_secondary_orientation: float = 0.24,
    max_off_line_fraction: float = 0.22,
    max_width_px: float = 5.5,
    max_branchpoints: int = 18,
) -> list[str]:
    reasons = []
    if (
        features["hough_orientation_clusters"] >= 2
        and features["hough_secondary_weight_fraction"] > max_secondary_orientation
    ):
        reasons.append("multi_orientation")
    if features["pca_off_line_fraction"] > max_off_line_fraction:
        reasons.append("off_axis_mass")
    if features["topology_estimated_width_px"] > max_width_px:
        reasons.append("too_wide")
    if features["topology_branchpoints"] > max_branchpoints:
        reasons.append("too_branched")
    return reasons
