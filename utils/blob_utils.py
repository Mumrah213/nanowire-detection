"""Shared blob detection helpers."""

from __future__ import annotations

import cv2
import numpy as np


def threshold_markers(gray: np.ndarray) -> np.ndarray:
    """Threshold bright markers from a dark background."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def component_anchor(
    labels: np.ndarray,
    label_idx: int,
    stat: np.ndarray,
    centroid: np.ndarray,
    structural_min_area: int = 120,
) -> tuple[float, float, str]:
    """Return the marker anchor for a connected component.

    Small dot-like markers are represented well by their centroid. Large
    structural L markers are better represented by the occupied outer corner
    of their bounding box, which is the lattice anchor.
    """
    x, y, w, h, area = [int(v) for v in stat]

    if area < structural_min_area or min(w, h) < 15:
        return float(centroid[0]), float(centroid[1]), "centroid"

    fill_ratio = area / float(w * h)
    if fill_ratio > 0.85:
        return float(centroid[0]), float(centroid[1]), "centroid"

    component = labels[y:y + h, x:x + w] == label_idx
    patch = max(4, min(w, h) // 5)
    edge_scores = {
        "top": int(component[:patch, :].sum()),
        "bottom": int(component[h - patch:, :].sum()),
        "left": int(component[:, :patch].sum()),
        "right": int(component[:, w - patch:].sum()),
    }
    edge_fill = {
        "top": edge_scores["top"] / float(patch * w),
        "bottom": edge_scores["bottom"] / float(patch * w),
        "left": edge_scores["left"] / float(patch * h),
        "right": edge_scores["right"] / float(patch * h),
    }
    strong_edges = {edge for edge, fill in edge_fill.items() if fill >= 0.45}
    if len(strong_edges) != 2:
        return float(centroid[0]), float(centroid[1]), "centroid"

    adjacent_corners = {
        frozenset(("top", "left")): "top_left",
        frozenset(("top", "right")): "top_right",
        frozenset(("bottom", "left")): "bottom_left",
        frozenset(("bottom", "right")): "bottom_right",
    }
    corner_name = adjacent_corners.get(frozenset(strong_edges))
    if corner_name is None:
        return float(centroid[0]), float(centroid[1]), "centroid"

    corner_scores = {
        "top_left": edge_scores["top"] * edge_scores["left"],
        "top_right": edge_scores["top"] * edge_scores["right"],
        "bottom_left": edge_scores["bottom"] * edge_scores["left"],
        "bottom_right": edge_scores["bottom"] * edge_scores["right"],
    }
    score = corner_scores[corner_name]
    if score == 0:
        return float(centroid[0]), float(centroid[1]), "centroid"

    corners = {
        "top_left": (x, y),
        "top_right": (x + w - 1, y),
        "bottom_left": (x, y + h - 1),
        "bottom_right": (x + w - 1, y + h - 1),
    }
    ax, ay = corners[corner_name]
    return float(ax), float(ay), f"corner:{corner_name}"


def detect_blob_components(
    gray: np.ndarray,
    min_area: int = 5,
    max_area: int = 5000,
) -> tuple[list[dict], np.ndarray]:
    """Detect marker components and return anchored blob dictionaries."""
    binary = threshold_markers(gray)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    blobs = []
    for i in range(1, num_labels):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if min_area <= area <= max_area:
            ax, ay, anchor_type = component_anchor(labels, i, stats[i], centroids[i])
            blobs.append({
                "x": round(float(ax), 2),
                "y": round(float(ay), 2),
                "centroid_x": round(float(centroids[i][0]), 2),
                "centroid_y": round(float(centroids[i][1]), 2),
                "anchor_type": anchor_type,
                "area": area,
                "width": w,
                "height": h,
                "x0": x,
                "y0": y,
            })

    blobs.sort(key=lambda b: (round(b["y"] / 10) * 10, b["x"]))
    return blobs, binary
