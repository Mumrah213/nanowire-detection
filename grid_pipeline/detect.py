"""Step 1: turn a grayscale SEM image into candidate dot / L binary masks.

Each function takes the image plus the `report` from sem_preprocess (which holds
the annotation-band cutoff and robust contrast stats) and returns a binary mask.
"""

import cv2
import numpy as np


def dark_dot_binary(gray: np.ndarray, report: dict, sigma_multiplier: float = 3.0) -> tuple[np.ndarray, float]:
    """Dark dots: pixels well below the median (dark markers on a bright field)."""
    cutoff_y = report["annotation_band"]["cutoff_y"]
    valid_mask = np.ones(gray.shape, dtype=bool)
    valid_mask[cutoff_y:, :] = False
    contrast = report["contrast"]
    threshold = contrast["median"] - sigma_multiplier * contrast["robust_sigma"]
    threshold = max(contrast["percentiles"]["p0.1"], threshold)
    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[(gray < threshold) & valid_mask] = 255
    return binary, float(threshold)


def l_candidate_binary(gray: np.ndarray, report: dict, sigma_multiplier: float = 4.0) -> tuple[np.ndarray, float]:
    """Bright L candidates: pixels well above the median, clipped to a sane band."""
    cutoff_y = report["annotation_band"]["cutoff_y"]
    valid_mask = np.ones(gray.shape, dtype=bool)
    valid_mask[cutoff_y:, :] = False
    contrast = report["contrast"]
    threshold = contrast["median"] + sigma_multiplier * contrast["robust_sigma"]
    threshold = max(contrast["percentiles"]["p95"], threshold)
    threshold = min(contrast["percentiles"]["p99.9"], threshold)
    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[(gray > threshold) & valid_mask] = 255
    return binary, float(threshold)


def compact_dot_candidates(binary: np.ndarray, cutoff_y: int) -> list[dict]:
    """Keep connected components that look like small, roughly-round grid dots."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    dots = []
    for label in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if y >= cutoff_y:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        if 8 <= area <= 90 and 3 <= w <= 14 and 3 <= h <= 14 and aspect <= 2.2:
            dots.append({
                "label": int(label),
                "x": float(centroids[label][0]),
                "y": float(centroids[label][1]),
                "area": int(area),
                "width": int(w),
                "height": int(h),
                "x0": int(x),
                "y0": int(y),
            })
    return dots


def adaptive_bright_binary(
    gray: np.ndarray,
    report: dict,
    base_binary: np.ndarray,
    args,
) -> tuple[np.ndarray, float]:
    """Illumination-flattened bright-dot mask via white top-hat.

    A single global threshold misses faint dots on the dimmer side of an
    illumination gradient. A morphological opening with a disk larger than a dot
    but smaller than the lattice spacing estimates the local background
    (gradient + large structures); subtracting it isolates the dots regardless
    of absolute brightness. Thresholded relative to robust sigma, then unioned
    with the existing global mask so nothing already detected is lost. Off-grid
    junk this admits is pruned downstream by lattice outlier removal.
    """
    cutoff_y = report["annotation_band"]["cutoff_y"]
    valid = np.zeros(gray.shape, dtype=bool)
    valid[:cutoff_y, :] = True
    k = max(7, int(args.adaptive_bright_kernel) | 1)  # odd disk diameter
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, elem)
    tophat = cv2.subtract(gray, opened).astype(np.float32)
    sigma = float(report["contrast"]["robust_sigma"])
    cut = max(args.adaptive_bright_min, args.adaptive_bright_sigma * sigma)
    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[(tophat > cut) & valid] = 255
    binary = cv2.bitwise_or(binary, base_binary)
    return binary, float(cut)
