#!/usr/bin/env python3
"""Pure (cv2/numpy) crop preprocessing for nanowire components.

These helpers turn a connected-component crop into the normalized/aligned input
the CNN expects. They are deliberately torch-free so the grid + nanowire
pipelines can import them (and run with the CNN disabled) without installing
torch. The training/inference code in ``train_classifier`` re-exports them.
"""

import cv2
import numpy as np


LABEL_TO_TARGET = {"bad": 0.0, "single": 1.0}


def pca_align_mask(img: np.ndarray, output_size: int | None = None) -> np.ndarray:
    """Convert a crop to a centered binary mask aligned along the vertical axis."""
    if output_size is None:
        output_size = int(img.shape[0])
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (0, 0), 0.6)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 3:
        return cv2.resize(mask, (output_size, output_size), interpolation=cv2.INTER_AREA)

    points = np.column_stack([xs, ys]).astype(np.float32)
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    angle_deg = np.degrees(np.arctan2(major[1], major[0]))
    # Rotate so the major axis is vertical (90 degrees in image coordinates).
    rotate_deg = 90.0 - angle_deg
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), rotate_deg, 1.0)
    aligned = cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)

    ys, xs = np.nonzero(aligned > 0)
    if len(xs) >= 3:
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        pad = max(4, int(0.18 * max(x1 - x0, y1 - y0)))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(aligned.shape[1], x1 + pad)
        y1 = min(aligned.shape[0], y1 + pad)
        aligned = aligned[y0:y1, x0:x1]

    canvas = np.zeros((output_size, output_size), dtype=np.uint8)
    scale = min(output_size / aligned.shape[1], output_size / aligned.shape[0])
    resized = cv2.resize(
        aligned,
        (max(1, int(aligned.shape[1] * scale)), max(1, int(aligned.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    yy = (output_size - resized.shape[0]) // 2
    xx = (output_size - resized.shape[1]) // 2
    canvas[yy:yy + resized.shape[0], xx:xx + resized.shape[1]] = resized
    return canvas


def robust_normalize_gray(img: np.ndarray) -> np.ndarray:
    pixels = img.astype(np.float32)
    lo, hi = np.percentile(pixels, [1.0, 99.5])
    if hi <= lo:
        lo, hi = float(pixels.min()), float(pixels.max())
    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)
    out = (pixels - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def pca_align_soft_gray(img: np.ndarray, output_size: int | None = None) -> np.ndarray:
    """Align using a binary foreground estimate, but preserve grayscale values."""
    if output_size is None:
        output_size = int(img.shape[0])
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = robust_normalize_gray(img)
    blur = cv2.GaussianBlur(gray, (0, 0), 0.6)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 3:
        return cv2.resize(gray, (output_size, output_size), interpolation=cv2.INTER_AREA)

    points = np.column_stack([xs, ys]).astype(np.float32)
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    angle_deg = np.degrees(np.arctan2(major[1], major[0]))
    rotate_deg = 90.0 - angle_deg
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), rotate_deg, 1.0)
    rotated_gray = cv2.warpAffine(gray, matrix, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
    rotated_mask = cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)

    ys2, xs2 = np.nonzero(rotated_mask > 0)
    if len(xs2) >= 3:
        x0, x1 = int(xs2.min()), int(xs2.max()) + 1
        y0, y1 = int(ys2.min()), int(ys2.max()) + 1
        pad = max(4, int(0.18 * max(x1 - x0, y1 - y0)))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(rotated_gray.shape[1], x1 + pad)
        y1 = min(rotated_gray.shape[0], y1 + pad)
        rotated_gray = rotated_gray[y0:y1, x0:x1]

    canvas = np.zeros((output_size, output_size), dtype=np.uint8)
    scale = min(output_size / rotated_gray.shape[1], output_size / rotated_gray.shape[0])
    resized = cv2.resize(
        rotated_gray,
        (max(1, int(rotated_gray.shape[1] * scale)), max(1, int(rotated_gray.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    yy = (output_size - resized.shape[0]) // 2
    xx = (output_size - resized.shape[1]) // 2
    canvas[yy:yy + resized.shape[0], xx:xx + resized.shape[1]] = resized
    return robust_normalize_gray(canvas)


def preprocess_image(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return img
    if mode == "pca_mask":
        return pca_align_mask(img, output_size=int(img.shape[0]))
    if mode == "soft_gray_pca":
        return pca_align_soft_gray(img, output_size=int(img.shape[0]))
    raise ValueError(f"Unknown preprocess mode: {mode}")
