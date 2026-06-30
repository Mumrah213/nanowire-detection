#!/usr/bin/env python3
"""SEM-specific preprocessing diagnostics for marker extraction."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def detect_annotation_band(gray: np.ndarray, min_body_fraction: float = 0.55) -> dict:
    """Detect a bottom annotation/scale bar band from row statistics.

    SEM metadata bands tend to have much higher row variance and many saturated
    bright/dark pixels compared with the image field. We search from the bottom
    upward for the first sustained transition back to image-like statistics.
    """
    h, _ = gray.shape
    body_end = int(h * min_body_fraction)
    reference = gray[:body_end]

    row_std = gray.std(axis=1)
    row_bright = (gray >= 245).mean(axis=1)
    row_dark = (gray <= 5).mean(axis=1)

    ref_std = float(np.median(row_std[:body_end]))
    ref_bright = float(np.median(row_bright[:body_end]))
    ref_dark = float(np.median(row_dark[:body_end]))

    std_threshold = max(ref_std * 3.0, ref_std + 15.0)
    bright_threshold = max(ref_bright * 8.0, 0.02)
    dark_threshold = max(ref_dark * 8.0, 0.02)

    suspicious = (
        (row_std > std_threshold)
        | (row_bright > bright_threshold)
        | (row_dark > dark_threshold)
    )

    cutoff = h
    run = 0
    for y in range(h - 1, int(h * 0.45), -1):
        if suspicious[y]:
            run += 1
        elif run >= 12:
            cutoff = y + 1
            break
        else:
            run = 0

    return {
        "cutoff_y": int(cutoff),
        "masked_fraction": float((h - cutoff) / h),
        "row_std_reference_median": ref_std,
        "row_std_threshold": float(std_threshold),
        "bright_fraction_reference_median": ref_bright,
        "bright_fraction_threshold": float(bright_threshold),
        "dark_fraction_reference_median": ref_dark,
        "dark_fraction_threshold": float(dark_threshold),
    }


def robust_contrast_and_threshold(gray: np.ndarray, valid_mask: np.ndarray) -> dict:
    """Compute reproducible contrast and threshold parameters."""
    pixels = gray[valid_mask]
    percentiles = {
        f"p{p:g}": float(np.percentile(pixels, p))
        for p in (0.1, 1, 5, 50, 95, 99, 99.5, 99.9)
    }

    median = percentiles["p50"]
    mad = float(np.median(np.abs(pixels.astype(np.float32) - median)))
    robust_sigma = 1.4826 * mad

    # Markers are high-tail objects. This threshold tracks SEM gain/contrast by
    # using both percentile and robust-sigma criteria.
    threshold = max(percentiles["p99"], median + 5.0 * robust_sigma)
    threshold = min(threshold, percentiles["p99.9"])

    lo = percentiles["p1"]
    hi = percentiles["p99.5"]
    if hi <= lo:
        hi = float(pixels.max())

    return {
        "percentiles": percentiles,
        "median": float(median),
        "mad": mad,
        "robust_sigma": float(robust_sigma),
        "contrast_low": float(lo),
        "contrast_high": float(hi),
        "threshold": float(threshold),
        "fraction_above_threshold": float((pixels > threshold).mean()),
        "fraction_saturated_low": float((pixels <= 0).mean()),
        "fraction_saturated_high": float((pixels >= 255).mean()),
    }


def preprocess(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    band = detect_annotation_band(gray)
    valid_mask = np.ones(gray.shape, dtype=bool)
    valid_mask[band["cutoff_y"]:, :] = False

    contrast = robust_contrast_and_threshold(gray, valid_mask)
    lo = contrast["contrast_low"]
    hi = contrast["contrast_high"]
    stretched = ((gray.astype(np.float32) - lo) / max(1.0, hi - lo) * 255.0)
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    stretched[~valid_mask] = 0

    threshold_value = contrast["threshold"]
    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[(gray > threshold_value) & valid_mask] = 255

    report = {
        "annotation_band": band,
        "contrast": contrast,
    }
    return stretched, binary, report


def draw_report_overlay(gray: np.ndarray, binary: np.ndarray, report: dict) -> np.ndarray:
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cutoff = report["annotation_band"]["cutoff_y"]
    overlay[cutoff:] = (overlay[cutoff:].astype(np.float32) * 0.25).astype(np.uint8)
    overlay[binary > 0] = (0, 255, 255)
    cv2.line(overlay, (0, cutoff), (overlay.shape[1] - 1, cutoff), (0, 0, 255), 2)

    c = report["contrast"]
    lines = [
        f"cutoff_y={cutoff}, masked={report['annotation_band']['masked_fraction']:.1%}",
        f"threshold={c['threshold']:.1f}, above={c['fraction_above_threshold']:.3%}",
        f"contrast p1..p99.5={c['contrast_low']:.1f}..{c['contrast_high']:.1f}",
        f"saturation low/high={c['fraction_saturated_low']:.3%}/{c['fraction_saturated_high']:.3%}",
    ]
    y = 24
    for line in lines:
        cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 24
    return overlay


def main():
    parser = argparse.ArgumentParser(description="SEM preprocessing diagnostics.")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--output-dir", default="experimental_sem_results/preprocess")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for image in args.images:
        path = Path(image)
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Could not read image: {path}")

        stretched, binary, report = preprocess(gray)
        report["image"] = str(path)
        reports.append(report)

        stem = path.stem
        cv2.imwrite(str(out_dir / f"{stem}_contrast.png"), stretched)
        cv2.imwrite(str(out_dir / f"{stem}_binary.png"), binary)
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), draw_report_overlay(gray, binary, report))
        (out_dir / f"{stem}_report.json").write_text(json.dumps(report, indent=2))

        band = report["annotation_band"]
        contrast = report["contrast"]
        print(
            f"{path}: cutoff_y={band['cutoff_y']} masked={band['masked_fraction']:.1%} "
            f"threshold={contrast['threshold']:.1f} above={contrast['fraction_above_threshold']:.3%} "
            f"contrast={contrast['contrast_low']:.1f}..{contrast['contrast_high']:.1f}"
        )

    (out_dir / "reports.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
