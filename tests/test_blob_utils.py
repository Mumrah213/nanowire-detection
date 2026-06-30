"""Unit tests for utils.blob_utils (thresholding, component anchoring)."""

import cv2
import numpy as np
import pytest

from utils import blob_utils as bu


def blank(h=120, w=160):
    return np.zeros((h, w), dtype=np.uint8)


# --- thresholding ---------------------------------------------------------

def test_threshold_markers_separates_bright_from_dark():
    gray = blank()
    cv2.rectangle(gray, (40, 40), (70, 70), 255, -1)
    binary = bu.threshold_markers(gray)
    assert set(np.unique(binary)).issubset({0, 255})
    assert binary[55, 55] == 255  # inside the bright square
    assert binary[10, 10] == 0    # background


# --- component anchoring --------------------------------------------------

def _single_component_stats(gray):
    binary = bu.threshold_markers(gray)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # component 0 is background; return the largest foreground one
    idx = 1 + int(np.argmax([stats[i, cv2.CC_STAT_AREA] for i in range(1, num)]))
    return labels, idx, stats[idx], centroids[idx]


def test_component_anchor_small_dot_uses_centroid():
    gray = blank()
    cv2.circle(gray, (80, 60), 4, 255, -1)
    labels, idx, stat, centroid = _single_component_stats(gray)
    ax, ay, kind = bu.component_anchor(labels, idx, stat, centroid)
    assert kind == "centroid"
    assert ax == pytest.approx(80, abs=2)
    assert ay == pytest.approx(60, abs=2)


def test_component_anchor_filled_square_uses_centroid():
    gray = blank()
    cv2.rectangle(gray, (50, 50), (90, 90), 255, -1)  # high fill ratio
    labels, idx, stat, centroid = _single_component_stats(gray)
    _, _, kind = bu.component_anchor(labels, idx, stat, centroid)
    assert kind == "centroid"


def test_component_anchor_l_shape_returns_corner():
    gray = blank()
    # An "L" opening to the bottom-right -> arms on top & left edges -> top_left corner
    cv2.rectangle(gray, (40, 40), (95, 52), 255, -1)  # top arm
    cv2.rectangle(gray, (40, 40), (52, 95), 255, -1)  # left arm
    labels, idx, stat, centroid = _single_component_stats(gray)
    ax, ay, kind = bu.component_anchor(labels, idx, stat, centroid)
    assert kind == "corner:top_left"
    assert ax == pytest.approx(40, abs=3)
    assert ay == pytest.approx(40, abs=3)


# --- full detection pipeline ----------------------------------------------

def test_detect_blob_components_counts_and_sorts():
    gray = blank()
    centers = [(30, 30), (100, 30), (30, 80), (100, 80)]
    for cx, cy in centers:
        cv2.circle(gray, (cx, cy), 5, 255, -1)
    blobs, binary = bu.detect_blob_components(gray, min_area=5, max_area=5000)
    assert len(blobs) == len(centers)
    # sorted by banded-y then x: first blob is top-left, last is bottom-right
    assert (blobs[0]["x"], blobs[0]["y"]) == pytest.approx((30, 30), abs=2)
    assert (blobs[-1]["x"], blobs[-1]["y"]) == pytest.approx((100, 80), abs=2)
    assert set(np.unique(binary)).issubset({0, 255})


def test_detect_blob_components_respects_area_filter():
    gray = blank()
    cv2.circle(gray, (40, 40), 2, 255, -1)    # tiny
    cv2.circle(gray, (100, 60), 12, 255, -1)  # large
    # window admits only the large blob
    blobs, _ = bu.detect_blob_components(gray, min_area=200, max_area=5000)
    assert len(blobs) == 1
    assert blobs[0]["x"] == pytest.approx(100, abs=3)


def test_detect_blob_components_blob_schema():
    gray = blank()
    cv2.circle(gray, (60, 60), 6, 255, -1)
    blobs, _ = bu.detect_blob_components(gray)
    assert blobs
    expected_keys = {
        "x", "y", "centroid_x", "centroid_y", "anchor_type",
        "area", "width", "height", "x0", "y0",
    }
    assert expected_keys <= set(blobs[0])
