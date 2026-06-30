"""Unit tests for utils.sem_preprocess (annotation band + contrast)."""

import numpy as np
import pytest

from utils import sem_preprocess as sp


def synthetic_sem(h=400, w=300, band_h=60, seed=0):
    """Calm image field with a noisy, high-contrast annotation band at the bottom.

    Mimics a real SEM frame: a fairly uniform mid-grey field with a few bright
    markers, plus a metadata strip at the bottom full of saturated text.
    """
    rng = np.random.default_rng(seed)
    gray = np.full((h, w), 110, dtype=np.uint8)
    gray = np.clip(gray + rng.integers(-6, 7, size=gray.shape), 0, 255).astype(np.uint8)
    # a couple of bright markers in the field
    gray[100:108, 50:58] = 255
    gray[200:208, 150:158] = 255
    # noisy annotation band: alternating saturated rows
    band = gray[h - band_h:, :]
    band[::2, :] = 255
    band[1::2, :] = 0
    gray[h - band_h:, :] = band
    return gray, h - band_h


# --- annotation band detection -------------------------------------------

def test_detect_annotation_band_finds_cutoff():
    gray, true_cut = synthetic_sem()
    band = sp.detect_annotation_band(gray)
    assert band["cutoff_y"] == pytest.approx(true_cut, abs=8)
    assert 0.0 < band["masked_fraction"] < 0.5


def test_detect_annotation_band_clean_image_masks_little():
    rng = np.random.default_rng(1)
    gray = np.clip(110 + rng.integers(-6, 7, size=(400, 300)), 0, 255).astype(np.uint8)
    band = sp.detect_annotation_band(gray)
    # nothing band-like -> cutoff at (or very near) the bottom
    assert band["masked_fraction"] < 0.05


# --- robust contrast ------------------------------------------------------

def test_robust_contrast_basic_invariants():
    gray, true_cut = synthetic_sem()
    valid = np.ones(gray.shape, dtype=bool)
    valid[true_cut:, :] = False
    c = sp.robust_contrast_and_threshold(gray, valid)
    assert c["contrast_low"] < c["median"] < c["contrast_high"]
    assert c["threshold"] >= c["median"]
    assert 0.0 <= c["fraction_above_threshold"] <= 1.0
    assert c["robust_sigma"] >= 0.0


def test_robust_contrast_threshold_above_field_marks_markers():
    gray, true_cut = synthetic_sem()
    valid = np.ones(gray.shape, dtype=bool)
    valid[true_cut:, :] = False
    c = sp.robust_contrast_and_threshold(gray, valid)
    # bright markers (255) sit above threshold; the ~110 field does not
    assert c["threshold"] < 255
    assert c["threshold"] > 120


# --- end to end -----------------------------------------------------------

def test_preprocess_outputs_shapes_and_masking():
    gray, true_cut = synthetic_sem()
    stretched, binary, report = sp.preprocess(gray)
    assert stretched.shape == gray.shape == binary.shape
    assert stretched.dtype == np.uint8 and binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 255})
    cut = report["annotation_band"]["cutoff_y"]
    # everything below the detected band must be zeroed in both outputs
    assert stretched[cut:, :].sum() == 0
    assert binary[cut:, :].sum() == 0


def test_preprocess_detects_field_markers():
    gray, _ = synthetic_sem()
    _, binary, _ = sp.preprocess(gray)
    # the two planted markers should survive as foreground
    assert binary[100:108, 50:58].max() == 255
    assert binary[200:208, 150:158].max() == 255
