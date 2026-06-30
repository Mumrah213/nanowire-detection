"""Unit tests for the pure helpers in grid_pipeline.l_markers."""

import numpy as np
import pytest

from grid_pipeline import l_markers as lm
from grid_pipeline.config import ORIENTATION_TO_SIGN_NM
from .conftest import make_lattice


# --- anchor_nm ------------------------------------------------------------

@pytest.mark.parametrize("orientation,sign", ORIENTATION_TO_SIGN_NM.items())
def test_anchor_nm_signs(orientation, sign):
    out = lm.anchor_nm(orientation, "small", small_l_nm=10000.0, big_l_nm=20000.0)
    assert out == [sign[0] * 10000.0, sign[1] * 10000.0]


def test_anchor_nm_big_uses_big_magnitude():
    assert lm.anchor_nm("UR", "big", 10000.0, 20000.0) == [20000.0, 20000.0]


def test_anchor_nm_rejects_unknown():
    assert lm.anchor_nm("XX", "small", 10000.0, 20000.0) is None
    assert lm.anchor_nm("UR", "ambiguous", 10000.0, 20000.0) is None


# --- size_from_l_area -----------------------------------------------------

def test_size_from_l_area_classifies_by_normalized_area():
    lattice = make_lattice(spacing=100.0)  # cell area 10000
    assert lm.size_from_l_area(200, lattice, small_max=0.03, big_min=0.12)["size"] == "small"
    assert lm.size_from_l_area(1500, lattice, small_max=0.03, big_min=0.12)["size"] == "big"
    assert lm.size_from_l_area(700, lattice, small_max=0.03, big_min=0.12)["size"] == "ambiguous"


def test_size_from_l_area_no_lattice():
    out = lm.size_from_l_area(500, None, 0.03, 0.12)
    assert out["size"] == "unknown" and out["area_norm"] is None


# --- l_template -----------------------------------------------------------

@pytest.mark.parametrize("orientation", list(ORIENTATION_TO_SIGN_NM))
def test_l_template_has_two_arms_on_correct_edges(orientation):
    size = 40
    t = lm.l_template(orientation, size, thickness_fraction=0.2)
    assert t.shape == (size, size)
    top_set = t[0, :].all()
    bottom_set = t[-1, :].all()
    left_set = t[:, 0].all()
    right_set = t[:, -1].all()
    assert top_set == ("U" in orientation)
    assert bottom_set == ("U" not in orientation)
    assert left_set == orientation.endswith("L")
    assert right_set == orientation.endswith("R")


# --- fixed_crop_window ----------------------------------------------------

def test_fixed_crop_window_uses_lattice_spacing():
    lattice = make_lattice(spacing=80.0)
    left, top, right, bottom, side = lm.fixed_crop_window(
        (100, 100, 20, 20), image_shape=(500, 500), lattice=lattice, crop_lattice_steps=1.0
    )
    assert side == 80
    assert right - left == side and bottom - top == side


def test_fixed_crop_window_clamps_to_image_bounds():
    lattice = make_lattice(spacing=80.0)
    # bbox near the top-left corner: window must not go negative
    left, top, right, bottom, side = lm.fixed_crop_window(
        (0, 0, 10, 10), image_shape=(200, 200), lattice=lattice, crop_lattice_steps=1.0
    )
    assert left >= 0 and top >= 0
    assert right <= 200 and bottom <= 200


def test_fixed_crop_window_never_smaller_than_component():
    left, top, right, bottom, side = lm.fixed_crop_window(
        (10, 10, 50, 60), image_shape=(400, 400), lattice=None, crop_lattice_steps=1.0
    )
    assert side >= max(50, 60) + 4
