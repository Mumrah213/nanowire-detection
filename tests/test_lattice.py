"""Unit tests for grid_pipeline.lattice geometry."""

import math

import pytest

from grid_pipeline import lattice as L
from grid_pipeline.config import GridConfig
from .conftest import lattice_point, make_lattice


# --- pixel <-> cell round trips ------------------------------------------

@pytest.mark.parametrize("i,j", [(0, 0), (3, 2), (-1, 4), (7, -3)])
def test_lattice_cell_recovers_integer_cells(square_lattice, i, j):
    x, y = lattice_point(square_lattice, i, j)
    assert L.lattice_cell((x, y), square_lattice) == (i, j)


def test_lattice_cell_handles_rotation(rotated_lattice):
    x, y = lattice_point(rotated_lattice, 4, -2)
    assert L.lattice_cell((x, y), rotated_lattice) == (4, -2)


def test_lattice_residual_zero_on_node(square_lattice):
    x, y = lattice_point(square_lattice, 2, 3)
    assert L.lattice_residual((x, y), square_lattice) == pytest.approx(0.0, abs=1e-6)


def test_lattice_residual_is_offset_distance(square_lattice):
    x, y = lattice_point(square_lattice, 2, 3)
    # nudge 3px right, 4px down -> residual should be 5px (still nearest the same node)
    res = L.lattice_residual((x + 3.0, y + 4.0), square_lattice)
    assert res == pytest.approx(5.0, abs=1e-6)


# --- quality ordering -----------------------------------------------------

def test_lattice_quality_none_is_lowest():
    assert L.lattice_quality(None) == (0, 0.0, 0)


def test_lattice_quality_good_beats_bad():
    good = make_lattice(n_inliers=20, n_total=24)
    bad = make_lattice(n_inliers=8, n_total=40)
    assert L.lattice_quality(good) > L.lattice_quality(bad)


def test_lattice_quality_good_flag_requires_ratio_and_count():
    flag, ratio, inliers = L.lattice_quality(make_lattice(n_inliers=20, n_total=24))
    assert flag == 1 and inliers == 20 and ratio == pytest.approx(20 / 24)
    # high ratio but too few inliers -> not flagged good
    flag2, _, _ = L.lattice_quality(make_lattice(n_inliers=6, n_total=6))
    assert flag2 == 0


# --- inlier filtering -----------------------------------------------------

def test_filter_lattice_inliers_splits_on_tolerance(square_lattice):
    on = dict(zip("xy", lattice_point(square_lattice, 1, 1)))
    far = lattice_point(square_lattice, 2, 2)
    off = {"x": far[0] + 25.0, "y": far[1]}  # 25px > tolerance
    accepted, rejected = L.filter_lattice_inliers(
        [on, off], square_lattice, tolerance_fraction=0.10, min_tolerance_px=8.0
    )
    assert len(accepted) == 1 and len(rejected) == 1
    assert rejected[0]["rejection_reason"] == "off_lattice"
    assert accepted[0]["lattice_residual_px"] == pytest.approx(0.0, abs=1e-6)


def test_filter_lattice_inliers_no_lattice_passes_through():
    blobs = [{"x": 1.0, "y": 2.0}]
    accepted, rejected = L.filter_lattice_inliers(blobs, None, 0.1, 8.0)
    assert accepted == blobs and rejected == []


# --- interior displacement rejection -------------------------------------

def test_reject_interior_displaced_drops_shifted_middle(square_lattice):
    p0 = dict(zip("xy", lattice_point(square_lattice, 0, 0)))
    p2 = dict(zip("xy", lattice_point(square_lattice, 2, 0)))
    mid = lattice_point(square_lattice, 1, 0)
    bad = {"x": mid[0], "y": mid[1] + 20.0}  # flanked but displaced 20px
    kept, removed = L.reject_interior_displaced([p0, p2, bad], square_lattice, tol_px=10.0)
    assert len(removed) == 1
    assert removed[0]["rejection_reason"] == "interior_displaced"
    assert {(round(d["x"]), round(d["y"])) for d in kept} == {
        (round(p0["x"]), round(p0["y"])),
        (round(p2["x"]), round(p2["y"])),
    }


def test_reject_interior_displaced_keeps_aligned_middle(square_lattice):
    pts = [dict(zip("xy", lattice_point(square_lattice, i, 0))) for i in range(3)]
    kept, removed = L.reject_interior_displaced(pts, square_lattice, tol_px=10.0)
    assert removed == [] and len(kept) == 3


# --- least-squares refit --------------------------------------------------

def test_refine_lattice_similarity_recovers_parameters():
    truth = make_lattice(origin=(33.0, 77.0), spacing=42.5, angle=math.radians(7.0))
    obs = [(i, j, *lattice_point(truth, i, j)) for i in range(4) for j in range(4)]
    # start from a deliberately wrong guess
    guess = make_lattice(origin=(0.0, 0.0), spacing=40.0, angle=0.0)
    refined = L.refine_lattice_similarity(obs, guess)
    assert refined["spacing"] == pytest.approx(truth["spacing"], abs=1e-4)
    assert refined["angle"] == pytest.approx(truth["angle"], abs=1e-4)
    assert refined["origin"][0] == pytest.approx(truth["origin"][0], abs=1e-3)
    assert refined["origin"][1] == pytest.approx(truth["origin"][1], abs=1e-3)


def test_refine_lattice_similarity_too_few_points_is_noop():
    guess = make_lattice()
    assert L.refine_lattice_similarity([(0, 0, 1.0, 2.0)], guess) is guess


# --- physical (um) coordinates -------------------------------------------

def test_marker_um_anchor_cell_is_anchor_value():
    ux, uy = L.marker_um(cell=(5, 5), sel_cell=(5, 5), sel_nm=[10000.0, -10000.0], pitch_nm=2500.0)
    assert (ux, uy) == pytest.approx((10.0, -10.0))


def test_marker_um_sign_convention():
    # +grid_i -> +um_x ; +grid_j -> -um_y (image y points down)
    base = L.marker_um((0, 0), (0, 0), [0.0, 0.0], pitch_nm=2500.0)
    plus_i = L.marker_um((1, 0), (0, 0), [0.0, 0.0], pitch_nm=2500.0)
    plus_j = L.marker_um((0, 1), (0, 0), [0.0, 0.0], pitch_nm=2500.0)
    assert base == pytest.approx((0.0, 0.0))
    assert plus_i == pytest.approx((2.5, 0.0))
    assert plus_j == pytest.approx((0.0, -2.5))


def test_pixel_to_um_matches_marker_um_on_nodes(square_lattice):
    pitch = 2500.0
    sel_cell = (0, 0)
    sel_px = lattice_point(square_lattice, *sel_cell)
    sel_nm = [0.0, 0.0]
    for i, j in [(0, 0), (2, 1), (-1, 3)]:
        px, py = lattice_point(square_lattice, i, j)
        cont = L.pixel_to_um(px, py, square_lattice, sel_px, sel_nm, pitch)
        disc = L.marker_um((i, j), sel_cell, sel_nm, pitch)
        assert cont == pytest.approx(disc, abs=1e-6)


# --- big-L prediction -----------------------------------------------------

def test_predict_big_l_positions_returns_four_corners(square_lattice):
    args = GridConfig()
    selected = {
        "anchor_px": lattice_point(square_lattice, 0, 0),
        "anchor_nm": [args.big_l_nm, args.big_l_nm],  # this L is the UR big-L
    }
    preds = L.predict_big_l_positions(selected, square_lattice, args)
    assert {o for *_xy, o in preds} == {"UL", "UR", "LR", "LL"}
    by_o = {o: (x, y) for x, y, o in preds}
    # the UR prediction must land on the selected L's own anchor pixel
    assert by_o["UR"] == pytest.approx(tuple(selected["anchor_px"]), abs=1e-6)


def test_predict_big_l_positions_needs_anchor(square_lattice):
    args = GridConfig()
    assert L.predict_big_l_positions({"anchor_px": None, "anchor_nm": None}, square_lattice, args) == []
    assert L.predict_big_l_positions(None, square_lattice, args) == []


# --- serialization --------------------------------------------------------

def test_serializable_lattice_is_json_friendly(square_lattice):
    out = L.serializable_lattice(square_lattice)
    assert out["spacing_px"] == pytest.approx(40.0)
    assert out["angle_deg"] == pytest.approx(0.0)
    assert isinstance(out["inlier_indices"], list)
    assert L.serializable_lattice(None) is None
