"""Unit tests for utils.grid_ransac (spacing detection + RANSAC fitting)."""

import math

import pytest

from utils import grid_ransac as gr


def make_grid(spacing=50.0, n=7, angle=0.0, origin=(20.0, 30.0)):
    ca, sa = math.cos(angle), math.sin(angle)
    pts = []
    for i in range(n):
        for j in range(n):
            x = origin[0] + i * spacing * ca - j * spacing * sa
            y = origin[1] + i * spacing * sa + j * spacing * ca
            pts.append((x, y))
    return pts


# --- spacing detection ----------------------------------------------------

def test_find_candidate_spacings_includes_unit_pitch():
    # The function returns the most populous distance peaks (which on a square
    # grid also include sqrt(2) diagonals and 2-step distances); the contract
    # that matters downstream is that the true unit pitch is among them.
    pts = make_grid(spacing=50.0, n=7)
    spacings = gr.find_candidate_spacings(pts, min_dist=40, max_dist=300)
    assert any(s == pytest.approx(50.0, abs=2.0) for s in spacings)


def test_find_candidate_spacings_empty_when_all_filtered():
    pts = make_grid(spacing=50.0, n=7)
    # window excludes every inter-blob distance (max distance on this grid is
    # well under 600px), so no candidate survives
    assert gr.find_candidate_spacings(pts, min_dist=600, max_dist=900) == []


def test_find_candidate_spacings_no_points():
    assert gr.find_candidate_spacings([], 20, 300) == []


# --- angle estimation -----------------------------------------------------

def test_estimate_spacing_angles_recovers_rotation():
    angle = math.radians(8.0)
    pts = make_grid(spacing=50.0, n=7, angle=angle)
    hyps = gr.estimate_spacing_angles(pts, [50.0])
    assert hyps
    spacing, best_angle, support = hyps[0]
    assert spacing == pytest.approx(50.0)
    # folded into +/-45deg; 8deg should come back directly
    assert math.degrees(best_angle) == pytest.approx(8.0, abs=1.0)
    assert support > 2


# --- single-hypothesis fit ------------------------------------------------

def test_fit_grid_to_blobs_2d_all_inliers_on_clean_grid():
    pts = make_grid(spacing=50.0, n=5)
    fitted = gr.fit_grid_to_blobs_2d(pts, 50.0, 50.0, pts[0], 0.0, tolerance=8.0)
    assert len(fitted) == len(pts)


def test_fit_grid_to_blobs_2d_excludes_outlier():
    pts = make_grid(spacing=50.0, n=5)
    pts_with_dirt = pts + [(pts[0][0] + 17.0, pts[0][1] + 13.0)]
    fitted = gr.fit_grid_to_blobs_2d(pts_with_dirt, 50.0, 50.0, pts[0], 0.0, tolerance=8.0)
    assert (len(pts_with_dirt) - 1) not in fitted
    assert len(fitted) == len(pts)


# --- full RANSAC ----------------------------------------------------------

def test_ransac_2d_grid_fitting_clean_grid():
    pts = make_grid(spacing=50.0, n=6)
    result = gr.ransac_2d_grid_fitting(pts, [50.0], n_iterations=200, tolerance=8.0)
    assert result["n_total"] == len(pts)
    assert result["n_inliers"] == len(pts)
    assert result["spacing"] == pytest.approx(50.0)
    assert result["origin"] is not None


def test_ransac_2d_grid_fitting_separates_dirt():
    pts = make_grid(spacing=50.0, n=6)
    dirt = [(5.0, 7.0), (333.0, 11.0)]  # off-grid specks
    result = gr.ransac_2d_grid_fitting(pts + dirt, [50.0], n_iterations=200, tolerance=8.0)
    assert result["n_inliers"] == len(pts)
    inliers = result["inlier_indices"]
    assert len(pts) not in inliers and len(pts) + 1 not in inliers
