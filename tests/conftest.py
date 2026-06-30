"""Shared fixtures and helpers for the test suite.

The tests here exercise the deterministic, pure-geometry parts of the pipeline
(lattice math, RANSAC spacing detection, blob anchoring, SEM preprocessing) on
small synthetic inputs, so they run fast and need no real SEM images, no GPU,
and no torch.
"""

import math

import numpy as np
import pytest


def make_lattice(origin=(100.0, 50.0), spacing=40.0, angle=0.0, n_inliers=20, n_total=24):
    """Build a lattice dict in the shape the pipeline functions expect."""
    return {
        "origin": origin,
        "spacing": spacing,
        "angle": angle,
        "n_inliers": n_inliers,
        "n_total": n_total,
        "inlier_indices": set(range(n_inliers)),
        "orientation_support": n_inliers,
    }


def lattice_point(lattice, i, j):
    """Pixel coordinate of integer cell (i, j) under `lattice` (no rounding)."""
    s = float(lattice["spacing"])
    a = float(lattice["angle"])
    ca, sa = math.cos(a), math.sin(a)
    ox, oy = lattice["origin"]
    x = ox + i * s * ca - j * s * sa
    y = oy + i * s * sa + j * s * ca
    return x, y


@pytest.fixture
def square_lattice():
    return make_lattice(origin=(100.0, 50.0), spacing=40.0, angle=0.0)


@pytest.fixture
def rotated_lattice():
    return make_lattice(origin=(120.0, 80.0), spacing=37.5, angle=math.radians(10.0))


@pytest.fixture
def grid_positions(square_lattice):
    """A clean 5x5 grid of pixel positions on `square_lattice`."""
    return [lattice_point(square_lattice, i, j) for i in range(5) for j in range(5)]
