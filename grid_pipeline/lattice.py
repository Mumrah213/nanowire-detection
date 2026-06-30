"""Step 2: fit the dot lattice, decide which dots are on-grid, and convert
between pixels, lattice cells, and physical (um) coordinates.

A "lattice" here is a dict with keys ``origin`` (px), ``spacing`` (px) and
``angle`` (rad), plus RANSAC bookkeeping (``n_total``/``n_inliers``/...).
"""

import math

import numpy as np

from utils.grid_ransac import find_candidate_spacings, ransac_2d_grid_fitting
from grid_pipeline.config import ORIENTATION_TO_SIGN_NM


# --- fitting -------------------------------------------------------------

def fit_lattice(dots: list[dict]) -> dict | None:
    positions = [(dot["x"], dot["y"]) for dot in dots]
    if len(positions) < 6:
        return None
    spacings = find_candidate_spacings(positions, min_dist=40, max_dist=300)
    if not spacings:
        return None
    lattice = ransac_2d_grid_fitting(positions, spacings, n_iterations=600, tolerance=10)
    if not lattice or int(lattice["n_inliers"]) < 6 or not lattice.get("spacing"):
        return None
    return lattice


def lattice_quality(lattice: dict | None) -> tuple[int, float, int]:
    if not lattice:
        return (0, 0.0, 0)
    total = max(1, int(lattice["n_total"]))
    inliers = int(lattice["n_inliers"])
    ratio = inliers / total
    return (int(ratio >= 0.50 and inliers >= 12), ratio, inliers)


def choose_dot_grid(bright_dots: list[dict], dark_dots: list[dict]) -> tuple[str, list[dict], dict | None]:
    bright_lattice = fit_lattice(bright_dots)
    dark_lattice = fit_lattice(dark_dots)
    if lattice_quality(dark_lattice) > lattice_quality(bright_lattice):
        return "dark", dark_dots, dark_lattice
    return "bright", bright_dots, bright_lattice


# --- on-grid vs off-grid classification ----------------------------------

def filter_lattice_inliers(
    blobs: list[dict],
    lattice: dict | None,
    tolerance_fraction: float,
    min_tolerance_px: float,
) -> tuple[list[dict], list[dict]]:
    if not lattice or not lattice.get("spacing"):
        return blobs, []
    tolerance_px = max(min_tolerance_px, tolerance_fraction * float(lattice["spacing"]))
    accepted = []
    rejected = []
    for blob in blobs:
        residual = lattice_residual((blob["x"], blob["y"]), lattice)
        item = {**blob, "lattice_residual_px": float(residual), "lattice_tolerance_px": float(tolerance_px)}
        if residual <= tolerance_px:
            accepted.append(item)
        else:
            item["rejection_reason"] = "off_lattice"
            rejected.append(item)
    return accepted, rejected


def classify_grid_markers(
    detected_dots: list[dict],
    lattice: dict | None,
    args,
) -> tuple[list[dict], list[dict], dict | None]:
    """Split detected markers into on-grid markers vs removed outliers/dirt.

    Philosophy: the grid is an ideal lattice. A detected marker near a node
    (within a deliberately loose tolerance) is that node's marker, even if it
    is physically shifted -- the shift is real signal we keep and report. A
    detection too far from every node is an outlier/dirt and is removed; it is
    never snapped onto. The lattice is then refit (least squares) to the
    accepted markers so the ideal grid best matches the real ones.
    Returns (accepted, rejected, refined_lattice).
    """
    if not lattice or not lattice.get("spacing"):
        return detected_dots, [], lattice

    accepted, rejected = filter_lattice_inliers(
        detected_dots, lattice,
        args.dot_lattice_tolerance_fraction, args.dot_min_lattice_tolerance_px,
    )
    if args.refine_lattice and len(accepted) >= 6:
        obs = [(*lattice_cell((d["x"], d["y"]), lattice), d["x"], d["y"]) for d in accepted]
        lattice = refine_lattice_similarity(obs, lattice)
        accepted, rejected = filter_lattice_inliers(
            detected_dots, lattice,
            args.dot_lattice_tolerance_fraction, args.dot_min_lattice_tolerance_px,
        )

    # One marker per cell: keep the detection closest to each node; the rest of
    # any cluster (typically dirt beside a real marker) is removed.
    best: dict[tuple[int, int], dict] = {}
    extra: list[dict] = []
    for dot in sorted(accepted, key=lambda d: d["lattice_residual_px"]):
        cell = lattice_cell((dot["x"], dot["y"]), lattice)
        if cell in best:
            extra.append({**dot, "rejection_reason": "duplicate_cell"})
        else:
            best[cell] = dot

    kept = list(best.values())
    interior_removed = []
    if args.reject_interior_displaced:
        tol_px = max(
            args.interior_min_displacement_px,
            args.interior_max_displacement_fraction * float(lattice["spacing"]),
        )
        kept, interior_removed = reject_interior_displaced(kept, lattice, tol_px)
    return kept, rejected + extra + interior_removed, lattice


def reject_interior_displaced(accepted: list[dict], lattice: dict, tol_px: float) -> tuple[list[dict], list[dict]]:
    """Remove markers pinned between accepted neighbors yet displaced from them.

    A marker flanked by accepted (green) neighbors on a row or column is fixed
    by the rigid lattice: a real marker there sits at the neighbors' midpoint.
    If it deviates by more than tol_px it is dirt / a misdetection, not a
    genuine shift, so it is removed. Removal is greedy (worst offender first,
    re-evaluated each step) so a marker is never dropped merely because a
    soon-to-be-removed displaced neighbor skewed its predicted position.
    """
    if not lattice or not lattice.get("spacing") or tol_px <= 0:
        return accepted, []
    cellmap = {lattice_cell((d["x"], d["y"]), lattice): d for d in accepted}
    removed = []
    while True:
        worst_cell, worst_dev = None, tol_px
        for (i, j), dot in cellmap.items():
            preds = []
            for a, b in (((i - 1, j), (i + 1, j)), ((i, j - 1), (i, j + 1))):
                if a in cellmap and b in cellmap:
                    da, db = cellmap[a], cellmap[b]
                    preds.append(((da["x"] + db["x"]) / 2.0, (da["y"] + db["y"]) / 2.0))
            if not preds:
                continue
            dev = max(math.hypot(dot["x"] - px, dot["y"] - py) for px, py in preds)
            if dev > worst_dev:
                worst_cell, worst_dev = (i, j), dev
        if worst_cell is None:
            break
        dot = cellmap.pop(worst_cell)
        removed.append({**dot, "rejection_reason": "interior_displaced", "interior_displacement_px": float(worst_dev)})
    return list(cellmap.values()), removed


def split_removed(rejected_dots: list[dict], lattice: dict | None, band_px: float) -> tuple[list[dict], int]:
    """Split removed detections into audit-worthy (drawn/stored) and noise (only
    counted). Audit-worthy = a cell duplicate (two candidates, one node -- did we
    keep the right one?) or a detection just past the acceptance tolerance (a
    possibly-shifted marker). Adaptive detection floods the frame with bright
    specks the lattice prunes correctly; those are noise, not worth showing.
    """
    if not lattice or not lattice.get("spacing"):
        return list(rejected_dots), 0
    near, noise = [], 0
    for dot in rejected_dots:
        res = dot.get("lattice_residual_px")
        if dot.get("rejection_reason") in ("duplicate_cell", "interior_displaced") or (res is not None and res <= band_px):
            near.append(dot)
        else:
            noise += 1
    return near, noise


# --- geometry: pixels <-> lattice cells ----------------------------------

def expected_lattice_points(lattice: dict, image_shape: tuple[int, int], padding_cells: int = 1) -> list[tuple[float, float]]:
    origin = lattice["origin"]
    spacing = float(lattice["spacing"])
    angle = float(lattice["angle"])
    ca, sa = math.cos(angle), math.sin(angle)

    corners = [
        (0.0, 0.0),
        (float(image_shape[1] - 1), 0.0),
        (float(image_shape[1] - 1), float(image_shape[0] - 1)),
        (0.0, float(image_shape[0] - 1)),
    ]

    gi_values = []
    gj_values = []
    for px, py in corners:
        dx = px - origin[0]
        dy = py - origin[1]
        gx = dx * ca + dy * sa
        gy = -dx * sa + dy * ca
        gi_values.append(gx / spacing)
        gj_values.append(gy / spacing)

    min_i = math.floor(min(gi_values)) - padding_cells
    max_i = math.ceil(max(gi_values)) + padding_cells
    min_j = math.floor(min(gj_values)) - padding_cells
    max_j = math.ceil(max(gj_values)) + padding_cells

    points = []
    for gi in range(min_i, max_i + 1):
        for gj in range(min_j, max_j + 1):
            x = origin[0] + gi * spacing * ca - gj * spacing * sa
            y = origin[1] + gi * spacing * sa + gj * spacing * ca
            if -spacing <= x < image_shape[1] + spacing and -spacing <= y < image_shape[0] + spacing:
                points.append((float(x), float(y)))
    return points


def lattice_cell(point: tuple[float, float], lattice: dict) -> tuple[int, int]:
    origin = lattice["origin"]
    spacing = float(lattice["spacing"])
    angle = float(lattice["angle"])
    ca, sa = math.cos(angle), math.sin(angle)
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    gx = dx * ca + dy * sa
    gy = -dx * sa + dy * ca
    return int(round(gx / spacing)), int(round(gy / spacing))


def lattice_residual(point: tuple[float, float], lattice: dict) -> float:
    origin = lattice["origin"]
    spacing = float(lattice["spacing"])
    angle = float(lattice["angle"])
    ca, sa = math.cos(angle), math.sin(angle)
    i, j = lattice_cell(point, lattice)
    expected_x = origin[0] + i * spacing * ca - j * spacing * sa
    expected_y = origin[1] + i * spacing * sa + j * spacing * ca
    return float(math.hypot(point[0] - expected_x, point[1] - expected_y))


def refine_lattice_similarity(observations: list[tuple], lattice: dict) -> dict:
    """Least-squares similarity refit from (cell_i, cell_j) -> (x, y) pairs.

    Keeps the (origin, spacing, angle) representation: a square lattice with one
    uniform scale plus a rotation is the right model for these grids (measured
    axis angle 90.4 deg, aspect 0.99), so a 4-DOF similarity is all that is
    needed to shave the few-pixel residual a single global RANSAC fit leaves
    toward the frame edges. Solves x = a*i - b*j + tx, y = b*i + a*j + ty.
    """
    if len(observations) < 4:
        return lattice
    rows, rhs = [], []
    for i, j, x, y in observations:
        rows.append([i, -j, 1.0, 0.0]); rhs.append(x)
        rows.append([j, i, 0.0, 1.0]); rhs.append(y)
    sol, *_ = np.linalg.lstsq(np.asarray(rows, float), np.asarray(rhs, float), rcond=None)
    a, b, tx, ty = (float(v) for v in sol)
    spacing = math.hypot(a, b)
    if spacing <= 1e-6:
        return lattice
    refined = dict(lattice)
    refined["origin"] = (tx, ty)
    refined["spacing"] = spacing
    refined["angle"] = math.atan2(b, a)
    return refined


def serializable_lattice(lattice: dict | None) -> dict | None:
    if not lattice:
        return None
    return {
        "origin": [float(lattice["origin"][0]), float(lattice["origin"][1])] if lattice.get("origin") else None,
        "spacing_px": float(lattice["spacing"]),
        "angle_rad": float(lattice["angle"]),
        "angle_deg": math.degrees(float(lattice["angle"])),
        "n_total": int(lattice["n_total"]),
        "n_inliers": int(lattice["n_inliers"]),
        "inlier_indices": sorted(int(idx) for idx in lattice["inlier_indices"]),
        "orientation_support": int(lattice.get("orientation_support", 0)),
    }


# --- physical (um) coordinates, anchored on the chosen L -----------------

def marker_um(cell: tuple[int, int], sel_cell: tuple[int, int], sel_nm: list, pitch_nm: float) -> tuple[float, float]:
    """Physical (x, y) in micrometers for a lattice cell, anchored on the chosen
    L. Convention: +um_x -> +grid_i, +um_y -> -grid_j (image y points down)."""
    i, j = cell
    si, sj = sel_cell
    ux = (float(sel_nm[0]) + (i - si) * pitch_nm) / 1000.0
    uy = (float(sel_nm[1]) - (j - sj) * pitch_nm) / 1000.0
    return ux, uy


def pixel_to_um(px: float, py: float, lattice: dict, sel_anchor_px, sel_anchor_nm, pitch_nm: float) -> tuple[float, float]:
    """Continuous physical (x, y) in um for any pixel, anchored on the chosen L.

    Projects the pixel offset from the L anchor onto the lattice basis (in
    fractional steps) and scales by the grid pitch. Same convention as
    marker_um (+um_x -> +grid_i, +um_y -> -grid_j) but not rounded to a cell, so
    it suits off-lattice objects such as nanowires.
    """
    s = float(lattice["spacing"])
    a = float(lattice["angle"])
    ca, sa = math.cos(a), math.sin(a)
    dx = float(px) - float(sel_anchor_px[0])
    dy = float(py) - float(sel_anchor_px[1])
    di = (dx * ca + dy * sa) / s
    dj = (-dx * sa + dy * ca) / s
    ux = (float(sel_anchor_nm[0]) + di * pitch_nm) / 1000.0
    uy = (float(sel_anchor_nm[1]) - dj * pitch_nm) / 1000.0
    return ux, uy


def predict_big_l_positions(selected: dict | None, lattice: dict | None, args) -> list[tuple[float, float, str]]:
    """Predict the four big-L elbow pixels from the selected L's anchor.

    L markers lie on the pattern diagonals at multiples of the small magnitude;
    the big Ls sit at (+/-big_nm, +/-big_nm). Given the selected L's known
    physical anchor and the lattice basis, place each big-L elbow relative to
    it. Convention from the dataset: +um_x -> +grid_i, +um_y -> -grid_j (image y
    points down). Returns (px, py, orientation) for each of the four big Ls.
    """
    if selected is None or not lattice or not lattice.get("spacing"):
        return []
    anchor = selected.get("anchor_px")
    anchor_nm = selected.get("anchor_nm")
    if not anchor or not anchor_nm:
        return []
    s = float(lattice["spacing"])
    a = float(lattice["angle"])
    ca, sa = math.cos(a), math.sin(a)
    basis_i = (s * ca, s * sa)
    basis_j = (-s * sa, s * ca)
    pitch = float(args.grid_pitch_nm)
    big_nm = float(args.big_l_nm)
    out = []
    for orientation, (sx, sy) in ORIENTATION_TO_SIGN_NM.items():
        d_x = sx * big_nm - float(anchor_nm[0])
        d_y = sy * big_nm - float(anchor_nm[1])
        di = d_x / pitch
        dj = -d_y / pitch
        px = anchor[0] + di * basis_i[0] + dj * basis_j[0]
        py = anchor[1] + di * basis_i[1] + dj * basis_j[1]
        out.append((float(px), float(py), orientation))
    return out
