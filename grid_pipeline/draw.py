"""Step 4: draw the diagnostic overlay (and the side panel of L crops).

All colours live here as COL_* constants; helpers draw hollow rings, the legend
bar, per-marker coordinate labels, and the final composited overlay.
"""

import math
from pathlib import Path

import cv2
import numpy as np

from grid_pipeline.lattice import expected_lattice_points, lattice_cell, marker_um


COL_DETECTED = (95, 185, 95)    # muted green - grid dot found by blob detection
COL_MISSING = (80, 80, 205)     # muted red - expected cell, no marker found (inspect here)
COL_L_BIG = (255, 0, 255)       # magenta- big L candidate
COL_L_SMALL = (0, 255, 255)     # yellow - small L candidate
COL_L_AMBIG = (0, 165, 255)     # orange - ambiguous-size L candidate
COL_L_REJECT = (150, 150, 150)  # gray   - rejected L candidate
COL_SELECTED = (255, 255, 255)  # white  - selected L (anchors the coordinates)
COL_OFFGRID = (180, 130, 60)    # steel  - detection removed as off-grid / dirt
COL_PRED_BIG_L = (255, 255, 0)  # cyan   - predicted big-L location (illustrative)


def draw_coord_label(img: np.ndarray, x: int, y: int, ux: float, uy: float, color: tuple) -> None:
    """Small '(x,y)' coordinate label centered below a marker, with a black halo."""
    text = f"({ux:.1f},{uy:.1f})"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    org = (x - tw // 2, y)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_ring(img: np.ndarray, x: int, y: int, color: tuple, r: int, thickness: int = 2) -> None:
    """Hollow ring with a thin black halo so the enclosed marker stays visible."""
    cv2.circle(img, (x, y), r + 1, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.circle(img, (x, y), r, color, thickness, cv2.LINE_AA)


def draw_legend_bar(img: np.ndarray, entries: list[tuple], pad: int = 10) -> np.ndarray:
    """Append a black strip below the image and lay out color->meaning entries.

    Drawn in added dead space so it never covers image data. Only entries that
    actually appear in the figure are passed in, keeping it non-intrusive.
    """
    if not entries:
        return img
    font, fs, row_h, gap, sw = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 26, 24, 7
    h, w = img.shape[:2]
    rows, cur, x = [], [], pad
    for color, label in entries:
        (tw, _), _ = cv2.getTextSize(label, font, fs, 1)
        ew = sw * 2 + 8 + tw + gap
        if cur and x + ew > w - pad:
            rows.append(cur); cur, x = [], pad
        cur.append((color, label, ew)); x += ew
    if cur:
        rows.append(cur)
    bar = np.full((pad + row_h * len(rows), w, 3), 28, np.uint8)
    y = pad + row_h // 2
    for row in rows:
        x = pad
        for color, label, ew in row:
            cx = x + sw + 2
            draw_ring(bar, cx, y, color, sw, 2)
            cv2.putText(bar, label, (cx + sw + 8, y + 5), font, fs, (235, 235, 235), 1, cv2.LINE_AA)
            x += ew
        y += row_h
    return np.vstack([img, bar])


def candidate_color(candidate: dict, is_selected: bool) -> tuple:
    if is_selected:
        return COL_SELECTED
    if not candidate.get("accepted_l_candidate", False):
        return COL_L_REJECT
    return {"big": COL_L_BIG, "small": COL_L_SMALL, "ambiguous": COL_L_AMBIG}.get(
        candidate.get("size"), COL_L_REJECT
    )


def build_l_panel(gray: np.ndarray, candidates: list[dict], selected: dict | None, panel_height: int, panel_width: int = 230) -> np.ndarray:
    """A side column of zoomed raw-grayscale crops of the L candidates.

    The main overlay's rings/boxes cover the markers they annotate; this panel
    shows the underlying pixels with a thin anchor crosshair and a color-matched
    label. Only the chosen L is shown -- and when both a big and a small L are
    present in the image, one of each -- leaving the rest of the column empty.
    """
    panel = np.full((panel_height, panel_width, 3), 24, np.uint8)
    cv2.putText(panel, "Chosen L", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    top0 = 34

    # Representative chosen L per size: the selected L for its own size, and the
    # best accepted candidate of the other size if that size also appears.
    sel_label = selected["label"] if selected else None
    reps: dict[str, dict] = {}
    for c in candidates:
        if c.get("accepted_l_candidate", False) and c.get("size") in ("big", "small"):
            s = c["size"]
            if s not in reps or float(c.get("confidence", 0.0)) > float(reps[s].get("confidence", 0.0)):
                reps[s] = c
    if selected is not None and selected.get("size") in ("big", "small"):
        reps[selected["size"]] = selected
    ordered = ([selected] if selected is not None else [])
    for s in ("big", "small"):
        c = reps.get(s)
        if c is not None and (sel_label is None or c["label"] != sel_label):
            ordered.append(c)
    if not ordered:
        cv2.putText(panel, "(no L selected)", (10, top0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        return panel

    margin, label_h = 8, 18
    # Fixed slot height so 1 or 2 crops render at a consistent, legible size and
    # the remainder of the column is intentionally left empty.
    thumb = min(panel_width - 2 * margin, 200)
    y = top0
    for candidate in ordered:
        is_sel = selected is not None and candidate["label"] == sel_label
        color = candidate_color(candidate, is_sel)
        x0, y0, w, h = candidate["bbox"]
        side = int(candidate.get("crop_side_px") or max(w, h) * 1.8)
        side = max(side, max(w, h) + 6)
        cx, cy = x0 + w / 2.0, y0 + h / 2.0
        left = int(max(0, round(cx - side / 2.0)))
        top = int(max(0, round(cy - side / 2.0)))
        right = min(gray.shape[1], left + side)
        bottom = min(gray.shape[0], top + side)
        crop = gray[top:bottom, left:right]
        if crop.size == 0:
            continue
        thumb_bgr = cv2.cvtColor(cv2.resize(crop, (thumb, thumb), interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR)
        ax, ay = candidate.get("anchor_px", (cx, cy))
        sx = thumb / max(1, right - left)
        sy = thumb / max(1, bottom - top)
        axp, ayp = int(round((ax - left) * sx)), int(round((ay - top) * sy))
        if 0 <= axp < thumb and 0 <= ayp < thumb:
            cv2.drawMarker(thumb_bgr, (axp, ayp), color, cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
        cv2.rectangle(thumb_bgr, (0, 0), (thumb - 1, thumb - 1), color, 2)
        panel[y:y + thumb, margin:margin + thumb] = thumb_bgr
        prefix = "SELECTED " if is_sel else ""
        label = "%s%s p=%.2f" % (prefix, candidate.get("label_size_orientation", "?"), float(candidate.get("confidence", 0.0)))
        cv2.putText(panel, label, (margin, y + thumb + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        y += thumb + label_h + margin

    return panel


def draw_overlay(gray: np.ndarray, dots: list[dict], lattice: dict | None, candidates: list[dict], selected: dict | None, report: dict, scale: dict, out_path: Path, rejected_dots: list[dict] | None = None, panel_path: Path | None = None, predicted_big_ls: list[tuple] | None = None, marker_coords: bool = True) -> None:
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cutoff = report["annotation_band"]["cutoff_y"]
    img[cutoff:] = (img[cutoff:].astype(np.float32) * 0.35).astype(np.uint8)

    # Ring radius sits just outside a marker so the dot stays visible inside.
    dot_r = max(9, int(round(0.10 * float(lattice["spacing"])))) if lattice and lattice.get("spacing") else 9

    # Removed outliers/dirt drawn first (underneath), so kept markers sit on top.
    rejected_dots = rejected_dots or []
    for dot in rejected_dots:
        x, y = int(round(dot["x"])), int(round(dot["y"]))
        draw_ring(img, x, y, COL_OFFGRID, dot_r, 1)

    # Physical-coordinate context: anchored only when an L was selected.
    coord_ctx = None
    if (marker_coords and selected is not None and selected.get("anchor_nm")
            and lattice and lattice.get("spacing")):
        sel_cell = lattice_cell((selected["anchor_px"][0], selected["anchor_px"][1]), lattice)
        coord_ctx = (sel_cell, selected["anchor_nm"], float(scale.get("grid_pitch_nm") or 0.0))

    n_detected = 0
    for dot in dots:
        x, y = int(round(dot["x"])), int(round(dot["y"]))
        n_detected += 1
        draw_ring(img, x, y, COL_DETECTED, dot_r, 2)
        if coord_ctx:
            ux, uy = marker_um(lattice_cell((dot["x"], dot["y"]), lattice), *coord_ctx)
            draw_coord_label(img, x, y + dot_r + 11, ux, uy, COL_DETECTED)

    expected_missing = 0
    if lattice and lattice.get("spacing"):
        for x_f, y_f in expected_lattice_points(lattice, gray.shape):
            x, y = int(round(x_f)), int(round(y_f))
            if any((dot["x"] - x_f) ** 2 + (dot["y"] - y_f) ** 2 <= (0.38 * float(lattice["spacing"])) ** 2 for dot in dots):
                continue
            if x < 0 or y < 0 or x >= gray.shape[1] or y >= gray.shape[0]:
                continue
            expected_missing += 1
            draw_ring(img, x, y, COL_MISSING, dot_r, 2)
            if coord_ctx:
                ux, uy = marker_um(lattice_cell((x_f, y_f), lattice), *coord_ctx)
                draw_coord_label(img, x, y + dot_r + 11, ux, uy, COL_MISSING)

    l_sizes_present = set()
    any_rejected_l = False
    for candidate in candidates:
        x, y, w, h = candidate["bbox"]
        is_selected = selected is not None and selected["label"] == candidate["label"]
        accepted = candidate.get("accepted_l_candidate", False)
        if accepted:
            l_sizes_present.add(candidate["size"])
        else:
            any_rejected_l = True
        color = {"big": COL_L_BIG, "small": COL_L_SMALL, "ambiguous": COL_L_AMBIG}.get(
            candidate["size"], COL_L_REJECT
        )
        if not accepted:
            color = COL_L_REJECT
        outline = COL_SELECTED if is_selected else color
        thickness = 2 if is_selected else 1
        cv2.rectangle(img, (x, y), (x + w, y + h), outline, thickness)
        ax, ay = [int(round(value)) for value in candidate["anchor_px"]]
        draw_ring(img, ax, ay, outline, dot_r + 3 if is_selected else dot_r, 2 if is_selected else 1)
        prefix = "SELECTED " if is_selected else ""
        if not is_selected and not candidate.get("accepted_l_candidate", False):
            prefix = "REJECT "
        label = f"{prefix}{candidate['label_size_orientation']} p={candidate['confidence']:.2f}"
        cv2.putText(img, label, (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, label, (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    # Predicted big-L positions (illustrative): draw an L glyph + label at each
    # predicted elbow that falls within (or just past) the frame, so partially
    # visible big Ls at the edges get labelled.
    pred_drawn = 0
    if predicted_big_ls and lattice and lattice.get("spacing"):
        arm = max(18, int(round(0.35 * float(lattice["spacing"]))))
        h_img, w_img = gray.shape
        for px_f, py_f, orientation in predicted_big_ls:
            if not (-arm <= px_f <= w_img + arm and -arm <= py_f <= h_img + arm):
                continue
            ex, ey = int(round(px_f)), int(round(py_f))
            hdir = -1 if orientation.endswith("R") else 1   # horizontal arm direction
            vdir = 1 if orientation.startswith("U") else -1  # vertical arm direction
            for (p0, p1) in (((ex, ey), (ex + hdir * arm, ey)), ((ex, ey), (ex, ey + vdir * arm))):
                cv2.line(img, p0, p1, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.line(img, p0, p1, COL_PRED_BIG_L, 2, cv2.LINE_AA)
            tag = f"pred big L {orientation}"
            cv2.putText(img, tag, (ex + 4, ey - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, tag, (ex + 4, ey - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_PRED_BIG_L, 1, cv2.LINE_AA)
            pred_drawn += 1

    if lattice:
        title = (
            f"grid spacing={scale['spacing_px']:.1f}px "
            f"px_per_nm={scale['px_per_nm']:.5f} "
            f"angle={math.degrees(float(lattice['angle'])):.2f}deg "
            f"inliers={lattice['n_inliers']}/{lattice['n_total']}"
        )
    else:
        title = "no grid found"
    cv2.putText(img, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)
    if lattice:
        footer = f"detected={n_detected} expected_missing={expected_missing}"
        cv2.putText(img, footer, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, footer, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    legend = []
    if n_detected:
        legend.append((COL_DETECTED, "grid marker"))
    if expected_missing:
        legend.append((COL_MISSING, "expected, not found"))
    if rejected_dots:
        legend.append((COL_OFFGRID, "off-grid removed (near-miss)"))
    if selected is not None:
        legend.append((COL_SELECTED, "selected L (anchor)"))
    if "big" in l_sizes_present:
        legend.append((COL_L_BIG, "big L"))
    if "small" in l_sizes_present:
        legend.append((COL_L_SMALL, "small L"))
    if "ambiguous" in l_sizes_present:
        legend.append((COL_L_AMBIG, "ambiguous L"))
    if any_rejected_l:
        legend.append((COL_L_REJECT, "rejected L"))
    if pred_drawn:
        legend.append((COL_PRED_BIG_L, "predicted big L"))

    cv2.imwrite(str(out_path), draw_legend_bar(img, legend))

    if panel_path is not None:
        panel = build_l_panel(gray, candidates, selected, img.shape[0])
        combined = draw_legend_bar(np.hstack([img, panel]), legend)
        cv2.imwrite(str(panel_path), combined)
    return expected_missing
