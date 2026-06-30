"""All tuning knobs for the grid pipeline in one place.

Edit a value here to tune the pipeline. Only the runtime essentials (input,
output dir, contrast-sweep on/off) are exposed on the command line; everything
else lives in this `GridConfig` so the code that uses it stays uncluttered.
"""

from dataclasses import dataclass

# L-marker orientation -> sign of its physical (x, y) anchor, for all 4 rotations.
ORIENTATION_TO_SIGN_NM = {
    "UL": (-1.0, 1.0),
    "UR": (1.0, 1.0),
    "LR": (1.0, -1.0),
    "LL": (-1.0, -1.0),
}

# Which two box edges each L orientation's arms lie on.
ORIENTATION_EDGES = {
    "UL": ("top", "left"),
    "UR": ("top", "right"),
    "LR": ("bottom", "right"),
    "LL": ("bottom", "left"),
}

OPPOSITE_EDGES = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


@dataclass
class GridConfig:
    # --- runtime (overridable on the command line) ---
    input: str = "experimental_sem"
    output_dir: str = "experimental_sem_results/grid_pipeline"
    contrast_sweep: bool = True

    # --- physical scale ---
    grid_pitch_nm: float = 2500.0      # one lattice step
    small_l_nm: float = 10000.0        # small L anchor magnitude (10 um)
    big_l_nm: float = 20000.0          # big L anchor magnitude (20 um)

    # --- dot / L detection thresholds (robust-sigma multipliers) ---
    dark_dot_sigma: float = 3.0
    l_sigma: float = 4.0
    dark_dot_sigma_values: str = "2.5,3.0,3.5"      # contrast-sweep grid
    l_sigma_values: str = "3.0,3.5,4.0,4.5,5.0"

    # --- adaptive bright-dot detection (illumination-flattened top-hat) ---
    adaptive_bright: bool = True
    adaptive_bright_kernel: int = 21   # top-hat disk diameter (px); > dot, < spacing
    adaptive_bright_sigma: float = 3.5  # threshold in robust sigmas above local bg
    adaptive_bright_min: float = 6.0    # absolute floor (intensity) on top-hat

    # --- lattice fit + loose-tolerance acceptance ---
    refine_lattice: bool = True                 # least-squares refit to accepted dots
    dot_lattice_tolerance_fraction: float = 0.10
    dot_min_lattice_tolerance_px: float = 8.0
    # interior-displacement rule: a marker flanked by accepted neighbours must sit
    # near their midpoint or it is removed as dirt.
    reject_interior_displaced: bool = True
    interior_max_displacement_fraction: float = 0.06
    interior_min_displacement_px: float = 5.0

    # --- L-candidate geometry gates ---
    min_l_area: int = 35
    max_l_area: int = 15000
    min_l_side: int = 8
    max_l_side: int = 220
    max_l_aspect: float = 4.0
    edge_fill_min: float = 0.30
    small_area_norm_max: float = 0.03
    big_area_norm_min: float = 0.12
    reject_ambiguous_size: bool = False

    # --- L orientation scoring ---
    orientation_score_min: float = 0.32
    orientation_margin_min: float = 0.025
    edge_score_weight: float = 0.55
    template_score_weight: float = 0.45
    template_size: int = 64
    crop_lattice_steps: float = 1.0

    # --- L lattice consistency / selection ---
    require_lattice_consistency: bool = True
    lattice_tolerance_fraction: float = 0.15
    big_lattice_tolerance_fraction: float = 0.25
    min_lattice_tolerance_px: float = 10.0
    # big/small L are the same corner at two scales: a small L must match a
    # detected big L's orientation and sit a fixed step inward.
    enforce_l_scale: bool = True
    l_scale_cell_tolerance: float = 1.0
    big_l_confidence_bonus: float = 0.20  # thumb on the scale for big Ls in conflicts

    # --- overlay extras ---
    marker_coords: bool = True   # per-marker (x,y) um labels
    predict_big_l: bool = True   # draw the four predicted big-L elbows
    l_panel: bool = True         # side panel of zoomed L crops
