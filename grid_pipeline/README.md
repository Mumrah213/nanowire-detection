# Grid coordinate pipeline

This module turns a raw SEM field into a physical (µm) coordinate frame anchored
on the device's fiducial markers. The stages are:

1. detect compact dot blobs
2. fit the grid/lattice
3. derive `px_per_nm` from grid pitch
4. segment non-dot L candidates
5. infer L orientation with geometry + binary-template scoring (`LR`, `LL`, `UR`, `UL`)
6. assign `small`/`big` from calibrated/grid-normalized L size

Run one image:

```bash
python -m grid_pipeline.pipeline experimental_sem/23.tif --output-dir experimental_sem_results/grid_pipeline_23
```

Run a directory:

```bash
python -m grid_pipeline.pipeline experimental_sem --output-dir experimental_sem_results/grid_pipeline_batch
```

Run the experimental batch with recorded parameters:

```bash
bash grid_pipeline/run_experimental_batch.sh
```

By default, each image is tested across multiple contrast/threshold settings:

```text
dark dot sigma: 2.5, 3.0, 3.5
L sigma:        3.0, 3.5, 4.0, 4.5, 5.0
```

The summary selects the highest-scoring outcome. All alternatives are kept under
`_contrast_sweep/<image_stem>/`.

The top-level result returns one answer:

```text
most_likely_candidate
```

That candidate is selected from accepted L candidates only. An accepted candidate
must pass orientation scoring, size classification, and lattice consistency.
In summaries, `l_candidates` means accepted grid-matching candidates only;
`l_like_components` includes rejected diagnostics as well.

Override parameters with environment variables:

```bash
ORIENTATION_SCORE_MIN=0.28 ORIENTATION_MARGIN_MIN=0.01 \
bash grid_pipeline/run_experimental_batch.sh
```

Override the contrast sweep:

```bash
DARK_DOT_SIGMA_VALUES=2.0,2.5,3.0 L_SIGMA_VALUES=2.5,3.0,3.5 \
bash grid_pipeline/run_experimental_batch.sh
```

The important outputs are:

- `overlays/*_overlay.png`: top-level winning overlays for visual review
- `details/*_pipeline.json`: top-level winning JSON details
- `summary.csv` / `summary.json`: one row per image
- `_contrast_sweep/<stem>/<variant>/`: contrast masks, crops, overlays, and full variant diagnostics
- `_diagnostics/<stem>/single/`: diagnostics when contrast sweep is disabled

Orientation is scored from two model-free signals:

- edge-fill geometry: which two adjacent component edges are occupied
- normalized binary-template similarity: IoU/coverage against `UL`, `UR`, `LR`, `LL` templates

Size is intentionally separate from orientation. The `small`/`big` decision uses
component area normalized by fitted grid spacing.

L candidates are rejected by default if their inferred anchor does not match
the fitted lattice:

```text
small/ambiguous: lattice residual <= max(10 px, 0.15 * grid spacing)
big:             lattice residual <= max(10 px, 0.25 * grid spacing)
```

Rejected L-like components are still shown in overlays as gray `REJECT` boxes
and listed in each `*_pipeline.json`.

Dot markers are stricter than L markers and are filtered down to the lattice
inliers with a tighter tolerance:

```text
dot residual <= max(2.5 px, 0.03 * grid spacing)
```

Accepted dot inliers are drawn as green circles in the overlay. Rejected
dot-like blobs are not shown. Missing expected lattice positions are drawn as
yellow circles.

Relax the lattice check only for debugging:

```bash
REQUIRE_LATTICE_CONSISTENCY=0 bash grid_pipeline/run_experimental_batch.sh
```

Tune the tolerance:

```bash
LATTICE_TOLERANCE_FRACTION=0.20 BIG_LATTICE_TOLERANCE_FRACTION=0.30 \
MIN_LATTICE_TOLERANCE_PX=12 \
bash grid_pipeline/run_experimental_batch.sh
```
