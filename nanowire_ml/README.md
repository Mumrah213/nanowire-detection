# Nanowire ML Prototype

Goal: train a conservative binary classifier for identifying suitable nanowires from SEM figures.

Primary labels serve to identify clean, isolated nanowires:

- `single`: one isolated nanowire - thin and long -> high aspect ratio
- `bad`: crossed wires, parallel bundles, messy clusters, short fragments, L/marker/text-like artifacts, and other non-single objects.

The current workflow is simple:

1. Generate synthetic train/validation/test crops.
2. Visually inspect contact sheets.
3. Train a small grayscale CNN.
4. Evaluate with threshold sweeps and mistake sheets.
5. Run qualitative prediction on real SEM components.

Example:

```bash
python nanowire_ml/generate_dataset.py --output-dir experimental_sem_results/nanowire_ml_dataset --train-per-binary 2000 --val-per-binary 400 --test-per-binary 400
python nanowire_ml/train_classifier.py --dataset-dir experimental_sem_results/nanowire_ml_dataset --output-dir experimental_sem_results/nanowire_ml_model --epochs 8
python nanowire_ml/evaluate_classifier.py --dataset-dir experimental_sem_results/nanowire_ml_dataset --checkpoint experimental_sem_results/nanowire_ml_model/best_model.pt --output-dir experimental_sem_results/nanowire_ml_eval
python nanowire_ml/predict_real_components.py experimental_sem/13.tif --checkpoint experimental_sem_results/nanowire_ml_model/best_model.pt --output-dir experimental_sem_results/nanowire_ml_real_13
```

High-contrast synthetic variant, closer to the L-marker synthetic dataset style:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib python nanowire_ml/generate_high_contrast_dataset.py --output-dir experimental_sem_results/nanowire_ml_high_contrast_dataset --train-per-binary 2000 --val-per-binary 400 --test-per-binary 400
PYTHONDONTWRITEBYTECODE=1 python nanowire_ml/train_classifier.py --dataset-dir experimental_sem_results/nanowire_ml_high_contrast_dataset --output-dir experimental_sem_results/nanowire_ml_high_contrast_model --epochs 8
PYTHONDONTWRITEBYTECODE=1 python nanowire_ml/evaluate_classifier.py --dataset-dir experimental_sem_results/nanowire_ml_high_contrast_dataset --checkpoint experimental_sem_results/nanowire_ml_high_contrast_model/best_model.pt --output-dir experimental_sem_results/nanowire_ml_high_contrast_eval
PYTHONDONTWRITEBYTECODE=1 python nanowire_ml/predict_real_components.py experimental_sem/13.tif --checkpoint experimental_sem_results/nanowire_ml_high_contrast_model/best_model.pt --output-dir experimental_sem_results/nanowire_ml_high_contrast_real_13 --crop-mode mask --crop-size 64 --threshold 0.85
```

For high-contrast models, `--crop-mode mask` is the fair real-SEM comparison because it converts connected components to white-on-black crops like the synthetic training data. Use the same `--crop-size` as the synthetic `--canvas-size`.

PCA-aligned + topology-veto variant:

```bash
PYTHONDONTWRITEBYTECODE=1 python nanowire_ml/train_classifier.py --dataset-dir experimental_sem_results/nanowire_ml_high_contrast_jagged_smoke_dataset --output-dir experimental_sem_results/nanowire_ml_pca_jagged_model --epochs 8 --preprocess-mode pca_mask
PYTHONDONTWRITEBYTECODE=1 python nanowire_ml/predict_real_components.py experimental_sem/13.tif --checkpoint experimental_sem_results/nanowire_ml_pca_jagged_model/best_model.pt --output-dir experimental_sem_results/nanowire_ml_pca_jagged_real_13 --crop-mode mask --crop-size 64 --preprocess-mode pca_mask --threshold 0.85 --topology-veto
```

The topology veto currently checks multiple Hough orientations, off-axis mass, estimated mask width, and skeleton branchpoints. Branchpoint count is useful on real SEM masks but noisy on jagged synthetics, so treat `--max-branchpoints` as a tuning parameter.

Soft grayscale PCA variant:

```bash
PYTHONDONTWRITEBYTECODE=1 python nanowire_ml/train_classifier.py --dataset-dir experimental_sem_results/nanowire_ml_high_contrast_jagged_smoke_dataset --output-dir experimental_sem_results/nanowire_ml_soft_gray_pca_model --epochs 8 --preprocess-mode soft_gray_pca
PYTHONDONTWRITEBYTECODE=1 python nanowire_ml/evaluate_classifier.py --dataset-dir experimental_sem_results/nanowire_ml_high_contrast_jagged_smoke_dataset --checkpoint experimental_sem_results/nanowire_ml_soft_gray_pca_model/best_model.pt --output-dir experimental_sem_results/nanowire_ml_soft_gray_pca_eval --preprocess-mode soft_gray_pca
PYTHONDONTWRITEBYTECODE=1 python nanowire_ml/predict_real_components.py experimental_sem/13.tif --checkpoint experimental_sem_results/nanowire_ml_soft_gray_pca_model/best_model.pt --output-dir experimental_sem_results/nanowire_ml_soft_gray_pca_real_13 --crop-mode raw --crop-size 64 --preprocess-mode soft_gray_pca --threshold 0.85
```

Initial comparison on `13.tif` at threshold `0.85`:

- `pca_mask`: accepts many real components unless topology veto is enabled.
- `pca_mask + topology_veto`: conservative and currently the most useful real-SEM behavior.
- `soft_gray_pca`: strong synthetic metrics, but too conservative on real raw crops with the current synthetic data.

Classical segmentation + skeleton-topology baseline:

```bash
PYTHONDONTWRITEBYTECODE=1 python nanowire_ml/segmentation_baseline.py experimental_sem/13.tif --output-dir experimental_sem_results/nanowire_segmentation_baseline_13_context61_auto --segmentation-mode auto
```

This pipeline is the current no-training alternative to the CNN: SEM preprocessing gives connected-component proposals, local segmentation refines each crop, skeleton/PCA/Hough features describe the object, and transparent topology rules decide whether it is a single nanowire candidate. It writes per-component debug panels, contact sheets, an image-wide overlay, and CSV/JSON metrics.

The output is tiered:

- `single`: hard-rule clean candidate.
- `review`: high-scoring linear candidate that failed only softer topology/isolation checks.
- `bad`: multi-orientation, broad, off-axis, too small, or otherwise structurally poor.

The contact sheets show raw crop, proposal, all local foreground, selected local segmentation, skeleton, pruned skeleton, and PCA axis. The script writes:

- `*_segmentation_contact_sheet.png`: all candidates.
- `*_segmentation_single_sheet.png`: green hard-rule singles only.
- `*_segmentation_review_sheet.png`: yellow high-scoring review candidates only.
- `*_segmentation_bad_sheet.png`: red rejected candidates only.
- `debug_panels/*_segmentation.png`: one full-resolution diagnostic panel per component.

On the current SEM smoke batch, the improved default produced:

- `13.tif`: `20 single`, `3 review`, `7 bad`
- `4.tif`: `8 single`, `0 review`, `7 bad`
- `9.tif`: `3 single`, `1 review`, `9 bad`
- `22.tif`: `4 single`, `2 review`, `12 bad`
- `23.tif`: `4 single`, `0 review`, `6 bad`
- `2_2.tif`: `18 single`, `4 review`, `11 bad`

The default hard single threshold is intentionally recall-oriented for rough SEM masks: pruned branch groups up to `7` are allowed, while broad/off-axis/multi-orientation components are still rejected. Nearby local foreground is retained as `context_warning` and shown in the sheet, but it no longer demotes an otherwise clean selected wire.

Fused blob + segmentation + CNN pipeline:

```bash
PYTHONDONTWRITEBYTECODE=1 python nanowire_ml/fused_nanowire_pipeline.py experimental_sem/13.tif --output-dir experimental_sem_results/nanowire_fused_13 --policy both
```

This is the combined production-style wrapper:

1. Blob/connected-component detection proposes candidates.
2. Local segmentation/topology assigns an interpretable tier and score.
3. CNN adds `cnn_single_prob` as secondary evidence.
4. Fusion writes separate `high_recall` and `high_precision` outputs.

The blob stage still uses SEM high-contrast preprocessing to get high-recall connected-component proposals. The fused wrapper then applies proposal-geometry gates before segmentation/CNN:

- very short proposals are rejected as likely markers/ticks/noise,
- very large electrode-like proposals are rejected,
- CNN cannot rescue a proposal rejected by these geometry gates.

Current fused smoke-batch behavior:

- `13.tif`: recall `22 single`, `0 review`, `8 bad`; precision `18 single`, `1 review`, `11 bad`
- `4.tif`: recall `8 single`, `0 review`, `7 bad`; precision `7 single`, `1 review`, `7 bad`
- `9.tif`: recall `4 single`, `1 review`, `8 bad`; precision `3 single`, `0 review`, `10 bad`
- `22.tif`: recall `4 single`, `0 review`, `14 bad`; precision `2 single`, `1 review`, `15 bad`
- `23.tif`: recall `4 single`, `0 review`, `6 bad`; precision `4 single`, `0 review`, `6 bad`
- `2_2.tif`: recall `15 single`, `1 review`, `17 bad`; precision `13 single`, `0 review`, `20 bad`

In `high_recall`, topology `single` cannot be vetoed by the CNN, and strong CNN/topology-score evidence can promote `review` to `single`. In `high_precision`, topology `single` usually needs CNN confirmation, context warnings remain `review`, and segmentation `review` is not automatically promoted.
