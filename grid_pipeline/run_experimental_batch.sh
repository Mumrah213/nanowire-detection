#!/usr/bin/env bash
set -euo pipefail

# Run the grid pipeline (contrast-sweep) over a folder of SEM images.
# Tuning lives in grid_pipeline/config.py, not on the command line.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_DIR="${INPUT_DIR:-experimental_sem}"
OUTPUT_DIR="${OUTPUT_DIR:-experimental_sem_results/grid_pipeline_batch}"

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" -m grid_pipeline.pipeline "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/run.log"

echo
echo "Wrote:"
echo "  $OUTPUT_DIR/summary.csv"
echo "  $OUTPUT_DIR/summary.json"
echo "  $OUTPUT_DIR/run.log"
