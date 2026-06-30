FROM python:3.12-slim

# Install uv (reproducible Python env manager)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy project source (uv needs README.md to build the package)
COPY . .

# Install deps + project into venv (frozen = exact versions from lock)
RUN uv sync --frozen

# Use the venv's Python by default
ENV PATH="/app/.venv/bin:$PATH"

# Default command — run the combined coordinate-candidates pipeline
# Override at runtime:
#   podman run nanowire_detection \
#     python nanowire_ml/coordinate_candidates.py input.tif --output-dir results
CMD ["python", "nanowire_ml/coordinate_candidates.py"]
