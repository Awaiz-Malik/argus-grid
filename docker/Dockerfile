# Single shared image for every Argus Grid service - compose.yaml varies the
# `command:` per service rather than maintaining 4 near-identical Dockerfiles.
FROM python:3.12-slim

# Runtime libs OpenCV's video decoding needs even in the "headless" build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY services ./services
COPY configs ./configs

# CPU-only torch first: these containers don't have GPU passthrough wired up,
# so pin the CPU wheel before ultralytics' own install would otherwise pull
# the much larger CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e .

# Bake in YOLO weights at build time so containers don't need network at first request.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

CMD ["python", "--version"]
