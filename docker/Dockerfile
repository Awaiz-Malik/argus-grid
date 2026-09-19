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

# CPU-only torch + torchvision first, both from the same CPU wheel index so
# their native ops match (installing them separately/mismatched breaks
# torchvision's custom ops, e.g. "operator torchvision::nms does not exist").
# These containers don't have GPU passthrough wired up, so this also avoids
# ultralytics pulling the much larger CUDA build.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e .

# ultralytics transitively pulls the GUI opencv-python build, which conflicts
# with our pinned opencv-python-headless in the same cv2 install location -
# same fix as the local venv (see README's Known Issue).
RUN pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir opencv-python-headless

# Bake in YOLO weights at build time so containers don't need network at first request.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

CMD ["python", "--version"]
