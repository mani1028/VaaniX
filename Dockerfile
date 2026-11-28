# Vaani X 0.2.0 — slim CPU image.
#
# What is in this image: Python 3.11, ffmpeg, the telugu-speech package and
# its [host] extra (faster-whisper + onnxruntime + safetensors +
# sentencepiece). No torch, no CUDA.
#
# What is NOT in this image: model weights. They are mounted at runtime
# (see docker-compose.yml) or placed in the cache volume
# /home/vaani/.cache/vaani. The entrypoint refuses to start with a clear
# message when a required model folder is absent; it never invents one.
#
# What IS in this image, besides the code: configs/ (vaani.yaml +
# detection.yaml — text only, no weights, no secrets). Without it the
# server silently runs on built-in defaults instead of the shipped
# configs/vaani.yaml, and POST /api/v1/detect always 503s because
# configs/detection.yaml is missing. See docs/LIGHTWEIGHT.md.
#
# Build (from the repo root):
#   docker build -t vaani-x:0.2.0-slim .
# Run (see docs/DEPLOYMENT_TIERS.md for the tiers):
#   docker run --rm -p 8000:8000 \
#     -e TELUGU_API_KEYS='choose-a-long-random-key' \
#     -v vaani-cache:/home/vaani/.cache/vaani \
#     -v "$PWD/models:/app/models:ro" \
#     -v "$PWD/lite:/app/lite:ro" \
#     vaani-x:0.2.0-slim
#
# A :gpu image is a separate tag and is NOT built by this Dockerfile.
# Image size is not claimed here: measure it with `docker images vaani-x`
# after building and record it in docs/DEPLOYMENT_TIERS.md.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/home/vaani/.local/bin:$PATH \
    TELUGU_HOST=0.0.0.0 \
    TELUGU_PORT=8000 \
    BHASHA_ENABLE_PLATFORM=0 \
    VAANI_CACHE_DIR=/home/vaani/.cache/vaani

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 vaani \
    && mkdir -p /app/models /app/lite /home/vaani/.cache/vaani \
    && chown -R vaani:vaani /app /home/vaani

WORKDIR /app
USER vaani

COPY --chown=vaani:vaani pyproject.toml README.md LICENSE index.html ./
COPY --chown=vaani:vaani src ./src
COPY --chown=vaani:vaani scripts ./scripts
COPY --chown=vaani:vaani configs ./configs

RUN pip install --user ".[host]" \
    && python -c "import telugu_speech; print(telugu_speech.__version__)"

COPY --chown=vaani:vaani scripts/docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

EXPOSE 8000
VOLUME ["/home/vaani/.cache/vaani"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4)"
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["telugu-api"]
