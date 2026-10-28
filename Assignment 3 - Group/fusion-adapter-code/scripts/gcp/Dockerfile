# GPU image for Vertex AI custom jobs: the environment of scripts/setup_env.sh, built for
# A100, H100 and B200, plus the pinned SAM 3D / V-JEPA 2 code. No weights, data or tokens:
# checkpoints and data come from the bucket mounted at /gcs/<bucket>.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git rsync \
    && rm -rf /var/lib/apt/lists/*
ENV PROJECT_DIR=/opt/vidsam3d \
    CODE_DIR=/opt/vidsam3d/fusion-adapter-code \
    TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0"
WORKDIR /opt/vidsam3d/fusion-adapter-code

# environment first (slow, rarely changes), then the code (fast)
COPY scripts/fetch_upstream.sh scripts/setup_env.sh scripts/
COPY requirements.txt requirements.gpu.txt ./
RUN bash scripts/fetch_upstream.sh --repos-only \
    && SKIP_GPU_CHECK=1 INSTALL_PACKAGE=0 bash scripts/setup_env.sh \
    && rm -rf "$PROJECT_DIR/pip-cache" "$PROJECT_DIR/conda/pkgs" "$PROJECT_DIR/Miniforge3.sh"
COPY . .
RUN "$PROJECT_DIR/envs/vidsam3d/bin/python" -m pip install --no-deps -e .

ENV VIDSAM3D_ENV=gcp
ENTRYPOINT ["bash", "scripts/run.sh"]
