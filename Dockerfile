# syntax=docker/dockerfile:1.7
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.32
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${UV_IMAGE} AS uv

FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

ARG MJLAB_REF=v1.6.0
ARG MJLAB_COMMIT=0fb8a681136be94ffc636a3dd423cabb97d91f10

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/gr3mini-tracking-venv \
    GR3MINI_USE_PREBUILT_ENV=1 \
    GR3MINI_TRACKING_ROOT=/workspace/gr3mini_tracking \
    MUJOCO_GL=egl \
    MPLBACKEND=Agg \
    PYOPENGL_PLATFORM=egl \
    PATH="/opt/gr3mini-tracking-venv/bin:${PATH}"

COPY --from=uv /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        ffmpeg \
        git \
        libegl1 \
        libgl1 \
        libgles2 \
        libglib2.0-0 \
        libgomp1 \
        libosmesa6 \
        libsm6 \
        libxext6 \
        libxrender1 \
        python3.12 \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/gr3mini_tracking

# The project currently declares mjlab as ../mjlab. Keep that source layout
# internal to the image so image users need not provide it themselves.
RUN git clone --depth 1 --branch "${MJLAB_REF}" https://github.com/mujocolab/mjlab.git /workspace/mjlab \
    && test "$(git -C /workspace/mjlab rev-parse HEAD)" = "${MJLAB_COMMIT}"

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY . ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

RUN python - <<'PY'
import gr3mini_tracking
import mjlab
import torch

print("gr3mini_tracking import OK")
print("mjlab import OK")
print("torch =", torch.__version__, "CUDA =", torch.version.cuda)
PY

COPY scripts/docker/gr3mini-entrypoint.sh /usr/local/bin/gr3mini-entrypoint
RUN chmod +x /usr/local/bin/gr3mini-entrypoint

ENTRYPOINT ["/usr/local/bin/gr3mini-entrypoint"]
CMD ["Gr3Mini-Tracking-Teacher"]
