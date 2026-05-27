# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.8.1
ARG PYTHON_VERSION=3.12
ARG UBUNTU_VERSION=22.04

# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: builder — compile vLLM wheel for H100 (sm_90a) only
# ──────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG PYTHON_VERSION
ARG TORCH_CUDA_ARCH_LIST="9.0a"
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG VLLM_REF=v0.21.0
ARG FLASHINFER_VERSION=0.6.8.post1
ARG MAX_JOBS=8
ARG NVCC_THREADS=4
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    CCACHE_DIR=/root/.ccache \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FLASHINFER_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS} \
    CUDA_HOME=/usr/local/cuda

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gnupg ca-certificates curl git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
        build-essential cmake ninja-build ccache \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION} 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel \
 && pip install \
        --index-url ${TORCH_INDEX_URL} \
        --extra-index-url ${PIP_INDEX_URL} \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
 && pip install \
        build packaging numpy ninja tqdm requests nvidia-ml-py \
        "apache-tvm-ffi>=0.1.6,!=0.1.8,!=0.1.8.post0,<0.2"

RUN mkdir -p /src /wheels

# ── FlashInfer: build jit-cache wheel with sm_90a-only AOT kernels ────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.ccache \
    git clone --depth 1 --branch v${FLASHINFER_VERSION} --recurse-submodules \
        https://github.com/flashinfer-ai/flashinfer.git /src/flashinfer \
 && cd /src/flashinfer \
 && pip install --no-build-isolation -e . \
 && python -m flashinfer.aot \
 && cd /src/flashinfer/flashinfer-jit-cache \
 && python -m build --no-isolation --wheel \
 && cp dist/*.whl /wheels/

# ── vLLM: build wheel with sm_90a-only kernels ────────────────────────────────
ENV VLLM_TARGET_DEVICE=cuda
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.ccache \
    git clone --depth 1 --branch ${VLLM_REF} \
        https://github.com/vllm-project/vllm.git /src/vllm \
 && cd /src/vllm \
 && python use_existing_torch.py \
 && pip install -r requirements/build/cuda.txt \
 && pip wheel --no-build-isolation --no-deps -w /wheels .


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime — slim, no nvcc, no build toolchain
# ──────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION} AS runtime

ARG PYTHON_VERSION
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG FLASHINFER_VERSION=0.6.8.post1
ARG PYTHON_JSON_LOGGER_VERSION=4.1.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
    VLLM_USAGE_SOURCE=vllm-slim-h100 \
    VLLM_ENABLE_CUDA_COMPATIBILITY=0 \
    VLLM_DO_NOT_TRACK=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gnupg ca-certificates curl libgomp1 libnuma1 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION} 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && apt-get purge -y software-properties-common gnupg \
    && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /tmp/wheels

RUN pip install --upgrade pip setuptools wheel \
 && pip install \
        --index-url ${TORCH_INDEX_URL} \
        --extra-index-url https://pypi.org/simple \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
 && pip install \
        flashinfer-python==${FLASHINFER_VERSION} \
        flashinfer-cubin==${FLASHINFER_VERSION} \
 && pip install python-json-logger==${PYTHON_JSON_LOGGER_VERSION} /tmp/wheels/*.whl \
 && flashinfer show-config \
 && flashinfer download-cubin \
 && apt-get purge -y curl && apt-get autoremove -y \
 && rm -rf /tmp/wheels /root/.cache/pip \
 && find /usr/lib/python3 /usr/local/lib/python3.12 -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true \
 && find /usr/local/lib/python3.12/site-packages -type d \( -name tests -o -name test -o -name examples \) -prune -exec rm -rf {} + 2>/dev/null || true

RUN mkdir -p /models

EXPOSE 8000

ENTRYPOINT ["vllm", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
