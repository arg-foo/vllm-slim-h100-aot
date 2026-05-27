# vllm-slim

Slim, H100-only ([compute capability 9.0a](https://developer.nvidia.com/cuda-gpus)) vLLM serving image, built for airgapped operation. Model weights are mounted from the host at runtime — the container never talks to Hugging Face, GitHub, or any external service.

- vLLM v0.21.0
- PyTorch 2.11.0 + CUDA 12.8
- FlashInfer 0.6.8.post1 with AOT-compiled kernels and pre-baked cubins
- Targets `sm_90a` only — will not run on Ada (4090), Ampere (A100), or Blackwell

---

## Prerequisites

**On the build host** (needs internet):
- Docker 24+ with BuildKit
- ~50 GB free disk for the build
- One-time HF login if you'll be staging gated models: `huggingface-cli login`

**On the runtime host** (can be fully airgapped):
- NVIDIA H100 (SXM or PCIe), driver ≥ 535
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Docker 24+
- Weights staged at a known host path (see below)

---

## Build the image

```bash
docker build -t vllm-slim:0.21.0 .
```

Build takes ~30–60 min on a 16-core box; the dominant cost is compiling vLLM and FlashInfer kernels for `sm_90a`. Resulting image is ~5 GB (no weights).

Override defaults if needed:
```bash
docker build \
  --build-arg VLLM_REF=v0.21.0 \
  --build-arg FLASHINFER_VERSION=0.6.8.post1 \
  --build-arg MAX_JOBS=16 \
  -t vllm-slim:0.21.0 .
```

---

## Stage model weights (Llama-3-70B example)

Llama-3-70B is gated — the staging host must have an HF account with access to `meta-llama/Meta-Llama-3-70B-Instruct`.

### On an internet-connected staging host

```bash
huggingface-cli login   # paste an HF token with read access to the repo

huggingface-cli download \
  meta-llama/Meta-Llama-3-70B-Instruct \
  --local-dir /staging/Llama-3-70B-Instruct \
  --local-dir-use-symlinks False
```

This downloads ~140 GB. Expected layout after completion:

```
/staging/Llama-3-70B-Instruct/
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── model.safetensors.index.json
└── model-00001-of-00030.safetensors ... model-00030-of-00030.safetensors
```

All of these files are required. Missing `tokenizer_config.json` will break chat templating; missing the `.index.json` will break shard loading.

### Optional: verify integrity

```bash
cd /staging/Llama-3-70B-Instruct
sha256sum -c <(huggingface-cli scan-cache --quiet)   # or compare against published hashes
```

### Transfer to the airgapped host

Whatever your approved transfer mechanism is (physical media, secure file transfer, approved one-way diode). The directory should end up at a stable path on the runtime host, e.g. `/srv/models/Llama-3-70B-Instruct/`.

---

## Run the container

### Single-host, 8x H100 SXM (TP=8) — the standard 70B serving config

```bash
docker run -d --name vllm-llama-3-70b \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  --restart unless-stopped \
  -p 8000:8000 \
  -v /srv/models/Llama-3-70B-Instruct:/models/Llama-3-70B:ro \
  vllm-slim:0.21.0 \
  /models/Llama-3-70B \
  --host 0.0.0.0 --port 8000 \
  --served-model-name llama-3-70b-instruct \
  --tensor-parallel-size 8 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

Notes on the flags:
- `--gpus all` — exposes all H100s to the container via NVIDIA Container Toolkit
- `--ipc=host` and `--shm-size=16g` — required for multi-GPU tensor parallel (NCCL uses shared memory between worker processes)
- `:ro` — weights mounted read-only; vLLM only reads
- The model path `/models/Llama-3-70B` *replaces* the default `CMD`, so `--host/--port` must be repeated
- `--served-model-name` controls the `model` field clients send in OpenAI API requests
- `--tensor-parallel-size 8` shards the 70B model across 8 H100s (~17.5 GB weights per GPU + KV cache)
- `--max-model-len 8192` caps context; raise carefully — KV cache grows linearly

### Single H100 (only viable with a quantized 70B checkpoint)

70B at fp16 needs ~140 GB of weight memory, which does not fit on one 80 GB H100. For single-GPU serving you must stage an AWQ/GPTQ/int4 checkpoint instead, e.g. `casperhansen/llama-3-70b-instruct-awq`:

```bash
docker run -d --name vllm-llama-3-70b-awq \
  --gpus '"device=0"' \
  -p 8000:8000 \
  -v /srv/models/Llama-3-70B-AWQ:/models/Llama-3-70B-AWQ:ro \
  vllm-slim:0.21.0 \
  /models/Llama-3-70B-AWQ \
  --host 0.0.0.0 --port 8000 \
  --quantization awq \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

---

## Verify the server is up

```bash
# Container logs — wait for "Uvicorn running on http://0.0.0.0:8000"
docker logs -f vllm-llama-3-70b

# Health check
curl http://localhost:8000/health

# List served model(s)
curl http://localhost:8000/v1/models

# Smoke test via OpenAI-compatible chat endpoint
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-70b-instruct",
    "messages": [{"role": "user", "content": "ping"}],
    "max_tokens": 16
  }'
```

First request will be slow (CUDA graph capture + KV cache warmup); subsequent requests should be sub-second to first token.

---

## Metrics (Prometheus scrape)

vLLM exposes a Prometheus-format `/metrics` endpoint on the **same port as the API** — no extra flag, no extra port, on by default. Scrape `http://<host>:8000/metrics` from your internal Prometheus (or VictoriaMetrics / Mimir / Thanos / Datadog OpenMetrics integration / whatever you run).

### Quick check the endpoint is live

```bash
curl -s http://localhost:8000/metrics | head -40
```

You should see standard Prometheus exposition: `# HELP`, `# TYPE`, then metric lines prefixed with `vllm:`.

### Headline metrics

| Metric | Type | What it tells you |
|---|---|---|
| `vllm:e2e_request_latency_seconds` | histogram | End-to-end request latency. p50/p95/p99 are your SLA dashboard. |
| `vllm:time_to_first_token_seconds` | histogram | TTFT — the user-perceived latency metric for streaming chat. |
| `vllm:inter_token_latency_seconds` | histogram | Streaming smoothness. Spikes here = batching pressure. |
| `vllm:num_requests_running` | gauge | Requests actively decoding. |
| `vllm:num_requests_waiting` | gauge | Queue depth. **Primary autoscaling signal** — sustained > 0 = under-provisioned. |
| `vllm:kv_cache_usage_perc` | gauge | KV cache occupancy 0–1. Climbing toward 1.0 = about to preempt. |
| `vllm:num_preemptions_total` | counter | Preemption count. Non-zero in steady state = memory-starved; raise `--gpu-memory-utilization` or lower `--max-num-seqs` / `--max-model-len`. |
| `vllm:prompt_tokens_total` | counter | Input throughput. |
| `vllm:generation_tokens_total` | counter | Output throughput. The real "tokens/sec" once you `rate()` it. |
| `vllm:request_success_total` | counter | Successful requests. Pair with HTTP 5xx counts for error rate. |

All metrics carry a `model_name` label matching `--served-model-name`, so a single Prometheus job can scrape multiple vLLM hosts serving different models and you can split by label.

### Prometheus scrape config

```yaml
# prometheus.yml on your internal monitoring host
scrape_configs:
  - job_name: vllm
    metrics_path: /metrics
    scrape_interval: 15s
    static_configs:
      - targets:
          - vllm-host-1.internal:8000
          - vllm-host-2.internal:8000
        labels:
          cluster: prod-east
          model: llama-3-70b-instruct
```

If your hosts are managed dynamically, swap `static_configs` for `file_sd_configs` or your service-discovery mechanism of choice.

### Exposing the port

`/metrics` rides on the same `:8000` you already publish for the API. The `docker run` example earlier already does `-p 8000:8000`, so the scrape endpoint is reachable at `http://<host>:8000/metrics` with no further changes. If your security posture requires separating the metrics surface from the API surface, front it with a reverse proxy that routes `/metrics` to a restricted listener and `/v1/*` to the public one.

### Dashboard starter queries

```promql
# Tokens/sec (output, per host)
rate(vllm:generation_tokens_total[1m])

# p95 TTFT
histogram_quantile(0.95, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le, model_name))

# Queue depth (sustained > 0 means add capacity)
vllm:num_requests_waiting

# KV cache headroom
1 - vllm:kv_cache_usage_perc

# Preemption rate
rate(vllm:num_preemptions_total[5m])
```

A community-maintained Grafana dashboard exists at [vllm-project/vllm](https://github.com/vllm-project/vllm/tree/main/examples/online_serving/prometheus_grafana) — copy `grafana.json` into your Grafana instance as a starting point.

---

## Logging

vLLM logs through Python's standard `logging`. By default it emits human-readable lines to
stdout. For structured JSON logs, point vLLM at a `logging.dictConfig` file via
`VLLM_LOGGING_CONFIG_PATH`.

`python-json-logger` (pinned at `4.1.0`) is baked into the image, so JSON logging works
without installing anything at runtime — important for the airgapped runtime host. The
image itself ships **no** logging config; you supply one at runtime so formatters stay
customizable per deployment.

### Enable JSON logging

A ready-to-use sample lives at [`logging.json`](logging.json) in this repo. Mount it and
set the env var:

```bash
docker run -d --name vllm-llama-3-70b \
  --gpus all --ipc=host --shm-size=16g \
  -p 8000:8000 \
  -v /srv/models/Llama-3-70B-Instruct:/models/Llama-3-70B:ro \
  -v $PWD/logging.json:/etc/vllm/logging.json:ro \
  -e VLLM_LOGGING_CONFIG_PATH=/etc/vllm/logging.json \
  vllm-slim:0.21.0 \
  /models/Llama-3-70B \
  --host 0.0.0.0 --port 8000 \
  --served-model-name llama-3-70b-instruct \
  --tensor-parallel-size 8
```

The sample formats `vllm` and `uvicorn` logs as JSON on stdout, quiets `uvicorn.access`
noise to WARNING, and includes the worker PID (`%(process)d`) so logs from tensor-parallel
worker processes can be split apart. Edit the fields to taste.

> **Formatter class path:** use `pythonjsonlogger.json.JsonFormatter` (the canonical path
> in `python-json-logger` 4.x). The older `pythonjsonlogger.jsonlogger.JsonFormatter` —
> still shown in some vLLM docs — works but emits a `DeprecationWarning`.

Other knobs:
- `VLLM_LOGGING_LEVEL=DEBUG|INFO|WARNING|ERROR` — quick level change without a config file.
- `VLLM_CONFIGURE_LOGGING=0` — disable vLLM's logging setup entirely (e.g. when you run the
  server in-process and configure `logging` yourself).

### Log rotation

Rotation is the **container runtime's** job, not an in-container file handler — vLLM
tensor-parallel runs spawn multiple worker processes that would race on a shared log file.
Keep logs on stdout (as the sample does) and let Docker's `json-file` driver rotate them:

```bash
docker run ... \
  --log-driver=json-file \
  --log-opt max-size=100m \
  --log-opt max-file=10 \
  --log-opt compress=true \
  vllm-slim:0.21.0 ...
```

Or make it the daemon default in `/etc/docker/daemon.json`:

```json
{
  "log-driver": "json-file",
  "log-opts": {"max-size": "100m", "max-file": "10", "compress": "true"}
}
```

That caps logs at 100 MB × 10 files = 1 GB with automatic rotation and compression, all
still visible through `docker logs` and shippable through any other log driver (fluentd,
awslogs, gcplogs, journald, …) later.

To pin a different `python-json-logger` version, rebuild with
`--build-arg PYTHON_JSON_LOGGER_VERSION=...` (see [Updating the image](#updating-the-image)).

---

## Common runtime flags

| Flag | Default | Notes |
|---|---|---|
| `--tensor-parallel-size N` | 1 | Shard model across N GPUs. Must divide attention head count. |
| `--max-model-len N` | model config | Hard cap on context length. Larger = more KV cache memory per request. |
| `--gpu-memory-utilization F` | 0.90 | Fraction of each GPU's memory vLLM may use. Lower if you share GPUs. |
| `--max-num-seqs N` | 256 | Max concurrent sequences in a batch. |
| `--dtype` | auto | `auto` picks `bfloat16` on H100. Override only if you know why. |
| `--quantization` | none | `awq`, `gptq`, `fp8`, etc. Must match the checkpoint. |
| `--enable-prefix-caching` | off | Reuses KV cache across requests with shared prefixes. Big win for chat. |
| `--api-key KEY` | none | Require clients to send `Authorization: Bearer KEY`. |

Full reference: `docker run --rm vllm-slim:0.21.0 --help`.

---

## Troubleshooting

**Container exits immediately with "NVIDIA driver not found"**
Install / fix the NVIDIA Container Toolkit on the host. Verify with `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi`.

**`CUDA error: no kernel image is available for execution on the device`**
You're trying to run the image on a non-H100 GPU. This build only contains `sm_90a` kernels — rebuild with a wider `TORCH_CUDA_ARCH_LIST` and `FLASHINFER_CUDA_ARCH_LIST` if you need to support other architectures.

**`OSError: Can't load tokenizer` or HTTP timeout at startup**
The model directory is missing tokenizer files, or `HF_HUB_OFFLINE=1` isn't taking effect (check the image is the one you just built). Re-stage the full model directory — partial copies are a common cause.

**Loads, but first request hangs forever**
On multi-GPU runs, this is almost always missing `--ipc=host` or insufficient `--shm-size` — NCCL can't establish the shared-memory rendezvous between worker processes.

**`AssertionError: <N> is not divisible by tensor_parallel_size`**
Your `--tensor-parallel-size` doesn't divide the model's attention head count (Llama-3-70B has 64 heads, so TP must be in {1, 2, 4, 8, 16, 32, 64}).

**Out-of-memory at startup**
Lower `--gpu-memory-utilization` (e.g. 0.85) or `--max-model-len`. Memory budget = weights + KV cache + activation buffers; KV cache scales with `max_model_len * max_num_seqs`.

**Container can't reach huggingface.co (expected!)**
That's by design — `HF_HUB_OFFLINE=1` and friends are set. If you see startup errors mentioning the Hub, the model directory is incomplete; do not "fix" by allowing egress.

---

## Updating models

Stage the new model directory alongside the old one, then restart the container with the new mount:

```bash
docker stop vllm-llama-3-70b && docker rm vllm-llama-3-70b
# re-run docker run ... with the new -v ... path
```

No image rebuild is required to swap models; the image is model-agnostic.

---

## Updating the image

When vLLM, PyTorch, or FlashInfer releases new versions:

```bash
docker build \
  --build-arg VLLM_REF=v0.22.0 \
  --build-arg TORCH_VERSION=2.12.0 \
  --build-arg FLASHINFER_VERSION=0.7.0 \
  -t vllm-slim:0.22.0 .
```

Verify the three pin against each other before bumping — vLLM pins exact `torch` and `flashinfer` versions in `requirements/cuda.txt`. Use mismatched versions and the build will silently install a torch from PyPI that doesn't match your CUDA wheels.
