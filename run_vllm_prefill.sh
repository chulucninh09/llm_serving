#!/bin/bash

export CUDA_DISABLE_PERF_BOOST=1
source .env

# Source the argument parser
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/parse_args.sh"

# Read arguments from vllm_args.sh
ARGS=()
parse_args_file "vllm_prefill_args.sh" ARGS

# ---- Safe, Speed‑Focused Env Vars ----
export NCCL_CUMEM_ENABLE=0                  # Critical: forces cudaMalloc for BAR1 P2P compatibility
# export NCCL_P2P_LEVEL=LOC                # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
export NCCL_SHM_DISABLE=0     # keep SHM enabled (this is the fast path)
export NCCL_ALGO=Ring
export PYTORCH_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'  # Required for stability with 1G hugepages
export VLLM_ENABLE_CUDAGRAPH_GC=1
export PYTHONHASHSEED=1
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_USE_FLASHINFER_SAMPLER=1        # FlashInfer sampler, faster than vLLM default on Ampere
# export VLLM_LOGGING_CONFIG_PATH=vllm_logging_config.json

# ---- NCCL Tuning for SYS/PCIe Topology ----
export CUDA_DEVICE_MAX_CONNECTIONS=4        # Serialized streams, lower PCIe contention on cross-socket UPI

# ---- vLLM Stability (Driver‑Dependent) ----
export VLLM_WORKER_MULTIPROC_METHOD=spawn    # NEW

# ---- FP8 & Memory ----
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
# export VLLM_TEST_FORCE_FP8_MARLIN=1
export VLLM_SKIP_MODEL_NAME_VALIDATION=1

# Clean stale FlashInfer cache
# rm -rf ~/.cache/
# rm vllm_server.log

# VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"int8","group_size":0}' \
# VLLM_USE_V2_MODEL_RUNNER=1 \
# ----- NIXL P/D (prefill ↔ decode) -----
# If EngineCore fails with: "UCX CUDA support was not found" or NIXL_ERR_BACKEND:
#   - Install UCX *with* CUDA UCT (system packages vary: ucx-cuda, libuct-cuda,
#     or use NVIDIA HPC-X / a container image that ships cuda_ipc).
#   - Do NOT set UCX_TLS=sm,self — that excludes GPU IPC needed for kv_buffer_device=cuda.
# Optional hardening (try if transfers are flaky):
# export UCX_MEMTYPE_CACHE=n
# export UCX_RCACHE_MAX_UNRELEASED=1024
# export UCX_RC_TIMEOUT=100ms
# export UCX_RC_TX_QUEUE_SLOTS=16384

# VLLM_ENFORCE_STRICT_TOOL_CALLING=1 \
# VLLM_SSM_CONV_STATE_LAYOUT=DS \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
uv run vllm serve \
    "${ARGS[@]}"