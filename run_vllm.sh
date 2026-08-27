#!/bin/bash

export CUDA_DISABLE_PERF_BOOST=1
source .env

# KV offload pins ~kv_offloading_size GiB of CPU RAM (split across TP ranks).
# Default memlock (~8 GiB) is too small for TP=2 with --kv-offloading-size 10.
ulimit -l unlimited 2>/dev/null || true

# Source the argument parser
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/parse_args.sh"

# Read arguments from vllm_args.sh (override with VLLM_ARGS_FILE for sweeps)
ARGS=()
parse_args_file "${VLLM_ARGS_FILE:-vllm_args.sh}" ARGS
unset VLLM_ARGS_FILE

# ---- Safe, Speed‑Focused Env Vars ----
# export NCCL_CUMEM_ENABLE=0                  # Critical: forces cudaMalloc for BAR1 P2P compatibility
export NCCL_P2P_LEVEL=LOC                # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
export NCCL_SHM_DISABLE=0     # keep SHM enabled (this is the fast path)
export NCCL_ALGO=Ring
export NCCL_PROTO=LL,Simple
# export PYTORCH_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'  # Required for stability with 1G hugepages
export VLLM_ENABLE_CUDAGRAPH_GC=1
export PYTHONHASHSEED=0
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_USE_FLASHINFER_SAMPLER=1        # FlashInfer sampler, faster than vLLM default on Ampere
# export VLLM_ALLREDUCE_USE_FLASHINFER=1
# export VLLM_LOGGING_CONFIG_PATH=vllm_logging_config.json

# ---- NCCL Tuning for SYS/PCIe Topology ----
export CUDA_DEVICE_MAX_CONNECTIONS=32        # Serialized streams, lower PCIe contention on cross-socket UPI

# ---- vLLM Stability (Driver‑Dependent) ----
# export VLLM_WORKER_MULTIPROC_METHOD=spawn    # NEW

# ---- FP8 & Memory ----
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
# export VLLM_TEST_FORCE_FP8_MARLIN=1
export VLLM_SKIP_MODEL_NAME_VALIDATION=1

# Clean stale FlashInfer cache
# rm -rf ~/.cache/
# rm vllm_server.log

# VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"int8","group_size":0}' \
# VLLM_HUMMING_ONLINE_QUANT_CONFIG='{"dtype":"int8","group_size":0}' \
# VLLM_HUMMING_INPUT_QUANT_CONFIG='{"a_dtype":"int8"}' \
# VLLM_SSM_CONV_STATE_LAYOUT=DS \
# NCCL_P2P_LEVEL=PIX \
# VLLM_HUMMING_USE_F16_ACCUM=1 \
# VLLM_FLASHINFER_ALLREDUCE_BACKEND="trtllm" \
# VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1 \
# VLLM_USE_V2_MODEL_RUNNER=1 \
# FLASHINFER_DISABLE_VERSION_CHECK=1 \
# Leftover vLLM SHM (KV offload mmap, psm_*, sem.*) fills /dev/shm across restarts.
# vLLM then fails with: Insufficient space in /dev/shm: N MiB required, M MiB free.
rm -rf /dev/shm/* /dev/shm/.[!.]* /dev/shm/..?* 2>/dev/null || true

# CUDA_VISIBLE_DEVICES=2,3 \
# VLLM_MARLIN_INPUT_DTYPE=int8 \
uv run vllm serve "${ARGS[@]}"



# docker run --gpus all -p 8000:8000 --rm \
#     --privileged \
#     -e VLLM_ENFORCE_STRICT_TOOL_CALLING=1 \
#     -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
#     -e VLLM_USE_FLASHINFER_SAMPLER=1 \
#     -e VLLM_ENABLE_CUDAGRAPH_GC=1 \
#     -e PYTHONHASHSEED=0 \
#     -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
#     -e VLLM_SKIP_MODEL_NAME_VALIDATION=1 \
#     -v /mnt/llm-data/huggingface:/root/.cache/huggingface \
#     -v ./templates:/vllm_templates \
#   ghcr.io/avesed/vllm-ampere-optimized:latest \
#         --marlin-input-dtype int8 \
#         "${ARGS[@]}"
