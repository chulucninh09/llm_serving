#!/bin/bash

export CUDA_DISABLE_PERF_BOOST=1
source .env

# Source the argument parser
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/parse_args.sh"

# Read arguments from vllm_args.sh
ARGS=()
parse_args_file "sglang_args.sh" ARGS


export NCCL_SHM_DISABLE=0     # keep SHM enabled (this is the fast path)
export NCCL_ALGO=Ring
export NCCL_PROTO=LL,Simple
export PYTHONHASHSEED=0

# ---- NCCL Tuning for SYS/PCIe Topology ----
export CUDA_DEVICE_MAX_CONNECTIONS=32

export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/mnt/llm-data/kv-cache/sglang

rm -rf /dev/shm/* /dev/shm/.[!.]* /dev/shm/..?* 2>/dev/null || true

numactl --interleave=0,1 uv run -m sglang.launch_server "${ARGS[@]}"