#!/bin/bash

export CUDA_DISABLE_PERF_BOOST=1
source .env

# Source the argument parser
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/parse_args.sh"

# Read arguments from vllm_args.sh
ARGS=()
parse_args_file "sglang_args.sh" ARGS


export NCCL_ALGO=Ring
export NCCL_PROTO=LL,Simple
# export PYTORCH_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'  # Required for stability with 1G hugepages
export PYTHONHASHSEED=0
# export VLLM_ALLREDUCE_USE_FLASHINFER=1
# export VLLM_LOGGING_CONFIG_PATH=vllm_logging_config.json
export PYTORCH_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'  # Required for stability with 1G hugepages

# ---- NCCL Tuning for SYS/PCIe Topology ----
export CUDA_DEVICE_MAX_CONNECTIONS=32        # Serialized streams, lower PCIe contention on cross-socket UPI

export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/mnt/llm-data/kv-cache/sglang

uv run -m sglang.launch_server "${ARGS[@]}"