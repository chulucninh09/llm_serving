#!/bin/bash

export CUDA_DISABLE_PERF_BOOST=1
source .env

# Source the argument parser
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/parse_args.sh"

# Read arguments from vllm_args.sh
ARGS=()
parse_args_file "sglang_args.sh" ARGS


# export NCCL_CUMEM_ENABLE=0                  # Critical: forces cudaMalloc for BAR1 P2P compatibility
# export NCCL_P2P_LEVEL=LOC                # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# export NCCL_SHM_DISABLE=0     # keep SHM enabled (this is the fast path)
# export NCCL_ALGO=Ring
# export PYTORCH_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'  # Required for stability with 1G hugepages
# export PYTHONHASHSEED=0

# FLASHINFER_DISABLE_VERSION_CHECK=1 \
# TORCH_COMPILE_SKIP_OPS=causal_conv1d_update \
uv run -m sglang.launch_server "${ARGS[@]}"