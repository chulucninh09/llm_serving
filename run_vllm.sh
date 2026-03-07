#!/bin/bash

export CUDA_DISABLE_PERF_BOOST=1

# Source the argument parser
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/parse_args.sh"

# Read arguments from vllm_args.sh
ARGS=()
parse_args_file "vllm_args.sh" ARGS


PYTHONHASHSEED=1 \
LMCACHE_CONFIG_FILE=$PWD/lmcache.yaml \
LMCACHE_LOG_LEVEL=WARNING \
NCCL_P2P_DISABLE=1 \
uv run vllm serve \
    "${ARGS[@]}"