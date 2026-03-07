#!/bin/bash

# Source the argument parser
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/parse_args.sh"

# Read arguments from vllm_args.sh
ARGS=()
parse_args_file "llama_args.sh" ARGS

export GGML_CUDA_GRAPH_OPT=1
# Run llama-server with the parsed arguments
./llama.cpp/build/bin/llama-server "${ARGS[@]}"