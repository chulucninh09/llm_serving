#!/bin/bash

# Source the argument parser
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/parse_args.sh"

# Read arguments from ik_llama_args.sh
ARGS=()
parse_args_file "ik_llama_args.sh" ARGS

# Run llama-server with the parsed arguments
./ik_llama/build/bin/llama-server "${ARGS[@]}"