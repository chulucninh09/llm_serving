#!/bin/bash

export CUDA_DISABLE_PERF_BOOST=1
# Read arguments from llama_args.sh, skipping comments and empty lines
ARGS=()
while IFS= read -r line; do
    # Skip comment lines (starting with #) and empty lines
    if [[ ! "$line" =~ ^[[:space:]]*# ]] && [[ -n "${line// }" ]]; then
        # Split the line into arguments and add them to ARGS array
        read -ra LINE_ARGS <<< "$line"
        ARGS+=("${LINE_ARGS[@]}")
    fi
done < llama_args.sh

# Run llama-server with the parsed arguments
taskset -c 0-7,10-15 ./llama.cpp/build/bin/llama-server "${ARGS[@]}"

