#!/bin/bash

export CUDA_DISABLE_PERF_BOOST=1
# Read arguments from vllm_args.sh, skipping comments and empty lines
ARGS=()
while IFS= read -r line; do
    # Skip comment lines (starting with #) and empty lines
    if [[ ! "$line" =~ ^[[:space:]]*# ]] && [[ -n "${line// }" ]]; then
        # Split the line into arguments and add them to ARGS array
        read -ra LINE_ARGS <<< "$line"
        ARGS+=("${LINE_ARGS[@]}")
    fi
done < vllm_args_embedding.sh

# Debug: Print arguments being passed (comment out in production)
# echo "Arguments to pass to Docker:"
# printf "'%s'\n" "${ARGS[@]}"

# Run vllm with the parsed arguments
# Use --entrypoint to ensure arguments are passed correctly
OMP_NUM_THREADS=8
PYTHONHASHSEED=0
uv run vllm serve \
    --compilation-config '{"cache_dir": "/mnt/llm-data/.cache/vllm"}' \
    "${ARGS[@]}"