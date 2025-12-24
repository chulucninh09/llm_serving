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
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./templates:/templates \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:nightly-f1c2c20136cca6ea8798a64855eaf52ee9a42210 \
    "${ARGS[@]}"