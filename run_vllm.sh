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
done < vllm_args.sh

# Debug: Print arguments being passed (comment out in production)
# echo "Arguments to pass to Docker:"
# printf "'%s'\n" "${ARGS[@]}"

# Run vllm with the parsed arguments
# Use --entrypoint to ensure arguments are passed correctly
docker run --runtime nvidia --gpus all \
    --rm \
    -v /mnt/llm-data/huggingface:/root/.cache/huggingface \
    -v ./templates:/templates \
    -v /root/.cache/vllm:/root/.cache/vllm \
    -v /root/.cache/torch:/root/.cache/torch \
    -v /tmp:/tmp \
    -v /root/.triton:/root/.triton \
    --env "HF_TOKEN=$HF_TOKEN" \
    -e OMP_NUM_THREADS=16 \
    --network host \
    --ipc=host \
    --entrypoint vllm \
    vllm/vllm-openai:v0.13.0 \
    serve \
    "${ARGS[@]}"