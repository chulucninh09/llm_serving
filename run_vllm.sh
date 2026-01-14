#!/bin/bash

export CUDA_DISABLE_PERF_BOOST=1

# Source the argument parser
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/parse_args.sh"

# Read arguments from vllm_args.sh
ARGS=()
parse_args_file "vllm_args.sh" ARGS

# Debug: Print arguments being passed (comment out in production)
# echo "Arguments to pass to Docker:"
# printf "'%s'\n" "${ARGS[@]}"

# Run vllm with the parsed arguments
# Use --entrypoint to ensure arguments are passed correctly
# docker run --runtime nvidia --gpus all \
#     --rm \
#     -v /mnt/llm-data/huggingface:/root/.cache/huggingface \
#     -v ./templates:/templates \
#     -v /mnt/llm-data/.cache/vllm:/root/.cache/vllm \
#     -v /mnt/llm-data/.cache/torch:/root/.cache/torch \
#     -v /mnt/llm-data/tmp:/tmp \
#     -v /mnt/llm-data/root/.triton:/root/.triton \
#     --env "HF_TOKEN=$HF_TOKEN" \
#     --env "PYTHONHASHSEED=0" \
#     -e OMP_NUM_THREADS=16 \
#     --network host \
#     --ipc=host \
#     --entrypoint vllm \
#     vllm/vllm-openai:nightly-96142f209453a381fcaf9d9d010bbf8711119a77 \
#     serve \
#     "${ARGS[@]}"

PYTHONHASHSEED=1 \
LMCACHE_CONFIG_FILE=$PWD/lmcache.yaml \
VLLM_SKIP_P2P_CHECK=1 \
LMCACHE_LOG_LEVEL=WARNING \
uv run vllm serve \
    "${ARGS[@]}"
    # Note: --compilation-config disabled when using EAGLE3 with expert parallelism
    # due to incompatibility with torch.compile. Uncomment below if not using EAGLE3:
    # --compilation-config '{"cache_dir": "/mnt/llm-data/.cache/vllm_llm"}'
    # --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1Dynamic","kv_role":"kv_both","kv_connector_module_path":"lmcache.integration.vllm.lmcache_connector_v1"}' \