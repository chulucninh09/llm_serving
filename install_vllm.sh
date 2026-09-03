set -e

# Fetch the latest vLLM nightly version from the wheel index
echo "Fetching latest vLLM nightly version..."
VLLM_VERSION=$(curl -sL https://wheels.vllm.ai/nightly/cu130/vllm/ \
    | grep -oP '(?<=>vllm-)[^<]+' \
    | head -1 \
    | sed 's/-cp38.*//')

if [ -z "$VLLM_VERSION" ]; then
    echo "ERROR: Could not determine vLLM nightly version from https://wheels.vllm.ai/nightly/cu130/vllm/"
    exit 1
fi
echo "Detected vLLM version: $VLLM_VERSION"

# Install other dependencies
# uv pip install lmcache xxhash torch-c-dlpack-ext --upgrade

# Install vLLM nightly build from wheels.vllm.ai first
# Nightly index must come first to prioritize nightly over stable releases
# uv pip install vllm transformers lmcache instanttensor xxhash nixl vllm-router runai-model-streamer \
#     --index-url https://wheels.vllm.ai/nightly/cu130/ \
#     --force-reinstall \
#     --torch-backend=cu130 \
#     --extra-index-url https://download.pytorch.org/whl/cu130 \
#     --extra-index-url https://pypi.org/simple \
#     --upgrade --prerelease=allow --index-strategy unsafe-best-match

# Pin cuda-tile to 1.6.0rc5: 1.6.0rc6 lacks a cp313-manylinux2014_x86_64 wheel,
# and the system runs Python 3.13 on x86_64 Linux.
uv pip install "vllm==$VLLM_VERSION" flashinfer-python transformers torchcodec lmcache instanttensor \
    xxhash nixl vllm-router runai-model-streamer "cuda-tile==1.6.0rc5" \
    --force-reinstall \
    --torch-backend=cu130 \
    --extra-index-url https://wheels.vllm.ai/nightly/cu130/ \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.nvidia.com/ \
    --upgrade --prerelease=allow --index-strategy unsafe-best-match

uv pip uninstall cupy-cuda12x

# uv pip install lmcache instanttensor xxhash nixl vllm-router \
#     --force-reinstall \
#     --upgrade --prerelease=allow --index-strategy unsafe-best-match

# OffloadingConnector + prefix caching on hybrid (Mamba) models (optional on vLLM >=0.23)
# bash "$(dirname "$0")/apply_vllm_offload_scheduler_patch.sh" || true
# bash "$(dirname "$0")/apply_vllm_offload_worker_patch.sh"

# Default thinking_token_budget via --reasoning-config
bash "$(dirname "$0")/apply_vllm_thinking_budget_patch.sh"

# Recognize Qwen3-based DSpark drafts (arch=DSparkDraftModel, model_type=qwen3)
# as Qwen3DSparkModel instead of routing them to the DeepSeek-V4 class.
# bash "$(dirname "$0")/apply_vllm_dspark_arch_patch.sh"

# DSpark + hybrid-Mamba prefix-cache reuse: keep Mamba groups out of the
# "flag all groups as draft" fallback (both the coordinator and the offloading
# connector), restoring align-mode Mamba prefix reuse.
# bash "$(dirname "$0")/apply_vllm_dspark_mamba_prefix_patch.sh"