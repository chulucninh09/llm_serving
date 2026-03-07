# Install other dependencies
uv pip install lmcache xxhash torch-c-dlpack-ext

# Install vLLM nightly build from wheels.vllm.ai first
# Nightly index must come first to prioritize nightly over stable releases
uv pip install vllm --torch-backend=cu130 \
    --extra-index-url https://wheels.vllm.ai/nightly/cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --upgrade --prerelease=allow --index-strategy unsafe-best-match

