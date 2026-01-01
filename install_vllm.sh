export CUDA_VERSION=130 # or other
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly/cu${CUDA_VERSION}