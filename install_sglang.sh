# uv pip install flashinfer-python
# uv pip install flashinfer-cubin --index-url https://flashinfer.ai/whl
# uv pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu130

uv pip install transformers flashinfer-python torch sglang sglang-kernel "cuda-tile==1.6.0rc5" \
    --torch-backend=cu130 \
    --extra-index-url https://docs.sglang.io/whl/cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.nvidia.com/ \
    --upgrade --prerelease=allow --index-strategy unsafe-best-match