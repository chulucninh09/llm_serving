# uv pip install flashinfer-python
# uv pip install flashinfer-cubin --index-url https://flashinfer.ai/whl
# uv pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu130

uv pip install transformers flashinfer-python torch sglang sglang-kernel "cuda-tile==1.6.0rc5" \
    --torch-backend=cu130 \
    --extra-index-url https://docs.sglang.io/whl/cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.nvidia.com/ \
    --upgrade --prerelease=allow --index-strategy unsafe-best-match

# Map Qwen3.5 pre-fusion GDN names (in_proj_a/b/qkv/z) onto fused in_proj_ba/qkvz
# so GPTQ ignore lists skip the 24-wide gating projection (vLLM #34697 analog).
bash "$(dirname "$0")/apply_sglang_qwen3_5_hf_mapper_patch.sh"