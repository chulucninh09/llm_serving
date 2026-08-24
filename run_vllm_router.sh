# After vllm instances are up on port 8100 (prefill) and 8200 (decode)
/root/llm_serving/.venv/bin/vllm-router \
  --policy consistent_hash \
  --vllm-pd-disaggregation \
  --prefill http://127.0.0.1:8100 \
  --decode http://127.0.0.1:8200 \
  --host 0.0.0.0 \
  --port 8000 \
  --intra-node-data-parallel-size 1