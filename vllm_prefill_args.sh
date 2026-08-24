# Tool call fixes
# https://forums.developer.nvidia.com/t/qwen3-5-tool-calling-finally-fixed-possibly/366451
# https://github.com/allanchan339/vLLM-Qwen3.5-27B
# https://github.com/nleve/vllm/blob/qwen3-implicit-tool-boundary/vllm/reasoning/qwen3_reasoning_parser.py
# https://github.com/vllm-project/vllm/issues/39056


--model Lorbus/Qwen3.6-27B-int4-AutoRound
--mm-encoder-tp-mode data
--mm-processor-cache-type shm
--reasoning-parser qwen3
--chat-template templates/qwen3.6-enhanced.jinja2
--tool-call-parser qwen3_coder
--override-generation-config '{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.0, "repetition_penalty": 1.0}'
--reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>", "thinking_token_budget": 3072}'
--default-chat-template-kwargs '{"preserve_thinking":true}'
# --speculative-config '{"method": "mtp", "num_speculative_tokens": 5}'
# --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.6-27B-DFlash", "num_speculative_tokens": 15}'
--dtype float16
--kv-cache-dtype fp8

# Common config
# --attention-backend FLASHINFER
--quantization humming
# MTP + instanttensor OOMs on this model; keep MTP enabled and use normal load.
# --load-format instanttensor
--tensor-parallel-size 2
--numa-bind
--max-model-len auto
--performance-mode throughput
--enable-prefix-caching
--disable-custom-all-reduce
# --prefix-caching-hash-algo xxhash
--kv-offloading-size 6
# --kv-offloading-backend native


--max-model-len auto
--host 0.0.0.0
--port 8100
--enable-chunked-prefill
--max-num-batched-tokens 8K
--max-cudagraph-capture-size 8
--max-num-seqs 8
--gpu-memory-utilization 0.95
--enable-auto-tool-choice
--served-model-name kCode
--async-scheduling
--trust-remote-code
--disable-access-log-for-endpoints /health,/metrics,/ping
--kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"kv_lease_duration": 180, "kv_recompute_threshold": 0}}'