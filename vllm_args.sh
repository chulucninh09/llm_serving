--model RedHatAI/Qwen3.8-27B-INT4
# --model RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead
--load-format instanttensor
--mm-encoder-tp-mode data
--mm-processor-cache-type lru
--reasoning-parser qwen3
--chat-template templates/qwen3.8-enhanced.jinja
--tool-call-parser qwen3_coder
--reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>", "thinking_token_budget": 4096}'
--default-chat-template-kwargs '{"preserve_thinking":true,"enable_thinking":true}'
--mamba-cache-mode all

--speculative_config.method dflash
--speculative_config.model z-lab/Qwen3.8-27B-DFlash2
--speculative_config.num_speculative_tokens 7
--speculative_config.draft_sample_method probabilistic

# --speculative_config.method dspark
# --speculative_config.num_speculative_tokens 7
# --speculative_config.draft_sample_method probabilistic
# --speculative_config.model RadixArk/Qwen3.8-27B-DSpark
# --speculative_config.model Doopeworld/Qwen3.8-27B-DSpark-vLLM

# --speculative_config.method mtp
# --speculative_config.num_speculative_tokens 1

--dtype bfloat16
--mamba-cache-dtype bfloat16
# --kv-cache-dtype bfloat16
--kv-cache-dtype int8_per_token_head
# --kv-cache-dtype float16
# --attention-backend TRITON_ATTN

# Common config
-tp 4
--numa-bind
--max-model-len auto
# --performance-mode throughput
--performance-mode interactivity
--enable-prefix-caching
# --disable-custom-all-reduce
# --prefix-caching-hash-algo xxhash
# --kv-transfer-config '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both", "kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector", "kv_connector_extra_config": {"lmcache.mp.host": "localhost", "lmcache.mp.port": 5555}}'
--kv-transfer-config '{"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"spec_name": "TieringOffloadingSpec", "cpu_bytes_to_use": 21474836480, "offload_prompt_only": false, "eviction_policy": "lru", "secondary_tiers": [{"type": "fs","root_dir": "/mnt/llm-data/kv-cache/vllm","n_read_threads": 16,"n_write_threads": 8}]}}'
# --kv-offloading-size 10
# --kv-offloading-backend native
--host 0.0.0.0
--port 8000
--enable-chunked-prefill
# --skip-mm-profiling
# https://docs.lmcache.ai/recipes/qwen3_5.html
--long-prefill-token-threshold 1024
--max-num-batched-tokens 6144
--max-cudagraph-capture-size 128
--cudagraph-capture-sizes 8 16 24 32 40 48 56 64 80 96 112 128
# --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [512, 1024, 2048, 4096], "cudagraph_num_of_warmups": 1, "compile_sizes": [512, 1024, 2048, 4096, 8192]}'
--max-num-seqs 12
--gpu-memory-utilization 0.975
--enable-auto-tool-choice
--served-model-name kCode
--async-scheduling
--trust-remote-code
--disable-access-log-for-endpoints /health,/metrics,/ping
--api-key $LLM_API_KEY
# --api-server-count 1
