# Tool call fixes
# https://forums.developer.nvidia.com/t/qwen3-5-tool-calling-finally-fixed-possibly/366451
# https://github.com/allanchan339/vLLM-Qwen3.5-27B
# https://github.com/nleve/vllm/blob/qwen3-implicit-tool-boundary/vllm/reasoning/qwen3_reasoning_parser.py
# https://github.com/vllm-project/vllm/issues/39056

# --model Qwen/Qwen3.6-27B-FP8
# --model Intel/Qwen3.6-27B-int4-AutoRound
# --model btbtyler09/Qwen3.6-27B-GPTQ-4bit
# --model webhie/Qwen3.6-27B-int4-AutoRound-Code
# --model Lorbus/Qwen3.6-27B-int4-AutoRound
# --model bottlecapai/ThinkingCap-Qwen3.6-27B-FP8
# --model josefprusa/ThinkingCap-Qwen3.6-27B-int4-AutoRound-v1
# --model cyankiwi/ThinkingCap-Qwen3.6-27B-AWQ-INT4
# --model Avesed/Qwen3.6-27B-INT4-W4A16
# --model unsloth/Qwen3.8-27B-NVFP4
# --model nicosuter/Qwen3.8-27B-AWQ
# --model biMEMO/Qwen3.8-27B-int4-AutoRound
# --model cyankiwi/Qwen3.8-27B-AWQ-INT4
--model philbert440/Qwen3.8-27B-W4A16-AWQ
# --model Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8
# --model Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound
# --attention-backend FLASH_ATTN
--load-format instanttensor
--mm-encoder-tp-mode data
--mm-processor-cache-type lru
--reasoning-parser qwen3
--chat-template templates/qwen3.8-enhanced.jinja2
--tool-call-parser qwen3_coder
# --override-generation-config '{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.0, "repetition_penalty": 1.0}'
--reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>", "thinking_token_budget": 16384}'
--default-chat-template-kwargs '{"preserve_thinking":true,"enable_thinking":true}'

# --speculative_config.method dflash
# --speculative_config.model z-lab/Qwen3.8-27B-DFlash2
# --speculative_config.num_speculative_tokens 7
# --speculative_config.draft_sample_method probabilistic

# --speculative_config.method dspark
# --speculative_config.draft_sample_method probabilistic
# --speculative_config.rejection_sample_method block

# --speculative_config.method mtp
# --speculative_config.num_speculative_tokens 1
--dtype float16
--kv-cache-dtype fp8_e4m3
# --kv-cache-dtype int8_per_token_head
# --kv-cache-dtype float16

# # # --model deepreinforce-ai/Ornith-1.0-35B-FP8
# # # --model Qwen/Qwen3.6-35B-A3B-FP8
# # # --model Intel/Qwen3.5-122B-A10B-int4-AutoRound
# --model Avesed/Qwen3.6-35B-A3B-INT4-W4A16
# # --enable-expert-parallel
# # # --enable-eplb
# # # --eplb-config '{"num_redundant_experts":4}'
# --mm-encoder-tp-mode data
# --mm-processor-cache-type shm
# --reasoning-parser qwen3
# --chat-template templates/qwen3.6-enhanced.jinja2
# # # --chat-template templates/qwen3.5-enhanced.jinja2
# --tool-call-parser qwen3_coder
# # --load-format instanttensor
# # # --override-generation-config '{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.00, "presence_penalty": 1.5, "repetition_penalty": 1.0}'
# --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>", "thinking_token_budget": 4096}'
# --default-chat-template-kwargs '{"preserve_thinking":true}'
# --speculative_config.num_speculative_tokens 7
# --speculative_config.model RedHatAI/Qwen3.6-35B-A3B-speculator.dspark
# --speculative_config.method dspark
# --speculative_config.draft_sample_method probabilistic
# --speculative_config.rejection_sample_method block
# # # --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.6-35B-A3B-DFlash", "num_speculative_tokens": 11}'
# --dtype float16
# # # --kv-cache-dtype fp8
# # # --kv-cache-dtype int8_per_token_head
# # # # --kv-cache-dtype turboquant_4bit_nc
# # # --language-model-only

# # --model cyankiwi/Muse-Glimmer-30B-AWQ-INT4
# # --max-model-len 256000
# # # --reasoning-parser muse_glimmer
# # # --tool-call-parser muse_glimmer
# # --load-format instanttensor
# # --default-chat-template-kwargs '{"preserve_thinking": true, "enable_thinking": true}'
# # # --speculative-config '{"method": "dflash", "model": "meta-models/Muse-Glimmer-30B-assistant", "num_speculative_tokens": 15}'
# # --dtype float16


# --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
# --quantization modelopt_fp4
# --mamba-cache-mode align
# --moe-backend humming
# --linear-backend humming
# --mamba-backend flashinfer

# --reasoning-parser nemotron_v3
# --tool-call-parser qwen3_coder
# --speculative_config.num_speculative_tokens 3
# --speculative_config.model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark
# --speculative_config.method dspark

# --kv-cache-dtype float16
# --dtype float16
# --mamba-ssm-cache-dtype float16

# --enable-mamba-cache-stochastic-rounding
# --mamba-cache-philox-rounds 5

# Common config
-tp 4
# -pp 2
# -dp 2
--numa-bind
--max-model-len auto
# --performance-mode throughput
--performance-mode interactivity
--enable-prefix-caching
# --disable-custom-all-reduce
# --prefix-caching-hash-algo xxhash
# --kv-transfer-config '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both", "kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector", "kv_connector_extra_config": {"lmcache.mp.host": "localhost", "lmcache.mp.port": 5555}}'
--kv-transfer-config '{"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"spec_name": "TieringOffloadingSpec", "cpu_bytes_to_use": 8917287424, "offload_prompt_only": false, "eviction_policy": "lru", "secondary_tiers": [{"type": "fs","root_dir": "/mnt/llm-data/kv-cache","n_read_threads": 16,"n_write_threads": 8}]}}'
# --kv-transfer-config '{"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"spec_name": "TieringOffloadingSpec", "cpu_bytes_to_use": 27917287424, "block_size": 1568, "offload_prompt_only": false, "eviction_policy": "lru"}}'
# --kv-offloading-size 10
# --kv-offloading-backend native
--host 0.0.0.0
--port 8000
--enable-chunked-prefill
# --skip-mm-profiling
# https://docs.lmcache.ai/recipes/qwen3_5.html
--max-num-batched-tokens 4096
--max-cudagraph-capture-size 32
--max-num-seqs 10
--gpu-memory-utilization 0.93
--enable-auto-tool-choice
--served-model-name kCode
--async-scheduling
--trust-remote-code
--disable-access-log-for-endpoints /health,/metrics,/ping
--api-key $VLLM_API_KEY
# --api-server-count 1
