# --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
# --tool-call-parser qwen3_coder
# --enable-expert-parallel
# --max-model-len 100000

--model Qwen/Qwen3.5-35B-A3B-FP8
--tool-call-parser qwen3_coder
--enable-expert-parallel
--language-model-only
--max-model-len 100000
--reasoning-parser qwen3
# --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'

# --model cyankiwi/Qwen3-Coder-Next-AWQ-4bit
# --tool-call-parser qwen3_coder
# --enable-expert-parallel
# --max-model-len 1024
# --cpu-offload-gb 60
# --kv-cache-memory-bytes 4G

# --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
# --tool-call-parser hermes
# --enable-expert-parallel
# --max-model-len 80000
# --limit-mm-per-prompt '{"image": 0, "video": {"count": 0}}'

# --model cyankiwi/QwenLong-L1.5-30B-A3B-AWQ-8bit # Failed mermaid test
# --tool-call-parser hermes
# --enable-expert-parallel
# --max-model-len 80000

# --model btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-4bit
# --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct
# --chat-template templates/qwen3coder.jinja2
# --tool-call-parser qwen3_coder
# --enable-expert-parallel
# --max-model-len 200000

# --model unsloth/GLM-4.7-Flash-FP8-Dynamic
# --tool-call-parser glm47
# --reasoning-parser glm45
# --max-model-len 30000
# --enable-expert-parallel
# --speculative-config.method mtp
# --speculative-config.num_speculative_tokens 1

# --model GadflyII/GLM-4.7-Flash-NVFP4
# --tool-call-parser glm47
# --reasoning-parser glm45
# --max-model-len 30000
# --enable-expert-parallel

# --model /mnt/llm-data/huggingface/hub/models--unsloth--Qwen3-Coder-30B-A3B-Instruct-1M-GGUF/snapshots/4ea9030716b3dc671dc0aafaedfb7c570babb60f/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q6_K_XL.gguf
# --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct
# --chat-template templates/qwen3coder.jinja2
# --tool-call-parser qwen3_coder
# --enable-expert-parallel
# --max-model-len 8192

# --model QuantTrio/Qwen3-Coder-30B-A3B-Instruct-GPTQ-Int8
# --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct
# --chat-template templates/qwen3coder.jinja2
# --tool-call-parser qwen3_coder
# --enable-expert-parallel
# --max-model-len 90000

# --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
# --tool-call-parser hermes
# --enable-expert-parallel
# --max-model-len 90000


# --model Qwen/Qwen3-VL-30B-A3B-Thinking-FP8
# --tool-call-parser hermes
# --reasoning-parser deepseek_r1
# --enable-expert-parallel
# --max-model-len 90000

# --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
# --tool-call-parser qwen3_coder
# --reasoning-parser nano_v3
# --enable-expert-parallel
# --max-model-len 90000
# --dtype float16

# --model mistralai/Devstral-Small-2-24B-Instruct-2512
# --tool-call-parser mistral
# --max-model-len 80000

# --model openai/gpt-oss-20b
# --tool-call-parser openai
# --enable-expert-parallel
# --max-model-len 100000


# --model /mnt/llm-data/huggingface/hub/models--unsloth--Qwen3-Coder-30B-A3B-Instruct-1M-GGUF/snapshots/4ea9030716b3dc671dc0aafaedfb7c570babb60f/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q6_K_XL.gguf
# --tool-call-parser qwen3_coder
# --enable-expert-parallel
# --max-model-len 50000
# --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct

# --model /mnt/llm-data/huggingface/QwenLong-L1.5-30B-A3B.Q8_0.gguf
# --tool-call-parser qwen3_coder
# --enable-expert-parallel
# --max-model-len 50000
# --tokenizer Tongyi-Zhiwen/QwenLong-L1.5-30B-A3B

# Common config
# --attention-backend FLASHINFER
--tensor-parallel-size 2
# --disable-hybrid-kv-cache-manager
--enable-prefix-caching
# --kv-offloading-size 10
# --kv-offloading-backend lmcache
# --prefix-caching-hash-algo xxhash
--host 0.0.0.0
--port 8000
--max-num-batched-tokens 4096
# --max-cudagraph-capture-size 8
--max-num-seqs 8
# --enable-chunked-prefill
# --long-prefill-token-threshold 1024
--gpu-memory-utilization 0.9
--enable-auto-tool-choice
--served-model-name kCode
--async-scheduling
# --trust-remote-code
# --disable-custom-all-reduce
# --disable-cascade-attn
# --block-size 32
# --disable-sliding-window
# --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1Dynamic","kv_role":"kv_both","kv_connector_module_path":"lmcache.integration.vllm.lmcache_connector_v1"}'