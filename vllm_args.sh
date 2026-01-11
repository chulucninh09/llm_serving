--model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
--tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct
--chat-template templates/qwen3coder.jinja2
--tool-call-parser qwen3_coder
--enable-expert-parallel
--max-model-len 85000

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
--attention-backend FLASHINFER
--tensor-parallel-size 2
# --pipeline-parallel-size 2
# --nnodes 2
--enable-prefix-caching
--kv-offloading-size 20
--kv-offloading-backend lmcache
--disable-hybrid-kv-cache-manager
--host 0.0.0.0
--port 8000
--max-num-batched-tokens 2048
--max-cudagraph-capture-size 4
--max-num-seqs 4
--enable-chunked-prefill
--long-prefill-token-threshold 4096
--gpu-memory-utilization 0.83
--enable-auto-tool-choice
--served-model-name kCode
--async-scheduling
--trust-remote-code
--prefix-caching-hash-algo xxhash
--disable-custom-all-reduce
--disable-cascade-attn
--block-size 32
--disable-sliding-window