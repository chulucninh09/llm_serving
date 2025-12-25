# -m /mnt/llm-data/huggingface/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf
# --model /mnt/llm-data/huggingface/Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL.gguf
--model QuantTrio/Qwen3-Coder-30B-A3B-Instruct-GPTQ-Int8
# --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
# --load-format runai_streamer
# --model /root/.cache/huggingface/Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL.gguf
# --load-format gguf
--tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct
--chat-template /templates/qwen3coder.jinja2
--tool-call-parser qwen3_coder

# --temp 0.7
# --top-p 0.8
# --top-k 20
# --repeat-penalty 1.05
# --min-p 0.01

# -m /mnt/llm-data/huggingface/Qwen3-Qwen3-30B-A3B-Instruct-2507-UD-Q8_K_XL.gguf
# --temp 0.7
# --top-p 0.8
# --top-k 20
# --repeat-penalty 1.1
# --min-p 0.01
# --n-cpu-moe 64

# -m /mnt/llm-data/huggingface/QwenLong-L1.5-30B-A3B.Q8_0.gguf
# --temp 0.7
# --top-p 0.95

# -m /mnt/llm-data/huggingface/Nemotron-3-Nano-30B-A3B-UD-Q6_K_XL.gguf
# --temp 0.6
# --top-p 0.95
# --repeat-penalty 1.1
# --min-p 0.01

# -m /mnt/llm-data/huggingface/Devstral-Small-2-24B-Instruct-2512-UD-Q6_K_XL.gguf
# --temp 0.15
# --min-p 0.01

# Common config
--tensor-parallel-size 2
--enable-prefix-caching
--kv-offloading-size 32
--kv-offloading-backend lmcache
--disable-hybrid-kv-cache-manager
--host 0.0.0.0
--port 8000
--max-model-len 110000
# --async-scheduling
--enable-chunked-prefill
--max-num-batched-tokens 8192
# --enforce-eager
# --distributed-executor-backend mp
--max-num-seqs 3
--gpu-memory-utilization 0.9
# --enable-eplb
--enable-expert-parallel
--enable-auto-tool-choice
--served-model-name kCode
