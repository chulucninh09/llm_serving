--model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
--tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct
--chat-template templates/qwen3coder.jinja2
--tool-call-parser qwen3_coder
--enable-expert-parallel
--max-model-len 90000

# --model mistralai/Devstral-Small-2-24B-Instruct-2512
# --tool-call-parser mistral
# --max-model-len 80000

# --model openai/gpt-oss-20b
# --tool-call-parser openai
# --enable-expert-parallel
# --max-model-len 100000

# Common config
--tensor-parallel-size 2
# --pipeline-parallel-size 2
# --nnodes 2
--enable-prefix-caching
--kv-offloading-size 16
--kv-offloading-backend lmcache
--disable-hybrid-kv-cache-manager
--host 0.0.0.0
--port 8000
--max-num-batched-tokens 1024
--max-cudagraph-capture-size 4
--max-num-seqs 4
--gpu-memory-utilization 0.87
--enable-auto-tool-choice
--served-model-name kCode
--block-size 32
--async-scheduling
--trust-remote-code
