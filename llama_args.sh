# -m /mnt/llm-data/huggingface/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf
-m /mnt/llm-data/huggingface/Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL.gguf
--temp 0.7
--top-p 0.8
--top-k 20
--repeat-penalty 1.05
--min-p 0.01

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
--fit-ctx 4096
--fit-target 1536
-c 95000
--no-mmap
# --mlock
# --mmap
--ctx-checkpoints 20
-cram 32768
-kvu
-np 1
--host 0.0.0.0
--port 8000
--jinja
--no-context-shift
-fa on
-b 4096
-ub 1024
--slot-prompt-similarity 0.1
--numa numactl
-t 16
--threads-http 4
--cache-reuse 256