# -m /mnt/llm-data/huggingface/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf
# -m /mnt/llm-data/huggingface/hub/Qwen3-VL-30B-A3B-Thinking-UD-Q6_K_XL.gguf
# -m /mnt/llm-data/huggingface/hub/models--unsloth--Qwen3-Coder-30B-A3B-Instruct-1M-GGUF/snapshots/4ea9030716b3dc671dc0aafaedfb7c570babb60f/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q6_K_XL.gguf
# --temp 0.7
# --top-p 0.8
# --top-k 20
# --repeat-penalty 1.05
# --min-p 0.00
# -ngl 999

# -m /mnt/llm-data/huggingface/Devstral-Small-2-24B-Instruct-2512-UD-Q6_K_XL.gguf
# --mmproj unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF/mmproj-F16.gguf
# --temp 0.15
# --repeat-penalty 1.1
# --min-p 0.01
# -ngl 999

# -m /mnt/llm-data/huggingface/QwenLong-L1.5-30B-A3B.Q8_0.gguf
# --temp 0.7
# --top-p 0.95

# -m /mnt/llm-data/huggingface/Nemotron-3-Nano-30B-A3B-UD-Q6_K_XL.gguf
# --temp 0.6
# --top-p 0.95
# --repeat-penalty 1.1
# --min-p 0.01

-m /mnt/llm-data/huggingface/hub/Qwen3-VL-30B-A3B-Thinking-UD-Q6_K_XL.gguf
--temp 1
--top-p 0.95
--presence-penalty 0.0
--top-k 20
--min-p 0.01
-ngl 999

# Common config
-c 85000
--no-mmap
--color
-sm graph
# --mlock
--run-time-repack
-sas
-mqkv
-cram 16384
--host 0.0.0.0
--port 8000
--jinja
-fa on
-b 2048
-ub 512
-amb 4
--numa numactl
--alias kCode
--slot-prompt-similarity 0.1
-ns 4
-np 4
-cb
--slot-save-path ./.slots
--no-context-shift