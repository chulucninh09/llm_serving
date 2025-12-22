-m /mnt/llm-data/huggingface/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf
--temp 0.7
--top-p 0.8
--top-k 20
--repeat-penalty 1.05

# -m /mnt/llm-data/huggingface/QwenLong-L1.5-30B-A3B.Q8_0.gguf
# --temp 0.7
# --top-p 0.95

# Common config
--fit-ctx 75000
--fit-target 1536
-ngl 99
--no-mmap
--ctx-checkpoints 16
-cram 16384
-kvu
-np 1
--host 0.0.0.0
--port 8000
--jinja
--no-context-shift
-fa on
-b 4096
-ub 1024
--slot-prompt-similarity 0.9
--slot-save-path .slots/
--numa numactl
-t 16
--threads-http 4