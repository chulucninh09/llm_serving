-m /mnt/llm-data/huggingface/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf
--n-cpu-moe 30

# Common config
--fit-ctx 75000
-ngl 99
--no-mmap
--ctx-checkpoints 16
-cram 16384
-kvu
-np 2
--host 0.0.0.0
--port 8000
--jinja
--no-context-shift
-fa on
-b 8192
-ub 2048
--slot-prompt-similarity 0.9
--slot-save-path .slots/