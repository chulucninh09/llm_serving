# -m /mnt/llm-data/huggingface/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf

-m /mnt/llm-data/huggingface/QwenLong-L1.5-30B-A3B.Q8_0.gguf
--temp 0.7
--top-p 0.95

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
--numa distribute
-t 16
--threads-http 4