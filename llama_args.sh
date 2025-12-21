-m /mnt/llm-data/huggingface/models/meta-llama/Llama-3.1-8B-Instruct/

# Common config
--fit-ctx 75000
-ngl 99
--swe-
--no-mmap
--ctx-checkpoints 16
-cram 16384
-kvu
-np 2
--host "0.0.0.0"
--port 8000
--jinja
--no-context-shift
-fa on
-b 8192
-u 2048
--slot-prompt-similarity 0.9