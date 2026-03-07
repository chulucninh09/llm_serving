
# -m /mnt/llm-data/huggingface/hub/Qwen3.5-35B-A3B-UD-Q8_K_XL.gguf
# -m /mnt/llm-data/huggingface/hub/Qwen3.5-35B-A3B-UD-Q6_K_XL.gguf
-m /mnt/llm-data/huggingface/hub/Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf
# -m /mnt/llm-data/huggingface/hub/Qwen3-Coder-Next-IQ4_XS.gguf
# -m /mnt/llm-data/huggingface/hub/Qwen3-Coder-Next-UD-Q3_K_XL.gguf
# -ot "\.([6-9]|1[0-9])\.ffn_(gate|up|down)_exps.=CPU"
# -ot "\.([0-9]|3[0-9])\.ffn_(gate|up|down)_exps.=CPU"
# -ts 0.95,1
-np 4
-c 360000
# -m /mnt/llm-data/huggingface/hub/Qwen3-Coder-Next-IQ4_XS.gguf
# -ot "\.(1[0-9])\.ffn_(up|down)_exps.=CPU"
# -ts 1.2,1
# -np 2
# -c 70000
# -m /mnt/llm-data/huggingface/hub/Qwen3-Coder-Next-UD-Q3_K_XL.gguf
# -ts 0.85,1
# -np 2
# -c 130000
# -ctk q8_0
# -ctv q8_0
--temp 0.6
--top-p 0.8
--top-k 20
--presence-penalty 1.5
# --frequency-penalty 1.0
--min-p 0.00
# --reasoning-budget 0
--chat-template-kwargs '{"enable_thinking": false}'
-ngl 999
# -ot "\.([6-9]|1[0-9])\.ffn_(up|down)_exps.=CPU"
# --n-cpu-moe 12
# -ts 1.33,1
# -ts 0.85,1


# -m /mnt/llm-data/huggingface/hub/Qwen3-Coder-Next-UD-Q3_K_XL.gguf
# # -m /mnt/llm-data/huggingface/hub/Qwen3-Coder-Next-IQ4_XS.gguf
# # -ot "\.([6-9]|1[0-9])\.ffn_(up|down)_exps.=CPU"
# # -ts 1.3,1
# # -ot "\.(1[0-9])\.ffn_(up|down)_exps.=CPU"
# # -ts 1.3,1
# -ts 20,28
# # -ctk q8_0
# # -ctv q8_0
# --temp 0
# --top-p 0.95
# --top-k 40
# --min-p 0.01
# --seed 3407
# --repeat-penalty 1.0
# -ngl 999

# Common config
--color
-sm layer
# --no-mmap
# --mmap
--mlock
# --run-time-repack
--scheduler_async
# --merge-qkv
# -cram 0
-cram 32768
--ctx-checkpoints 256
--ctx-checkpoints-interval 2048
# --cache-ram-n-min 1024
# --cache-ram-similarity 0.7
--host 0.0.0.0
--port 8000
--jinja
-fa on
--numa numactl
--alias kCode
-b 4096
-ub 2048
-cb
--no-context-shift
--defrag-thold 0.2
# --k-cache-hadamard
# -muge
# --graph-reuse
# -no-ooae
# --grouped-expert-routing
--slot-save-path ./slots
--reasoning-tokens auto