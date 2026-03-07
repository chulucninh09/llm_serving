# -m /mnt/llm-data/huggingface/hub/Qwen3.5-35B-A3B-UD-Q6_K_XL.gguf
# # -m /mnt/llm-data/huggingface/hub/Qwen3-Coder-Next-IQ4_XS.gguf
# # -m /mnt/llm-data/huggingface/hub/Qwen3-Coder-Next-UD-Q3_K_XL.gguf
# # -ot "\.([6-9]|1[0-9])\.ffn_(gate|up|down)_exps.=CPU"
# # -ot "\.([0-9]|3[0-9])\.ffn_(gate|up|down)_exps.=CPU"
# -ts 0.95,1
# -np -1
# -c 90000
# # -m /mnt/llm-data/huggingface/hub/Qwen3-Coder-Next-IQ4_XS.gguf
# # -ot "\.(1[0-9])\.ffn_(up|down)_exps.=CPU"
# # -ts 1.2,1
# # -np 2
# # -c 70000
# # -m /mnt/llm-data/huggingface/hub/Qwen3-Coder-Next-UD-Q3_K_XL.gguf
# # -ts 0.85,1
# # -np 2
# # -c 130000
# # -ctk q8_0
# # -ctv q8_0
# --temp 1.0
# --top-p 0.95
# --top-k 40
# --min-p 0.01
# --seed 3407
# --reasoning-budget 0
# -ngl 999
# # -ot "\.([6-9]|1[0-9])\.ffn_(up|down)_exps.=CPU"
# # --n-cpu-moe 12
# # -ts 1.33,1
# # -ts 0.85,1

-m /mnt/llm-data/huggingface/hub/Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf
# -m /mnt/llm-data/huggingface/hub/Qwen3-Coder-Next-IQ4_XS.gguf
# -m /mnt/llm-data/huggingface/hub/Qwen3-Coder-Next-UD-Q3_K_XL.gguf
# -ot "\.([6-9]|1[0-9])\.ffn_(gate|up|down)_exps.=CPU"
# -ot "\.([0-9]|3[0-9])\.ffn_(gate|up|down)_exps.=CPU"
# -ts 0.95,1
-np 2
-c 480000
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
--chat-template-file templates/qwen35.jinja2

# -m /mnt/llm-data/huggingface/Nemotron-3-Nano-30B-A3B-UD-Q6_K_XL.gguf
# --temp 0.6
# --top-p 0.95
# --repeat-penalty 1.1
# --min-p 0.01

# -m /mnt/llm-data/huggingface/Devstral-Small-2-24B-Instruct-2512-UD-Q6_K_XL.gguf
# --temp 0.15
# --min-p 0.01

# Common config
# --fit-ctx 80000
# --fit-target 1900
-fit off
--mlock
--mmap
# --no-mmap
# --direct-io
# -sm row
--ctx-checkpoints 256
-cram 32768
# -kvu
--host 0.0.0.0
--port 8000
--jinja
-fa on
-b 4096
-ub 2048
--alias kCode
--cont-batching
--slot-save-path /root/llm_serving/.slots