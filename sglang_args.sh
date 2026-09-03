# --model-path palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4
# --context-length 150000
# --quantization moe_wna16
# --kv-cache-dtype fp8_e4m3
# --tool-call-parser qwen3_coder
# --reasoning-parser qwen3
# # --speculative-algorithm EAGLE
# # --speculative-num-steps 3
# # --speculative-eagle-topk 1
# # --speculative-num-draft-tokens 1
# # --speculative-algorithm DFLASH
# # --speculative-draft-model-path z-lab/Qwen3.6-35B-A3B-DFlash
# # --speculative-draft-num-speculative-tokens 15
# # --speculative-dflash-draft-window-size 2048


# --model-path philbert440/Qwen3.8-27B-W4A16-AWQ
# --model barrydeen/Qwen3.8-27B-AWQ-4bit
# --model-path pearsonkyle/Qwen3.8-27B-GPTQ-W4A16
# --model-path RedHatAI/Qwen3.8-27B-INT4
--model-path RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead
--enable-multimodal
--mm-process-config '{"image":{"min_pixels":65536,"max_pixels":3211264}}'
--image-processor-backend pil
--mm-enable-dp-encoder
--mm-preprocess-cache-size-mb 256
--kv-cache-dtype fp8_e4m3
--reasoning-parser qwen3
--tool-call-parser qwen3_coder
--chat-template templates/qwen3.8-enhanced.jinja
# --speculative-algorithm EAGLE
# --speculative-num-steps 3
# --speculative-eagle-topk 1
# --speculative-num-draft-tokens 4
# --speculative-algorithm DFLASH
# --speculative-draft-model-path z-lab/Qwen3.8-27B-DFlash2
# --speculative-num-draft-tokens 8
--speculative-algorithm DSPARK 
--speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark
--speculative-adaptive
# --mamba-full-memory-ratio 0.6
--max-mamba-cache-size 48
--mamba-radix-cache-strategy extra_buffer
--mamba-track-interval 512
--mamba-ssm-dtype bfloat16
--dtype bfloat16
--preferred-sampling-params '{"custom_params": {"thinking_budget": 2048}}'


# Common config
--tp 4
--port 8000
--host 0.0.0.0
--served-model-name kCode
--trust-remote-code
--disable-custom-all-reduce
# --model-impl sglang
--api-key $LLM_API_KEY
--enable-metrics
--mem-fraction-static 0.82
--max-running-requests 12
--schedule-policy lpm
--trust-remote-code

# Batching
--enable-profile-cuda-graph
--enable-cudagraph-gc
--enable-mixed-chunk

--chunked-prefill-size 1024
--max-prefill-tokens 6144
--cuda-graph-bs-prefill 512 1024 2048 3072 4096 5120 6144

# --cuda-graph-max-bs-decode 12

# Hierarchical cache
--enable-session-radix-cache
--enable-hierarchical-cache
--hicache-size 4
--hicache-io-backend direct
--hicache-storage-backend file
--hicache-storage-backend-extra-config '{"max_size":"60G","eviction_ratio":0.9}'
--page-size 64

# Performance
# --enable-dynamic-batch-tokenizer
