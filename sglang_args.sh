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
--model-path RedHatAI/Qwen3.8-27B-INT4
--enable-multimodal
--image-processor-backend pil
--mm-enable-dp-encoder
--mm-preprocess-cache-size-mb 512
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
--max-mamba-cache-size 160
--mamba-radix-cache-strategy extra_buffer_lazy
--mamba-ssm-dtype bfloat16
--dtype bfloat16
--preferred-sampling-params '{"custom_params": {"thinking_budget": 4096}}'


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

# Batching
# --enable-dynamic-chunking
--chunked-prefill-size 2048
--max-prefill-tokens 4096
--max-running-requests 4
--mem-fraction-static 0.80
# --cuda-graph-bs 4
# --piecewise-cuda-graph-max-tokens 8192
--cuda-graph-max-bs-prefill 2048
--cuda-graph-max-bs-decode 32
--enable-profile-cuda-graph
--enable-cudagraph-gc
# --enable-torch-compile
# --schedule-policy lpm
--trust-remote-code

# Hierarchical cache
--enable-session-radix-cache
--enable-hierarchical-cache
--hicache-size 5
--hicache-io-backend direct
--enable-lmcache
--lmcache-config-file lmcache.yaml
# --hicache-storage-backend file

# Performance
# --enable-dynamic-batch-tokenizer

# Scheduler
# --mamba-radix-cache-strategy extra_buffer