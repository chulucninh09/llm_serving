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

# --model-path btbtyler09/Qwen3.6-27B-GPTQ-4bit
# --model-path Qwen/Qwen3.6-27B-FP8
# --model-path Lorbus/Qwen3.6-27B-int4-AutoRound
--model-path cyankiwi/Qwen3.6-27B-AWQ-INT4
# --model-path Qwen/Qwen3.6-35B-A3B-FP8
# --model-path RedHatAI/Muse-Glimmer-30B-FP8-block
# --model-path cyankiwi/Muse-Glimmer-30B-AWQ-INT4
# --context-length 80000
# --quantization moe_wna16
# --kv-cache-dtype fp8_e4m3
--tool-call-parser qwen3_coder
--reasoning-parser qwen3
--speculative-algorithm EAGLE
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 1
# --speculative-algorithm DFLASH
# --speculative-draft-model-path z-lab/Qwen3.6-35B-A3B-DFlash
# --speculative-draft-num-speculative-tokens 15
# --speculative-dflash-draft-window-size 2048


# Common config
--tp 4
# --ep 4
--port 8000
--host 0.0.0.0
--served-model-name kCode
--trust-remote-code
--disable-custom-all-reduce
--model-impl sglang
# --load-format fastsafetensors
# --enable-torch-compile
# --load-format instanttensor
# --dtype float16

# Batching
--chunked-prefill-size 2048
--max-prefill-tokens 8192
# --enable-dynamic-chunking
--max-running-requests 4
--mem-fraction-static 0.9
# --cuda-graph-bs 4
--cuda-graph-bs-decode 4
# --piecewise-cuda-graph-max-tokens 8192
--cuda-graph-max-bs-prefill 8192
# --enable-profile-cuda-graph
# --enable-cudagraph-gc
# --enable-torch-compile
# --schedule-policy lpm
--trust-remote-code

# Hierarchical cache
# --enable-hierarchical-cache
# --hicache-size 10
# --hicache-io-backend direct

# Performance
# --enable-dynamic-batch-tokenizer

# Scheduler
# --mamba-radix-cache-strategy extra_buffer