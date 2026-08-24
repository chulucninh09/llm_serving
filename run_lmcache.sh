export PYTHONHASHSEED=0

lmcache server \
    --chunk-size 1600 \
    --separate-object-groups \
    --l1-size-gb 26 \
    --l2-adapter '{"type": "fs_native", "base_path": "/mnt/llm-data/kv-cache", "max_capacity_gb": 40}' \
    --eviction-policy LRU \
    --http-port 9999 \
    --max-workers 1