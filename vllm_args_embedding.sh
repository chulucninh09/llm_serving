--model jinaai/jina-code-embeddings-0.5b

# Common config
-dp 1
--host 0.0.0.0
--port 8001
--max-model-len 8192
--max-num-seqs 4
--max-cudagraph-capture-size 4
--gpu-memory-utilization 0.05
--served-model-name kCodeEmbedding
--trust-remote-code
