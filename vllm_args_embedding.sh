--model jinaai/jina-code-embeddings-0.5b

# Common config
--attention-backend FLASHINFER
-dp 2
--host 0.0.0.0
--port 8001
--max-model-len 8192
--max-num-seqs 16
--max-cudagraph-capture-size 16
--gpu-memory-utilization 0.06
--served-model-name kCodeEmbedding
--trust-remote-code
