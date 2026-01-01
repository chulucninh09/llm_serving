--model jinaai/jina-code-embeddings-0.5b

# Common config
--tensor-parallel-size 2
--host 0.0.0.0
--port 8001
--max-model-len 8192
--max-num-seqs 40
--max-cudagraph-capture-size 40
--gpu-memory-utilization 0.05
--served-model-name kCodeEmbedding
--trust-remote-code
