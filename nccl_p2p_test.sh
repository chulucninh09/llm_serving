# Install NCCL tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=0 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local

# Test P2P bandwidth directly — WITHOUT vLLM
NCCL_CUMEM_ENABLE=0 \
NCCL_P2P_LEVEL=SYS \
NCCL_DEBUG=INFO \
./build/all_reduce_perf -b 1M -e 1G -f 2 -g 2 2>&1 | tee nccl_p2p_test.log