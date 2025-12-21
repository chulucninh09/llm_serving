cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86" -DLLAMA_CURL=OFF -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=FLAME
taskset -c 0-7,10-15 cmake --build build --config Release -j 16