cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86" -DLLAMA_CURL=OFF
cmake --build build --config Release -j 8