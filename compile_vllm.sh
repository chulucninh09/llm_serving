# https://docs.vllm.ai/en/stable/contributing/incremental_build/#prerequisites
uv venv --python 3.12 --seed
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto
uv pip install -r requirements/build.txt --torch-backend=auto
python3 tools/generate_cmake_presets.py
cmake --preset release
cmake --build --preset release --target install
