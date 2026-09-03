    VLLM_TRITON_USE_TD: bool | None = None
    VLLM_GPU_SYNC_CHECK: Literal["warn", "error"] | None = None
    # Opt-in override to allow an FP8 KV cache on sub-SM89 NVIDIA devices
    # (e.g. SM80/SM86 RTX). Native fp8e4nv tensor cores don't exist there, so
    # the Triton backend stores fp8e4m3 and dequantizes in-kernel. This is
    # slower than bf16 and can lose accuracy; enable only for experiments that
    # specifically need "FULL decode in fp8 kv cache".
    VLLM_ALLOW_FP8_KV_CACHE_BELOW_SM89: bool = False
