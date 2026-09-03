    if kv_cache_dtype.startswith("fp8"):
        # Opt-in override for sub-SM89 NVIDIA devices (see envs). Without it,
        # native fp8e4nv is required (SM89+) and the backend falls back to
        # bf16/int8 on older GPUs.
        if os.environ.get("VLLM_ALLOW_FP8_KV_CACHE_BELOW_SM89", "").strip() in {
            "1",
            "true",
            "True",
            "on",
            "yes",
        }:
            return True
        return current_platform.has_device_capability(89) or current_platform.is_xpu()
    if kv_cache_dtype == "bfloat16":
        return current_platform.has_device_capability(80) or current_platform.is_xpu()
    return True
