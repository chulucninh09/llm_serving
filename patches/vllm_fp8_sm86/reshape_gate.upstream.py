    if kv_cache_dtype.startswith("fp8"):
        return current_platform.has_device_capability(89) or current_platform.is_xpu()
    if kv_cache_dtype == "bfloat16":
        return current_platform.has_device_capability(80) or current_platform.is_xpu()
    return True
