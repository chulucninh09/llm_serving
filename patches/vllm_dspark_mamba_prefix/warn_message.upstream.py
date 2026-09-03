    logger.warning(
        "Speculative decoding (method=%s) is enabled but no KV cache group "
        "could be identified as the draft model's, so every group -- "
        "including Mamba groups %s -- will be treated as a draft group. A "
        "Mamba group cannot satisfy the widened lookup window that implies, "
        "so prefix-cache reuse across requests will be disabled and any "
        "external KV offload tier will store without ever serving a hit.",
        spec_config.method,
        mamba_groups,
    )
