    logger.warning(
        "Speculative decoding (method=%s) is enabled but no KV cache group "
        "could be identified as the draft model's, so every attention group "
        "will be treated as a draft group. Mamba groups %s are excluded, so "
        "only attention-group prefix-cache reuse is widened (its trailing "
        "block is dropped); Mamba groups keep full prefix reuse.",
        spec_config.method,
        mamba_groups,
    )
