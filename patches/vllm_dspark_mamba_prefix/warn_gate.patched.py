    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_eagle_block_drop():
        return
    if any(group.is_eagle_group for group in kv_cache_groups):
        return
