        use_eagle_block_drop = (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.use_eagle_block_drop()
        )
        if use_eagle_block_drop and not eagle_groups:
            eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))
