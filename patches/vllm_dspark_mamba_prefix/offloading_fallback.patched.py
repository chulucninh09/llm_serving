        use_eagle_block_drop = (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.use_eagle_block_drop()
        )
        if use_eagle_block_drop and not eagle_groups:
            # Same shape as KVCacheCoordinator: keep the last-block drop off
            # recurrent-state (Mamba) groups, whose align-mode checkpoints are
            # only taken at exact chunk boundaries and can never satisfy an
            # eagle-widened lookup window.
            eagle_groups = {
                idx
                for idx, g in enumerate(kv_cache_config.kv_cache_groups)
                if not is_mamba_group(g)
            }
            if eagle_groups:
                logger.info(
                    "Treating all attention groups as draft groups (drafter KV "
                    "group not identifiable); Mamba groups keep full prefix reuse."
                )
