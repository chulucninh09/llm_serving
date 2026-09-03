        # KV cache group indices that get the EAGLE last-block drop.
        self.eagle_group_ids: set[int] = {
            i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group
        }
        # Conservatively fall back to flagging all groups when no group is
        # flagged -- except recurrent-state (Mamba) groups. Align-mode
        # checkpoints are only taken at exact chunk/boundary positions, so a
        # Mamba group can never satisfy an eagle-widened lookup window and
        # would lose cross-request prefix reuse entirely.
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = {
                i
                for i, g in enumerate(kv_cache_config.kv_cache_groups)
                if not is_mamba_group(g)
            }
            if self.eagle_group_ids:
                logger.info(
                    "Treating all attention groups as draft groups (drafter KV "
                    "group not identifiable); Mamba groups keep full prefix reuse."
                )
