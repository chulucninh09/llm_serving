    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        return kv_cache_spec.kv_cache_specs.values()
    return (kv_cache_spec,)


def is_mamba_group(group: KVCacheGroupSpec) -> bool:
    """Whether a KV cache group is (or wraps) recurrent-state Mamba layers.

    Recurrent-state groups have no stable slot layout across a widened lookup
    window: align-mode checkpoints are only taken at exact chunk boundaries,
    so an eagle-widened window can never be satisfied. Consumers that fall
    back to flagging all draft groups therefore skip these groups so their
    prefix reuse is preserved.
    """
    return any(
        isinstance(spec, MambaSpec) for spec in iter_layer_specs(group.kv_cache_spec)
    )


def is_full_attention_spec(kv_cache_spec: KVCacheSpec) -> bool:
