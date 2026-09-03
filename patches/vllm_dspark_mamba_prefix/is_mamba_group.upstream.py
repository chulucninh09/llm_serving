    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        return kv_cache_spec.kv_cache_specs.values()
    return (kv_cache_spec,)


def is_full_attention_spec(kv_cache_spec: KVCacheSpec) -> bool:
