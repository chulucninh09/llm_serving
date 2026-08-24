    def register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ):
        kv_cache_config = self.spec.kv_cache_config
        num_blocks = kv_cache_config.num_blocks

        # layer_name -> (num_blocks, page_size_bytes) tensor
        tensors_per_block: dict[str, tuple[torch.Tensor, ...]] = {}
        # layer_name -> size of (un-padded) page in bytes
        unpadded_page_size_bytes: dict[str, int] = {}
        # layer_name -> size of page in bytes
        page_size_bytes: dict[str, int] = {}
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            group_layer_names = kv_cache_group.layer_names
            group_kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(group_kv_cache_spec, UniformTypeKVCacheSpecs):
                per_layer_specs = group_kv_cache_spec.kv_cache_specs
            else:
                per_layer_specs = {}
            for layer_name in group_layer_names:
                layer_kv_cache_spec = per_layer_specs.get(
                    layer_name, group_kv_cache_spec
                )
                if isinstance(layer_kv_cache_spec, AttentionSpec):
                    layer_kv_cache = kv_caches[layer_name]
                    assert isinstance(layer_kv_cache, torch.Tensor)

                    page = layer_kv_cache_spec.page_size_bytes
                    elem_size = layer_kv_cache.element_size()
                    byte_offset = layer_kv_cache.storage_offset() * elem_size
                    block_stride_bytes = layer_kv_cache.stride(0) * elem_size
                    tensors_per_block[layer_name] = (
                        torch.tensor(
                            [],
                            dtype=torch.int8,
                            device=layer_kv_cache.device,
                        ).set_(
                            layer_kv_cache.untyped_storage(),
                            byte_offset,
                            (num_blocks, page),
                            (block_stride_bytes, 1),
                        ),
                    )
                    page_size_bytes[layer_name] = layer_kv_cache_spec.page_size_bytes
                    unpadded_page_size_bytes[layer_name] = (
                        layer_kv_cache_spec.real_page_size_bytes
                    )

                elif isinstance(layer_kv_cache_spec, MambaSpec):
                    state_tensors = kv_caches[layer_name]
                    assert isinstance(state_tensors, list)

                    # re-construct the raw (num_blocks, page_size) tensor
                    # from the first state tensor
                    assert len(state_tensors) > 0
                    first_state_tensor = state_tensors[0]
                    assert first_state_tensor.storage_offset() == 0
                    tensor = (
                        torch.tensor(
                            [],
                            dtype=torch.int8,
                            device=first_state_tensor.device,
                        )
                        .set_(first_state_tensor.untyped_storage())
                        .view((num_blocks, layer_kv_cache_spec.page_size_bytes))
                    )
                    tensors_per_block[layer_name] = (tensor,)

                    page_size_bytes[layer_name] = layer_kv_cache_spec.page_size_bytes
                    unpadded_page_size_bytes[layer_name] = replace(
                        layer_kv_cache_spec, page_size_padded=None
                    ).page_size_bytes

                else:
                    raise NotImplementedError

        packed_kv_cache_tensor = next(
            (
                t
                for t in kv_cache_config.kv_cache_tensors
                if t.block_stride and t.shared_by
            ),
            None,
        )
        if packed_kv_cache_tensor is not None:
            (tensor,) = tensors_per_block[packed_kv_cache_tensor.shared_by[0]]
            block_stride = tensor.stride(0)
            packed_tensor = tensor.as_strided(
                (num_blocks, block_stride),
                (block_stride, 1),
                storage_offset=0,
            )
            self._init_worker(
                CanonicalKVCaches(
                    [CanonicalKVCacheTensor(packed_tensor, block_stride)],
                    [
                        [CanonicalKVCacheRef(0, block_stride)]
                        for _ in kv_cache_config.kv_cache_groups
                    ],
                )
            )
            return

        block_tensors: list[CanonicalKVCacheTensor] = []
        block_data_refs: dict[str, list[CanonicalKVCacheRef]] = defaultdict(list)
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            # Filter to layers that were actually processed above.
            # Packed KV allocation emits KVCacheTensor entries for
            # every (tuple_idx, page_size) slot; slots where no group has a
            # layer at that index produce an empty shared_by (reserved memory
            # with no corresponding model layer).
            tensor_layer_names = [
                n for n in kv_cache_tensor.shared_by if n in tensors_per_block
            ]
            if not tensor_layer_names:
                continue

            # Hybrid models (e.g. Qwen3.6) may list layers in the same shared_by
            # slot with different per-block tensor views. Group by view identity
            # instead of requiring a single stride across all layers.
            view_groups: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(
                list
            )
            for layer_name in tensor_layer_names:
                view = tensors_per_block[layer_name][0]
                view_groups[(view.data_ptr(), view.stride())].append(layer_name)

            for group_layers in view_groups.values():
                assert len({len(tensors_per_block[n]) for n in group_layers}) == 1

                first_layer_name = group_layers[0]
                for tensor in tensors_per_block[first_layer_name]:
                    block_tensors.append(
                        CanonicalKVCacheTensor(
                            tensor=tensor,
                            page_size_bytes=page_size_bytes[first_layer_name],
                        )
                    )

                    curr_tensor_idx = len(block_tensors) - 1
                    for layer_name in group_layers:
                        block_data_refs[layer_name].append(
                            CanonicalKVCacheRef(
                                tensor_idx=curr_tensor_idx,
                                page_size_bytes=unpadded_page_size_bytes[layer_name],
                            )
                        )

        group_data_refs: list[list[CanonicalKVCacheRef]] = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            group_refs: list[CanonicalKVCacheRef] = []
            for layer_name in kv_cache_group.layer_names:
                group_refs += block_data_refs[layer_name]
            group_data_refs.append(group_refs)

        canonical_kv_caches = CanonicalKVCaches(
            tensors=block_tensors,
            group_data_refs=group_data_refs,
        )

        self._init_worker(canonical_kv_caches)
