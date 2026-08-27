        # in_proj_ba is 2 * num_v_heads wide (96, 24 at TP=4). Marlin requires
        # size_n % 64 == 0, so this gating projection must stay unquantized.
        # Checkpoints keep in_proj_a/b in bf16; compressed-tensors ignore lists
        # often name the parent linear_attn module, which does not match this
        # fused child. Skip quant here instead of rewriting ignore names.
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[num_v_heads, num_v_heads],
            bias=False,
            quant_config=None,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
