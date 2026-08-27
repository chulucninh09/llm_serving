        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[num_v_heads, num_v_heads],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
