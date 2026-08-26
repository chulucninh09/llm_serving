    # Checkpoint ignore lists use pre-fusion names (in_proj_qkv/z, in_proj_b/a).
    # Map them onto the fused runtime tensors so GPTQ/Marlin skip in_proj_ba
    # instead of trying to quantize the 24-wide gating projection.
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            "in_proj_qkv": "in_proj_qkvz",
            "in_proj_z": "in_proj_qkvz",
            "in_proj_b": "in_proj_ba",
            "in_proj_a": "in_proj_ba",
        },
    )
