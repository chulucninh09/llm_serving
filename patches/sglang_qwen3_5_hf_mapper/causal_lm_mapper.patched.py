    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            "in_proj_qkv": "in_proj_qkvz",
            "in_proj_z": "in_proj_qkvz",
            "in_proj_b": "in_proj_ba",
            "in_proj_a": "in_proj_ba",
        },
    )

    supported_lora_modules = [
