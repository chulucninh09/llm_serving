    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        # Qwen3-based DSpark heads (e.g. RadixArk/Qwen3.8-27B-DSpark) declare
        # DSparkDraftModel, which routes to the DeepSeek-V4 speculator class
        # and gets model_type rewritten to deepseek_v4, breaking the load.
        # model_type=qwen3 identifies them as Qwen3 DSpark drafts; declare
        # Qwen3DSparkModel so the registry selects qwen3_dspark instead.
        if (
            hf_config.architectures
            and hf_config.architectures[0] == "DSparkDraftModel"
            and getattr(hf_config, "model_type", None) == "qwen3"
        ):
            hf_config.update({"architectures": ["Qwen3DSparkModel"]})

        initial_architecture = hf_config.architectures[0]
