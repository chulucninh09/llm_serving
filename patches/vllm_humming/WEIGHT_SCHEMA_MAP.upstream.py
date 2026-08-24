WEIGHT_SCHEMA_MAP: dict[str, type[BaseWeightSchema]] = {
    "awq": AWQWeightSchema,
    "bitnet": BitnetWeightSchema,
    "compressed-tensors": CompressedTensorsWeightSchema,
    "fp8": Fp8WeightSchema,
    "gptq": GPTQWeightSchema,
    "humming": HummingWeightSchema,
    "modelopt": ModeloptWeightSchema,
    "mxfp4": Mxfp4WeightSchema,
    "gpt_oss_mxfp4": GptOssMxfp4WeightSchema,
}
