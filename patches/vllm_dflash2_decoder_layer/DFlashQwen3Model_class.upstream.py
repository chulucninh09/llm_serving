@support_torch_compile
class DFlashQwen3Model(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
