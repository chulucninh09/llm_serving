@support_torch_compile
class DFlashQwen3Model(nn.Module):
    decoder_layer_cls = DFlashQwen3DecoderLayer

    hf_to_vllm_mapper = WeightsMapper(
