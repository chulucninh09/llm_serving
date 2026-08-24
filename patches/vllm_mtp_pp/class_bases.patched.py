class Qwen3_5MTP(LocalArgmaxMixin, nn.Module, SupportsMultiModal, SupportsPP):
    packed_modules_mapping = {
