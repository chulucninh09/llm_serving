        return get_linear_quant_method(
            self, layer, prefix=prefix, linear_method_cls=GPTQMarlinLinearMethod
        )

    def apply_weight_name_mapper(self, hf_to_sglang_mapper):
        if self.dynamic:
            self.dynamic = hf_to_sglang_mapper.apply_dict(self.dynamic)

    def get_linear_scheme(self, layer: torch.nn.Module):
        return GPTQMarlinLinearScheme(self)
