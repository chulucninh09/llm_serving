        return get_linear_quant_method(
            self, layer, prefix=prefix, linear_method_cls=GPTQLinearMethod
        )

    def apply_weight_name_mapper(self, hf_to_sglang_mapper):
        # GPTQModel ignore rules live in `dynamic` as "-:<regex>" keys using
        # checkpoint names. Alias them onto fused GDN projections.
        if self.dynamic:
            self.dynamic = hf_to_sglang_mapper.apply_dict(self.dynamic)

    def get_linear_scheme(self, layer: torch.nn.Module):
        return GPTQLinearScheme(self)
