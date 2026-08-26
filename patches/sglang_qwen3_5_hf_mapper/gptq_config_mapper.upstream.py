        return get_linear_quant_method(
            self, layer, prefix=prefix, linear_method_cls=GPTQLinearMethod
        )

    def get_linear_scheme(self, layer: torch.nn.Module):
        return GPTQLinearScheme(self)
