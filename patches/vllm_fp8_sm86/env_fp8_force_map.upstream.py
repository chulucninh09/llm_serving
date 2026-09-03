    "VLLM_TRITON_USE_TD": lambda: {"1": True, "0": False}.get(
        os.getenv("VLLM_TRITON_USE_TD", "").strip()
    ),
