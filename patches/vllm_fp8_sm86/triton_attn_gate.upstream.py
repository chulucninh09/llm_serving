            if self.kv_cache_dtype.startswith("fp8") and not (
                current_platform.has_device_capability(89)
            ):
                suggested = (
                    "float16" if (cap is None or cap.to_int() < 80) else "bfloat16"
                )
                raise ValueError(
                    f"FP8 KV cache is not supported by the Triton attention backend "
                    f"on {dev} (compute capability {cap_str}); native FP8 (fp8e4nv) "
                    f"requires SM89+. Re-run with --kv-cache-dtype {suggested}."
                )
