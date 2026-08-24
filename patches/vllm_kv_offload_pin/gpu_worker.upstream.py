    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_cpu_blocks: int,
        mmap_region: SharedOffloadRegion | None = None,
    ):
        pin_memory = PIN_MEMORY
        self.pin_thread: threading.Thread | None = None
        self._manually_pinned_tensors: list[torch.Tensor] = []

        logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
        self._mmap_region = mmap_region

        gpu_tensors: list[torch.Tensor] = []
        self.cpu_tensors: list[torch.Tensor] = []
        for kv_cache_tensor in kv_caches.tensors:
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            cpu_page_size_bytes = gpu_page_size_bytes * block_size_factor

            if mmap_region is not None:
                cpu_tensor = mmap_region.create_next_view(cpu_page_size_bytes)
            else:
                t0 = time.monotonic()
                cpu_tensor = torch.zeros(
                    (num_cpu_blocks, cpu_page_size_bytes),
                    dtype=torch.int8,
                    device="cpu",
                    # CUDA/ROCm memory is registered asynchronously below.
                    # Pinning here would block worker initialization; other
                    # hardware need PyTorch allocation-time pinning.
                    pin_memory=PIN_MEMORY and not current_platform.is_cuda_alike(),
                )
                logger.debug(
                    "torch.zeros tensor %d×%d (%.2f GB): %.3f s",
                    num_cpu_blocks,
                    cpu_page_size_bytes,
                    num_cpu_blocks * cpu_page_size_bytes / 1e9,
                    time.monotonic() - t0,
                )

            gpu_tensors.append(gpu_tensor)
            self.cpu_tensors.append(cpu_tensor)

        if pin_memory:
            if not current_platform.is_cuda_alike():
                logger.info(
                    "Skipping host registration on %s; cudaHostRegister is only "
                    "available on CUDA/ROCm.",
                    current_platform.device_name,
                )
            else:
                self.pin_thread = threading.Thread(
                    target=self._pin_cpu_tensors,
                    name="CPUTensorPinThread",
                )
                self.pin_thread.start()
                logger.info("Starting to pin memory in background...")

        self._store_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=self.cpu_tensors,
            block_size_factor=block_size_factor,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            gpu_to_cpu=True,
            mmap_region=mmap_region,
            pin_thread=self.pin_thread,
            manually_pinned_tensors=self._manually_pinned_tensors,
        )

        self._load_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=self.cpu_tensors,
            block_size_factor=block_size_factor,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            gpu_to_cpu=False,
        )
