        if self.draft_model_config:
            # V1 places the entire draft on the last PP rank
            # (gpu_model_runner.py). The head is not pipeline-parallel, so
            # do not require SupportsPP when the target uses -pp > 1.
            verify_parallel = self.draft_parallel_config
            if verify_parallel.pipeline_parallel_size > 1:
                from vllm.config.utils import replace as config_replace

                verify_parallel = config_replace(
                    verify_parallel, pipeline_parallel_size=1
                )
            self.draft_model_config.verify_with_parallel_config(verify_parallel)
