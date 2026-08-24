    def _attach_iteration_details(
        self,
        outputs: dict[int, EngineCoreOutputs],
        iteration_details: SchedulerIterationDetails | None,
    ) -> None:
        if iteration_details is None:
            return

        if (eco := next(iter(outputs.values()), None)) is None:
            outputs[0] = eco = EngineCoreOutputs()
        if eco.scheduler_stats is None:
            eco.scheduler_stats = self._make_iteration_details_stats(iteration_details)
        else:
            eco.scheduler_stats.iteration_details = iteration_details

        # Console stats live on EngineCore so --api-server-count > 1 still
        # prints complete per-rank throughput / KV / queue stats.
        if not self.log_stats:
            return

        from vllm.v1.metrics.loggers import LoggingStatLogger
        from vllm.v1.metrics.stats import IterationStats

        console_logger = getattr(self, "_console_stat_logger", None)
        if console_logger is None:
            engine_index = self.vllm_config.parallel_config.data_parallel_rank
            console_logger = LoggingStatLogger(
                self.vllm_config, engine_index=engine_index
            )
            self._console_stat_logger = console_logger
            self._last_console_stat_log = time.monotonic()

        iteration_stats = IterationStats()
        if not iteration_details.is_dummy:
            iteration_stats.prompt_token_stats.computed = (
                iteration_details.num_ctx_tokens
            )
            iteration_stats.num_generation_tokens = (
                iteration_details.num_generation_tokens
            )

        console_logger.record(
            eco.scheduler_stats,
            iteration_stats,
            engine_idx=console_logger.engine_index,
        )
        now = time.monotonic()
        if now - self._last_console_stat_log >= envs.VLLM_LOG_STATS_INTERVAL:
            console_logger.log()
            self._last_console_stat_log = now
