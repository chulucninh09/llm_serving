    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_idxs: list[int] | None = None,
        custom_stat_loggers: list[StatLoggerFactory] | None = None,
        enable_default_loggers: bool = True,
        aggregate_engine_logging: bool = False,
        client_count: int = 1,
    ):
        self.engine_indexes = engine_idxs if engine_idxs else [0]
        self.stat_loggers: list[AggregateStatLoggerBase] = []
        stat_logger_factories: list[StatLoggerFactory] = []
        if custom_stat_loggers is not None:
            stat_logger_factories.extend(custom_stat_loggers)
        if enable_default_loggers and logger.isEnabledFor(logging.INFO):
            if client_count > 1:
                logger.info(
                    "AsyncLLM created with api_server_count more than 1; "
                    "console stats are logged from each EngineCore instead."
                )
            else:
                default_logger_factory = (
                    AggregatedLoggingStatLogger
                    if aggregate_engine_logging
                    else LoggingStatLogger
                )
                stat_logger_factories.append(default_logger_factory)
        custom_prometheus_logger: bool = False
        for stat_logger_factory in stat_logger_factories:
            if isinstance(stat_logger_factory, type) and issubclass(
                stat_logger_factory, AggregateStatLoggerBase
            ):
                global_stat_logger = stat_logger_factory(
                    vllm_config=vllm_config,
                    engine_indexes=self.engine_indexes,
                )
                if isinstance(global_stat_logger, PrometheusStatLogger):
                    custom_prometheus_logger = True
            else:
                # per engine logger
                global_stat_logger = PerEngineStatLoggerAdapter(
                    vllm_config=vllm_config,
                    engine_indexes=self.engine_indexes,
                    per_engine_stat_logger_factory=stat_logger_factory,  # type: ignore[arg-type]
                )
            self.stat_loggers.append(global_stat_logger)
        if not custom_prometheus_logger:
            self.stat_loggers.append(
                PrometheusStatLogger(vllm_config, self.engine_indexes)
            )
