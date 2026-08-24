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
