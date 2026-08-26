    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType | EngineInput,
        params: SamplingParams | PoolingParams,
        supported_tasks: tuple[SupportedTask, ...],
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        resumable: bool = False,
        session_id: str | None = None,
    ) -> EngineCoreRequest:
        self._validate_params(params, supported_tasks)
        self._validate_lora(lora_request)

        parallel_config = self.vllm_config.parallel_config
        dp_size = parallel_config.data_parallel_size
        dp_local_size = parallel_config.data_parallel_size_local
        num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
        if data_parallel_rank is not None and not (0 <= data_parallel_rank < num_ranks):
            raise VLLMValidationError(
                f"data_parallel_rank {data_parallel_rank} "
                f"is out of range [0, {num_ranks})."
            )

        if isinstance(prompt, dict) and "type" in prompt:
            if arrival_time is None:
                arrival_time = prompt.get("arrival_time", time.time())  # type: ignore[assignment]

            engine_input: EngineInput = prompt  # type: ignore[assignment]
        else:
            logger.warning_once(
                "Passing raw prompts to InputProcessor is deprecated "
                "and will be removed in the future. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            if arrival_time is None:
                arrival_time = time.time()

            renderer = self.renderer
            model_config = self.model_config

            parsed_prompt = parse_model_prompt(model_config, prompt)
            tok_params = renderer.default_cmpl_tok_params.with_kwargs(
                **(tokenization_kwargs or {})
            )

            (engine_input,) = renderer.render_cmpl(
                [parsed_prompt],
                tok_params,
            )

        current_platform.validate_request(engine_input, params)

        encoder_input, decoder_input = split_enc_dec_input(engine_input)
        self._validate_model_inputs(encoder_input, decoder_input)

        # Mypy can be conservative for TypedDict unions; normalize access.
        if decoder_input["type"] == "embeds":
            prompt_embeds = decoder_input["prompt_embeds"]
            prompt_token_ids = decoder_input.get("prompt_token_ids")
            prompt_is_token_ids = decoder_input.get("is_token_ids")
        else:
            prompt_token_ids = decoder_input["prompt_token_ids"]
            prompt_embeds = None
            prompt_is_token_ids = None

        sampling_params = None
        pooling_params = None
        if isinstance(params, SamplingParams):
            # TODO: can we avoid cloning here in multiproc case?
            sampling_params = params.clone()
            if sampling_params.thinking_token_budget is None:
                reasoning_config = self.vllm_config.reasoning_config
                if (
                    reasoning_config is not None
                    and reasoning_config.thinking_token_budget is not None
                ):
                    sampling_params.thinking_token_budget = (
                        reasoning_config.thinking_token_budget
                    )
            # If unset max tokens, then generate up to the max_model_len.
            if sampling_params.max_tokens is None:
                seq_len = length_from_prompt_token_ids_or_embeds(
                    prompt_token_ids, prompt_embeds
                )
                sampling_params.max_tokens = self.model_config.max_model_len - seq_len

            sampling_params.update_from_generation_config(
                self.generation_config_fields,
                self.renderer.get_eos_token_id(),
            )
            if self.tokenizer is not None:
                sampling_params.update_from_tokenizer(self.tokenizer)
            if sampling_params.trace_decode_token_ids:
                self._normalize_trace_replay_params(
                    sampling_params,
                    length_from_prompt_token_ids_or_embeds(
                        prompt_token_ids, prompt_embeds
                    ),
                )
        else:
            pooling_params = params.clone()

        # Multimodal related.
        mm_features: list[MultiModalFeatureSpec] | None = None

        if decoder_input["type"] == "multimodal":
            decoder_mm_inputs = decoder_input["mm_kwargs"]
            decoder_mm_positions = decoder_input["mm_placeholders"]
            decoder_mm_hashes = decoder_input["mm_hashes"]

            if not all(
                isinstance(leaf, str) for leaf in json_iter_leaves(decoder_mm_hashes)
            ):
                raise ValueError(
                    f"mm_hashes must contain only strings, got: {decoder_mm_hashes}. "
                    "This is likely due to an incorrect custom implementation of "
                    "MultiModalProcessor.apply method."
                )

            # Merge and flatten multimodal placeholders, hashes and inputs
            # from dictionaries to lists, and sort them by each item's position
            # in the input sequence.
            sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)

            mm_features = []
            for modality, idx in sorted_mm_idxs:
                base_mm_hash = decoder_mm_hashes[modality][idx]
                mm_features.append(
                    MultiModalFeatureSpec(
                        data=decoder_mm_inputs[modality][idx],
                        modality=modality,
                        identifier=self._get_mm_identifier(
                            base_mm_hash,
                            lora_request,
                        ),
                        mm_position=decoder_mm_positions[modality][idx],
                        mm_hash=base_mm_hash,
                    )
                )

        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            prompt_embeds=prompt_embeds,
            prompt_is_token_ids=prompt_is_token_ids,
            mm_features=mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            cache_salt=decoder_input.get("cache_salt"),
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            trace_headers=trace_headers,
            resumable=resumable,
            session_id=session_id,
        )
