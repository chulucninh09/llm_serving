#!/usr/bin/env python3
"""LiteLLM streaming bench: long shared prefix (APC) + short unique turn, then decode tok/s."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm

litellm.drop_params = True

ROOT = Path(__file__).resolve().parent.parent

CONTEXTS = list(range(10_000, 230_001, 70_000))
MAX_TOKENS = 200
TRIALS = 3
CCU_LEVELS = (1,)

WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu river mountain valley forest canyon harbor meadow glacier "
    "nebula comet orbit photon quark neutron proton electron plasma gravity tide "
    "current circuit voltage resistor capacitor transistor kernel scheduler cache "
    "token context latency throughput decode prefill stream chunk batch pipeline"
).split()


@dataclass
class RequestResult:
    trial: int
    stream_idx: int
    prompt_tokens: int | None
    completion_tokens: int | None
    ttft_s: float | None
    e2e_s: float | None
    prefill_tps: float | None
    decode_tps: float | None
    error: str | None = None
    suffix: str = ""
    cached_tokens: int | None = None


def progress(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    print(f"{ts} {msg}", file=sys.stderr, flush=True)


def _fmt_result(r: RequestResult) -> str:
    if r.error:
        err = r.error.replace("\n", " ")
        return f"ERR {err[:160]}"
    parts: list[str] = []
    if r.ttft_s is not None:
        parts.append(f"ttft={r.ttft_s:.2f}s")
    if r.cached_tokens is not None and r.prompt_tokens:
        pct = 100.0 * r.cached_tokens / r.prompt_tokens
        parts.append(f"cached={r.cached_tokens}/{r.prompt_tokens} ({pct:.0f}%)")
    if r.prefill_tps is not None:
        parts.append(f"prefill={r.prefill_tps:.0f} tok/s")
    if r.decode_tps is not None:
        parts.append(f"decode={r.decode_tps:.1f} tok/s")
    if r.completion_tokens is not None:
        parts.append(f"out={r.completion_tokens}")
    return " ".join(parts) or "ok"


def random_seed_text(rng: random.Random, n_words: int = 480) -> str:
    parts = [rng.choice(WORDS) for _ in range(n_words)]
    # Mix punctuation so the tokenizer does not collapse into a tiny vocab run.
    for i in range(12, n_words, 17):
        parts[i] = parts[i] + rng.choice([".", ",", ";", ":"])
    return " ".join(parts) + "\n"


def extra_body() -> dict[str, Any]:
    return {
        "ignore_eos": True,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
        },
    }


def messages_for(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def litellm_kwargs(base_url: str, api_key: str, model: str) -> dict[str, Any]:
    return {
        "model": f"openai/{model}",
        "api_base": base_url.rstrip("/"),
        "api_key": api_key,
    }


def _delta_has_token(delta: Any) -> bool:
    if delta is None:
        return False
    for attr in ("content", "reasoning_content", "reasoning"):
        val = getattr(delta, attr, None)
        if val:
            return True
    extra = getattr(delta, "model_extra", None) or {}
    if extra.get("reasoning_content") or extra.get("reasoning") or extra.get("content"):
        return True
    if isinstance(delta, dict):
        return bool(
            delta.get("content")
            or delta.get("reasoning_content")
            or delta.get("reasoning")
        )
    return False


def _usage_tokens(usage: Any) -> tuple[int | None, int | None]:
    if usage is None:
        return None, None
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is None and isinstance(usage, dict):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
    return prompt, completion


def _cached_tokens(usage: Any) -> int | None:
    if usage is None:
        return None
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None and isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
    if details is None:
        return None
    if isinstance(details, dict):
        val = details.get("cached_tokens")
    else:
        val = getattr(details, "cached_tokens", None)
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


async def complete_once(
    *,
    base_url: str,
    api_key: str,
    model: str,
    content: str,
    max_tokens: int,
    stream: bool,
    timeout: float,
) -> tuple[int | None, int | None, float | None, float, str | None, int | None]:
    """Returns prompt_tokens, completion_tokens, ttft_s, e2e_s, error, cached_tokens."""
    kwargs = {
        **litellm_kwargs(base_url, api_key, model),
        "messages": messages_for(content),
        "max_tokens": max_tokens,
        "timeout": timeout,
        "extra_body": extra_body(),
    }
    start = time.perf_counter()
    ttft: float | None = None
    prompt_tokens = None
    completion_tokens = None
    cached_tokens = None
    try:
        if not stream:
            resp = await litellm.acompletion(**kwargs)
            e2e = time.perf_counter() - start
            usage = getattr(resp, "usage", None)
            prompt_tokens, completion_tokens = _usage_tokens(usage)
            cached_tokens = _cached_tokens(usage)
            return prompt_tokens, completion_tokens, e2e, e2e, None, cached_tokens

        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        stream_resp = await litellm.acompletion(**kwargs)
        async for chunk in stream_resp:
            if ttft is None:
                choices = getattr(chunk, "choices", None) or []
                if choices and _delta_has_token(getattr(choices[0], "delta", None)):
                    ttft = time.perf_counter() - start
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens, completion_tokens = _usage_tokens(usage)
                cached = _cached_tokens(usage)
                if cached is not None:
                    cached_tokens = cached
        e2e = time.perf_counter() - start
        if ttft is None:
            ttft = e2e
        return prompt_tokens, completion_tokens, ttft, e2e, None, cached_tokens
    except Exception as exc:  # noqa: BLE001 — record and continue the sweep
        e2e = time.perf_counter() - start
        return prompt_tokens, completion_tokens, ttft, e2e, f"{type(exc).__name__}: {exc}", cached_tokens


async def prompt_token_count(
    *,
    base_url: str,
    api_key: str,
    model: str,
    content: str,
    timeout: float,
) -> int:
    prompt_tokens, _, _, _, error, _ = await complete_once(
        base_url=base_url,
        api_key=api_key,
        model=model,
        content=content,
        max_tokens=1,
        stream=False,
        timeout=timeout,
    )
    if error:
        raise RuntimeError(f"token-count probe failed: {error}")
    if not prompt_tokens:
        raise RuntimeError("token-count probe returned no usage.prompt_tokens")
    return prompt_tokens


async def calibrate_prompt(
    *,
    base_url: str,
    api_key: str,
    model: str,
    target: int,
    rng: random.Random,
    timeout: float,
    rel_tol: float = 0.01,
    abs_tol: int = 64,
    label: str = "",
    base: str = "",
) -> tuple[str, int]:
    """Grow a stable prefix until vLLM reports ~target prompt tokens.

    Longer contexts extend the previous prefix so automatic prefix caching
    matches an agentic session whose history only grows.
    """
    progress(f"{label}calibrate {target} tok: probing tokenizer...")
    tol = max(abs_tol, int(target * rel_tol))
    n_base = 0
    if base:
        n_base = await prompt_token_count(
            base_url=base_url, api_key=api_key, model=model, content=base, timeout=timeout
        )
        if n_base >= target - tol:
            progress(f"{label}calibrate {target} tok: reused prefix ({n_base} tok)")
            return base, n_base

    seed = random_seed_text(rng)
    n1 = await prompt_token_count(
        base_url=base_url, api_key=api_key, model=model, content=seed, timeout=timeout
    )
    n2 = await prompt_token_count(
        base_url=base_url,
        api_key=api_key,
        model=model,
        content=seed + seed,
        timeout=timeout,
    )
    per_copy = n2 - n1
    if per_copy <= 0:
        per_copy = n1
        overhead = 0
    else:
        overhead = max(0, n1 - per_copy)
    need = max(1, target - n_base - (0 if base else overhead))
    copies = max(1, math.ceil(need / per_copy))
    extra = seed * copies
    body = base + extra
    actual = await prompt_token_count(
        base_url=base_url, api_key=api_key, model=model, content=body, timeout=timeout
    )
    if actual > 0 and abs(actual - target) > tol:
        extra_tokens = max(1, actual - n_base)
        scale = max(1, target - n_base) / extra_tokens
        new_len = max(len(seed), int(len(extra) * scale))
        if new_len <= len(extra):
            extra = extra[:new_len]
        else:
            more = math.ceil((new_len - len(extra)) / len(seed))
            extra = (extra + seed * more)[:new_len]
        body = base + extra
        actual = await prompt_token_count(
            base_url=base_url, api_key=api_key, model=model, content=body, timeout=timeout
        )
    progress(f"{label}calibrate {target} tok: got {actual} prompt tokens")
    return body, actual


def metrics_from(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    ttft_s: float | None,
    e2e_s: float | None,
    cached_tokens: int | None = None,
) -> tuple[float | None, float | None]:
    prefill = None
    decode = None
    uncached = prompt_tokens
    if prompt_tokens is not None and cached_tokens:
        uncached = max(0, prompt_tokens - cached_tokens)
    if uncached and ttft_s and ttft_s > 0:
        prefill = uncached / ttft_s
    if (
        completion_tokens is not None
        and completion_tokens > 1
        and e2e_s is not None
        and ttft_s is not None
        and e2e_s > ttft_s
    ):
        decode = (completion_tokens - 1) / (e2e_s - ttft_s)
    return prefill, decode


def agent_turn_suffix(trial: int, stream_idx: int) -> str:
    uid = uuid.uuid4().hex[:8]
    return (
        f"\n\n<user_turn trial={trial} stream={stream_idx} id={uid}>\n"
        "Continue from the current conversation. Give the next concrete action only.\n"
    )


async def timed_request(
    *,
    base_url: str,
    api_key: str,
    model: str,
    body: str,
    trial: int,
    stream_idx: int,
    timeout: float,
) -> RequestResult:
    suffix = agent_turn_suffix(trial, stream_idx)
    prompt_tokens, completion_tokens, ttft_s, e2e_s, error, cached_tokens = await complete_once(
        base_url=base_url,
        api_key=api_key,
        model=model,
        content=body + suffix,
        max_tokens=MAX_TOKENS,
        stream=True,
        timeout=timeout,
    )
    prefill, decode = metrics_from(
        prompt_tokens, completion_tokens, ttft_s, e2e_s, cached_tokens
    )
    return RequestResult(
        trial=trial,
        stream_idx=stream_idx,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_s=ttft_s,
        e2e_s=e2e_s,
        prefill_tps=prefill,
        decode_tps=decode,
        error=error,
        suffix=suffix.strip(),
        cached_tokens=cached_tokens,
    )


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "avg": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "n": float(len(values)),
    }


def cell_aggregate(results: list[RequestResult]) -> dict[str, Any]:
    ok = [r for r in results if r.error is None]
    def col(attr: str) -> list[float]:
        out: list[float] = []
        for r in ok:
            val = getattr(r, attr)
            if val is not None:
                out.append(float(val))
        return out

    return {
        "n_ok": len(ok),
        "n_err": len(results) - len(ok),
        "ttft_s": summarize(col("ttft_s")),
        "prefill_tps": summarize(col("prefill_tps")),
        "decode_tps": summarize(col("decode_tps")),
        "prompt_tokens": summarize(col("prompt_tokens")),
        "completion_tokens": summarize(col("completion_tokens")),
        "cached_tokens": summarize(col("cached_tokens")),
        "errors": [r.error for r in results if r.error],
    }


async def warm_prefix_cache(
    *,
    base_url: str,
    api_key: str,
    model: str,
    body: str,
    timeout: float,
    label: str = "",
    context: int | None = None,
) -> None:
    ctx = f"{context} tok " if context is not None else ""
    progress(f"{label}{ctx}warming prefix cache...")
    t0 = time.perf_counter()
    prompt_tokens, _, _, _, error, cached = await complete_once(
        base_url=base_url,
        api_key=api_key,
        model=model,
        content=body,
        max_tokens=1,
        stream=False,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - t0
    if error:
        progress(f"{label}{ctx}prefix warm FAILED: {error}")
        return
    hit = ""
    if prompt_tokens:
        cached_s = cached if cached is not None else 0
        hit = f" cached={cached_s}/{prompt_tokens}"
    progress(f"{label}{ctx}prefix warm: {elapsed:.1f}s{hit}")


async def run_cell(
    *,
    base_url: str,
    api_key: str,
    model: str,
    body: str,
    ccu: int,
    trials: int,
    timeout: float,
    label: str = "",
    context: int | None = None,
) -> list[RequestResult]:
    results: list[RequestResult] = []
    ctx = f"{context} tok " if context is not None else ""
    for trial in range(trials):
        progress(f"{label}{ctx}CCU={ccu} trial {trial + 1}/{trials}: starting ({ccu} stream{'s' if ccu != 1 else ''})")
        t0 = time.perf_counter()
        batch = await asyncio.gather(
            *[
                timed_request(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    body=body,
                    trial=trial,
                    stream_idx=i,
                    timeout=timeout,
                )
                for i in range(ccu)
            ]
        )
        elapsed = time.perf_counter() - t0
        results.extend(batch)
        streams = " | ".join(
            f"s{r.stream_idx} {_fmt_result(r)}" for r in batch
        )
        progress(
            f"{label}{ctx}CCU={ccu} trial {trial + 1}/{trials}: "
            f"{elapsed:.1f}s  {streams}"
        )
    return results


def timeout_for_context(context: int) -> float:
    # Chunked prefill at 4096 tokens plus 200 decode tokens; keep generous headroom.
    return max(180.0, context / 80.0 + 120.0)


async def bench_config(
    *,
    base_url: str,
    api_key: str,
    model: str,
    contexts: list[int],
    ccu_levels: list[int],
    trials: int,
    seed: int,
    label: str = "",
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    skip_from: int | None = None
    calibrated: dict[int, tuple[str, int]] = {}
    growing_prefix = ""
    n_ctx = len(contexts)
    prefix = f"{label} " if label else ""

    def emit(row: dict[str, Any]) -> None:
        rows.append(row)
        if on_row is not None:
            on_row(row)

    for ctx_i, context in enumerate(contexts, start=1):
        if skip_from is not None and context >= skip_from:
            progress(f"{prefix}skip {context} tok (after failure at {skip_from})")
            emit(
                {
                    "type": "skipped",
                    "context": context,
                    "reason": f"skipped after OOM/error at context {skip_from}",
                }
            )
            continue
        progress(f"{prefix}context {ctx_i}/{n_ctx}: {context} tok")
        try:
            body, actual = await calibrate_prompt(
                base_url=base_url,
                api_key=api_key,
                model=model,
                target=context,
                rng=rng,
                timeout=timeout_for_context(context),
                label=prefix,
                base=growing_prefix,
            )
            calibrated[context] = (body, actual)
            growing_prefix = body
            await warm_prefix_cache(
                base_url=base_url,
                api_key=api_key,
                model=model,
                body=body,
                timeout=timeout_for_context(context),
                label=prefix,
                context=context,
            )
        except Exception as exc:  # noqa: BLE001
            skip_from = context
            progress(f"{prefix}calibrate {context} tok FAILED: {type(exc).__name__}: {exc}")
            emit(
                {
                    "type": "calibrate_error",
                    "context": context,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        for ccu in ccu_levels:
            results = await run_cell(
                base_url=base_url,
                api_key=api_key,
                model=model,
                body=body,
                ccu=ccu,
                trials=trials,
                timeout=timeout_for_context(context),
                label=prefix,
                context=context,
            )
            agg = cell_aggregate(results)
            dec = (agg.get("decode_tps") or {}).get("median") if isinstance(agg.get("decode_tps"), dict) else None
            ttft = (agg.get("ttft_s") or {}).get("median") if isinstance(agg.get("ttft_s"), dict) else None
            dec_s = f"{dec:.1f} tok/s" if isinstance(dec, (int, float)) else "n/a"
            ttft_s = f"{ttft:.2f}s" if isinstance(ttft, (int, float)) else "n/a"
            cached = (agg.get("cached_tokens") or {}).get("median") if isinstance(agg.get("cached_tokens"), dict) else None
            cache_s = ""
            if isinstance(cached, (int, float)) and actual:
                cache_s = f" median_cached={cached:.0f}/{actual} ({100.0 * cached / actual:.0f}%)"
            progress(
                f"{prefix}{context} tok CCU={ccu}: done "
                f"ok={agg['n_ok']}/{agg['n_ok'] + agg['n_err']} "
                f"median_ttft={ttft_s} median_decode={dec_s}{cache_s}"
            )
            oomish = any(
                err and any(tok in err.lower() for tok in ("oom", "cuda", "out of memory"))
                for err in agg["errors"]
            )
            emit(
                {
                    "type": "cell",
                    "context": context,
                    "calibrated_prompt_tokens": actual,
                    "ccu": ccu,
                    "trials": trials,
                    "aggregate": agg,
                    "requests": [asdict(r) for r in results],
                }
            )
            if oomish:
                skip_from = context
                progress(f"{prefix}{context} tok: OOM/CUDA error — skipping longer contexts")
                break
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", required=True)
    p.add_argument("--model", default="kCode")
    p.add_argument("--contexts", default=",".join(str(c) for c in CONTEXTS))
    p.add_argument("--ccu", default=",".join(str(c) for c in CCU_LEVELS))
    p.add_argument("--trials", type=int, default=TRIALS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


async def async_main(ns: argparse.Namespace) -> int:
    contexts = [int(x) for x in ns.contexts.split(",") if x.strip()]
    ccu_levels = [int(x) for x in ns.ccu.split(",") if x.strip()]
    rows = await bench_config(
        base_url=ns.base_url,
        api_key=ns.api_key,
        model=ns.model,
        contexts=contexts,
        ccu_levels=ccu_levels,
        trials=ns.trials,
        seed=ns.seed,
    )
    payload = {"rows": rows}
    text = json.dumps(payload, indent=2)
    if ns.out:
        ns.out.parent.mkdir(parents=True, exist_ok=True)
        ns.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {ns.out}", file=sys.stderr)
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    return asyncio.run(async_main(ns))


if __name__ == "__main__":
    sys.exit(main())
