#!/usr/bin/env python3
"""
Enhanced Stress test script for self-hosted LLM server
This script sends concurrent requests to test server capacity with long context messages
Logs actual tokens returned from server to verify context setup
Uses Python's built-in asyncio for asynchronous operations
"""

import asyncio
import json
import math
import os
import random
import time
import aiohttp
import argparse
import logging
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent


def load_api_key(cli_key: Optional[str] = None) -> str:
    """Resolve API key from CLI, env, or vllm_args.sh (same key the server uses)."""
    if cli_key:
        return cli_key
    for env_name in ("LLM_API_KEY", "OPENAI_API_KEY"):
        val = os.environ.get(env_name)
        if val:
            return val
    args_path = SCRIPT_DIR / "vllm_args.sh"
    if args_path.is_file():
        for raw in args_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if parts[0] == "--api-key" and len(parts) == 2:
                return parts[1].strip().strip("'\"")
    return ""


def _delta_has_token(delta: Optional[Dict]) -> bool:
    if not isinstance(delta, dict):
        return False
    return bool(
        delta.get("content")
        or delta.get("reasoning_content")
        or delta.get("reasoning")
    )


SCOUT_WORDS = 480
FOCUS_SUFFIX = (
    "\n <FOCUS_HERE>Repeat previous sentence infinitively until I say stop.</FOCUS_HERE>"
)
RANDOM_WORDS = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "computer", "science",
    "artificial", "intelligence", "machine", "learning", "deep", "neural", "network", "algorithm",
    "data", "analysis", "statistical", "model", "prediction", "accuracy", "performance", "optimization",
    "development", "programming", "software", "hardware", "architecture", "design", "implementation",
    "testing", "deployment", "monitoring", "maintenance", "security", "privacy", "encryption", "decryption",
    "database", "storage", "retrieval", "processing", "computation", "simulation", "experiment", "research",
    "discovery", "innovation", "technology", "application", "interface", "user", "experience", "design",
    "framework", "library", "module", "component", "integration", "compatibility", "scalability", "reliability",
]


def generate_seed_text(rng: random.Random, n_words: int = SCOUT_WORDS) -> str:
    """Small sample used to scout tokenizer token counts, then scaled up."""
    parts = [rng.choice(RANDOM_WORDS) for _ in range(n_words)]
    for i in range(12, n_words, 17):
        parts[i] = parts[i] + rng.choice([".", ",", ";", ":"])
    return " ".join(parts) + "\n"


def scale_text(seed: str, copies: int, target_chars: Optional[int] = None) -> str:
    if not seed:
        raise ValueError("seed text is empty")
    body = seed * max(1, copies)
    if target_chars is None:
        return body
    target_chars = max(len(seed), target_chars)
    if len(body) < target_chars:
        more = math.ceil((target_chars - len(body)) / len(seed))
        body = body + seed * more
    return body[:target_chars]


def _generate_seed_worker(seed: int) -> str:
    return generate_seed_text(random.Random(seed))


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMStressTester:
    def __init__(self, server_url: str, ccu: int = 4, 
                 total_requests: int = 100, request_timeout: int = 300,
                 ctx: int = 40000, max_tokens: int = 150,
                 mode: Optional[str] = None, fixed_prefix: Optional[str] = None,
                 num_workers: Optional[int] = None, api_key: Optional[str] = None):
        self.server_url = server_url
        self.ccu = ccu
        self.total_requests = total_requests
        self.request_timeout = request_timeout
        self.ctx = ctx
        self.max_tokens = max_tokens
        self.mode = mode  # 'pp' for prompt processing, 'tg' for token generation
        self.fixed_prefix = fixed_prefix  # Fixed prefix for token generation mode
        self.num_workers = num_workers  # Number of worker processes for message generation
        self.api_key = api_key or ""
        self.results = []
        self.cache_warmed = False  # Track if cache has been warmed for -tg mode
        self.pre_generated_messages = []  # Store pre-generated random messages
        self.wall_time = 0.0
        self.scout_copies = 1
        self.scout_target_chars = 0
        self.calibrated_prompt_tokens = 0

    def _auth_headers(self) -> Dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def _client_timeout(self) -> aiohttp.ClientTimeout:
        # Idle timeout between SSE chunks, not a wall-clock cap on the whole
        # generation. That keeps long decode from timing out while still
        # failing if the server goes silent.
        connect = min(30, self.request_timeout)
        return aiohttp.ClientTimeout(
            total=None,
            connect=connect,
            sock_connect=connect,
            sock_read=self.request_timeout,
        )
        
    def _full_content(self, prefix: str) -> str:
        return prefix + FOCUS_SUFFIX

    def _chat_payload(self, content: str, max_tokens: int) -> Dict:
        return {
            "model": "kCode",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "ignore_eos": True,
        }

    def generate_long_message(self, context_tokens: int) -> str:
        """Fallback local scale when a scout has not run yet. Prefer calibrate_prefix()."""
        seed = generate_seed_text(random.Random())
        copies = max(1, math.ceil(max(1, context_tokens) / SCOUT_WORDS))
        return scale_text(seed, copies)
    
    async def _count_prompt_tokens(self, session: aiohttp.ClientSession, prefix: str) -> int:
        """Scout request: max_tokens=1, return server-reported prompt tokens."""
        payload = self._chat_payload(self._full_content(prefix), max_tokens=1)
        async with session.post(self.server_url, json=payload) as response:
            data = await self._read_api_response(response)
        tokens = (data.get("usage") or {}).get("prompt_tokens") or 0
        if int(tokens) <= 0:
            raise RuntimeError(f"scout request returned no prompt_tokens: {data}")
        return int(tokens)

    async def calibrate_prefix(
        self,
        session: aiohttp.ClientSession,
        target: int,
        seed: Optional[str] = None,
    ) -> Tuple[str, int]:
        """Scout a small sample, then scale copies/length to ~target prompt tokens."""
        if seed is None:
            seed = generate_seed_text(random.Random())
        tol = max(64, int(target * 0.01))
        logger.info(
            f"Scouting token count with a {len(seed.split())}-word sample "
            f"(target {target} prompt tokens)..."
        )
        n1 = await self._count_prompt_tokens(session, seed)
        logger.info(f"Scout sample: {n1} prompt tokens")
        if n1 >= target - tol:
            body = seed
            actual = n1
            if n1 > target + tol:
                new_len = max(1, int(len(seed) * (target / n1)))
                body = seed[:new_len]
                actual = await self._count_prompt_tokens(session, body)
                logger.info(f"Trimmed scout sample to {new_len} chars: {actual} prompt tokens")
            self.scout_copies = 1
            self.scout_target_chars = len(body)
            self.calibrated_prompt_tokens = actual
            return body, actual

        n2 = await self._count_prompt_tokens(session, seed + seed)
        logger.info(f"Scout doubled sample: {n2} prompt tokens")
        per_copy = n2 - n1
        if per_copy <= 0:
            per_copy = n1
            overhead = 0
        else:
            overhead = max(0, n1 - per_copy)
        need = max(1, target - overhead)
        copies = max(1, math.ceil(need / per_copy))
        body = scale_text(seed, copies)
        actual = await self._count_prompt_tokens(session, body)
        logger.info(f"Scaled to {copies} copies: {actual} prompt tokens (target {target})")
        if actual > 0 and abs(actual - target) > tol:
            scale = target / actual
            new_len = max(len(seed), int(len(body) * scale))
            body = scale_text(seed, copies, new_len)
            actual = await self._count_prompt_tokens(session, body)
            logger.info(f"Adjusted to {new_len} chars: {actual} prompt tokens")
        self.scout_copies = copies
        self.scout_target_chars = len(body)
        self.calibrated_prompt_tokens = actual
        return body, actual

    async def pre_generate_messages(
        self, session: aiohttp.ClientSession, num_messages: int, num_workers: Optional[int] = None
    ):
        """Scout once, then generate unique prefixes scaled to the target context."""
        if num_workers is None:
            num_workers = self.num_workers if self.num_workers is not None else multiprocessing.cpu_count()

        await self.calibrate_prefix(session, self.ctx)
        copies = self.scout_copies
        target_chars = self.scout_target_chars
        logger.info(
            f"Pre-generating {num_messages} unique prefixes "
            f"({copies} copies, {target_chars} chars) using {num_workers} processes..."
        )
        start_time = time.time()
        args_list = [random.randint(0, 2**31) for _ in range(num_messages)]
        with multiprocessing.Pool(processes=num_workers) as pool:
            seeds = pool.map(_generate_seed_worker, args_list)
        self.pre_generated_messages = [
            scale_text(seed, copies, target_chars) for seed in seeds
        ]
        elapsed_time = time.time() - start_time
        logger.info(
            f"Pre-generated {num_messages} messages in {elapsed_time:.2f} seconds "
            f"(scouted {self.calibrated_prompt_tokens} prompt tokens)"
        )

    async def _read_api_response(self, response: aiohttp.ClientResponse) -> Dict:
        """Parse JSON and raise if the API request failed (HTTP or error body)."""
        body = await response.text()
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}
        if response.status >= 400 or not isinstance(data, dict) or "error" in data:
            snippet = body[:500] if body else "(empty body)"
            error = data.get("error") if isinstance(data, dict) else None
            detail = error if error is not None else snippet
            raise RuntimeError(f"API request failed (HTTP {response.status}): {detail}")
        return data

    async def _stream_chat_completion(
        self, session: aiohttp.ClientSession, payload: Dict, start_time: float
    ) -> Dict:
        """Stream tokens (SSE) so tok/s uses TTFT and long generations do not hit total timeout."""
        stream_payload = {
            **payload,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        async with session.post(self.server_url, json=stream_payload) as response:
            if response.status >= 400:
                await self._read_api_response(response)
                raise RuntimeError(f"API request failed (HTTP {response.status})")

            prompt_tokens = 0
            completion_tokens = 0
            counted_completion = 0
            total_tokens = 0
            ttft = None
            saw_done = False
            saw_chunk = False

            async for raw in response.content:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    saw_done = True
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid SSE chunk: {data_str[:300]}") from exc
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error"):
                    raise RuntimeError(f"API request failed: {chunk['error']}")
                saw_chunk = True
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if _delta_has_token(delta):
                        if ttft is None:
                            ttft = time.time() - start_time
                        counted_completion += 1
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens") or prompt_tokens
                    if usage.get("completion_tokens") is not None:
                        completion_tokens = usage.get("completion_tokens") or 0
                    total_tokens = usage.get("total_tokens") or total_tokens

            e2e = time.time() - start_time
            if not saw_chunk and not saw_done:
                raise RuntimeError("Stream ended with no SSE data")
            if not completion_tokens:
                completion_tokens = counted_completion
            if ttft is None:
                ttft = e2e
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens or (prompt_tokens + completion_tokens),
                "ttft": ttft,
                "e2e": e2e,
            }
    
    async def send_preflight_request(self, session: aiohttp.ClientSession) -> bool:
        """Send pre-flight request to warm up cache for token generation mode"""
        if self.cache_warmed:
            return True
            
        try:
            payload = self._chat_payload(self._full_content(self.fixed_prefix), max_tokens=1)
            
            async with session.post(self.server_url, json=payload) as response:
                await self._read_api_response(response)
                self.cache_warmed = True
                logger.info("Pre-flight cache warming request completed")
                return True
        except Exception as e:
            logger.error(f"Pre-flight request failed: {str(e)}")
            raise
    
    async def send_request(self, session: aiohttp.ClientSession, request_id: int) -> Dict:
        """Send a single request and return timing and token information"""
        start_time = time.time()
        
        # Handle different modes
        if self.mode == 'pp':
            # Prompt processing mode: randomize every request, max_tokens=1
            if self.pre_generated_messages and request_id <= len(self.pre_generated_messages):
                long_message = self.pre_generated_messages[request_id - 1]
            else:
                long_message = scale_text(
                    generate_seed_text(random.Random()),
                    self.scout_copies,
                    self.scout_target_chars or None,
                )
            max_tokens = 1
        elif self.mode == 'tg':
            # Token generation mode: use fixed prefix (cache should already be warmed)
            if self.fixed_prefix is None:
                raise ValueError("fixed_prefix must be provided for token generation mode (-tg)")
            long_message = self.fixed_prefix
            max_tokens = self.max_tokens
        else:
            # Mixed mode (default): randomize prefix like pp, use full max_tokens like tg
            if self.pre_generated_messages and request_id <= len(self.pre_generated_messages):
                long_message = self.pre_generated_messages[request_id - 1]
            else:
                long_message = scale_text(
                    generate_seed_text(random.Random()),
                    self.scout_copies,
                    self.scout_target_chars or None,
                )
            max_tokens = self.max_tokens
        
        payload = self._chat_payload(self._full_content(long_message), max_tokens)
        
        try:
            response_data = await self._stream_chat_completion(session, payload, start_time)
            prompt_tokens = response_data["prompt_tokens"]
            completion_tokens = response_data["completion_tokens"]
            total_tokens = response_data["total_tokens"]
            ttft = response_data["ttft"]
            duration = response_data["e2e"]

            if self.mode == "pp":
                denom = ttft if ttft > 0 else duration
                tokens_per_sec = prompt_tokens / denom if denom > 0 else 0
            else:
                decode_s = duration - ttft
                if completion_tokens > 1 and decode_s > 0:
                    tokens_per_sec = (completion_tokens - 1) / decode_s
                else:
                    tokens_per_sec = 0

            result = {
                "request_id": request_id,
                "status": "SUCCESS",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "duration": duration,
                "ttft": ttft,
                "tokens_per_sec": tokens_per_sec,
            }

            mode_label = "Prompt Processing" if self.mode == "pp" else ("Token Generation" if self.mode == "tg" else "Mixed")
            logger.info(
                f"Request {request_id}: SUCCESS [{mode_label}] "
                f"(Prompt: {prompt_tokens}, Completion: {completion_tokens}, "
                f"Total: {total_tokens}, TTFT: {ttft:.3f}s, Time: {duration:.3f}s, "
                f"Tok/sec: {tokens_per_sec:.2f})"
            )

            return result

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time

            result = {
                "request_id": request_id,
                "status": "FAILED",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "duration": duration,
                "ttft": 0,
                "tokens_per_sec": 0,
            }

            logger.error(f"Request {request_id}: FAILED (Time: {duration:.3f}s, Error: {str(e)})")
            return result
    
    async def run_ccu(self) -> List[Dict]:
        """Run concurrent requests with semaphore for limiting concurrency"""
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(limit=self.ccu)
        timeout = self._client_timeout()
        
        async with aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout,
            headers=self._auth_headers(),
        ) as session:
            # Create a semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(self.ccu)
            
            async def limited_send_request(request_id):
                async with semaphore:
                    return await self.send_request(session, request_id)
            
            logger.info("Configuration:")
            logger.info(f"  Server URL: {self.server_url}")
            logger.info(f"  API Key: {'set' if self.api_key else 'MISSING'}")
            logger.info(f"  Mode: {self.mode.upper() if self.mode else 'MIXED'}")
            logger.info(f"  Concurrent Requests: {self.ccu}")
            logger.info(f"  Total Requests: {self.total_requests}")
            logger.info(f"  Context Size: ~{self.ctx} tokens (scout + scale)")
            if self.mode == 'pp':
                logger.info(f"  Max Tokens per Request: 1 (Prompt Processing Mode)")
            else:
                logger.info(f"  Max Tokens per Request: {self.max_tokens}")
            logger.info("  ignore_eos: enabled")
            if self.mode == 'tg' and self.fixed_prefix:
                logger.info(f"  Fixed Prefix: {self.fixed_prefix[:100]}..." if len(self.fixed_prefix) > 100 else f"  Fixed Prefix: {self.fixed_prefix}")
            elif self.mode != 'tg' and self.mode != 'pp':
                logger.info(f"  Prefix: Randomized (Mixed Mode)")
            logger.info(f"  Request Timeout: {self.request_timeout} seconds idle between stream chunks")
            logger.info("  Streaming: enabled")
            if self.mode in ('pp', 'mixed'):
                workers_info = self.num_workers if self.num_workers is not None else multiprocessing.cpu_count()
                logger.info(f"  Message Generation Workers: {workers_info}")
            logger.info("")
            
            if self.mode in ('pp', 'mixed'):
                await self.pre_generate_messages(session, self.total_requests)
            
            if self.mode == 'tg':
                seed = self.fixed_prefix
                self.fixed_prefix, actual = await self.calibrate_prefix(
                    session, self.ctx, seed=seed
                )
                logger.info(f"Calibrated -tg prefix to {actual} prompt tokens")
                if not self.cache_warmed:
                    logger.info("Sending pre-flight request to warm cache...")
                    await self.send_preflight_request(session)
            
            logger.info(f"Sending {self.ccu} concurrent requests, {self.total_requests} total...")
            
            # Create tasks for all requests
            tasks = [limited_send_request(i) for i in range(1, self.total_requests + 1)]
            
            wall_start = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            self.wall_time = time.time() - wall_start
            
            for i, result in enumerate(results, start=1):
                if isinstance(result, Exception):
                    logger.error(f"Request {i}: FAILED (Error: {result})")
                    self.results.append({
                        "request_id": i,
                        "status": "FAILED",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "duration": 0,
                        "ttft": 0,
                        "tokens_per_sec": 0,
                    })
                else:
                    self.results.append(result)
            
            return self.results
    
    def calculate_statistics(self) -> Dict:
        """Calculate statistics from results"""
        successful_results = [r for r in self.results if r.get('status') == 'SUCCESS']
        
        if not successful_results:
            return {}
        
        # Extract token values
        prompt_tokens_list = [r['prompt_tokens'] for r in successful_results]
        completion_tokens_list = [r['completion_tokens'] for r in successful_results]
        total_tokens_list = [r['total_tokens'] for r in successful_results]
        tokens_per_sec_list = [r['tokens_per_sec'] for r in successful_results]
        ttft_list = [r['ttft'] for r in successful_results if r.get('ttft') is not None]
        sum_prompt = sum(prompt_tokens_list)
        sum_completion = sum(completion_tokens_list)
        sum_total = sum(total_tokens_list)
        
        # Calculate statistics
        stats = {
            "average_prompt_tokens": sum(prompt_tokens_list) / len(prompt_tokens_list),
            "average_completion_tokens": sum(completion_tokens_list) / len(completion_tokens_list),
            "average_total_tokens": sum(total_tokens_list) / len(total_tokens_list),
            "average_tokens_per_sec": sum(tokens_per_sec_list) / len(tokens_per_sec_list),
            "average_ttft": (sum(ttft_list) / len(ttft_list)) if ttft_list else 0,
            "min_prompt_tokens": min(prompt_tokens_list),
            "max_prompt_tokens": max(prompt_tokens_list),
            "min_completion_tokens": min(completion_tokens_list),
            "max_completion_tokens": max(completion_tokens_list),
            "min_total_tokens": min(total_tokens_list),
            "max_total_tokens": max(total_tokens_list),
            "min_tokens_per_sec": min(tokens_per_sec_list),
            "max_tokens_per_sec": max(tokens_per_sec_list),
            "min_ttft": min(ttft_list) if ttft_list else 0,
            "max_ttft": max(ttft_list) if ttft_list else 0,
            "total_requests": len(self.results),
            "successful_requests": len(successful_results),
            "success_rate": (len(successful_results) / len(self.results)) * 100,
            "wall_time": self.wall_time,
            "sum_prompt_tokens": sum_prompt,
            "sum_completion_tokens": sum_completion,
            "sum_total_tokens": sum_total,
            "prefill_throughput": sum_prompt / self.wall_time if self.wall_time > 0 else 0,
            "decode_throughput": sum_completion / self.wall_time if self.wall_time > 0 else 0,
            "total_throughput": sum_total / self.wall_time if self.wall_time > 0 else 0,
            "request_throughput": len(successful_results) / self.wall_time if self.wall_time > 0 else 0,
        }
        
        return stats
    
    def print_report(self):
        """Print detailed report"""
        print("\n=== FINAL REPORT ===")
        print("Processing results...")
        
        stats = self.calculate_statistics()
        
        if not stats:
            print("No successful results found. Please check if the test ran correctly.")
            return
        
        print("\nDetailed Token Statistics:")
        print("--------------------------")
        mode_label = "Prompt Processing" if self.mode == 'pp' else ("Token Generation" if self.mode == 'tg' else "Mixed")
        print(f"Mode: {mode_label}")
        if self.calibrated_prompt_tokens:
            print(f"Calibrated Prompt Tokens: {self.calibrated_prompt_tokens} (target {self.ctx})")
        print(f"Average Prompt Tokens: {stats['average_prompt_tokens']:.0f}")
        print(f"Average Completion Tokens: {stats['average_completion_tokens']:.0f}")
        print(f"Average Total Tokens: {stats['average_total_tokens']:.0f}")
        print(f"Average TTFT: {stats['average_ttft']:.3f}s")
        if self.mode == 'pp':
            print(f"Average Prompt Processing Tokens/Second: {stats['average_tokens_per_sec']:.2f}")
        elif self.mode == 'tg':
            print(f"Average Decode Tokens/Second: {stats['average_tokens_per_sec']:.2f}")
        else:
            print(f"Average Decode Tokens/Second (Mixed Mode): {stats['average_tokens_per_sec']:.2f}")
        print(f"Min Prompt Tokens: {stats['min_prompt_tokens']}")
        print(f"Max Prompt Tokens: {stats['max_prompt_tokens']}")
        print(f"Min Completion Tokens: {stats['min_completion_tokens']}")
        print(f"Max Completion Tokens: {stats['max_completion_tokens']}")
        print(f"Min Total Tokens: {stats['min_total_tokens']}")
        print(f"Max Total Tokens: {stats['max_total_tokens']}")
        print(f"Min Tokens/Second: {stats['min_tokens_per_sec']:.2f}")
        print(f"Max Tokens/Second: {stats['max_tokens_per_sec']:.2f}")
        print(f"Min TTFT: {stats['min_ttft']:.3f}s")
        print(f"Max TTFT: {stats['max_ttft']:.3f}s")

        print("\nThroughput (successful tokens / wall time):")
        print("------------------------------------------")
        print(f"Wall Time: {stats['wall_time']:.3f}s")
        print(f"Prefill Throughput: {stats['prefill_throughput']:.2f} tok/s "
              f"({stats['sum_prompt_tokens']} prompt tokens)")
        print(f"Decode Throughput: {stats['decode_throughput']:.2f} tok/s "
              f"({stats['sum_completion_tokens']} completion tokens)")
        print(f"Total Throughput: {stats['total_throughput']:.2f} tok/s "
              f"({stats['sum_total_tokens']} tokens)")
        print(f"Request Throughput: {stats['request_throughput']:.2f} req/s")
        
        print(f"\nSuccess Rate: {stats['success_rate']:.2f}% ({stats['successful_requests']}/{stats['total_requests']} requests)")
        
        # Log summary for verification of context setup
        print("\n=== CONTEXT SETUP VERIFICATION ===")
        print("This section verifies that the context setup is working correctly by checking:")
        print("1. Prompt tokens match the scouted context size")
        print("2. Completion tokens reach max_tokens (ignore_eos avoids early stop)")
        print("3. Total tokens are prompt + completion tokens")
        print("")
        
        # Show first few results to verify context setup
        print("Sample results (first 5 requests):")
        successful_results = [r for r in self.results if r.get('status') == 'SUCCESS']
        for result in successful_results[:5]:
            print(f"  Prompt: {result['prompt_tokens']}, Completion: {result['completion_tokens']}, "
                  f"Total: {result['total_tokens']}, TTFT: {result.get('ttft', 0):.3f}s, "
                  f"Tok/sec: {result['tokens_per_sec']:.2f}")

async def main():
    parser = argparse.ArgumentParser(description='LLM Stress Test Script')
    parser.add_argument('--server-url', default='http://localhost:8000/v1/chat/completions',
                       help='Server URL (default: http://localhost:8000/v1/chat/completions)')
    parser.add_argument('-pp', '--prompt-processing', action='store_true',
                       help='Prompt processing mode: count prompt processing tok/sec, max_tokens=1, randomized requests')
    parser.add_argument('-tg', '--token-generation', action='store_true',
                       help='Token generation mode: use fixed prefix, pre-flight cache, measure generation speed')
    parser.add_argument('-ccu', type=int, default=6,
                       help='Number of concurrent requests (default: 6)')
    parser.add_argument('--total-requests', type=int, default=25,
                       help='Total number of requests to send (default: 25)')
    parser.add_argument('--request-timeout', type=int, default=300,
                       help='Idle timeout in seconds between streamed chunks, and for scout/prefill (default: 300). Not a cap on total generation time.')
    parser.add_argument('-ctx', type=int, default=70000,
                       help='Desired prompt tokens. A small sample is scouted, then scaled to this length (default: 70000)')
    parser.add_argument('--max-tokens', type=int, default=500,
                       help='Maximum tokens to generate per request (default: 2000, ignored in -pp mode)')
    parser.add_argument('--fixed-prefix', type=str, default=None,
                       help='Fixed prefix for token generation mode (-tg). If not provided, generates one based on context-size')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes for generating random sequences (default: number of CPU cores)')
    parser.add_argument('--api-key', default=None,
                       help='API key for Authorization header. Defaults to VLLM_API_KEY, OPENAI_API_KEY, or --api-key from vllm_args.sh')
    
    args = parser.parse_args()
    api_key = load_api_key(args.api_key)
    if not api_key:
        logger.error("No API key found. Pass --api-key, set VLLM_API_KEY/OPENAI_API_KEY, or add --api-key to vllm_args.sh")
        raise SystemExit(1)
    
    # Determine mode
    mode = None
    fixed_prefix = args.fixed_prefix
    
    if args.prompt_processing and args.token_generation:
        logger.error("Cannot use both -pp and -tg modes simultaneously")
        return
    
    if args.prompt_processing:
        mode = 'pp'
    elif args.token_generation:
        mode = 'tg'
    else:
        # Default to mixed mode: randomized prefix with full max_tokens
        mode = 'mixed'
    
    # Create tester instance
    tester = LLMStressTester(
        server_url=args.server_url,
        ccu=args.ccu,
        total_requests=args.total_requests,
        request_timeout=args.request_timeout,
        ctx=args.ctx,
        max_tokens=args.max_tokens,
        mode=mode,
        fixed_prefix=fixed_prefix,
        num_workers=args.workers,
        api_key=api_key,
    )
    
    # Run the stress test
    logger.info("Starting enhanced stress test for LLM server...")
    
    # Run asynchronously
    await tester.run_ccu()
    
    # Print final report
    tester.print_report()

    failed = [r for r in tester.results if r.get("status") != "SUCCESS"]
    if failed:
        raise RuntimeError(
            f"{len(failed)}/{len(tester.results)} API requests failed"
        )

if __name__ == "__main__":
    # Check if aiohttp is available
    try:
        import aiohttp
        asyncio.run(main())
    except ImportError:
        print("Error: aiohttp library is required for this script.")
        print("Please install it with: pip install aiohttp")
        exit(1)
