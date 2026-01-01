#!/usr/bin/env python3
"""
Enhanced Stress test script for self-hosted LLM server
This script sends concurrent requests to test server capacity with long context messages
Logs actual tokens returned from server to verify context setup
Uses Python's built-in asyncio for asynchronous operations
"""

import asyncio
import json
import time
import aiohttp
import argparse
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMStressTester:
    def __init__(self, server_url: str, concurrent_requests: int = 4, 
                 total_requests: int = 100, request_timeout: int = 30,
                 context_size: int = 40000, max_tokens: int = 150,
                 mode: Optional[str] = None, fixed_prefix: Optional[str] = None):
        self.server_url = server_url
        self.concurrent_requests = concurrent_requests
        self.total_requests = total_requests
        self.request_timeout = request_timeout
        self.context_size = context_size
        self.max_tokens = max_tokens
        self.mode = mode  # 'pp' for prompt processing, 'tg' for token generation
        self.fixed_prefix = fixed_prefix  # Fixed prefix for token generation mode
        
        # Store results
        self.results = []
        self.cache_warmed = False  # Track if cache has been warmed for -tg mode
        
    def generate_long_message(self, context_tokens: int) -> str:
        """Generate a long human message with random but meaningful words to prevent caching"""
        import random
        
        # List of random words to use for generating varied content
        random_words = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "computer", "science",
            "artificial", "intelligence", "machine", "learning", "deep", "neural", "network", "algorithm",
            "data", "analysis", "statistical", "model", "prediction", "accuracy", "performance", "optimization",
            "development", "programming", "software", "hardware", "architecture", "design", "implementation",
            "testing", "deployment", "monitoring", "maintenance", "security", "privacy", "encryption", "decryption",
            "database", "storage", "retrieval", "processing", "computation", "simulation", "experiment", "research",
            "discovery", "innovation", "technology", "application", "interface", "user", "experience", "design",
            "framework", "library", "module", "component", "integration", "compatibility", "scalability", "reliability"
        ]
        
        # Create random sentences with varied content
        # Approximate 1 token = 4 characters for English text
        target_chars = context_tokens * 4
        
        # Generate random sentences using the word list
        words = []
        while len(' '.join(words)) < target_chars:
            # Add a random number of words (between 1 and 10) to make it more varied
            num_words = random.randint(1, 10)
            for _ in range(num_words):
                words.append(random.choice(random_words))
        
        # Join words into sentences with proper punctuation
        sentence_parts = []
        current_sentence = []
        for i, word in enumerate(words):
            current_sentence.append(word)
            # Add period every 10-20 words to create natural sentences
            if (i + 1) % random.randint(10, 20) == 0:
                sentence_parts.append(' '.join(current_sentence) + '.')
                current_sentence = []
        
        # Add any remaining words to a final sentence
        if current_sentence:
            sentence_parts.append(' '.join(current_sentence) + '.')
        
        # Combine sentences with some structure
        structured_text = "User query: " + ' '.join(sentence_parts[:len(sentence_parts)//2]) + \
                         "\n\nAssistant response: " + ' '.join(sentence_parts[len(sentence_parts)//2:])
        
        # Trim to exact target character count
        if len(structured_text) > target_chars:
            structured_text = structured_text[:target_chars]
            
        return structured_text
    
    async def send_preflight_request(self, session: aiohttp.ClientSession) -> bool:
        """Send pre-flight request to warm up cache for token generation mode"""
        if self.cache_warmed:
            return True
            
        try:
            payload = {
                "model": "kCode",
                "messages": [
                    {
                        "role": "user",
                        "content": self.fixed_prefix
                    }
                ],
                "max_tokens": 1,  # Just need to process the prompt
                "temperature": 0.7
            }
            
            async with session.post(self.server_url, json=payload, timeout=aiohttp.ClientTimeout(total=self.request_timeout)) as response:
                await response.json()
                self.cache_warmed = True
                logger.info("Pre-flight cache warming request completed")
                return True
        except Exception as e:
            logger.warning(f"Pre-flight request failed: {str(e)}, continuing anyway...")
            return False
    
    async def send_request(self, session: aiohttp.ClientSession, request_id: int) -> Dict:
        """Send a single request and return timing and token information"""
        start_time = time.time()
        
        # Handle different modes
        if self.mode == 'pp':
            # Prompt processing mode: randomize every request, max_tokens=1
            long_message = self.generate_long_message(self.context_size)
            max_tokens = 1
        elif self.mode == 'tg':
            # Token generation mode: use fixed prefix (cache should already be warmed)
            if self.fixed_prefix is None:
                raise ValueError("fixed_prefix must be provided for token generation mode (-tg)")
            long_message = self.fixed_prefix
            max_tokens = self.max_tokens
        else:
            # Mixed mode (default): randomize prefix like pp, use full max_tokens like tg
            long_message = self.generate_long_message(self.context_size)
            max_tokens = self.max_tokens
        
        # Prepare request payload
        payload = {
            "model": "kCode",
            "messages": [
                {
                    "role": "user",
                    "content": long_message
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            # Send request
            async with session.post(self.server_url, json=payload, timeout=aiohttp.ClientTimeout(total=self.request_timeout)) as response:
                response_data = await response.json()
                
                # Parse token information from response
                prompt_tokens = response_data.get('usage', {}).get('prompt_tokens', 0)
                completion_tokens = response_data.get('usage', {}).get('completion_tokens', 0)
                total_tokens = response_data.get('usage', {}).get('total_tokens', 0)
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Calculate tokens per second based on mode
                if self.mode == 'pp':
                    # Prompt processing: calculate based on prompt tokens
                    tokens_per_sec = prompt_tokens / duration if duration > 0 else 0
                elif self.mode == 'tg':
                    # Token generation: calculate based on completion tokens (generation speed)
                    tokens_per_sec = completion_tokens / duration if duration > 0 else 0
                else:
                    # Mixed mode: calculate based on completion tokens (generation speed with randomized prefix)
                    tokens_per_sec = completion_tokens / duration if duration > 0 else 0
                
                result = {
                    "request_id": request_id,
                    "status": "SUCCESS",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "duration": duration,
                    "tokens_per_sec": tokens_per_sec
                }
                
                mode_label = "Prompt Processing" if self.mode == 'pp' else ("Token Generation" if self.mode == 'tg' else "Mixed")
                logger.info(f"Request {request_id}: SUCCESS [{mode_label}] "
                           f"(Prompt: {prompt_tokens}, Completion: {completion_tokens}, "
                           f"Total: {total_tokens}, Time: {duration:.3f}s, Tok/sec: {tokens_per_sec:.2f})")
                
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
                "tokens_per_sec": 0
            }
            
            logger.error(f"Request {request_id}: FAILED (Time: {duration:.3f}s, Error: {str(e)})")
            return result
    
    async def run_concurrent_requests(self) -> List[Dict]:
        """Run concurrent requests with semaphore for limiting concurrency"""
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(limit=self.concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        
        async with aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout
        ) as session:
            # Create a semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(self.concurrent_requests)
            
            async def limited_send_request(request_id):
                async with semaphore:
                    return await self.send_request(session, request_id)
            
            logger.info("Configuration:")
            logger.info(f"  Server URL: {self.server_url}")
            logger.info(f"  Mode: {self.mode.upper() if self.mode else 'MIXED'}")
            logger.info(f"  Concurrent Requests: {self.concurrent_requests}")
            logger.info(f"  Total Requests: {self.total_requests}")
            logger.info(f"  Context Size: ~{self.context_size} tokens")
            if self.mode == 'pp':
                logger.info(f"  Max Tokens per Request: 1 (Prompt Processing Mode)")
            else:
                logger.info(f"  Max Tokens per Request: {self.max_tokens}")
            if self.mode == 'tg' and self.fixed_prefix:
                logger.info(f"  Fixed Prefix: {self.fixed_prefix[:100]}..." if len(self.fixed_prefix) > 100 else f"  Fixed Prefix: {self.fixed_prefix}")
            elif self.mode != 'tg' and self.mode != 'pp':
                logger.info(f"  Prefix: Randomized (Mixed Mode)")
            logger.info(f"  Request Timeout: {self.request_timeout} seconds")
            logger.info("")
            
            # For token generation mode, send pre-flight request first to warm cache
            if self.mode == 'tg' and not self.cache_warmed:
                logger.info("Sending pre-flight request to warm cache...")
                await self.send_preflight_request(session)
            
            logger.info(f"Sending {self.concurrent_requests} concurrent requests, {self.total_requests} total...")
            
            # Create tasks for all requests
            tasks = [limited_send_request(i) for i in range(1, self.total_requests + 1)]
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and collect results
            for result in results:
                if not isinstance(result, Exception):
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
        
        # Calculate statistics
        stats = {
            "average_prompt_tokens": sum(prompt_tokens_list) / len(prompt_tokens_list),
            "average_completion_tokens": sum(completion_tokens_list) / len(completion_tokens_list),
            "average_total_tokens": sum(total_tokens_list) / len(total_tokens_list),
            "average_tokens_per_sec": sum(tokens_per_sec_list) / len(tokens_per_sec_list),
            "min_prompt_tokens": min(prompt_tokens_list),
            "max_prompt_tokens": max(prompt_tokens_list),
            "min_completion_tokens": min(completion_tokens_list),
            "max_completion_tokens": max(completion_tokens_list),
            "min_total_tokens": min(total_tokens_list),
            "max_total_tokens": max(total_tokens_list),
            "min_tokens_per_sec": min(tokens_per_sec_list),
            "max_tokens_per_sec": max(tokens_per_sec_list),
            "total_requests": len(self.results),
            "successful_requests": len(successful_results),
            "success_rate": (len(successful_results) / len(self.results)) * 100
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
        print(f"Average Prompt Tokens: {stats['average_prompt_tokens']:.0f}")
        print(f"Average Completion Tokens: {stats['average_completion_tokens']:.0f}")
        print(f"Average Total Tokens: {stats['average_total_tokens']:.0f}")
        if self.mode == 'pp':
            print(f"Average Prompt Processing Tokens/Second: {stats['average_tokens_per_sec']:.2f}")
        elif self.mode == 'tg':
            print(f"Average Generation Tokens/Second: {stats['average_tokens_per_sec']:.2f}")
        else:
            print(f"Average Generation Tokens/Second (Mixed Mode): {stats['average_tokens_per_sec']:.2f}")
        print(f"Min Prompt Tokens: {stats['min_prompt_tokens']}")
        print(f"Max Prompt Tokens: {stats['max_prompt_tokens']}")
        print(f"Min Completion Tokens: {stats['min_completion_tokens']}")
        print(f"Max Completion Tokens: {stats['max_completion_tokens']}")
        print(f"Min Total Tokens: {stats['min_total_tokens']}")
        print(f"Max Total Tokens: {stats['max_total_tokens']}")
        print(f"Min Tokens/Second: {stats['min_tokens_per_sec']:.2f}")
        print(f"Max Tokens/Second: {stats['max_tokens_per_sec']:.2f}")
        
        print(f"\nSuccess Rate: {stats['success_rate']:.2f}% ({stats['successful_requests']}/{stats['total_requests']} requests)")
        
        # Log summary for verification of context setup
        print("\n=== CONTEXT SETUP VERIFICATION ===")
        print("This section verifies that the context setup is working correctly by checking:")
        print("1. Prompt tokens are approximately correct (~6000 as configured)")
        print("2. Completion tokens are reasonable (should be around 350 as configured)")
        print("3. Total tokens are prompt + completion tokens")
        print("")
        
        # Show first few results to verify context setup
        print("Sample results (first 5 requests):")
        successful_results = [r for r in self.results if r.get('status') == 'SUCCESS']
        for result in successful_results[:5]:
            print(f"  Prompt: {result['prompt_tokens']}, Completion: {result['completion_tokens']}, "
                  f"Total: {result['total_tokens']}, Tok/sec: {result['tokens_per_sec']:.2f}")

async def main():
    parser = argparse.ArgumentParser(description='LLM Stress Test Script')
    parser.add_argument('--server-url', default='http://localhost:8000/v1/chat/completions',
                       help='Server URL (default: http://localhost:8000/v1/chat/completions)')
    parser.add_argument('-pp', '--prompt-processing', action='store_true',
                       help='Prompt processing mode: count prompt processing tok/sec, max_tokens=1, randomized requests')
    parser.add_argument('-tg', '--token-generation', action='store_true',
                       help='Token generation mode: use fixed prefix, pre-flight cache, measure generation speed')
    parser.add_argument('--concurrent-requests', type=int, default=4,
                       help='Number of concurrent requests (default: 3)')
    parser.add_argument('--total-requests', type=int, default=40,
                       help='Total number of requests to send (default: 50)')
    parser.add_argument('--request-timeout', type=int, default=180,
                       help='Request timeout in seconds (default: 180)')
    parser.add_argument('--context-size', type=int, default=6000,
                       help='Desired context window in tokens (default: 6000)')
    parser.add_argument('--max-tokens', type=int, default=350,
                       help='Maximum tokens to generate per request (default: 350, ignored in -pp mode)')
    parser.add_argument('--fixed-prefix', type=str, default=None,
                       help='Fixed prefix for token generation mode (-tg). If not provided, generates one based on context-size')
    
    args = parser.parse_args()
    
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
        # Generate fixed prefix if not provided
        if fixed_prefix is None:
            tester_temp = LLMStressTester(
                server_url=args.server_url,
                context_size=args.context_size
            )
            fixed_prefix = tester_temp.generate_long_message(args.context_size)
            logger.info(f"Generated fixed prefix of ~{args.context_size} tokens for token generation mode")
    else:
        # Default to mixed mode: randomized prefix with full max_tokens
        mode = 'mixed'
    
    # Create tester instance
    tester = LLMStressTester(
        server_url=args.server_url,
        concurrent_requests=args.concurrent_requests,
        total_requests=args.total_requests,
        request_timeout=args.request_timeout,
        context_size=args.context_size,
        max_tokens=args.max_tokens,
        mode=mode,
        fixed_prefix=fixed_prefix
    )
    
    # Run the stress test
    logger.info("Starting enhanced stress test for LLM server...")
    
    # Run asynchronously
    await tester.run_concurrent_requests()
    
    # Print final report
    tester.print_report()

if __name__ == "__main__":
    # Check if aiohttp is available
    try:
        import aiohttp
        asyncio.run(main())
    except ImportError:
        print("Error: aiohttp library is required for this script.")
        print("Please install it with: pip install aiohttp")
        exit(1)
