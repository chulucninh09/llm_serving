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
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMStressTester:
    def __init__(self, server_url: str, concurrent_requests: int = 4, 
                 total_requests: int = 100, request_timeout: int = 30,
                 context_size: int = 40000, max_tokens: int = 150):
        self.server_url = server_url
        self.concurrent_requests = concurrent_requests
        self.total_requests = total_requests
        self.request_timeout = request_timeout
        self.context_size = context_size
        self.max_tokens = max_tokens
        
        # Store results
        self.results = []
        
    def generate_long_message(self, context_tokens: int) -> str:
        """Generate a long human message to approximate context size"""
        base_prompt = "Explain the concept of attention mechanism in transformers. Be about 500 words."
        
        # Create a long message by repeating and expanding the base prompt
        # Approximate 1 token = 4 characters for English text
        target_chars = context_tokens * 4
        base_chars = len(base_prompt)
        
        # Calculate how many times we need to repeat the base prompt
        repetitions = target_chars // base_chars
        
        # Build the long message
        long_message = base_prompt
        for i in range(repetitions):
            long_message += f". This is additional content to increase the context size. The attention mechanism in transformers allows the model to focus on different parts of the input sequence when generating each token. It's a crucial component that enables models to handle long-range dependencies in text. The mechanism works by computing attention scores between all positions in the sequence, allowing the model to weigh the importance of different words when generating each output token. This is particularly important for understanding context in long documents or conversations. The scaled dot-product attention is the most common variant used in transformer architectures. It computes attention weights by taking the dot product of query and key vectors, scaling by the square root of the key dimension, and applying softmax to obtain probability distributions. These probabilities are then used to weight the value vectors to produce the final output. This attention mechanism is what makes transformers so effective at understanding relationships between words regardless of their distance in the sequence."
        
        # Trim to approximate target length
        trimmed_message = long_message[:target_chars]
        
        # Add a safety check to prevent extremely long messages
        if len(trimmed_message) > 100000:
            trimmed_message = trimmed_message[:100000]
            
        return trimmed_message
    
    async def send_request(self, session: aiohttp.ClientSession, request_id: int) -> Dict:
        """Send a single request and return timing and token information"""
        start_time = time.time()
        
        # Generate a long message to approximate context size
        long_message = self.generate_long_message(self.context_size)
        
        # Prepare request payload
        payload = {
            "model": "kCode",
            "messages": [
                {
                    "role": "user",
                    "content": long_message
                }
            ],
            "max_tokens": self.max_tokens,
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
                
                # Calculate tokens per second
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
                
                logger.info(f"Request {request_id}: SUCCESS "
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
        # Create a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.concurrent_requests)
        
        async def limited_send_request(request_id):
            async with semaphore:
                return await self.send_request(session, request_id)
        
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(limit=self.concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        
        async with aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout
        ) as session:
            logger.info("Configuration:")
            logger.info(f"  Server URL: {self.server_url}")
            logger.info(f"  Concurrent Requests: {self.concurrent_requests}")
            logger.info(f"  Total Requests: {self.total_requests}")
            logger.info(f"  Context Size: ~{self.context_size} tokens")
            logger.info(f"  Max Tokens per Request: {self.max_tokens}")
            logger.info(f"  Request Timeout: {self.request_timeout} seconds")
            logger.info("")
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
        print(f"Average Prompt Tokens: {stats['average_prompt_tokens']:.0f}")
        print(f"Average Completion Tokens: {stats['average_completion_tokens']:.0f}")
        print(f"Average Total Tokens: {stats['average_total_tokens']:.0f}")
        print(f"Average Tokens/Second: {stats['average_tokens_per_sec']:.2f}")
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
    parser.add_argument('--concurrent-requests', type=int, default=3,
                       help='Number of concurrent requests (default: 3)')
    parser.add_argument('--total-requests', type=int, default=50,
                       help='Total number of requests to send (default: 50)')
    parser.add_argument('--request-timeout', type=int, default=180,
                       help='Request timeout in seconds (default: 180)')
    parser.add_argument('--context-size', type=int, default=6000,
                       help='Desired context window in tokens (default: 6000)')
    parser.add_argument('--max-tokens', type=int, default=350,
                       help='Maximum tokens to generate per request (default: 350)')
    
    args = parser.parse_args()
    
    # Create tester instance
    tester = LLMStressTester(
        server_url=args.server_url,
        concurrent_requests=args.concurrent_requests,
        total_requests=args.total_requests,
        request_timeout=args.request_timeout,
        context_size=args.context_size,
        max_tokens=args.max_tokens
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
