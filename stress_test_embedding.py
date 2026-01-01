#!/usr/bin/env python3
"""
Simple Stress test script for self-hosted embedding server
This script sends concurrent requests to test server capacity with long context messages
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

class EmbeddingStressTester:
    def __init__(self, server_url: str, concurrent_requests: int = 4, 
                 total_requests: int = 100, request_timeout: int = 30,
                 context_size: int = 40000):
        self.server_url = server_url
        self.concurrent_requests = concurrent_requests
        self.total_requests = total_requests
        self.request_timeout = request_timeout
        self.context_size = context_size
        
        # Store results
        self.results = []
        # Track total tokens and time for system-wide metrics
        self.total_prompt_tokens = 0
        self.total_duration = 0
        
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
    
    async def send_request(self, session: aiohttp.ClientSession, request_id: int) -> Dict:
        """Send a single request and return timing and token information"""
        start_time = time.time()
        
        # Generate message with desired context size
        long_message = self.generate_long_message(self.context_size)
        
        # Prepare request payload for embedding endpoint
        payload = {
            "model": "kCodeEmbedding",
            "input": long_message
        }
        
        try:
            # Send request
            async with session.post(self.server_url, json=payload, timeout=aiohttp.ClientTimeout(total=self.request_timeout)) as response:
                response_data = await response.json()
                
                # Parse token information from response (embedding responses include usage info)
                prompt_tokens = response_data.get('usage', {}).get('prompt_tokens', 0)
                total_tokens = response_data.get('usage', {}).get('total_tokens', 0)
                completion_tokens = 0  # No completion tokens in embedding responses
                
                # If no usage info, we'll set to 0 or calculate from input length
                if prompt_tokens == 0 and total_tokens == 0:
                    # Estimate tokens from input length (approximate)
                    prompt_tokens = len(long_message.split())  # rough estimate
                    total_tokens = prompt_tokens
                    completion_tokens = 0
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Calculate tokens processed per second (prompt_tokens / duration)
                tokens_per_sec = prompt_tokens / duration if duration > 0 else 0
                
                # Update system-wide metrics
                self.total_prompt_tokens += prompt_tokens
                self.total_duration += duration
                
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
            logger.info(f"  Concurrent Requests: {self.concurrent_requests}")
            logger.info(f"  Total Requests: {self.total_requests}")
            logger.info(f"  Context Size: ~{self.context_size} tokens")
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
        
        # Add system-wide metrics
        if self.total_duration > 0:
            stats["system_total_prompt_tokens"] = self.total_prompt_tokens
            stats["system_total_duration"] = self.total_duration
            stats["system_tokens_per_sec"] = self.total_prompt_tokens / self.total_duration
        
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
        
        # Add system-wide metrics if available
        if "system_total_prompt_tokens" in stats:
            print(f"\nSystem-wide Metrics:")
            print(f"  Total Prompt Tokens Processed: {stats['system_total_prompt_tokens']:.0f}")
            print(f"  Total Duration: {stats['system_total_duration']:.2f} seconds")
            print(f"  System Tokens/Second: {stats['system_tokens_per_sec']:.2f}")
        
        print(f"\nSuccess Rate: {stats['success_rate']:.2f}% ({stats['successful_requests']}/{stats['total_requests']} requests)")
        
        # Show first few results to verify context setup
        print("\nSample results (first 5 requests):")
        successful_results = [r for r in self.results if r.get('status') == 'SUCCESS']
        for result in successful_results[:5]:
            print(f"  Prompt: {result['prompt_tokens']}, Completion: {result['completion_tokens']}, "
                  f"Total: {result['total_tokens']}, Tok/sec: {result['tokens_per_sec']:.2f}")

async def main():
    parser = argparse.ArgumentParser(description='Embedding Stress Test Script')
    parser.add_argument('--server-url', default='http://localhost:8001/v1/embeddings',
                       help='Server URL (default: http://localhost:8001/v1/embeddings)')
    parser.add_argument('--concurrent-requests', type=int, default=50,
                       help='Number of concurrent requests (default: 50)')
    parser.add_argument('--total-requests', type=int, default=200,
                       help='Total number of requests to send (default: 200)')
    parser.add_argument('--request-timeout', type=int, default=180,
                       help='Request timeout in seconds (default: 180)')
    parser.add_argument('--context-size', type=int, default=6000,
                       help='Desired context window in tokens (default: 6000)')
    
    args = parser.parse_args()
    
    # Create tester instance
    tester = EmbeddingStressTester(
        server_url=args.server_url,
        concurrent_requests=args.concurrent_requests,
        total_requests=args.total_requests,
        request_timeout=args.request_timeout,
        context_size=args.context_size
    )
    
    # Run the stress test
    logger.info("Starting stress test for embedding server...")
    
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