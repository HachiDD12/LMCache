#!/usr/bin/env python3
"""
Cache simulation script for blend server.
Models caching behavior by processing requests in order and tracking cache hits.
"""

import argparse
import json
import sys
from typing import List, Dict, Tuple
import statistics

# Third Party
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


class CacheSimulator:
    def __init__(self, model: str, blend_special_str: str = " # # "):
        """Initialize the cache simulator with the same tokenizer as the server"""
        self.model = model
        self.blend_special_str = blend_special_str
        
        # Initialize tokenizer (same as server)
        self.tokenizer = MistralTokenizer.from_hf_hub(model)
        self.tekkenizer = self.tokenizer.instruct_tokenizer.tokenizer
        
        # Pre-encode the blend special string
        self.blend_tokens = self.tekkenizer.encode(self.blend_special_str, bos=False, eos=False)
        
        # Cache: maps chunk token sequence (as tuple) to whether it's been seen
        self.cache: Dict[Tuple[int, ...], bool] = {}
        
        # Statistics
        self.request_stats: List[Dict] = []
    
    def get_stitched_tokens_for_msg_content(self, msg_content: str) -> List[Tuple[List[int], int]]:
        """
        Get stitched tokens for a single message content.
        Matches server's get_stitched_tokens_for_msg_content logic.
        Returns list of (chunk_tokens, chunk_length) tuples.
        """
        chunks_data = []
        # Split on blend special string
        msg_chunks = msg_content.split(self.blend_special_str)
        
        for chunk in msg_chunks:
            chunk_tokens = self.tekkenizer.encode(chunk, bos=False, eos=False)
            chunk_length = len(chunk_tokens)
            chunks_data.append((chunk_tokens, chunk_length))
        
        return chunks_data
    
    def process_request(self, request_data: Dict, request_id: str) -> Dict:
        """
        Process a single request and return cache statistics.
        Matches server's messages_to_prompt_blend_by_chunk logic exactly.
        
        Args:
            request_data: Request data from log file
            request_id: Unique identifier for this request
            
        Returns:
            Dictionary with cache statistics
        """
        messages = request_data.get('messages', [])
        
        # Get BOS and EOS tokens (as in server)
        dummy_tokens = self.tekkenizer.encode(" ", bos=True, eos=True)
        bos, eos = dummy_tokens[0], dummy_tokens[-1]
        
        # Track statistics
        total_tokens = 1  # Start with BOS
        hit_tokens = 0
        
        # Process each message (matching server logic exactly)
        # Server: get_stitched_tokens_for_msg_content returns tokens with blend between chunks, but removes last blend
        # Then messages_to_prompt_blend_by_chunk adds blend after each message, then removes last blend and adds EOS
        for msg_idx, message in enumerate(messages):
            msg_content = message.get('content', '')
            msg_chunks_data = self.get_stitched_tokens_for_msg_content(msg_content)
            
            # Process chunks within this message
            # get_stitched_tokens_for_msg_content adds blend after each chunk, then removes the last one
            for chunk_idx, (chunk_tokens, chunk_length) in enumerate(msg_chunks_data):
                chunk_tuple = tuple(chunk_tokens)
                
                # Check cache hit
                is_hit = chunk_tuple in self.cache
                
                if is_hit:
                    hit_tokens += chunk_length
                else:
                    # Add to cache (unlimited size)
                    self.cache[chunk_tuple] = True
                
                total_tokens += chunk_length
                
                # Add blend tokens after each chunk (except after last chunk of message)
                # This matches get_stitched_tokens_for_msg_content which adds blend then removes last one
                if chunk_idx < len(msg_chunks_data) - 1:
                    total_tokens += len(self.blend_tokens)
            
            # After processing all chunks in message, add blend tokens (as in server)
            # This matches messages_to_prompt_blend_by_chunk which adds blend after each message
            # (The last one will be removed at the end)
            total_tokens += len(self.blend_tokens)
        
        # Remove last blend tokens and add EOS (as in server: res = prompt_tokens[:-len(blend_tokens)] + [eos])
        total_tokens = total_tokens - len(self.blend_tokens) + 1  # Remove last blend, add EOS
        
        # Calculate hit ratio
        hit_ratio = hit_tokens / total_tokens if total_tokens > 0 else 0.0
        
        stats = {
            'request_id': request_id,
            'tokens_hit': hit_tokens,
            'total_tokens': total_tokens,
            'hit_ratio': hit_ratio
        }
        
        self.request_stats.append(stats)
        return stats
    
    def process_log_file(self, log_file: str):
        """Process all requests from the log file in order"""
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except Exception as e:
            print(f"Error loading log file {log_file}: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Process each log entry in order
        for i, log_entry in enumerate(logs):
            request_data = log_entry.get('request', {})
            response_data = log_entry.get('response', {})
            
            # Get request ID from response or generate one
            request_id = response_data.get('id', f"request_{i}")
            
            # Process the request
            stats = self.process_request(request_data, request_id)
            
            # Output per-request stats
            print(f"{stats['request_id']}\t{stats['tokens_hit']}\t{stats['total_tokens']}\t{stats['hit_ratio']:.6f}")
    
    def print_summary(self):
        """Print summary statistics to stdout"""
        if not self.request_stats:
            print("No requests processed.", file=sys.stderr)
            return
        
        hit_ratios = [stats['hit_ratio'] for stats in self.request_stats]
        
        # Calculate statistics
        min_ratio = min(hit_ratios)
        max_ratio = max(hit_ratios)
        mean_ratio = statistics.mean(hit_ratios)
        median_ratio = statistics.median(hit_ratios)
        
        # Calculate quartiles (q1 and q3)
        sorted_ratios = sorted(hit_ratios)
        n = len(sorted_ratios)
        q1_idx = n // 4
        q3_idx = (3 * n) // 4
        q1_ratio = sorted_ratios[q1_idx] if n > 0 else 0.0
        q3_ratio = sorted_ratios[q3_idx] if n > 0 else 0.0
        
        # Output summary to stdout in machine-readable format
        print("format: SUMMARY <num_requests> <min> <max> <mean> <median> <q1> <q3>")
        print(f"SUMMARY\t{len(self.request_stats)}\t{min_ratio:.6f}\t{max_ratio:.6f}\t{mean_ratio:.6f}\t{median_ratio:.6f}\t{q1_ratio:.6f}\t{q3_ratio:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate cache behavior from blend server log")
    parser.add_argument(
        "--log-file",
        required=True,
        help="Path to the JSON log file from blend server"
    )
    parser.add_argument(
        "--model",
        default="mistralai/Devstral-Small-2507",
        help="Model to use for tokenization (must match server model)"
    )
    parser.add_argument(
        "--blend-special-str",
        default=" # # ",
        help="Special separator for blending chunks (must match server)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create simulator
    simulator = CacheSimulator(
        model=args.model,
        blend_special_str=args.blend_special_str
    )
    
    # Print header to stdout
    print("request_id\ttokens_hit\ttotal_tokens\thit_ratio")
    
    # Process log file
    simulator.process_log_file(args.log_file)
    
    # Print summary to stdout
    simulator.print_summary()


if __name__ == "__main__":
    main()

