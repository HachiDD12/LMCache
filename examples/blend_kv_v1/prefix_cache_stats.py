#!/usr/bin/env python3
"""
Prefix Cache Statistics Module

Implements a trie-based prefix cache simulator that tracks theoretical upper bound
on prefix cache hit rates. Uses 16-token blocks following vLLM convention.

This module can be attached to BlendServer to measure how many tokens in each request
would theoretically be cache hits if we had infinite cache and perfect prefix matching.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PrefixCacheRequestStats:
    """Statistics for a single request's prefix cache performance"""
    hit_tokens: int
    total_tokens: int
    hit_ratio: float
    
    def __repr__(self) -> str:
        return f"PrefixCacheRequestStats(hit_tokens={self.hit_tokens}, total_tokens={self.total_tokens}, hit_ratio={self.hit_ratio:.6f})"


class PrefixCacheTrieNode:
    """A node in the prefix cache trie"""
    __slots__ = ['children']
    
    def __init__(self):
        # Maps token block (as tuple) -> child node
        self.children: Dict[Tuple[int, ...], 'PrefixCacheTrieNode'] = {}


class PrefixCacheStats:
    """
    Prefix Cache Statistics Tracker
    
    Uses a trie data structure where each edge represents a block of tokens.
    This allows efficient prefix matching and insertion.
    
    Following vLLM convention, we use 16-token blocks by default.
    
    The cache is assumed to be infinitely large (no eviction policy).
    
    Usage:
        cache_stats = PrefixCacheStats(block_size=16)
        
        # For each request:
        stats = cache_stats.get_request_cache_stats(prompt_tokens)
        print(f"Hit tokens: {stats.hit_tokens}, Hit ratio: {stats.hit_ratio}")
    """
    
    def __init__(self, block_size: int = 16):
        """
        Initialize the prefix cache stats tracker.
        
        Args:
            block_size: Number of tokens per block (default: 16, following vLLM convention)
        """
        self.block_size = block_size
        self.root = PrefixCacheTrieNode()
        
        # Track cumulative statistics
        self.total_requests = 0
        self.total_hit_tokens = 0
        self.total_tokens = 0
    
    def _tokenize_to_blocks(self, prompt_tokens: List[int]) -> List[Tuple[int, ...]]:
        """
        Split prompt tokens into blocks of size block_size.
        
        The last block may be smaller than block_size if tokens don't divide evenly.
        
        Args:
            prompt_tokens: List of token IDs
            
        Returns:
            List of token blocks as tuples (for hashability)
        """
        blocks = []
        for i in range(0, len(prompt_tokens), self.block_size):
            block = tuple(prompt_tokens[i:i + self.block_size])
            blocks.append(block)
        return blocks
    
    def get_request_cache_stats(self, prompt_tokens: List[int]) -> PrefixCacheRequestStats:
        """
        Calculate prefix cache statistics for a request and update the cache.
        
        This method:
        1. Traverses the trie to find the longest matching prefix
        2. Counts how many tokens are cache hits
        3. Inserts any non-matching blocks into the trie
        4. Returns statistics
        
        Args:
            prompt_tokens: List of token IDs for the request
            
        Returns:
            PrefixCacheRequestStats with hit_tokens, total_tokens, and hit_ratio
        """
        if not prompt_tokens:
            return PrefixCacheRequestStats(hit_tokens=0, total_tokens=0, hit_ratio=0.0)
        
        # Convert to blocks
        blocks = self._tokenize_to_blocks(prompt_tokens)
        
        # Traverse trie to find longest matching prefix
        hit_tokens = 0
        current_node = self.root
        match_depth = 0  # Number of blocks that matched
        
        for block in blocks:
            if block in current_node.children:
                # This block is a cache hit
                hit_tokens += len(block)
                current_node = current_node.children[block]
                match_depth += 1
            else:
                # No match - stop traversing
                break
        
        # Insert remaining (unmatched) blocks into the trie
        for block in blocks[match_depth:]:
            new_node = PrefixCacheTrieNode()
            current_node.children[block] = new_node
            current_node = new_node
        
        # Calculate statistics
        total_tokens = len(prompt_tokens)
        hit_ratio = hit_tokens / total_tokens if total_tokens > 0 else 0.0
        
        # Update cumulative statistics
        self.total_requests += 1
        self.total_hit_tokens += hit_tokens
        self.total_tokens += total_tokens
        
        return PrefixCacheRequestStats(
            hit_tokens=hit_tokens,
            total_tokens=total_tokens,
            hit_ratio=hit_ratio
        )
    
    def get_cumulative_stats(self) -> Dict:
        """
        Get cumulative statistics across all requests.
        
        Returns:
            Dictionary with total_requests, total_hit_tokens, total_tokens, 
            and overall_hit_ratio
        """
        overall_hit_ratio = (
            self.total_hit_tokens / self.total_tokens 
            if self.total_tokens > 0 else 0.0
        )
        return {
            'total_requests': self.total_requests,
            'total_hit_tokens': self.total_hit_tokens,
            'total_tokens': self.total_tokens,
            'overall_hit_ratio': overall_hit_ratio
        }
    
    def reset(self):
        """Reset the cache and statistics to initial state"""
        self.root = PrefixCacheTrieNode()
        self.total_requests = 0
        self.total_hit_tokens = 0
        self.total_tokens = 0
    
    def get_cache_size_blocks(self) -> int:
        """
        Get the number of unique token blocks stored in the cache.
        
        This is a measure of cache memory usage (in terms of blocks).
        
        Returns:
            Number of blocks in the cache
        """
        def count_nodes(node: PrefixCacheTrieNode) -> int:
            count = len(node.children)
            for child in node.children.values():
                count += count_nodes(child)
            return count
        
        return count_nodes(self.root)


# Standalone testing
if __name__ == "__main__":
    # Simple test
    print("Testing PrefixCacheStats...")
    
    cache = PrefixCacheStats(block_size=4)  # Using smaller blocks for testing
    
    # First request - should be all misses
    tokens1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    stats1 = cache.get_request_cache_stats(tokens1)
    print(f"Request 1: {stats1}")
    assert stats1.hit_tokens == 0, "First request should have no hits"
    
    # Second request with same prefix - should have hits
    tokens2 = [1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 23]
    stats2 = cache.get_request_cache_stats(tokens2)
    print(f"Request 2: {stats2}")
    assert stats2.hit_tokens == 8, "Second request should hit first 8 tokens (2 blocks)"
    
    # Third request with different prefix - should have no hits
    tokens3 = [100, 200, 300, 400]
    stats3 = cache.get_request_cache_stats(tokens3)
    print(f"Request 3: {stats3}")
    assert stats3.hit_tokens == 0, "Third request should have no hits"
    
    # Fourth request same as first - should have all hits
    tokens4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    stats4 = cache.get_request_cache_stats(tokens4)
    print(f"Request 4: {stats4}")
    assert stats4.hit_tokens == 10, "Fourth request should hit all 10 tokens"
    
    # Print cumulative stats
    print(f"\nCumulative stats: {cache.get_cumulative_stats()}")
    print(f"Cache size (blocks): {cache.get_cache_size_blocks()}")
    
    print("\nAll tests passed!")

