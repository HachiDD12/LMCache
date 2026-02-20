# Standard
import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
import signal
import uuid
import traceback
from typing import List, Optional
from datetime import datetime

# Third Party
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, AsyncLLMEngine
from vllm.config import KVTransferConfig
from vllm.inputs import TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from dataclasses import asdict
import uvicorn
import torch

# Devstral-specific imports
from mistral_common.protocol.instruct.messages import SystemMessage, UserMessage, AssistantMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest as MistralChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from huggingface_hub import hf_hub_download

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder

# Local
from prefix_cache_stats import PrefixCacheStats, PrefixCacheRequestStats


class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="The model to use for completion")
    messages: List[ChatMessage] = Field(..., description="The messages to complete")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(0.95, description="Top-p sampling parameter")
    n: Optional[int] = Field(1, description="Number of completions to generate")
    max_tokens: Optional[int] = Field(100, description="Maximum tokens to generate")
    tools: Optional[List[dict]] = Field(None, description="Tools for function calling")


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict


def setup_environment_variables(
    use_disk: bool = False, blend_special_str: str = " # # "
):
    """Setup environment variables for LMCache configuration"""
    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    # Blending related config
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"

    if use_disk:
        # Disable local CPU backend in LMCache
        os.environ["LMCACHE_LOCAL_CPU"] = "False"

        # Set the maximum size of the local CPU buffer size to 5GB
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"

        # Enable local disk backend in LMCache
        os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"

        # Set the maximum size of the local disk size to 10GB
        os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "10"
    else:
        # Enable local CPU backend in LMCache
        os.environ["LMCACHE_LOCAL_CPU"] = "True"

        # Set the maximum size of the local CPU size to 5GB
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"


@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str):
    """Build LLM with LMCache integration"""
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )
    
    # --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --tensor-parallel-size 2
    llm_args = AsyncEngineArgs(
        model=model,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        tensor_parallel_size=4,         # TODO: add this back in once have 2 GPUs
        kv_transfer_config=ktc,
        max_model_len=128000,           # TODO: change this to 128000 once hosting devstral
        gpu_memory_utilization=0.7,
        enable_prefix_caching=False,
        seed=42,                        # For reproducibility
    )

    # llm = LLM(**asdict(llm_args))
    llm = AsyncLLMEngine.from_engine_args(llm_args)
    try:
        yield llm
    finally:
        # Clean up lmcache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


class BlendServer:
    def __init__(self, 
                 model: str, 
                 use_disk: bool = False, 
                 blend_special_str: str = " # # ", 
                 log_file: str = "chat_requests.json", 
                 enable_ttft: bool = False,
                 blend_granularity: str = "chunk",
                 enable_prefix_cache_stats: bool = False,
                 prefix_cache_block_size: int = 16,
                 min_chunk_size: int = 0,
                 chunk_dist_file: str = None,
                 enable_fixed_size_chunk: bool = False):
        self.model = model
        self.use_disk = use_disk
        self.blend_special_str = blend_special_str
        self.blend_granularity = blend_granularity
        self.lmcache_connector = "LMCacheConnectorV1"
        self.log_file = log_file
        self.enable_ttft = enable_ttft
        self.enable_prefix_cache_stats = enable_prefix_cache_stats
        self.min_chunk_size = min_chunk_size
        self.chunk_dist_file = chunk_dist_file
        self.chunk_sizes: List[int] = []  # Track chunk sizes for distribution
        self.enable_fixed_size_chunk = enable_fixed_size_chunk
        
        # Initialize prefix cache stats module if enabled
        if enable_prefix_cache_stats:
            self.prefix_cache_stats = PrefixCacheStats(block_size=prefix_cache_block_size)
        else:
            self.prefix_cache_stats = None
        
        # Setup environment variables
        setup_environment_variables(use_disk, blend_special_str)
        
        # Load system prompt for Devstral
        self.system_prompt = self.load_system_prompt(model, "SYSTEM_PROMPT.txt")
        
        # Initialize Devstral tokenizer
        self.tokenizer = MistralTokenizer.from_hf_hub(model)
        self.tekkenizer = self.tokenizer.instruct_tokenizer.tokenizer
        
        # Initialize LLM with LMCache
        self.llm_context = build_llm_with_lmcache(self.lmcache_connector, model)
        self.llm = self.llm_context.__enter__()
        
        # Create FastAPI app
        self.app = FastAPI(title="LMCache Blend Server", version="1.0.0")
        self.setup_routes()
    
    def log_request(self, request: ChatCompletionRequest, response: ChatCompletionResponse, generation_time: float):
        """Log the request and response to the JSON file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request": {
                "model": request.model,
                "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
                "temperature": request.temperature,
                "top_p": request.top_p,
                "n": request.n,
                "max_tokens": request.max_tokens,
                "tools": request.tools
            },
            "response": {
                "id": response.id,
                "choices": response.choices,
                "usage": response.usage
            },
            "generation_time": generation_time
        }
        
        try:
            # Load existing logs if file exists
            logs = []
            if os.path.exists(self.log_file):
                try:
                    with open(self.log_file, 'r') as f:
                        logs = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    logs = []
            
            # Append new log entry
            logs.append(log_entry)
            
            # Write back to file
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            print(f"[WARNING] Failed to log request: {e}")
    
    def load_system_prompt(self, repo_id: str, filename: str) -> str:
        """Load system prompt from HuggingFace hub"""
        # try:
        #     file_path = hf_hub_download(repo_id=repo_id, filename=filename)
        #     with open(file_path, "r") as file:
        #         system_prompt = file.read()
        #     print(f"Loaded system prompt: {system_prompt}")
        #     return system_prompt
        # except Exception as e:
        #     print(f"Warning: Could not load system prompt from {repo_id}/{filename}: {e}")
        return "You are a very helpful assistant."
    
    def setup_routes(self):
        """Setup API routes"""
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            if self.enable_ttft:
                return await self.handle_chat_completion_ttft(request)
            else:
                return await self.handle_chat_completion(request)
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "model": self.model}
    
    def messages_to_prompt(self, messages: List[ChatMessage]) -> List[int]:
        """Convert chat messages to tokenized prompt using Devstral tokenizer"""
        # Convert our ChatMessage format to Mistral format
        mistral_messages = []
        
        for message in messages:
            if message.role == "system":
                mistral_messages.append(SystemMessage(content=message.content))
            elif message.role == "user":
                mistral_messages.append(UserMessage(content=message.content))
            elif message.role == "assistant":
                mistral_messages.append(AssistantMessage(content=message.content))
        
        # Create Mistral chat completion request
        mistral_request = MistralChatCompletionRequest(messages=mistral_messages)
        
        # Tokenize using Devstral tokenizer
        tokenized = self.tokenizer.encode_chat_completion(mistral_request)
        
        # print(f"@@@@@ Prompt tokens naive: {tokenized.tokens}")
        return tokenized.tokens
    
    def messages_to_prompt_no_blend(self, messages: List[ChatMessage]) -> List[int]:
        """Convert chat messages to tokenized prompt with no blending.
        Splits messages on the special string, removes them, concatenates everything,
        and encodes using Devstral tokenizer.
        """
        # Convert our ChatMessage format to Mistral format, removing special strings
        mistral_messages = []
        
        for message in messages:
            # Split on special string and remove it, then concatenate
            cleaned_content = message.content.split(self.blend_special_str)
            cleaned_content = "".join(cleaned_content)  # Remove special strings and concatenate
            
            if message.role == "system":
                mistral_messages.append(SystemMessage(content=cleaned_content))
            elif message.role == "user":
                mistral_messages.append(UserMessage(content=cleaned_content))
            elif message.role == "assistant":
                mistral_messages.append(AssistantMessage(content=cleaned_content))
        
        # Create Mistral chat completion request
        mistral_request = MistralChatCompletionRequest(messages=mistral_messages)
        
        # Tokenize using Devstral tokenizer
        tokenized = self.tokenizer.encode_chat_completion(mistral_request)
        
        # print(f"@@@@@ Prompt tokens no blend: {tokenized.tokens}")
        return tokenized.tokens
    
    def messages_to_prompt_blend_by_msg(self, messages: List[ChatMessage]) -> List[int]:
        """Convert chat messages to tokenized prompt using Devstral tokenizer as well as default Tekkenizer
        for the blend special string, such that each message is a separate blend chunk.
        """
        # Pre-encode the blend special string
        blend_tokens = self.tokenizer.instruct_tokenizer.tokenizer.encode(self.blend_special_str, bos=False, eos=False)
        
        # Convert our ChatMessage format to Mistral format
        prompt_tokens = []
        for message in messages:
            mistral_messages = []
            if message.role == "system":
                mistral_messages.append(SystemMessage(content=message.content))
            elif message.role == "user":
                mistral_messages.append(UserMessage(content=message.content))
            elif message.role == "assistant":
                mistral_messages.append(AssistantMessage(content=message.content))
        
            # Create Mistral chat completion request
            mistral_request = MistralChatCompletionRequest(messages=mistral_messages)

            # Tokenize using Devstral tokenizer
            msg_tokens = self.tokenizer.encode_chat_completion(mistral_request).tokens
            prompt_tokens.extend(msg_tokens)
        
            # Add blend special string
            prompt_tokens.extend(blend_tokens)
        
        # print(f"@@@@@ Prompt tokens blend by msg: {prompt_tokens[:-len(blend_tokens)]}")
        return prompt_tokens[:-len(blend_tokens)]
    
    def get_stitched_tokens_for_msg_content(self, msg_content: str, blend_tokens: List[int]) -> List[int]:
        """Get stitched tokens for a single message content.
        
        If min_chunk_size > 0:
          - If enable_fixed_size_chunk is True: chunks are split/padded to be exactly 
            min_chunk_size tokens. Large chunks are split into multiple fixed-size chunks,
            and the remainder is padded.
          - If enable_fixed_size_chunk is False: chunks smaller than min_chunk_size are 
            padded with pad tokens to reach the minimum size.
        
        Also tracks chunk sizes for distribution analysis if chunk_dist_file is set.
        """
        tokens = []
        # split on blend special string
        msg_chunks = msg_content.split(self.blend_special_str)
        print(f"[INFO] msg split into {len(msg_chunks)} chunks")
        actual_chunks = 0
        
        # Get pad token id for padding small chunks
        pad_id = self.tekkenizer.pad_id
        
        for chunk in msg_chunks:
            chunk_tokens = self.tekkenizer.encode(chunk, bos=False, eos=False)
            original_len = len(chunk_tokens)
            
            if self.enable_fixed_size_chunk and self.min_chunk_size > 0:
                # Fixed-size chunking: split large chunks and pad small ones
                fixed_size = self.min_chunk_size
                
                # Split chunk_tokens into fixed-size pieces
                for i in range(0, len(chunk_tokens), fixed_size):
                    piece = chunk_tokens[i:i + fixed_size]
                    piece_original_len = len(piece)
                    
                    # Pad if this piece is smaller than fixed_size
                    if len(piece) < fixed_size:
                        padding_needed = fixed_size - len(piece)
                        piece = piece + [pad_id] * padding_needed
                    
                    tokens.extend(piece)
                    tokens.extend(blend_tokens)
                    
                    # Track original piece size (before padding)
                    if self.chunk_dist_file:
                        self.chunk_sizes.append(piece_original_len)
                    actual_chunks += 1
                
                # Handle empty chunks (encode returns empty list)
                if len(chunk_tokens) == 0:
                    # Pad an empty chunk to fixed_size
                    piece = [pad_id] * fixed_size
                    tokens.extend(piece)
                    tokens.extend(blend_tokens)
                    if self.chunk_dist_file:
                        self.chunk_sizes.append(0)
                    actual_chunks += 1
            else:
                # Original behavior: only pad small chunks
                # Pad chunk with pad tokens if below min_chunk_size
                if self.min_chunk_size > 0 and len(chunk_tokens) < self.min_chunk_size:
                    padding_needed = self.min_chunk_size - len(chunk_tokens)
                    chunk_tokens = chunk_tokens + [pad_id] * padding_needed
                
                tokens.extend(chunk_tokens)
                tokens.extend(blend_tokens)
                
                # Track chunk size (original size before padding for analysis)
                if self.chunk_dist_file:
                    self.chunk_sizes.append(original_len)
                actual_chunks += 1
        
        print(f"[INFO] Actual chunks: {actual_chunks}")
        return tokens[:-len(blend_tokens)]
    
    def messages_to_prompt_blend_by_chunk(self, messages: List[ChatMessage]) -> List[int]:
        """Convert chat messages to tokenized prompt using Devstral tokenizer as well as default Tekkenizer.
        Extracts chunks by splitting on the blend special string, tokenizes each chunk, and then 
        stitches them back together with the blend special string.
        """
        # Pre-encode the blend special string
        blend_tokens = self.tekkenizer.encode(self.blend_special_str, bos=False, eos=False)
        dummy_tokens = self.tekkenizer.encode(" ", bos=True, eos=True)
        bos, eos = dummy_tokens[0], dummy_tokens[-1]
        # print(f"@@@@@ Blend tokens: {blend_tokens}")
        
        # encode each message chunk separately then stitch
        prompt_tokens = [bos]
        for message in messages:
            # TODO: add role to the tokens (see mistral control tokens)
            if message.role == "system":
                msg_tokens = self.get_stitched_tokens_for_msg_content(message.content, blend_tokens)
            elif message.role == "user":
                msg_tokens = self.get_stitched_tokens_for_msg_content(message.content, blend_tokens)
            elif message.role == "assistant":
                msg_tokens = self.get_stitched_tokens_for_msg_content(message.content, blend_tokens)
        
            # Create Mistral chat completion request
            # mistral_request = MistralChatCompletionRequest(messages=mistral_messages)

            # Tokenize using Devstral tokenizer
            # msg_tokens = self.tokenizer.encode_chat_completion(mistral_request).tokens
            prompt_tokens.extend(msg_tokens)
        
            # # Add blend special string
            # prompt_tokens.extend(blend_tokens)

        res = prompt_tokens[:-len(blend_tokens)] + [eos]
        # print(f"@@@@@ Prompt tokens blend by chunk: {res}")
        return res
    
    def handle_timeout(self, signum, frame):
        raise Exception("Generation timed out")
    
    async def handle_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle chat completion request"""
        print(f"@@@@@ Handling chat completion request of n = {request.n}")
        try:
            # Convert messages to tokenized prompt based on blend granularity
            if self.blend_granularity == "none":
                prompt_tokens = self.messages_to_prompt_no_blend(request.messages)
            elif self.blend_granularity == "msg":
                prompt_tokens = self.messages_to_prompt_blend_by_msg(request.messages)
            elif self.blend_granularity == "chunk":
                prompt_tokens = self.messages_to_prompt_blend_by_chunk(request.messages)
            else:
                # Default to chunk if invalid granularity
                prompt_tokens = self.messages_to_prompt_blend_by_chunk(request.messages)
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                n=request.n,
                max_tokens=request.max_tokens
            )
            
            # Generate response
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(300)
            start_time = time.time()
            outputs = self.llm.generate(
                prompt=TokensPrompt(prompt_token_ids=prompt_tokens), 
                sampling_params=sampling_params,
                request_id=str(uuid.uuid4())
            )
            end_time = time.time()
            generation_time = end_time - start_time
            signal.alarm(0)
            print(f"[INFO] Generation time: {generation_time} seconds for {len(prompt_tokens)} tokens")
            # Extract generated text
            
            # print(f"@@@@@ Outputs: {outputs}")
            # print(f"@@@@@ Outputs[0]: {outputs[0]}")
            
            choices = []
            for i, completion in enumerate(outputs[0].outputs):
                choices.append({
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": completion.text
                    },
                    "finish_reason": "stop"
                })
                
            # Calculate usage
            input_tokens = len(prompt_tokens)
            # For Devstral, we need to decode the generated text properly
            # The generated text is already decoded from vLLM, so we just count tokens
            # output_tokens = len(self.tokenizer.encode_chat_completion(generated_text))
            output_tokens = sum(len(completion.text) for completion in outputs[0].outputs)
            
            # Create response
            response = ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=request.model,
                choices=choices,
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }
            )
            
            print(f"@@@@@ Response # of choices: {len(choices)}")
            
            # Log the request and response
            self.log_request(request, response, generation_time)
            
            return response
            
        except Exception as e:
            signal.alarm(0)
            print(f"[ERROR] Error handling chat completion, type: {type(e)}, message: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
    
    async def handle_chat_completion_ttft(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle chat completion request with TTFT measurement"""
        print(f"@@@@@ Handling chat completion request of n = {request.n}")
        try:
            # Convert messages to tokenized prompt based on blend granularity
            if self.blend_granularity == "none":
                prompt_tokens: List[int] = self.messages_to_prompt_no_blend(request.messages)
            elif self.blend_granularity == "msg":
                prompt_tokens: List[int] = self.messages_to_prompt_blend_by_msg(request.messages)
            elif self.blend_granularity == "chunk":
                prompt_tokens: List[int] = self.messages_to_prompt_blend_by_chunk(request.messages)
            else:
                # Default to chunk if invalid granularity
                prompt_tokens: List[int] = self.messages_to_prompt_blend_by_chunk(request.messages)
            
            # Calculate prefix cache statistics before generation (if enabled)
            prefix_cache_request_stats: Optional[PrefixCacheRequestStats] = None
            if self.prefix_cache_stats is not None:
                prefix_cache_request_stats = self.prefix_cache_stats.get_request_cache_stats(prompt_tokens)
                # Print stats to stderr in similar format to TTFT
                print(f"[INFO] Prefix cache hit tokens: {prefix_cache_request_stats.hit_tokens}", file=sys.stderr)
                print(f"[INFO] Prefix cache hit ratio: {prefix_cache_request_stats.hit_ratio:.4f}", file=sys.stderr)
                sys.stderr.flush()
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                n=request.n,
                max_tokens=request.max_tokens,
                output_kind=RequestOutputKind.DELTA,
            )
            
            # Variables for timing
            start_time = time.perf_counter()
            first_token_time = None
            ttft = None
            # Store text for each completion (n completions)
            completions_text = [""] * request.n
            
            async def _stream_once() -> None:
                nonlocal first_token_time, ttft
                async for output in self.llm.generate(
                    prompt=TokensPrompt(prompt_token_ids=prompt_tokens),
                    sampling_params=sampling_params,
                    request_id=str(uuid.uuid4()),
                ):
                    # Handle multiple completions (n > 1)
                    for i, completion_output in enumerate(output.outputs):
                        if i < len(completions_text):
                            chunk_text = completion_output.text
                            if chunk_text:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                    ttft = first_token_time - start_time
                                    print(f"[INFO] TTFT: {ttft:.4f} seconds")
                                completions_text[i] += chunk_text
                    
                    if output.finished:
                        break
            
            # Enforce a hard timeout for the whole streaming operation (e.g., 600s)
            await asyncio.wait_for(_stream_once(), timeout=600.0)
            end_time = time.perf_counter()
            generation_time = end_time - start_time
            print(f"[INFO] Generation time: {generation_time:.4f} seconds for {len(prompt_tokens)} tokens")
            
            # Build choices from all completions
            choices = []
            for i, text in enumerate(completions_text):
                choices.append({
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": text
                    },
                    "finish_reason": "stop"
                })
            
            # Calculate usage
            input_tokens = len(prompt_tokens)
            output_tokens = sum(len(text) for text in completions_text)
            
            # Create response
            response = ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=request.model,
                choices=choices,
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "ttft": ttft,
                }
            )
            
            print(f"@@@@@ Response # of choices: {len(choices)}")
            sys.stdout.flush()
            
            # Log the request and response
            self.log_request(request, response, generation_time)
            
            # Write chunk distribution after each request
            self.write_chunk_distribution()
            
            return response
            
        except asyncio.TimeoutError:
            msg = "Request timed out after 600 seconds."
            print(f"[ERROR] {msg}")
            raise HTTPException(status_code=500, detail=msg)
        except Exception as e:
            print(f"[ERROR] Error handling chat completion, type: {type(e)}, message: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
    
    def load_mock_requests(self, mock_file: str) -> List[dict]:
        """Load mock requests from a JSON log file"""
        try:
            with open(mock_file, 'r') as f:
                logs = json.load(f)
            
            # Extract requests from logs
            requests = []
            for log_entry in logs:
                if 'request' in log_entry:
                    # Convert back to ChatCompletionRequest format
                    request_data = log_entry['request']
                    messages = [ChatMessage(role=msg['role'], content=msg['content']) 
                              for msg in request_data.get('messages', [])]
                    
                    mock_request = {
                        'request': ChatCompletionRequest(
                            model=request_data.get('model', self.model),
                            messages=messages,
                            temperature=request_data.get('temperature', 0.7),
                            top_p=request_data.get('top_p', 0.95),
                            n=request_data.get('n', 1),
                            max_tokens=request_data.get('max_tokens', 100),
                            tools=request_data.get('tools')
                        ),
                        'original_timestamp': log_entry.get('timestamp'),
                        'original_generation_time': log_entry.get('generation_time')
                    }
                    requests.append(mock_request)
            
            print(f"[MOCK] Loaded {len(requests)} requests from {mock_file}")
            return requests
            
        except Exception as e:
            print(f"[ERROR] Failed to load mock requests from {mock_file}: {e}")
            return []
    
    async def run_mock_experiment(self, mock_file: str):
        """Run a mock experiment by replaying requests from the log file"""
        print(f"[MOCK] Starting mock experiment with file: {mock_file}")
        
        # Load mock requests
        mock_requests = self.load_mock_requests(mock_file)
        if not mock_requests:
            print("[MOCK] No valid requests found, exiting mock mode")
            return
        
        # Sort by original timestamp if available
        mock_requests.sort(key=lambda x: x.get('original_timestamp', ''))
        
        print(f"[MOCK] Replaying {len(mock_requests)} requests...")
        
        total_original_time = 0
        total_mock_time = 0
        total_ttft = 0
        successful_requests = 0
        fails = 0
        
        for i, mock_data in enumerate(mock_requests):
            request = mock_data['request']
            original_time = mock_data.get('original_generation_time', 0)
            original_timestamp = mock_data.get('original_timestamp', 'unknown')
            
            print(f"[MOCK] Processing request {i+1}/{len(mock_requests)} (original: {original_timestamp})")
            print(f"[MOCK] Request: {len(request.messages)} messages, max_tokens: {request.max_tokens}")
            
            try:
                # Process the request (TTFT will be measured and printed during generation)
                start_time = time.time()
                response = await self.handle_chat_completion_mock(request, i)
                end_time = time.time()
                
                mock_time = end_time - start_time
                ttft = response.usage.get('ttft', None)
                if ttft is not None:
                    total_ttft += ttft
                
                total_mock_time += mock_time
                total_original_time += original_time
                successful_requests += 1
                
                print(f"[MOCK] Request {i+1} completed:")
                print(f"  Original time: {original_time:.2f}s")
                print(f"  Mock time: {mock_time:.2f}s")
                print(f"  TTFT: {ttft:.4f}s" if ttft is not None else "  TTFT: N/A")  # Time To First Token
                print(f"  Speedup: {original_time/mock_time:.2f}x" if mock_time > 0 else "  Speedup: N/A")
                print(f"  Response tokens: {response.usage['completion_tokens']}")
                
            except Exception as e:
                print(f"[MOCK] Request {i+1} failed: {e}")
                print(f"[MOCK] Traceback: {traceback.format_exc()}")
                fails += 1
                if fails > 5:
                    print(f"[MOCK] Too many failures, exiting mock experiment")
                    break
            sys.stdout.flush()
        
        # Print summary
        print(f"\n[MOCK] Experiment Summary:")
        print(f"  Total requests: {len(mock_requests)}")
        print(f"  Successful: {successful_requests}")
        print(f"  Failed: {len(mock_requests) - successful_requests}")
        if successful_requests > 0:
            print(f"  Average original time: {total_original_time/successful_requests:.2f}s")
            print(f"  Average mock time: {total_mock_time/successful_requests:.2f}s")
            if total_ttft > 0:
                print(f"  Average TTFT: {total_ttft/successful_requests:.4f}s")  # Time To First Token
            if total_mock_time > 0:
                print(f"  Overall speedup: {total_original_time/total_mock_time:.2f}x")

    async def handle_chat_completion_mock(self, request: ChatCompletionRequest, rid: int) -> ChatCompletionResponse:
        """Mock version of handle_chat_completion for mock mode with TTFT measurement."""
        print(f"@@@@@ Handling chat completion request {rid} of n = {request.n} (mock)")
        # Convert messages to tokenized prompt based on blend granularity
        if self.blend_granularity == "none":
            prompt_tokens: List[int] = self.messages_to_prompt_no_blend(request.messages)
        elif self.blend_granularity == "msg":
            prompt_tokens: List[int] = self.messages_to_prompt_blend_by_msg(request.messages)
        elif self.blend_granularity == "chunk":
            prompt_tokens: List[int] = self.messages_to_prompt_blend_by_chunk(request.messages)
        else:
            # Default to chunk if invalid granularity
            prompt_tokens: List[int] = self.messages_to_prompt_blend_by_chunk(request.messages)

        # Calculate prefix cache statistics before generation (if enabled)
        prefix_cache_request_stats: Optional[PrefixCacheRequestStats] = None
        if self.prefix_cache_stats is not None:
            prefix_cache_request_stats = self.prefix_cache_stats.get_request_cache_stats(prompt_tokens)
            # Print stats to stderr in similar format to TTFT
            print(f"[INFO] Prefix cache hit tokens: {prefix_cache_request_stats.hit_tokens}", file=sys.stderr)
            print(f"[INFO] Prefix cache hit ratio: {prefix_cache_request_stats.hit_ratio:.4f}", file=sys.stderr)
            sys.stderr.flush()
            
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            max_tokens=request.max_tokens,
            output_kind=RequestOutputKind.DELTA,
        )

        # Variables for timing
        start_time = time.perf_counter()
        first_token_time = None
        ttft = None
        words = ""

        async def _stream_once() -> None:
            nonlocal words, first_token_time, ttft
            async for output in self.llm.generate(
                prompt=TokensPrompt(prompt_token_ids=prompt_tokens),
                sampling_params=sampling_params,
                request_id=str(rid),
            ):
                chunk_tokens = output.outputs[0].text  # assumes single completion
                if chunk_tokens:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                        ttft = first_token_time - start_time
                        print(f"[INFO] TTFT: {ttft:.4f} seconds")
                    words += chunk_tokens
                if output.finished:
                    break

        try:
            # Enforce a hard timeout for the whole streaming operation (e.g., 300s)
            await asyncio.wait_for(_stream_once(), timeout=300.0)
            end_time = time.perf_counter()
            generation_time = end_time - start_time
            print(f"[INFO] Generation time: {generation_time:.4f} seconds for {len(prompt_tokens)} tokens")
            print(f"[INFO] Final output: {words}")

            # Build choices (mock: duplicate same text n times)
            choices = []
            for i in range(request.n):
                choices.append({
                    "index": i,
                    "message": {"role": "assistant", "content": words},
                    "finish_reason": "stop",
                })

            # Mock usage accounting
            input_tokens = len(prompt_tokens)
            output_tokens = len(words) * request.n  # NOTE: mock approximation

            response = ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=request.model,
                choices=choices,
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "ttft": ttft,
                },
            )

            print(f"@@@@@ Response # of choices: {len(choices)}")
            sys.stdout.flush()
            return response

        except asyncio.TimeoutError:
            # If your LLM client supports cancel/abort, call it here (best-effort cleanup)
            # e.g., await self.llm.abort(request_id=str(rid))   # if available
            msg = f"Request {rid} timed out after 300 seconds."
            print(f"[ERROR] {msg}")
            await self.llm.abort(request_id=str(rid))
        except Exception as e:
            print(f"[ERROR] Error handling chat completion, type: {type(e)}, message: {str(e)}")
            raise
    
    async def handle_chat_completion_mock_v0(self, request: ChatCompletionRequest, rid: int) -> ChatCompletionResponse:
        """Mock version of handle_chat_completion for mock mode with TTFT measurement"""
        print(f"@@@@@ Handling chat completion request {rid} of n = {request.n} (mock)")
        try:
            # Convert messages to tokenized prompt based on blend granularity
            if self.blend_granularity == "none":
                prompt_tokens = self.messages_to_prompt_no_blend(request.messages)
            elif self.blend_granularity == "msg":
                prompt_tokens = self.messages_to_prompt_blend_by_msg(request.messages)
            elif self.blend_granularity == "chunk":
                prompt_tokens = self.messages_to_prompt_blend_by_chunk(request.messages)
            else:
                # Default to chunk if invalid granularity
                prompt_tokens = self.messages_to_prompt_blend_by_chunk(request.messages)
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                n=request.n,
                max_tokens=request.max_tokens,
                output_kind=RequestOutputKind.DELTA
            )
            
            # Generate response
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(300)
            start_time = time.time()
            first_token_time = None
            words = ""
            ttft = None
            
            # Stream through the outputs to measure TTFT
            async for output in self.llm.generate(
                prompt=TokensPrompt(prompt_token_ids=prompt_tokens), 
                sampling_params=sampling_params,
                request_id=str(rid)
            ):
                # if first_token_time is None:
                #     print(f"@@@@@ 1st RequestOutput: {output}")
                # print(f"@@@@@ Token: {output.outputs[0].text}")
                chunk_tokens = output.outputs[0].text     # assumes single completion
                if chunk_tokens is not None:
                    if first_token_time is None and chunk_tokens != "":
                        first_token_time = time.time()
                        ttft = first_token_time - start_time
                        print(f"[INFO] TTFT: {ttft:.4f} seconds")
                    words += chunk_tokens
                if output.finished:
                    # print(f"@@@@@ last RequestOutput: {output}")
                    break

            end_time = time.time()
            generation_time = end_time - start_time
            signal.alarm(0)
            
            print(f"[INFO] Generation time: {generation_time} seconds for {len(prompt_tokens)} tokens")
            print(f"[INFO] Final output: {words}")
            
            choices = []
            for i in range(request.n):
                choices.append({
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": words
                    },
                    "finish_reason": "stop"
                })
                
            # Calculate usage
            input_tokens = len(prompt_tokens)
            output_tokens = len(words) * request.n
            
            # Create response
            response = ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=request.model,
                choices=choices,
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "ttft": ttft
                }
            )
            
            print(f"@@@@@ Response # of choices: {len(choices)}")
            
            return response
            
        except Exception as e:
            signal.alarm(0)
            print(f"[ERROR] Error handling chat completion, type: {type(e)}, message: {str(e)}")
            raise e
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the server"""
        uvicorn.run(self.app, host=host, port=port)
    
    def write_chunk_distribution(self):
        """Write chunk size distribution to file if enabled"""
        if self.chunk_dist_file and self.chunk_sizes:
            try:
                dist_data = {
                    "chunk_sizes": self.chunk_sizes,
                    "total_chunks": len(self.chunk_sizes),
                    "min_size": min(self.chunk_sizes),
                    "max_size": max(self.chunk_sizes),
                    "mean_size": sum(self.chunk_sizes) / len(self.chunk_sizes),
                    "min_chunk_size_setting": self.min_chunk_size,
                    "fixed_size_chunking_enabled": self.enable_fixed_size_chunk
                }
                with open(self.chunk_dist_file, 'w') as f:
                    json.dump(dist_data, f, indent=2)
            except Exception as e:
                print(f"[WARNING] Failed to write chunk distribution: {e}")
    
    def __del__(self):
        """Cleanup when server is destroyed"""
        if hasattr(self, 'llm_context'):
            self.llm_context.__exit__(None, None, None)


def parse_args():
    parser = argparse.ArgumentParser(description="LMCache Blend Server")
    parser.add_argument(
        "--model",
        # default="mistralai/Mistral-7B-Instruct-v0.2",
        default="mistralai/Devstral-Small-2507",
        help="Model to use for inference"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "-d",
        "--use-disk",
        action="store_true",
        help="Use disk as LMCache backend"
    )
    parser.add_argument(
        "-b",
        "--blend-special-str",
        default=" # # ",
        help="Special separator for blending chunks"
    )
    parser.add_argument(
        "--log-file",
        default="~/chat_requests.json",
        help="Path to the JSON log file for request logging"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Enable mock mode to replay requests from log file"
    )
    parser.add_argument(
        "--mock-file",
        help="Path to the JSON log file to replay in mock mode (required if --mock is set)"
    )
    parser.add_argument(
        "--enable-ttft",
        action="store_true",
        help="Enable TTFT (Time To First Token) measurement in non-mock mode"
    )
    parser.add_argument(
        "--blend-granularity",
        default="chunk",
        choices=["none", "msg", "chunk"],
        help="Blend granularity: 'none' for no blending (remove special strings), 'msg' for message-level blending, 'chunk' for chunk-level blending (default: chunk)"
    )
    parser.add_argument(
        "--enable-prefix-cache-stats",
        action="store_true",
        help="Enable prefix cache statistics tracking (theoretical upper bound on cache hits)"
    )
    parser.add_argument(
        "--prefix-cache-block-size",
        type=int,
        default=16,
        help="Block size for prefix cache statistics (default: 16, following vLLM convention)"
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=0,
        help="Minimum chunk size (in tokens). Chunks smaller than this will be padded with pad tokens to reach the minimum size, allowing each chunk to benefit from caching. (default: 0, meaning no padding)"
    )
    parser.add_argument(
        "--chunk-dist-file",
        type=str,
        default=None,
        help="Path to write chunk size distribution (JSON format). Only used with chunk blend granularity."
    )
    parser.add_argument(
        "--enable-fixed-size-chunk",
        action="store_true",
        help="Enable fixed-size chunking. When enabled, min-chunk-size becomes THE chunk size: large chunks are split into fixed-size pieces and remainders are padded. Requires --min-chunk-size > 0."
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate mock mode arguments
    if args.mock and not args.mock_file:
        print("Error: --mock flag requires --mock-file to be specified")
        return
    
    # Validate fixed-size chunk arguments
    if args.enable_fixed_size_chunk and args.min_chunk_size <= 0:
        print("Error: --enable-fixed-size-chunk requires --min-chunk-size > 0")
        return
    
    # Create and run server
    server = BlendServer(
        model=args.model,
        use_disk=args.use_disk,
        blend_special_str=args.blend_special_str,
        log_file=args.log_file,
        enable_ttft=args.enable_ttft,
        blend_granularity=args.blend_granularity,
        enable_prefix_cache_stats=args.enable_prefix_cache_stats,
        prefix_cache_block_size=args.prefix_cache_block_size,
        min_chunk_size=args.min_chunk_size,
        chunk_dist_file=args.chunk_dist_file,
        enable_fixed_size_chunk=args.enable_fixed_size_chunk
    )
    
    print(f"Starting LMCache Blend Server on {args.host}:{args.port}")
    print(f"Model: {args.model}")
    print(f"Use disk: {args.use_disk}")
    print(f"Blend special string: {args.blend_special_str}")
    print(f"Blend granularity: {args.blend_granularity}")
    print(f"Request log file: {args.log_file}")
    print(f"TTFT measurement enabled: {args.enable_ttft}")
    print(f"Prefix cache stats enabled: {args.enable_prefix_cache_stats}")
    if args.enable_prefix_cache_stats:
        print(f"Prefix cache block size: {args.prefix_cache_block_size}")
    print(f"Min chunk size: {args.min_chunk_size}")
    print(f"Fixed-size chunking enabled: {args.enable_fixed_size_chunk}")
    if args.chunk_dist_file:
        print(f"Chunk distribution file: {args.chunk_dist_file}")
    
    if args.mock:
        print(f"Mock mode enabled - replaying requests from: {args.mock_file}")
        import asyncio
        asyncio.run(server.run_mock_experiment(args.mock_file))
    else:
        server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main() 