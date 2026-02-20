# Standard
import argparse
import asyncio
import contextlib
import json
import os
import time
import signal
import uuid
import sys
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

def build_llm_vllm(model: str, enable_prefix_caching: bool = False):
    """Build LLM without LMCache integration"""
    print(f"@@@@@ Building LLM with vllm")
    # --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --tensor-parallel-size 2
    llm_args = AsyncEngineArgs(
        model=model,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        tensor_parallel_size=4,         # TODO: add this back in once have 2 GPUs
        # kv_transfer_config=ktc,
        max_model_len=128000,           # TODO: change this to 128000 once hosting devstral
        gpu_memory_utilization=0.7,
        enable_prefix_caching=enable_prefix_caching,
        seed=42,                        # For reproducibility
    )

    # llm = LLM(**asdict(llm_args))
    print(f"@@@@@ Creating AsyncLLMEngine...")
    
    sys.stdout.flush()  # Force flush to see where we are
    llm = AsyncLLMEngine.from_engine_args(llm_args)
    print(f"@@@@@ Returning llm object")
    sys.stdout.flush()
    return llm


class BlendServer:
    def __init__(self, 
                 model: str, 
                 use_disk: bool = False, 
                 blend_special_str: str = " # # ", 
                 log_file: str = "chat_requests.json", 
                 enable_prefix_caching: bool = False):
        self.model = model
        self.use_disk = use_disk
        self.blend_special_str = blend_special_str
        self.log_file = log_file
        self.enable_prefix_caching = enable_prefix_caching
        
        # Setup environment variables
        setup_environment_variables(use_disk, blend_special_str)
        
        # Load system prompt for Devstral
        self.system_prompt = self.load_system_prompt(model, "SYSTEM_PROMPT.txt")
        
        # Initialize Devstral tokenizer
        self.tokenizer = MistralTokenizer.from_hf_hub(model)
        self.tekkenizer = self.tokenizer.instruct_tokenizer.tokenizer
        
        # Initialize LLM without LMCache
        print(f"@@@@@ About to create LLM...")
        
        sys.stdout.flush()
        print(f"@@@@@ Calling build_llm_vllm...")
        sys.stdout.flush()
        try:
            result = build_llm_vllm(model, enable_prefix_caching)
            print(f"@@@@@ build_llm_vllm returned!")
            sys.stdout.flush()
            self.llm_vllm = result
            print(f"@@@@@ Assignment complete, engine object: {type(self.llm_vllm)}")
            sys.stdout.flush()
            print(f"@@@@@ LLM initialized (vllm)")
            sys.stdout.flush()
            # Create FastAPI app
            self.app = FastAPI(title="vLLM Server", version="1.0.0")
            print(f"@@@@@ FastAPI app created")
            sys.stdout.flush()
            self.setup_routes()
            print(f"@@@@@ FastAPI app setup complete")
            sys.stdout.flush()
        except Exception as e:
            print(f"@@@@@ ERROR in build_llm_vllm: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise
        
    
    def log_request(self, request: ChatCompletionRequest, response: ChatCompletionResponse, generation_time: float, ttft: float):
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
            "generation_time": generation_time,
            "ttft": ttft
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
        """Get stitched tokens for a single message content"""
        tokens = []
        # split on blend special string
        msg_chunks = msg_content.split(self.blend_special_str)
        print(f"[INFO] split into {len(msg_chunks)} chunks")
        for chunk in msg_chunks:
            tokens.extend(self.tekkenizer.encode(chunk, bos=False, eos=False))
            tokens.extend(blend_tokens)
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
        
            # Add blend special string
            prompt_tokens.extend(blend_tokens)

        # res = prompt_tokens[:-len(blend_tokens)] + [eos]
        res = prompt_tokens[:-len(blend_tokens)]
        # print(f"@@@@@ Prompt tokens blend by chunk: {res}")
        return res
    
    def get_prompt_token_length(self, messages: List[ChatMessage]) -> int:
        """Get the token length of the prompt"""
        prompt_tokens = self.messages_to_prompt_blend_by_chunk(messages)
        return len(prompt_tokens)
    
    def handle_timeout(self, signum, frame):
        raise Exception("Generation timed out")
    
    async def handle_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle chat completion request with TTFT measurement"""
        print(f"@@@@@ Handling chat completion request of n = {request.n}")
        sys.stdout.flush()
        try:
            # Convert messages to tokenized prompt (use naive tokenization for no-lmcache version)
            prompt_tokens = self.messages_to_prompt(request.messages)
            
            # Create sampling parameters with DELTA output for streaming
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                n=request.n,
                max_tokens=request.max_tokens,
                output_kind=RequestOutputKind.DELTA
            )
            
            # Generate response with TTFT measurement
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(300)
            start_time = time.time()
            first_token_time = None
            ttft = None
            words = ""
            
            # Stream through the outputs to measure TTFT
            async for output in self.llm_vllm.generate(
                prompt=TokensPrompt(prompt_token_ids=prompt_tokens), 
                sampling_params=sampling_params,
                request_id=str(uuid.uuid4())
            ):
                chunk_tokens = output.outputs[0].text  # assumes single completion
                if chunk_tokens is not None:
                    if first_token_time is None and chunk_tokens != "":
                        first_token_time = time.time()
                        ttft = first_token_time - start_time
                        print(f"[INFO] TTFT: {ttft:.4f} seconds")
                    words += chunk_tokens
                if output.finished:
                    break
            
            sys.stdout.flush()
            end_time = time.time()
            generation_time = end_time - start_time
            signal.alarm(0)
            
            print(f"[INFO] Generation time: {generation_time} seconds for {len(prompt_tokens)} tokens")
            sys.stdout.flush()
            
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
            sys.stdout.flush()
            
            # Log the request and response with TTFT
            self.log_request(request, response, generation_time, ttft if ttft is not None else 0.0)
            
            return response
            
        except Exception as e:
            signal.alarm(0)
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
            sys.stdout.flush()
            return
        
        # Sort by original timestamp if available
        mock_requests.sort(key=lambda x: x.get('original_timestamp', ''))
        
        print(f"[MOCK] Replaying {len(mock_requests)} requests...")
        sys.stdout.flush()
        
        total_original_time = 0
        total_mock_time = 0
        total_ttft = 0
        successful_requests = 0
        
        for i, mock_data in enumerate(mock_requests):
            request = mock_data['request']
            original_time = mock_data.get('original_generation_time', 0)
            original_timestamp = mock_data.get('original_timestamp', 'unknown')
            
            print(f"[MOCK] Processing request {i+1}/{len(mock_requests)} (original: {original_timestamp})")
            print(f"[MOCK] Request: {len(request.messages)} messages, max_tokens: {request.max_tokens}")
            sys.stdout.flush()
            
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
        sys.stdout.flush()
    
    async def handle_chat_completion_mock(self, request: ChatCompletionRequest, rid: int) -> ChatCompletionResponse:
        """Mock version of handle_chat_completion for mock mode with TTFT measurement"""
        print(f"@@@@@ Handling chat completion request {rid} of n = {request.n} (mock)")
        sys.stdout.flush()
        try:
            # Convert messages to tokenized prompt (use naive tokenization for no-lmcache version)
            prompt_tokens = self.messages_to_prompt(request.messages)
            print(f"@@@@@ Prompt tokens (length: {len(prompt_tokens)}): {prompt_tokens}")
            sys.stdout.flush()
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                n=request.n,
                max_tokens=request.max_tokens,
                output_kind=RequestOutputKind.DELTA
            )
            # print(f"@@@@@ Sampling params: {sampling_params}")
            # sys.stdout.flush()
            # Generate response
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(300)
            # print(f"@@@@@ Signal alarm set to 300 seconds")
            # sys.stdout.flush()
            start_time = time.time()
            first_token_time = None
            words = ""
            ttft = None
            
            # Stream through the outputs to measure TTFT
            async for output in self.llm_vllm.generate(
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
            sys.stdout.flush()
            end_time = time.time()
            generation_time = end_time - start_time
            signal.alarm(0)
            
            print(f"[INFO] Generation time: {generation_time} seconds for {len(prompt_tokens)} tokens")
            print(f"[INFO] Final output: {words}")
            sys.stdout.flush()
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
            sys.stdout.flush()
            return response
            
        except Exception as e:
            signal.alarm(0)
            print(f"[ERROR] Error handling chat completion, type: {type(e)}, message: {str(e)}")
            raise e
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the server"""
        print(f"@@@@@ About to run server")
        sys.stdout.flush()
        uvicorn.run(self.app, host=host, port=port)
        print(f"@@@@@ Server run complete")
        sys.stdout.flush()
    
    def __del__(self):
        """Cleanup when server is destroyed"""
        if hasattr(self, 'llm_vllm'):
            # AsyncLLMEngine cleanup is handled automatically
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Server")
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
        "--enable-prefix-caching",
        action="store_true",
        help="Enable vLLM prefix caching"
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
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"@@@@@ main() started")
    
    # Validate mock mode arguments
    if args.mock and not args.mock_file:
        print("Error: --mock flag requires --mock-file to be specified")
        return
    
    print(f"@@@@@ About to create BlendServer...")
    # Create and run server
    server = BlendServer(
        model=args.model,
        use_disk=args.use_disk,
        blend_special_str=args.blend_special_str,
        log_file=args.log_file,
        enable_prefix_caching=args.enable_prefix_caching
    )
    print(f"@@@@@ Enable prefix caching: {args.enable_prefix_caching}")
    sys.stdout.flush()
    
    print(f"@@@@@ BlendServer created successfully")
    print(f"Starting vLLM Server on {args.host}:{args.port}")
    print(f"Model: {args.model}")
    print(f"Request log file: {args.log_file}")
    sys.stdout.flush()
    
    if args.mock:
        print(f"Mock mode enabled - replaying requests from: {args.mock_file}")
        sys.stdout.flush()
        import asyncio
        asyncio.run(server.run_mock_experiment(args.mock_file))
    else:
        server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main() 