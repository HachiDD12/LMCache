# Standard
import argparse
import contextlib
import json
import os
import time
import signal
from typing import List, Optional

# Third Party
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs
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
    llm_args = EngineArgs(
        model=model,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        tensor_parallel_size=4, # TODO: add this back in once have 2 GPUs
        kv_transfer_config=ktc,
        max_model_len=128000,    # TODO: change this to 128000 once hosting devstral
        gpu_memory_utilization=0.7,
        enable_prefix_caching=False,
    )
    
    # # mistralai/Mistral-7B-Instruct-v0.2 args
    # llm_args = EngineArgs(
    #     model=model,
    #     tokenizer_mode="mistral",
    #     kv_transfer_config=ktc,
    #     max_model_len=32000,
    #     gpu_memory_utilization=0.8,
    #     enable_prefix_caching=False,
    # )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        # Clean up lmcache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


class BlendServer:
    def __init__(self, model: str, use_disk: bool = False, blend_special_str: str = " # # "):
        self.model = model
        self.use_disk = use_disk
        self.blend_special_str = blend_special_str
        self.lmcache_connector = "LMCacheConnectorV1"
        
        # Setup environment variables
        setup_environment_variables(use_disk, blend_special_str)
        
        # Load system prompt for Devstral
        self.system_prompt = self.load_system_prompt(model, "SYSTEM_PROMPT.txt")
        
        # Initialize Devstral tokenizer
        self.tokenizer = MistralTokenizer.from_hf_hub(model)
        
        # Initialize LLM with LMCache
        self.llm_context = build_llm_with_lmcache(self.lmcache_connector, model)
        self.llm = self.llm_context.__enter__()
        
        # Create FastAPI app
        self.app = FastAPI(title="LMCache Blend Server", version="1.0.0")
        self.setup_routes()
    
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
        
        return tokenized.tokens
    
    def messages_to_prompt_blend(self, messages: List[ChatMessage]) -> List[int]:
        """Convert chat messages to tokenized prompt using Devstral tokenizer as well as default Tekkenizer
        for the blend special string, such that each message is a separate blend chunk.
        """
        # Convert our ChatMessage format to Mistral format
        for message in messages:
            mistral_messages = []
            prompt_tokens = []
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
            blend_tokens = self.tokenizer.instruct_tokenizer.tokenizer.encode(self.blend_special_str, bos=False, eos=False)
            prompt_tokens.extend(blend_tokens)
        
        return prompt_tokens
    
    def handle_timeout(self, signum, frame):
        raise Exception("Generation timed out")
    
    async def handle_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle chat completion request"""
        print(f"@@@@@ Handling chat completion request of n = {request.n}")
        try:
            # Convert messages to tokenized prompt
            prompt_tokens = self.messages_to_prompt_blend(request.messages)
            
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
                prompt_token_ids=prompt_tokens, 
                sampling_params=sampling_params,
            )
            end_time = time.time()
            signal.alarm(0)
            print(f"[INFO] Generation time: {end_time - start_time} seconds for {len(prompt_tokens)} tokens")
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
            
            return response
            
        except Exception as e:
            signal.alarm(0)
            print(f"[ERROR] Error handling chat completion, type: {type(e)}, message: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the server"""
        uvicorn.run(self.app, host=host, port=port)
    
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
        default="# #",
        help="Special separator for blending chunks"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create and run server
    server = BlendServer(
        model=args.model,
        use_disk=args.use_disk,
        blend_special_str=args.blend_special_str
    )
    
    print(f"Starting LMCache Blend Server on {args.host}:{args.port}")
    print(f"Model: {args.model}")
    print(f"Use disk: {args.use_disk}")
    print(f"Blend special string: {args.blend_special_str}")
    
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main() 