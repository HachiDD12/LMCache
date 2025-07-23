# LMCache Blend Server

This directory contains a FastAPI server that implements LMCache blending functionality, mimicking the behavior of `blend.py` but serving as an API endpoint.

## Files

- `blend_server.py` - The main server implementation
- `client_example.py` - Example client demonstrating how to use the server
- `blend.py` - Original blending example (for reference)

## Features

- **LMCache Integration**: Uses LMCache for KV cache management with blending capabilities
- **OpenAI-Compatible API**: Implements `/v1/chat/completions` endpoint
- **Configurable Backend**: Supports both CPU and disk backends for LMCache
- **Blending Support**: Implements the same blending logic as the original `blend.py`

## Quick Start

### 1. Start the Server

```bash
# Basic usage with default settings
python blend_server.py

# With custom model
python blend_server.py --model mistralai/Mistral-7B-Instruct-v0.2

# With disk backend
python blend_server.py --use-disk

# With custom blend separator
python blend_server.py --blend-special-str "###"

# Custom host and port
python blend_server.py --host 0.0.0.0 --port 8000
```

### 2. Test with Client

```bash
# Run the example client
python client_example.py
```

### 3. Use with Your Own Client

The server accepts requests in the same format as your example:

```python
import requests
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json", "Authorization": "Bearer token"}

# Note: model is fixed by the backend, but still required in request
model = "mistralai/Mistral-7B-Instruct-v0.2"  # Will be ignored by server

messages = [
    {"role": "system", "content": "You are a very helpful assistant."},
    {"role": "user", "content": "How are you today?"},
]

data = {
    "model": model,  # Required but ignored
    "messages": messages, 
    "temperature": 0.15,
    "max_tokens": 100
}

response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()
print(result["choices"][0]["message"]["content"])
```

## Server Configuration

### Environment Variables

The server automatically sets up LMCache environment variables:

- `LMCACHE_CHUNK_SIZE`: Set to 256 tokens per chunk
- `LMCACHE_ENABLE_BLENDING`: Enabled for blending functionality
- `LMCACHE_BLEND_SPECIAL_STR`: Configurable blend separator
- `LMCACHE_USE_LAYERWISE`: Enabled for layerwise processing

### Backend Options

**CPU Backend (default):**
- Uses local CPU for LMCache storage
- Maximum size: 5GB

**Disk Backend (--use-disk):**
- Uses local disk for LMCache storage
- Maximum size: 10GB
- Path: `file://local_disk/`

## API Endpoints

### POST /v1/chat/completions

Main chat completion endpoint.

**Request Body:**
```json
{
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "top_p": 0.95,
  "max_tokens": 100
}
```

**Response:**
```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 10,
    "total_tokens": 25
  }
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "mistralai/Mistral-7B-Instruct-v0.2"
}
```

## Blending Logic

The server implements the same blending logic as `blend.py`:

1. **Message Tokenization**: Each message is tokenized separately
2. **Blend Separators**: Special separators are inserted between messages
3. **Chunk Management**: LMCache manages chunks of 256 tokens
4. **Cache Reuse**: Identical chunks are reused across requests

## Differences from Original blend.py

1. **API Interface**: Provides HTTP API instead of direct function calls
2. **Model Fixed**: Model is configured at server startup, not per request
3. **Async Processing**: Uses FastAPI for async request handling
4. **Error Handling**: Proper HTTP error responses
5. **Usage Tracking**: Tracks token usage for each request

## Requirements

- Python 3.8+
- FastAPI
- uvicorn
- vllm
- transformers
- requests (for client)
- lmcache

## Troubleshooting

1. **Server won't start**: Check if the model is available and GPU memory is sufficient
2. **Connection refused**: Ensure the server is running on the correct host/port
3. **Model loading errors**: Verify the model name and internet connection for downloads
4. **LMCache errors**: Check environment variables and backend configuration 