#!/usr/bin/env python3
"""
Example client for LMCache Blend Server

This client demonstrates how to interact with the blend server using the same
format as the user's example, but with the model fixed by the backend.
"""

import requests
import json
import time


def chat_with_blend_server(
    url: str = "http://localhost:80/v1/chat/completions",
    system_prompt: str = "You are a very helpful assistant.",
    user_message: str = "How are you today?",
    temperature: float = 0.15,
    max_tokens: int = 100
):
    """Send a chat completion request to the blend server"""
    
    headers = {
        "Content-Type": "application/json", 
        "Authorization": "Bearer token"  # Note: server doesn't validate this currently
    }

    # Note: model is fixed by the backend, but we still need to provide it in the request
    # The server will use its configured model regardless of what's sent
    model = "mistralai/Devstral-Small-2507"  # This will be ignored by the server

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    data = {
        "model": model,  # Will be ignored by the server
        "messages": messages, 
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    print(f"Sending request to {url}")
    print(f"System prompt: {system_prompt}")
    print(f"User message: {user_message}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_tokens}")
    print("-" * 50)

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        
        result = response.json()
        assistant_message = result["choices"][0]["message"]["content"]
        usage = result["usage"]
        
        print(f"Assistant response: {assistant_message}")
        print(f"Usage: {usage}")
        print("-" * 50)
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None


def demonstrate_blending():
    """Demonstrate the blending capabilities with multiple requests"""
    
    url = "http://localhost:80/v1/chat/completions"
    
    # First request
    print("=== FIRST REQUEST ===")
    chat_with_blend_server(
        url=url,
        system_prompt="You are a very helpful assistant.",
        user_message="Hello, how are you?" * 10,  # Repeat to create chunks
        temperature=0.0,
        max_tokens=20
    )
    
    time.sleep(1)
    
    # Second request with different order
    print("\n=== SECOND REQUEST (different order) ===")
    chat_with_blend_server(
        url=url,
        system_prompt="You are a very helpful assistant.",
        user_message="Hello, what's up?" * 10,  # Different content
        temperature=0.0,
        max_tokens=20
    )
    
    time.sleep(1)
    
    # Third request - repeat first to show caching
    print("\n=== THIRD REQUEST (repeat first) ===")
    chat_with_blend_server(
        url=url,
        system_prompt="You are a very helpful assistant.",
        user_message="Hello, how are you?" * 10,  # Same as first request
        temperature=0.0,
        max_tokens=20
    )


def main():
    """Main function to run the client example"""
    
    print("LMCache Blend Server Client Example")
    print("=" * 50)
    
    # Check if server is running
    try:
        health_response = requests.get("http://localhost:80/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"Server is healthy! Model: {health_data['model']}")
        else:
            print("Server health check failed")
            return
    except requests.exceptions.ConnectionError:
        print("Cannot connect to server. Make sure the blend server is running on localhost:80")
        print("Start it with: python blend_server.py")
        return
    
    # Run demonstration
    demonstrate_blending()


if __name__ == "__main__":
    main() 