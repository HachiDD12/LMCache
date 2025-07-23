#!/usr/bin/env python3
"""
Devstral Client Example for LMCache Blend Server

This example shows how to use the blend server with the exact format from the Devstral example,
but using the server instead of direct model inference.
"""

import requests
import json
from huggingface_hub import hf_hub_download


def load_system_prompt(repo_id: str, filename: str) -> str:
    """Load system prompt from HuggingFace hub (same as original example)"""
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()
    return system_prompt


def chat_with_devstral_server(
    url: str = "http://localhost:80/v1/chat/completions",
    model_id: str = "mistralai/Devstral-Small-2507",
    user_command: str = "<your-command>",
    max_tokens: int = 1000,
    temperature: float = 0.15
):
    """Send a chat completion request to the Devstral blend server"""
    
    headers = {
        "Content-Type": "application/json", 
        "Authorization": "Bearer token"
    }

    # Load system prompt (same as original example)
    system_prompt = load_system_prompt(model_id, "SYSTEM_PROMPT.txt")
    
    print(f"System prompt loaded: {system_prompt[:100]}...")
    print(f"User command: {user_command}")
    print("-" * 50)

    # Create messages in the same format as the original example
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_command},
    ]

    data = {
        "model": model_id,  # Will be ignored by server but required
        "messages": messages, 
        "temperature": temperature,
        "max_tokens": max_tokens
    }

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


def demonstrate_devstral_integration():
    """Demonstrate Devstral integration with multiple commands"""
    
    url = "http://localhost:80/v1/chat/completions"
    model_id = "mistralai/Devstral-Small-2507"
    
    # Test different commands
    commands = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Write a short poem about AI",
        "Explain quantum computing in simple terms"
    ]
    
    for i, command in enumerate(commands, 1):
        print(f"\n=== COMMAND {i} ===")
        chat_with_devstral_server(
            url=url,
            model_id=model_id,
            user_command=command,
            max_tokens=200,
            temperature=0.1
        )


def main():
    """Main function to run the Devstral client example"""
    
    print("Devstral LMCache Blend Server Client Example")
    print("=" * 60)
    
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
    demonstrate_devstral_integration()


if __name__ == "__main__":
    main() 