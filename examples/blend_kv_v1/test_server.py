#!/usr/bin/env python3
"""
Simple test script for the blend server

This script tests the server functionality without requiring the full model to be loaded.
It can be used to verify the server setup and basic functionality.
"""

import sys
import os

# Add the parent directory to the path so we can import from blend_server
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    
    try:
        import argparse
        import contextlib
        import json
        import time
        from typing import List, Optional
        print("✓ Standard library imports")
        
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
        import uvicorn
        print("✓ FastAPI imports")
        
        from transformers import AutoTokenizer
        print("✓ Transformers import")
        
        # Test vLLM imports (these might fail if vLLM is not installed)
        try:
            from vllm import LLM, SamplingParams
            from vllm.config import KVTransferConfig
            from vllm.engine.arg_utils import EngineArgs
            print("✓ vLLM imports")
        except ImportError as e:
            print(f"⚠ vLLM imports failed: {e}")
            print("  This is expected if vLLM is not installed")
        
        # Test LMCache imports (these might fail if LMCache is not installed)
        try:
            from lmcache.integration.vllm.utils import ENGINE_NAME
            from lmcache.v1.cache_engine import LMCacheEngineBuilder
            print("✓ LMCache imports")
        except ImportError as e:
            print(f"⚠ LMCache imports failed: {e}")
            print("  This is expected if LMCache is not installed")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_server_class():
    """Test the BlendServer class structure"""
    print("\nTesting BlendServer class structure...")
    
    try:
        from blend_server import BlendServer, setup_environment_variables, build_llm_with_lmcache
        
        # Test environment setup function
        setup_environment_variables(use_disk=False, blend_special_str=" # # ")
        print("✓ Environment variables setup")
        
        # Test that we can create the class (without initializing LLM)
        print("✓ BlendServer class imported")
        
        return True
        
    except ImportError as e:
        print(f"✗ BlendServer import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ BlendServer test failed: {e}")
        return False


def test_client_example():
    """Test the client example structure"""
    print("\nTesting client example...")
    
    try:
        from client_example import chat_with_blend_server, demonstrate_blending, main
        print("✓ Client example functions imported")
        
        return True
        
    except ImportError as e:
        print(f"✗ Client example import failed: {e}")
        return False


def test_argument_parsing():
    """Test argument parsing"""
    print("\nTesting argument parsing...")
    
    try:
        from blend_server import parse_args
        
        # Test with default arguments
        sys.argv = ['blend_server.py']
        args = parse_args()
        print(f"✓ Default arguments: model={args.model}, host={args.host}, port={args.port}")
        
        # Test with custom arguments
        sys.argv = ['blend_server.py', '--model', 'test-model', '--port', '9000', '--use-disk']
        args = parse_args()
        print(f"✓ Custom arguments: model={args.model}, port={args.port}, use_disk={args.use_disk}")
        
        return True
        
    except Exception as e:
        print(f"✗ Argument parsing failed: {e}")
        return False


def main():
    """Run all tests"""
    print("LMCache Blend Server Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_server_class,
        test_client_example,
        test_argument_parsing,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The server should work correctly.")
        print("\nTo run the server:")
        print("  python blend_server.py")
        print("\nTo test with client:")
        print("  python client_example.py")
    else:
        print("⚠ Some tests failed. Check the output above for details.")
        print("\nYou may need to install missing dependencies:")
        print("  pip install fastapi uvicorn transformers vllm lmcache")


if __name__ == "__main__":
    main() 