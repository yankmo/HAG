#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ollama 连接测试脚本
"""

import requests
import json
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import get_config
from ollama import embeddings, chat, Message

def test_direct_api():
    """直接测试 Ollama HTTP API"""
    try:
        config = get_config()
        print("=== 测试直接 HTTP API ===")
        response = requests.get(f"{config.ollama.base_url}/api/tags", timeout=10)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"可用模型: {[model['name'] for model in data['models']]}")
            return True
        else:
            print(f"API 错误: {response.text}")
            return False
    except Exception as e:
        print(f"HTTP API 测试失败: {e}")
        return False

def test_ollama_client():
    """测试 ollama Python 客户端"""
    try:
        config = get_config()
        print("\n=== 测试 ollama Python 客户端 ===")
        
        # 测试 embeddings
        print("测试 embeddings...")
        response = embeddings(model=config.ollama.embedding_model, prompt='Hello world')
        print(f"Embeddings 成功，向量维度: {len(response['embedding'])}")
        
        # 测试 chat
        print("测试 chat...")
        response = chat(config.ollama.default_model, [Message(role='user', content='Hello')])
        print(f"Chat 成功: {response['message']['content'][:50]}...")
        
        return True
    except Exception as e:
        print(f"ollama 客户端测试失败: {e}")
        return False

def test_with_timeout():
    """测试带超时的连接"""
    try:
        config = get_config()
        print("\n=== 测试带超时的连接 ===")
        import ollama
        
        # 设置客户端超时
        client = ollama.Client(host=config.ollama.base_url, timeout=config.ollama.timeout)
        
        # 测试 embeddings
        response = client.embeddings(model=config.ollama.embedding_model, prompt='test')
        print(f"带超时的 embeddings 成功，向量维度: {len(response['embedding'])}")
        
        return True
    except Exception as e:
        print(f"带超时的测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始 Ollama 连接诊断...\n")
    
    # 测试 1: 直接 HTTP API
    api_ok = test_direct_api()
    
    # 测试 2: ollama Python 客户端
    client_ok = test_ollama_client()
    
    # 测试 3: 带超时的连接
    timeout_ok = test_with_timeout()
    
    print(f"\n=== 诊断结果 ===")
    print(f"HTTP API: {'✓' if api_ok else '✗'}")
    print(f"Python 客户端: {'✓' if client_ok else '✗'}")
    print(f"带超时连接: {'✓' if timeout_ok else '✗'}")
    
    if not client_ok:
        print("\n建议解决方案:")
        print("1. 重启 Ollama 应用程序")
        print("2. 检查防火墙设置")
        print("3. 尝试使用不同的超时设置")
        print("4. 检查 ollama 库版本: pip show ollama")