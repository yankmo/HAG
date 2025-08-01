#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置系统使用示例
演示如何在项目中使用统一的配置管理
"""

import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import ConfigManager


def demonstrate_config_usage():
    """演示配置系统的使用"""
    print("🔧 HAG项目配置系统使用示例")
    print("=" * 50)
    
    # 初始化配置管理器
    config = ConfigManager()
    
    print("📋 当前配置信息:")
    print(f"Neo4j URI: {config.neo4j.uri}")
    print(f"Neo4j 用户名: {config.neo4j.username}")
    print(f"Neo4j 密码: {'已设置' if config.neo4j.password else '未设置'}")
    print(f"Neo4j 数据库: {config.neo4j.database}")
    print()
    
    print(f"Ollama 基础URL: {config.ollama.base_url}")
    print(f"Ollama 默认模型: {config.ollama.default_model}")
    print(f"Ollama 嵌入模型: {config.ollama.embedding_model}")
    print(f"Ollama 超时时间: {config.ollama.timeout}秒")
    print(f"Ollama 生成API: {config.ollama.api_generate_url}")
    print(f"Ollama 标签API: {config.ollama.api_tags_url}")
    print()
    
    print(f"Weaviate URL: {config.weaviate.url}")
    print(f"Weaviate 主机: {config.weaviate.host}")
    print(f"Weaviate 端口: {config.weaviate.port}")
    print(f"Weaviate 元数据API: {config.weaviate.meta_url}")
    print()
    
    print(f"应用调试模式: {config.app.debug}")
    print(f"日志级别: {config.app.log_level}")
    print(f"最大块大小: {config.app.max_chunk_size}")
    print(f"块重叠大小: {config.app.chunk_overlap}")
    print(f"最大结果数: {config.app.max_results}")
    print()


def demonstrate_service_connections():
    """演示如何使用配置连接各种服务"""
    print("🔗 服务连接示例")
    print("=" * 50)
    
    config = ConfigManager()
    
    # Neo4j连接示例
    print("📊 Neo4j连接示例:")
    print(f"from py2neo import Graph")
    print(f"graph = Graph('{config.neo4j.uri}', auth={config.neo4j.to_auth_tuple()})")
    print()
    
    # Ollama连接示例
    print("🤖 Ollama连接示例:")
    print(f"import ollama")
    print(f"client = ollama.Client(host='{config.ollama.base_url}')")
    print(f"# 使用默认模型: {config.ollama.default_model}")
    print(f"# 使用嵌入模型: {config.ollama.embedding_model}")
    print()
    
    # Weaviate连接示例
    print("🔍 Weaviate连接示例:")
    print(f"import weaviate")
    print(f"client = weaviate.Client(url='{config.weaviate.url}')")
    print()


def check_environment_variables():
    """检查环境变量设置"""
    print("🌍 环境变量检查")
    print("=" * 50)
    
    env_vars = [
        'NEO4J_URI',
        'NEO4J_USERNAME', 
        'NEO4J_PASSWORD',
        'NEO4J_DATABASE',
        'OLLAMA_BASE_URL',
        'OLLAMA_DEFAULT_MODEL',
        'OLLAMA_EMBEDDING_MODEL',
        'OLLAMA_TIMEOUT',
        'WEAVIATE_URL',
        'WEAVIATE_HOST',
        'WEAVIATE_PORT',
        'DEBUG',
        'LOG_LEVEL'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        status = "✅ 已设置" if value else "❌ 未设置"
        print(f"{var}: {status}")
        if value and var != 'NEO4J_PASSWORD':  # 不显示密码
            print(f"  值: {value}")
    print()


def demonstrate_config_priority():
    """演示配置优先级"""
    print("📋 配置优先级说明")
    print("=" * 50)
    print("配置加载优先级（从高到低）:")
    print("1. 环境变量 (.env文件)")
    print("2. 配置文件 (config.yaml)")
    print("3. 代码中的默认值")
    print()
    print("建议:")
    print("- 开发环境：使用.env文件设置敏感信息（如密码）")
    print("- 生产环境：使用环境变量")
    print("- 通用配置：使用config.yaml文件")
    print()


if __name__ == "__main__":
    demonstrate_config_usage()
    demonstrate_service_connections()
    check_environment_variables()
    demonstrate_config_priority()