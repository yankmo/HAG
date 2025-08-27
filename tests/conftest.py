#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest配置文件 - 基本测试配置
"""

import pytest
import os
import sys
from unittest.mock import Mock

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@pytest.fixture
def test_config():
    """基本测试配置"""
    return {
        'ollama': {
            'base_url': 'http://localhost:11434',
            'model': 'test_model'
        }
    }





@pytest.fixture
def mock_embedding_service():
    """模拟向量化服务"""
    service = Mock()
    service.embed_text.return_value = [0.1] * 384
    return service


@pytest.fixture
def mock_vector_store():
    """模拟向量存储"""
    store = Mock()
    store.search.return_value = [{'content': 'test content', 'score': 0.9}]
    return store


@pytest.fixture
def mock_graph_service():
    """模拟图谱服务"""
    service = Mock()
    service.search_entities_by_name.return_value = [{'name': 'test_entity', 'type': 'test'}]
    return service


@pytest.fixture
def mock_llm_service():
    """模拟LLM服务"""
    service = Mock()
    service.generate_response.return_value = "test response"
    return service