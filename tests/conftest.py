#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest配置文件 - 全局测试配置和fixtures
提供测试环境配置、公共fixtures和测试工具
"""

import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock
import logging

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 配置测试日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@pytest.fixture(scope="session")
def test_config():
    """测试配置fixture"""
    return {
        'neo4j': {
            'uri': 'bolt://localhost:7687',
            'username': 'neo4j',
            'password': 'test_password',
            'database': 'test_db'
        },
        'weaviate': {
            'url': 'http://localhost:8080',
            'api_key': None
        },
        'ollama': {
            'base_url': 'http://localhost:11434',
            'model': 'gemma3:4b',
            'embedding_model': 'nomic-embed-text'
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 1,  # 使用测试数据库
            'password': None
        },
        'cache': {
            'type': 'lru',
            'max_size': 1000,
            'ttl': 3600
        },
        'weight_manager': {
            'strategies': {
                'fixed': {'doc_weight': 0.6, 'graph_weight': 0.4},
                'adaptive': {'base_doc_weight': 0.5, 'base_graph_weight': 0.5},
                'performance_based': {'initial_doc_weight': 0.6, 'initial_graph_weight': 0.4}
            },
            'ab_test': {'enabled': True, 'test_ratio': 0.5}
        }
    }


@pytest.fixture(scope="session")
def temp_dir():
    """临时目录fixture"""
    temp_path = tempfile.mkdtemp(prefix="hag_test_")
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_embedding_service():
    """模拟向量化服务fixture"""
    service = Mock()
    service.embed_text.return_value = [0.1] * 384  # 384维向量
    service.embed_documents.return_value = [[0.1] * 384, [0.2] * 384]
    service.get_embedding_dimension.return_value = 384
    return service


@pytest.fixture
def mock_vector_store():
    """模拟向量存储fixture"""
    store = Mock()
    store.search.return_value = [
        {
            'content': '人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。',
            'score': 0.95,
            'metadata': {'source': 'ai_basics.txt', 'chunk_id': 1}
        },
        {
            'content': '机器学习是人工智能的一个子领域，专注于算法和统计模型的开发。',
            'score': 0.88,
            'metadata': {'source': 'ml_intro.txt', 'chunk_id': 2}
        }
    ]
    store.add_documents.return_value = True
    store.delete_documents.return_value = True
    store.get_stats.return_value = {
        'total_documents': 1000,
        'total_vectors': 1000,
        'index_size': '50MB'
    }
    return store


@pytest.fixture
def mock_graph_service():
    """模拟图谱服务fixture"""
    service = Mock()
    
    # 模拟实体搜索
    service.search_entities_by_name.return_value = [
        {
            'name': '人工智能',
            'type': 'Technology',
            'description': '模拟人类智能的技术',
            'properties': {'field': 'Computer Science'}
        },
        {
            'name': '机器学习',
            'type': 'Technology',
            'description': 'AI的子领域',
            'properties': {'field': 'Computer Science'}
        }
    ]
    
    # 模拟关系搜索
    service.search_relationships_by_query.return_value = [
        {
            'source': '人工智能',
            'target': '机器学习',
            'type': 'INCLUDES',
            'description': '人工智能包含机器学习',
            'relevance_score': 0.92
        }
    ]
    
    # 模拟实体关系获取
    service.get_entity_relationships.return_value = {
        'relationships': [
            {
                'entity': '人工智能',
                'related_entity': '机器学习',
                'relation_type': 'INCLUDES',
                'relation_description': '人工智能包含机器学习'
            }
        ]
    }
    
    service.get_stats.return_value = {
        'total_nodes': 1000,
        'total_relationships': 2500,
        'query_count': 150
    }
    
    return service


@pytest.fixture
def mock_llm_service():
    """模拟LLM服务fixture"""
    service = Mock()
    service.generate_response.return_value = (
        "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"
        "它包括机器学习、深度学习、自然语言处理等多个子领域。AI的目标是让机器能够理解、学习和推理。"
    )
    service.generate_response_async.return_value = service.generate_response.return_value
    return service


@pytest.fixture
def mock_cache_manager():
    """模拟缓存管理器fixture"""
    cache = Mock()
    cache_data = {}
    
    def mock_get(key):
        return cache_data.get(key)
    
    def mock_set(key, value, ttl=None):
        cache_data[key] = value
        return True
    
    def mock_delete(key):
        return cache_data.pop(key, None) is not None
    
    def mock_clear():
        cache_data.clear()
        return True
    
    cache.get.side_effect = mock_get
    cache.set.side_effect = mock_set
    cache.delete.side_effect = mock_delete
    cache.clear.side_effect = mock_clear
    cache.get_stats.return_value = {
        'hits': 100,
        'misses': 50,
        'hit_rate': 0.67,
        'size': len(cache_data)
    }
    
    return cache


@pytest.fixture
def mock_weight_manager():
    """模拟权重管理器fixture"""
    manager = Mock()
    
    def mock_get_weights(query, doc_results=None, graph_results=None, **kwargs):
        # 根据结果数量动态调整权重
        doc_count = len(doc_results) if doc_results else 0
        graph_count = len(graph_results) if graph_results else 0
        
        if doc_count > graph_count:
            return {'doc_weight': 0.7, 'graph_weight': 0.3}
        elif graph_count > doc_count:
            return {'doc_weight': 0.3, 'graph_weight': 0.7}
        else:
            return {'doc_weight': 0.5, 'graph_weight': 0.5}
    
    manager.get_weights.side_effect = mock_get_weights
    manager.set_strategy.return_value = True
    manager.get_current_strategy.return_value = 'adaptive'
    manager.get_performance_stats.return_value = {
        'strategy': 'adaptive',
        'avg_response_time': 0.5,
        'success_rate': 0.95
    }
    
    return manager


@pytest.fixture
def sample_documents():
    """示例文档数据fixture"""
    return [
        {
            'content': '人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。',
            'metadata': {'source': 'ai_intro.txt', 'author': 'AI研究团队', 'date': '2024-01-01'}
        },
        {
            'content': '机器学习是人工智能的一个子领域，专注于算法和统计模型的开发，使计算机能够在没有明确编程的情况下学习和改进。',
            'metadata': {'source': 'ml_basics.txt', 'author': 'ML专家', 'date': '2024-01-02'}
        },
        {
            'content': '深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的工作方式，在图像识别、自然语言处理等领域取得了突破性进展。',
            'metadata': {'source': 'dl_overview.txt', 'author': 'DL研究者', 'date': '2024-01-03'}
        }
    ]


@pytest.fixture
def sample_entities():
    """示例实体数据fixture"""
    return [
        {
            'name': '人工智能',
            'type': 'Technology',
            'description': '模拟人类智能的技术领域',
            'properties': {
                'field': 'Computer Science',
                'established': '1956',
                'applications': ['机器学习', '自然语言处理', '计算机视觉']
            }
        },
        {
            'name': '机器学习',
            'type': 'Technology',
            'description': '使计算机能够学习的算法和方法',
            'properties': {
                'field': 'Computer Science',
                'parent': '人工智能',
                'types': ['监督学习', '无监督学习', '强化学习']
            }
        },
        {
            'name': '深度学习',
            'type': 'Technology',
            'description': '基于神经网络的机器学习方法',
            'properties': {
                'field': 'Computer Science',
                'parent': '机器学习',
                'architecture': ['CNN', 'RNN', 'Transformer']
            }
        }
    ]


@pytest.fixture
def sample_relationships():
    """示例关系数据fixture"""
    return [
        {
            'source': '人工智能',
            'target': '机器学习',
            'type': 'INCLUDES',
            'description': '人工智能包含机器学习作为其子领域',
            'properties': {'strength': 0.9, 'established': '1960s'}
        },
        {
            'source': '机器学习',
            'target': '深度学习',
            'type': 'INCLUDES',
            'description': '机器学习包含深度学习作为其分支',
            'properties': {'strength': 0.8, 'established': '2000s'}
        },
        {
            'source': '深度学习',
            'target': '神经网络',
            'type': 'USES',
            'description': '深度学习使用神经网络作为基础架构',
            'properties': {'strength': 0.95, 'dependency': 'high'}
        }
    ]


@pytest.fixture
def sample_queries():
    """示例查询数据fixture"""
    return [
        {
            'question': '什么是人工智能？',
            'expected_entities': ['人工智能'],
            'expected_keywords': ['计算机科学', '智能', '系统']
        },
        {
            'question': '机器学习和深度学习的区别是什么？',
            'expected_entities': ['机器学习', '深度学习'],
            'expected_keywords': ['算法', '神经网络', '学习']
        },
        {
            'question': '人工智能有哪些应用领域？',
            'expected_entities': ['人工智能'],
            'expected_keywords': ['应用', '领域', '技术']
        }
    ]


# 测试标记定义
pytest_plugins = []


def pytest_configure(config):
    """pytest配置钩子"""
    # 注册自定义标记
    config.addinivalue_line(
        "markers", "unit: 单元测试标记"
    )
    config.addinivalue_line(
        "markers", "integration: 集成测试标记"
    )
    config.addinivalue_line(
        "markers", "performance: 性能测试标记"
    )
    config.addinivalue_line(
        "markers", "slow: 慢速测试标记"
    )
    config.addinivalue_line(
        "markers", "requires_services: 需要外部服务的测试"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试收集项"""
    # 为性能测试添加slow标记
    for item in items:
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # 为集成测试添加requires_services标记
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.requires_services)


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """自动设置测试环境"""
    # 设置测试环境变量
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    
    # 禁用外部服务连接（在单元测试中）
    monkeypatch.setenv("DISABLE_EXTERNAL_SERVICES", "true")


@pytest.fixture
def performance_threshold():
    """性能阈值配置fixture"""
    return {
        'single_query_max_time': 5.0,  # 单查询最大时间（秒）
        'concurrent_success_rate': 0.8,  # 并发成功率
        'memory_growth_limit': 100.0,  # 内存增长限制（MB）
        'cache_hit_rate_min': 0.5,  # 最小缓存命中率
        'qps_min': 1.0,  # 最小QPS
        'p95_response_time': 10.0,  # P95响应时间（秒）
        'cpu_usage_max': 80.0,  # 最大CPU使用率（%）
    }


class TestDataManager:
    """测试数据管理器"""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.data_dir = os.path.join(temp_dir, "test_data")
        os.makedirs(self.data_dir, exist_ok=True)
    
    def create_test_file(self, filename: str, content: str) -> str:
        """创建测试文件"""
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    def get_test_file_path(self, filename: str) -> str:
        """获取测试文件路径"""
        return os.path.join(self.data_dir, filename)
    
    def cleanup(self):
        """清理测试数据"""
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)


@pytest.fixture
def test_data_manager(temp_dir):
    """测试数据管理器fixture"""
    manager = TestDataManager(temp_dir)
    yield manager
    manager.cleanup()


# 跳过条件定义
skip_if_no_services = pytest.mark.skipif(
    os.getenv("DISABLE_EXTERNAL_SERVICES") == "true",
    reason="外部服务被禁用"
)

skip_if_slow = pytest.mark.skipif(
    os.getenv("SKIP_SLOW_TESTS") == "true",
    reason="跳过慢速测试"
)


# 测试报告钩子（需要安装pytest-html插件）
# def pytest_html_report_title(report):
#     """自定义HTML报告标题"""
#     report.title = "HAG系统测试报告"


# def pytest_html_results_summary(prefix, summary, postfix):
#     """自定义HTML报告摘要"""
#     prefix.extend([
#         "<p>HAG (Hybrid AI Graph) 系统测试结果</p>",
#         "<p>测试覆盖：单元测试、集成测试、性能测试</p>"
#     ])


def pytest_runtest_makereport(item, call):
    """测试运行报告钩子"""
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item


def pytest_runtest_setup(item):
    """测试设置钩子"""
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail(f"previous test failed ({previousfailed.name})")


def pytest_sessionfinish(session, exitstatus):
    """测试会话结束钩子"""
    # 可以在这里添加测试结束后的清理工作
    pass