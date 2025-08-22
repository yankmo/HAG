#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端集成测试 - 完整检索流程测试
测试HAG系统的完整检索流程，包括数据库集成和API响应
"""

import pytest
import asyncio
import time
import json
import tempfile
import os
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import dataclass

# 导入被测试的模块
try:
    from api import HAGIntegratedAPI, IntegratedResponse, RetrievalStep
    from src.services import (
        HybridRetrievalService, RetrievalService, GraphRetrievalService,
        OllamaEmbeddingService, OllamaLLMService
    )
    from src.knowledge.vector_storage import WeaviateVectorStore
    from src.knowledge.neo4j_vector_storage import Neo4jVectorStore
    from config import get_config
except ImportError as e:
    pytest.skip(f"无法导入必要模块: {e}", allow_module_level=True)


class TestE2ERetrieval:
    """端到端检索流程测试"""
    
    @pytest.fixture
    def mock_config(self):
        """模拟配置"""
        config = Mock()
        config.neo4j = {
            'uri': 'bolt://localhost:7687',
            'username': 'neo4j',
            'password': 'password',
            'database': 'neo4j'
        }
        config.weaviate = {
            'url': 'http://localhost:8080',
            'api_key': None
        }
        config.ollama = {
            'base_url': 'http://localhost:11434',
            'model': 'gemma3:4b'
        }
        return config
    
    @pytest.fixture
    def mock_embedding_service(self):
        """模拟向量化服务"""
        service = Mock(spec=OllamaEmbeddingService)
        service.embed_text.return_value = [0.1] * 384  # 模拟384维向量
        service.embed_documents.return_value = [[0.1] * 384, [0.2] * 384]
        return service
    
    @pytest.fixture
    def mock_vector_store(self):
        """模拟向量存储"""
        store = Mock(spec=WeaviateVectorStore)
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
        return store
    
    @pytest.fixture
    def mock_graph_service(self):
        """模拟图谱检索服务"""
        service = Mock(spec=GraphRetrievalService)
        
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
            },
            {
                'source': '机器学习',
                'target': '深度学习',
                'type': 'INCLUDES',
                'description': '机器学习包含深度学习',
                'relevance_score': 0.87
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
    def mock_retrieval_service(self, mock_embedding_service, mock_vector_store):
        """模拟检索服务"""
        service = Mock(spec=RetrievalService)
        
        # 模拟混合搜索结果
        hybrid_result = Mock()
        hybrid_result.hybrid_results = [
            Mock(content='人工智能是计算机科学的一个分支', score=0.95),
            Mock(content='机器学习是人工智能的一个子领域', score=0.88)
        ]
        service.search_hybrid.return_value = hybrid_result
        
        service.get_stats.return_value = {
            'total_queries': 100,
            'avg_response_time': 0.5,
            'cache_hit_rate': 0.75
        }
        
        return service
    
    @pytest.fixture
    def mock_hybrid_service(self):
        """模拟混合检索服务"""
        service = Mock(spec=HybridRetrievalService)
        
        # 模拟混合检索结果
        hybrid_result = Mock()
        hybrid_result.documents = [
            {
                'content': '人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。',
                'score': 0.95,
                'metadata': {'source': 'ai_basics.txt'}
            }
        ]
        hybrid_result.entities = [
            {
                'name': '人工智能',
                'type': 'Technology',
                'properties': {'field': 'Computer Science'}
            }
        ]
        hybrid_result.relationships = [
            {
                'source': '人工智能',
                'target': '机器学习',
                'type': 'INCLUDES',
                'description': '人工智能包含机器学习'
            }
        ]
        hybrid_result.metadata = {
            'doc_weight': 0.6,
            'graph_weight': 0.4,
            'strategy': 'adaptive'
        }
        
        service.search_hybrid.return_value = hybrid_result
        return service
    
    @pytest.fixture
    def mock_llm_service(self):
        """模拟LLM服务"""
        service = Mock(spec=OllamaLLMService)
        service.generate_response.return_value = (
            "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"
            "它包括机器学习、深度学习、自然语言处理等多个子领域。AI的目标是让机器能够理解、学习和推理。"
        )
        return service
    
    @pytest.fixture
    def hag_api(self, mock_config, mock_embedding_service, mock_vector_store, 
                mock_retrieval_service, mock_graph_service, mock_hybrid_service, mock_llm_service):
        """创建HAG API实例"""
        with patch('api.get_config', return_value=mock_config), \
             patch('api.OllamaEmbeddingService', return_value=mock_embedding_service), \
             patch('api.WeaviateVectorStore', return_value=mock_vector_store), \
             patch('api.RetrievalService', return_value=mock_retrieval_service), \
             patch('api.GraphRetrievalService', return_value=mock_graph_service), \
             patch('api.HybridRetrievalService', return_value=mock_hybrid_service), \
             patch('api.OllamaLLMService', return_value=mock_llm_service), \
             patch('api.RAGPipeline'):
            
            api = HAGIntegratedAPI()
            return api

    def test_complete_retrieval_flow(self, hag_api):
        """测试完整的检索流程"""
        question = "什么是人工智能？"
        
        # 执行查询
        result = hag_api.query(question)
        
        # 验证响应结构
        assert isinstance(result, IntegratedResponse)
        assert result.answer is not None
        assert len(result.answer) > 0
        assert isinstance(result.sources, dict)
        assert isinstance(result.metadata, dict)
        assert isinstance(result.retrieval_process, list)
        
        # 验证检索步骤
        expected_steps = [
            "问题向量化",
            "文档检索", 
            "实体检索",
            "关系检索",
            "混合检索整合",
            "答案生成"
        ]
        
        actual_steps = [step.step_name for step in result.retrieval_process]
        for expected_step in expected_steps:
            assert expected_step in actual_steps
        
        # 验证每个步骤都有必要的属性
        for step in result.retrieval_process:
            assert isinstance(step, RetrievalStep)
            assert step.step_name is not None
            assert step.duration >= 0
            assert step.status in ['success', 'error', 'warning']
            assert step.result_count >= 0
    
    def test_sources_integration(self, hag_api):
        """测试数据源整合"""
        question = "人工智能和机器学习的关系是什么？"
        
        result = hag_api.query(question)
        
        # 验证数据源结构
        assert 'documents' in result.sources
        assert 'entities' in result.sources
        assert 'relationships' in result.sources
        
        # 验证文档数据
        documents = result.sources['documents']
        if documents:
            for doc in documents:
                assert 'content' in doc
                assert 'score' in doc
                assert 'metadata' in doc
        
        # 验证实体数据
        entities = result.sources['entities']
        if entities:
            for entity in entities:
                assert 'name' in entity
                assert 'type' in entity
                assert 'properties' in entity
        
        # 验证关系数据
        relationships = result.sources['relationships']
        if relationships:
            for rel in relationships:
                assert 'source' in rel
                assert 'target' in rel
                assert 'type' in rel
                assert 'description' in rel
    
    def test_metadata_tracking(self, hag_api):
        """测试元数据跟踪"""
        question = "深度学习的应用领域有哪些？"
        
        result = hag_api.query(question)
        
        # 验证元数据
        metadata = result.metadata
        assert 'question' in metadata
        assert metadata['question'] == question
        assert 'processing_method' in metadata
        assert 'total_processing_time' in metadata
        assert metadata['total_processing_time'] >= 0
    
    def test_error_handling(self, hag_api):
        """测试错误处理"""
        # 模拟服务异常
        hag_api.embedding_service.embed_text.side_effect = Exception("向量化服务异常")
        
        question = "测试错误处理"
        result = hag_api.query(question)
        
        # 验证错误处理
        assert isinstance(result, IntegratedResponse)
        assert "错误" in result.answer or "异常" in result.answer
        
        # 验证检索步骤中记录了错误
        error_steps = [step for step in result.retrieval_process if step.status == 'error']
        assert len(error_steps) > 0
    
    def test_performance_tracking(self, hag_api):
        """测试性能跟踪"""
        question = "测试性能跟踪"
        
        start_time = time.time()
        result = hag_api.query(question)
        end_time = time.time()
        
        # 验证性能指标
        total_duration = sum(step.duration for step in result.retrieval_process)
        assert total_duration > 0
        assert total_duration <= (end_time - start_time) + 0.1  # 允许小误差
        
        # 验证每个步骤的时间记录
        for step in result.retrieval_process:
            assert step.start_time > 0
            assert step.end_time > step.start_time
            assert step.duration == step.end_time - step.start_time
    
    def test_system_status(self, hag_api):
        """测试系统状态获取"""
        status = hag_api.get_system_status()
        
        # 验证状态结构
        assert isinstance(status, dict)
        assert 'status' in status
        assert 'services' in status
        
        # 验证服务状态
        services = status['services']
        expected_services = ['weaviate', 'neo4j', 'ollama']
        for service in expected_services:
            assert service in services
    
    def test_concurrent_queries(self, hag_api):
        """测试并发查询"""
        questions = [
            "什么是人工智能？",
            "机器学习的基本概念是什么？",
            "深度学习和传统机器学习的区别？"
        ]
        
        # 并发执行查询
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(hag_api.query, q) for q in questions]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 验证所有查询都成功
        assert len(results) == len(questions)
        for result in results:
            assert isinstance(result, IntegratedResponse)
            assert len(result.answer) > 0
    
    def test_langchain_runnable_integration(self, hag_api):
        """测试LangChain Runnable集成"""
        question = "解释一下神经网络的工作原理"
        
        # 验证runnable_chain存在
        assert hasattr(hag_api, 'runnable_chain')
        assert hag_api.runnable_chain is not None
        
        # 执行查询
        result = hag_api.query(question)
        
        # 验证LangChain管道被正确调用
        assert isinstance(result, IntegratedResponse)
        assert result.answer is not None
        
        # 验证答案生成步骤存在
        answer_steps = [step for step in result.retrieval_process if step.step_name == "答案生成"]
        assert len(answer_steps) == 1
        assert answer_steps[0].status == "success"


class TestDatabaseIntegration:
    """数据库集成测试"""
    
    @pytest.fixture
    def mock_neo4j_driver(self):
        """模拟Neo4j驱动"""
        driver = Mock()
        session = Mock()
        result = Mock()
        
        # 模拟查询结果
        record = Mock()
        record.get.return_value = {
            'name': '人工智能',
            'type': 'Technology',
            'description': '模拟人类智能的技术'
        }
        result.__iter__ = Mock(return_value=iter([record]))
        
        session.run.return_value = result
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=None)
        
        driver.session.return_value = session
        return driver
    
    @pytest.fixture
    def mock_weaviate_client(self):
        """模拟Weaviate客户端"""
        client = Mock()
        
        # 模拟搜索结果
        search_result = {
            'data': {
                'Get': {
                    'Document': [
                        {
                            'content': '人工智能是计算机科学的一个分支',
                            '_additional': {'distance': 0.05}
                        }
                    ]
                }
            }
        }
        
        client.query.get.return_value.with_near_text.return_value.with_limit.return_value.do.return_value = search_result
        return client
    
    def test_neo4j_connection_handling(self, mock_neo4j_driver):
        """测试Neo4j连接处理"""
        with patch('neo4j.GraphDatabase.driver', return_value=mock_neo4j_driver):
            from src.knowledge.neo4j_vector_storage import Neo4jVectorStore
            
            store = Neo4jVectorStore()
            
            # 测试连接
            assert store.driver is not None
            
            # 测试查询执行
            result = store.search_entities("人工智能")
            assert isinstance(result, list)
    
    def test_weaviate_connection_handling(self, mock_weaviate_client):
        """测试Weaviate连接处理"""
        with patch('weaviate.Client', return_value=mock_weaviate_client):
            from src.knowledge.vector_storage import WeaviateVectorStore
            
            store = WeaviateVectorStore()
            
            # 测试连接
            assert store.client is not None
            
            # 测试搜索
            results = store.search("人工智能", limit=5)
            assert isinstance(results, list)
    
    def test_database_error_recovery(self):
        """测试数据库错误恢复"""
        # 模拟数据库连接失败
        with patch('neo4j.GraphDatabase.driver', side_effect=Exception("连接失败")):
            with pytest.raises(Exception):
                from src.knowledge.neo4j_vector_storage import Neo4jVectorStore
                Neo4jVectorStore()
    
    def test_connection_pooling(self, mock_neo4j_driver):
        """测试连接池管理"""
        with patch('neo4j.GraphDatabase.driver', return_value=mock_neo4j_driver):
            from src.knowledge.neo4j_vector_storage import Neo4jVectorStore
            
            # 创建多个实例，验证连接池
            stores = [Neo4jVectorStore() for _ in range(5)]
            
            # 验证所有实例都能正常工作
            for store in stores:
                assert store.driver is not None


class TestAPIResponseValidation:
    """API响应验证测试"""
    
    def test_response_schema_validation(self):
        """测试响应模式验证"""
        # 创建测试响应
        response = IntegratedResponse(
            answer="测试答案",
            sources={
                "documents": [],
                "entities": [],
                "relationships": []
            },
            metadata={"question": "测试问题"},
            retrieval_process=[]
        )
        
        # 验证响应结构
        assert hasattr(response, 'answer')
        assert hasattr(response, 'sources')
        assert hasattr(response, 'metadata')
        assert hasattr(response, 'retrieval_process')
    
    def test_json_serialization(self):
        """测试JSON序列化"""
        from dataclasses import asdict
        
        # 创建检索步骤
        step = RetrievalStep(
            step_name="测试步骤",
            step_description="测试描述",
            start_time=time.time(),
            end_time=time.time() + 1,
            duration=1.0,
            status="success",
            result_count=5,
            details={"test": "value"}
        )
        
        # 测试序列化
        step_dict = asdict(step)
        json_str = json.dumps(step_dict)
        
        # 验证可以反序列化
        parsed = json.loads(json_str)
        assert parsed['step_name'] == "测试步骤"
        assert parsed['status'] == "success"
    
    def test_error_response_format(self):
        """测试错误响应格式"""
        error_response = IntegratedResponse(
            answer="处理查询时出现错误: 测试错误",
            sources={"documents": [], "entities": [], "relationships": []},
            metadata={"error": "测试错误"},
            retrieval_process=[]
        )
        
        # 验证错误响应格式
        assert "错误" in error_response.answer
        assert "error" in error_response.metadata
        assert error_response.metadata["error"] == "测试错误"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])