"""HybridRetrievalService 单元测试"""

import pytest
from unittest.mock import Mock, AsyncMock
from dataclasses import dataclass
from typing import List, Dict, Any

from src.services.hybrid_retrieval_service import HybridRetrievalService
from src.services.weight_manager import QueryContext
from src.services.common_types import SearchResult
from src.services.weight_manager import WeightStrategy


class TestHybridRetrievalService:
    """测试HybridRetrievalService类"""
    
    @pytest.fixture
    def mock_doc_service(self):
        """模拟文档检索服务"""
        service = Mock()
        service.search_hybrid = Mock()
        service.search_hybrid_async = AsyncMock()
        return service
    
    @pytest.fixture
    def mock_graph_service(self):
        """模拟图检索服务"""
        service = Mock()
        service.search_entities = AsyncMock(return_value=[
            SearchResult(id="entity1", content="Entity", score=0.9, distance=0.1, metadata={'type': 'entity'})
        ])
        service.search_relationships = AsyncMock(return_value=[
            SearchResult(id="rel1", content="Relationship", score=0.7, distance=0.3, metadata={'type': 'relationship'})
        ])
        return service
    
    @pytest.fixture
    def mock_hybrid_result(self):
        """模拟混合检索结果"""
        @dataclass
        class MockResult:
            content: str
            metadata: Dict[str, Any]
            score: float
        
        @dataclass
        class MockHybridResult:
            hybrid_results: List[MockResult]
        
        return MockHybridResult([
            MockResult("测试文档1", {"id": "doc1"}, 0.9),
            MockResult("测试文档2", {"id": "doc2"}, 0.8)
        ])
    
    @pytest.fixture
    def mock_intent_result(self):
        """模拟意图识别结果"""
        @dataclass
        class MockIntentResult:
            intent_type: str
            confidence: float
            entities: List[str]
            relations: List[str]
        
        return MockIntentResult(
            intent_type="查询",
            confidence=0.85,
            entities=["实体1", "实体2"],
            relations=["关系1"]
        )
    
    @pytest.fixture
    def basic_config(self):
        """基础配置"""
        return {
            'doc_weight': 0.6,
            'graph_weight': 0.4,
            'weight_strategy': WeightStrategy.STATIC,
            'enable_dynamic_weights': False,
            'enable_ab_testing': False,
            'enable_concurrent_queries': False,
            'max_workers': 2,
            'enable_performance_monitoring': False
        }
    
    @pytest.fixture
    def hybrid_service(self, mock_doc_service, mock_graph_service, basic_config):
        """创建HybridRetrievalService实例"""
        return HybridRetrievalService(
            document_retrieval_service=mock_doc_service,
            graph_retrieval_service=mock_graph_service,
            **basic_config
        )


    
    def test_init_basic(self, mock_doc_service, mock_graph_service):
        """测试基础初始化"""
        service = HybridRetrievalService(
            document_retrieval_service=mock_doc_service,
            graph_retrieval_service=mock_graph_service
        )
        
        assert service.doc_service == mock_doc_service
        assert service.graph_service == mock_graph_service