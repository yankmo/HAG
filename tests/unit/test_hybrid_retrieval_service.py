"""HybridRetrievalService 单元测试"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# 导入被测试的类
from src.services.hybrid_retrieval_service import HybridRetrievalService, HybridResult
from src.services.weight_manager import WeightStrategy


class TestHybridRetrievalService:
    """HybridRetrievalService 测试类"""
    
    @pytest.fixture
    def mock_doc_service(self):
        """模拟文档检索服务"""
        service = Mock()
        service.search_hybrid = Mock()
        service.search_hybrid_async = AsyncMock()
        return service
    
    @pytest.fixture
    def mock_graph_service(self):
        """模拟图谱检索服务"""
        service = Mock()
        service.recognize_intent = Mock()
        service.recognize_intent_async = AsyncMock()
        service.search_entities_by_name = Mock()
        service.search_entities_by_name_async = AsyncMock()
        service.search_relationships_by_type = Mock()
        service.search_relationships_by_type_async = AsyncMock()
        service.search_relationships_by_query = Mock()
        service.search_relationships_by_query_async = AsyncMock()
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
        assert service.default_doc_weight == 0.6
        assert service.default_graph_weight == 0.4
        assert service.current_doc_weight == 0.6
        assert service.current_graph_weight == 0.4
        assert service.enable_concurrent_queries is True
        assert service.max_workers == 4
        assert service.enable_dynamic_weights is False
        assert service.enable_ab_testing is False
    
    def test_init_with_custom_weights(self, mock_doc_service, mock_graph_service):
        """测试自定义权重初始化"""
        service = HybridRetrievalService(
            document_retrieval_service=mock_doc_service,
            graph_retrieval_service=mock_graph_service,
            doc_weight=0.7,
            graph_weight=0.3
        )
        
        assert service.default_doc_weight == 0.7
        assert service.default_graph_weight == 0.3
        assert service.current_doc_weight == 0.7
        assert service.current_graph_weight == 0.3
    
    def test_init_with_dynamic_weights(self, mock_doc_service, mock_graph_service):
        """测试动态权重初始化"""
        with patch('src.services.hybrid_retrieval_service.DynamicWeightManager') as mock_weight_manager:
            mock_manager_instance = Mock()
            mock_manager_instance.set_strategy = Mock()
            mock_weight_manager.return_value = mock_manager_instance
            
            service = HybridRetrievalService(
                document_retrieval_service=mock_doc_service,
                graph_retrieval_service=mock_graph_service,
                enable_dynamic_weights=True,
                weight_strategy=WeightStrategy.ADAPTIVE
            )
            
            assert service.enable_dynamic_weights is True
            assert service.weight_manager is not None
            mock_weight_manager.assert_called_once()
    
    def test_init_with_ab_testing(self, mock_doc_service, mock_graph_service):
        """测试A/B测试初始化"""
        with patch('src.services.hybrid_retrieval_service.ABTestManager') as mock_ab_manager:
            mock_manager_instance = Mock()
            mock_ab_manager.return_value = mock_manager_instance
            
            ab_config = {
                'storage_path': './test_ab_data',
                'significance_level': 0.01,
                'auto_cleanup': False
            }
            
            service = HybridRetrievalService(
                document_retrieval_service=mock_doc_service,
                graph_retrieval_service=mock_graph_service,
                enable_ab_testing=True,
                ab_test_config=ab_config
            )
            
            assert service.enable_ab_testing is True
            assert service.ab_test_manager is not None
            mock_ab_manager.assert_called_once_with(
                storage_path='./test_ab_data',
                significance_level=0.01,
                auto_cleanup=False
            )
    
    def test_init_with_cache_config(self, mock_doc_service, mock_graph_service):
        """测试缓存配置初始化"""
        with patch('src.services.hybrid_retrieval_service.IntelligentCacheManager') as mock_cache:
            mock_cache_instance = Mock()
            mock_cache.return_value = mock_cache_instance
            
            cache_config = {'redis_url': 'redis://localhost:6379'}
            
            service = HybridRetrievalService(
                document_retrieval_service=mock_doc_service,
                graph_retrieval_service=mock_graph_service,
                cache_config=cache_config
            )
            
            assert service.cache_manager is not None
            mock_cache.assert_called_once_with(cache_config)
    
    @pytest.mark.asyncio
    async def test_search_hybrid_basic(self, hybrid_service, mock_hybrid_result, mock_intent_result):
        """测试基础混合检索"""
        # 设置模拟返回值
        hybrid_service.doc_service.search_hybrid_async.return_value = mock_hybrid_result
        hybrid_service.graph_service.recognize_intent_async.return_value = mock_intent_result
        hybrid_service.graph_service.search_entities_by_name_async.return_value = [
            {'id': 'entity1', 'name': '实体1', 'score': 0.9}
        ]
        hybrid_service.graph_service.search_relationships_by_type_async.return_value = [
            {'id': 'rel1', 'type': '关系1', 'score': 0.8}
        ]
        hybrid_service.graph_service.search_relationships_by_query_async.return_value = []
        
        # 执行测试
        result = await hybrid_service.search_hybrid("测试查询", top_k=10)
        
        # 验证结果
        assert isinstance(result, HybridResult)
        assert len(result.documents) == 2
        assert len(result.entities) == 1
        assert len(result.relationships) == 1
        assert result.combined_score > 0
        assert result.metadata['query'] == "测试查询"
        assert result.metadata['doc_weight'] == 0.6
        assert result.metadata['graph_weight'] == 0.4
        assert result.metadata['weight_strategy'] == WeightStrategy.STATIC
        assert result.metadata['dynamic_weights_enabled'] is False
    
    @pytest.mark.asyncio
    async def test_search_hybrid_with_dynamic_weights(self, mock_doc_service, mock_graph_service, mock_hybrid_result):
        """测试动态权重混合检索"""
        with patch('src.services.hybrid_retrieval_service.DynamicWeightManager') as mock_weight_manager:
            mock_manager_instance = Mock()
            mock_manager_instance.calculate_weights = Mock(return_value={'doc_weight': 0.8, 'graph_weight': 0.2})
            mock_manager_instance.calculate_weights_async = AsyncMock(return_value={'doc_weight': 0.8, 'graph_weight': 0.2})
            mock_manager_instance.set_strategy = Mock()
            mock_weight_manager.return_value = mock_manager_instance
            
            service = HybridRetrievalService(
                document_retrieval_service=mock_doc_service,
                graph_retrieval_service=mock_graph_service,
                enable_dynamic_weights=True,
                enable_concurrent_queries=False
            )
            
            # 设置模拟返回值
            mock_doc_service.search_hybrid_async.return_value = mock_hybrid_result
            mock_graph_service.recognize_intent_async.return_value = Mock(
                intent_type="查询", confidence=0.8, entities=[], relations=[]
            )
            mock_graph_service.search_entities_by_name_async.return_value = []
            mock_graph_service.search_relationships_by_query_async.return_value = []
            
            # 执行测试
            result = await service.search_hybrid("测试查询", top_k=10)
            
            # 验证动态权重被应用
            assert result.metadata['doc_weight'] == 0.8
            assert result.metadata['graph_weight'] == 0.2
            assert result.metadata['dynamic_weights_enabled'] is True
            mock_manager_instance.calculate_weights.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_hybrid_with_ab_testing(self, mock_doc_service, mock_graph_service, mock_hybrid_result):
        """测试A/B测试混合检索"""
        with patch('src.services.hybrid_retrieval_service.ABTestManager') as mock_ab_manager:
            mock_manager_instance = Mock()
            mock_manager_instance.assign_strategy = Mock(return_value={
                'strategy': WeightStrategy.QUALITY_DRIVEN,
                'experiment_id': 'exp_123'
            })
            mock_manager_instance.record_result = Mock()
            mock_ab_manager.return_value = mock_manager_instance
            
            service = HybridRetrievalService(
                document_retrieval_service=mock_doc_service,
                graph_retrieval_service=mock_graph_service,
                enable_ab_testing=True,
                enable_concurrent_queries=False
            )
            
            # 设置模拟返回值
            mock_doc_service.search_hybrid_async.return_value = mock_hybrid_result
            mock_graph_service.recognize_intent_async.return_value = Mock(
                intent_type="查询", confidence=0.8, entities=[], relations=[]
            )
            mock_graph_service.search_entities_by_name_async.return_value = []
            mock_graph_service.search_relationships_by_query_async.return_value = []
            
            # 执行测试
            result = await service.search_hybrid("测试查询", top_k=10, user_id="user123")
            
            # 验证A/B测试
            assert result.metadata['ab_test_enabled'] is True
            assert result.metadata['assigned_strategy'] == WeightStrategy.QUALITY_DRIVEN
            assert result.metadata['experiment_id'] == 'exp_123'
            mock_manager_instance.assign_strategy.assert_called_once_with("user123", "测试查询")
            mock_manager_instance.record_result.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_hybrid_concurrent_mode(self, mock_doc_service, mock_graph_service, mock_hybrid_result):
        """测试并发模式混合检索"""
        service = HybridRetrievalService(
            document_retrieval_service=mock_doc_service,
            graph_retrieval_service=mock_graph_service,
            enable_concurrent_queries=True,
            max_workers=2
        )
        
        # 设置模拟返回值
        mock_doc_service.search_hybrid_async.return_value = mock_hybrid_result
        mock_graph_service.recognize_intent_async.return_value = Mock(
            intent_type="查询", confidence=0.8, entities=[], relations=[]
        )
        mock_graph_service.search_entities_by_name_async.return_value = []
        mock_graph_service.search_relationships_by_query_async.return_value = []
        
        # 执行测试
        result = await service.search_hybrid("测试查询", top_k=10)
        
        # 验证并发查询统计
        assert service.query_stats['concurrent_queries'] == 1
        assert service.query_stats['total_queries'] == 1
    
    @pytest.mark.asyncio
    async def test_search_documents_async(self, hybrid_service, mock_hybrid_result):
        """测试异步文档检索"""
        hybrid_service.doc_service.search_hybrid_async.return_value = mock_hybrid_result
        
        result = await hybrid_service._search_documents_async("测试查询", 5)
        
        assert 'documents' in result
        assert 'total_score' in result
        assert len(result['documents']) == 2
        assert result['documents'][0]['content'] == "测试文档1"
        assert result['documents'][0]['score'] == 0.9
        assert result['documents'][0]['source'] == 'weaviate'
        assert result['total_score'] == 1.7  # 0.9 + 0.8
    
    @pytest.mark.asyncio
    async def test_search_documents_async_fallback(self, hybrid_service, mock_hybrid_result):
        """测试异步文档检索降级"""
        # 移除异步方法，测试降级到同步方法
        del hybrid_service.doc_service.search_hybrid_async
        hybrid_service.doc_service.search_hybrid.return_value = mock_hybrid_result
        
        result = await hybrid_service._search_documents_async("测试查询", 5)
        
        assert 'documents' in result
        assert len(result['documents']) == 2
        hybrid_service.doc_service.search_hybrid.assert_called_once_with("测试查询", 5)
    
    @pytest.mark.asyncio
    async def test_search_documents_async_error_handling(self, hybrid_service):
        """测试异步文档检索错误处理"""
        hybrid_service.doc_service.search_hybrid_async.side_effect = Exception("连接失败")
        
        result = await hybrid_service._search_documents_async("测试查询", 5)
        
        assert result['documents'] == []
        assert result['total_score'] == 0.0
    
    @pytest.mark.asyncio
    async def test_search_graph_async_with_intent(self, hybrid_service, mock_intent_result):
        """测试带意图识别的异步图谱检索"""
        hybrid_service.graph_service.recognize_intent_async.return_value = mock_intent_result
        hybrid_service.graph_service.search_entities_by_name_async.return_value = [
            {'id': 'entity1', 'name': '实体1', 'score': 0.9}
        ]
        hybrid_service.graph_service.search_relationships_by_type_async.return_value = [
            {'id': 'rel1', 'type': '关系1', 'score': 0.8}
        ]
        hybrid_service.graph_service.search_relationships_by_query_async.return_value = []
        
        result = await hybrid_service._search_graph_async("测试查询", 10, include_intent=True)
        
        assert 'entities' in result
        assert 'relationships' in result
        assert 'intent' in result
        assert result['intent']['intent'] == "查询"
        assert result['intent']['confidence'] == 0.85
        assert len(result['entities']) == 1
        assert len(result['relationships']) == 1
    
    @pytest.mark.asyncio
    async def test_search_graph_async_without_intent(self, hybrid_service):
        """测试不带意图识别的异步图谱检索"""
        hybrid_service.graph_service.search_entities_by_name_async.return_value = [
            {'id': 'entity1', 'name': '实体1', 'score': 0.9}
        ]
        hybrid_service.graph_service.search_relationships_by_query_async.return_value = [
            {'id': 'rel1', 'type': '关系1', 'score': 0.8}
        ]
        
        result = await hybrid_service._search_graph_async("测试查询", 10, include_intent=False)
        
        assert result['intent'] is None
        assert len(result['entities']) == 1
        assert len(result['relationships']) == 1
    
    @pytest.mark.asyncio
    async def test_search_graph_async_error_handling(self, hybrid_service):
        """测试异步图谱检索错误处理"""
        hybrid_service.graph_service.recognize_intent_async.side_effect = Exception("Neo4j连接失败")
        
        result = await hybrid_service._search_graph_async("测试查询", 10)
        
        assert result['entities'] == []
        assert result['relationships'] == []
        assert result['intent'] is None
    
    def test_calculate_combined_score(self, hybrid_service):
        """测试综合得分计算"""
        doc_results = {
            'documents': [
                {'score': 0.9},
                {'score': 0.8}
            ],
            'total_score': 1.7
        }
        
        graph_results = {
            'entities': [
                {'score': 0.7}
            ],
            'relationships': [
                {'score': 0.6}
            ]
        }
        
        score = hybrid_service._calculate_combined_score(doc_results, graph_results)
        
        # 验证得分计算逻辑
        assert isinstance(score, float)
        assert score > 0
    
    def test_calculate_dynamic_weights(self, mock_doc_service, mock_graph_service):
        """测试动态权重计算"""
        with patch('src.services.hybrid_retrieval_service.DynamicWeightManager') as mock_weight_manager:
            mock_manager_instance = Mock()
            mock_manager_instance.calculate_weights = Mock(return_value={'doc_weight': 0.7, 'graph_weight': 0.3})
            mock_manager_instance.set_strategy = Mock()
            mock_weight_manager.return_value = mock_manager_instance
            
            service = HybridRetrievalService(
                document_retrieval_service=mock_doc_service,
                graph_retrieval_service=mock_graph_service,
                enable_dynamic_weights=True
            )
            
            weights = service._calculate_dynamic_weights("测试查询")
            
            assert weights['doc_weight'] == 0.7
            assert weights['graph_weight'] == 0.3
            mock_manager_instance.calculate_weights.assert_called_once_with("测试查询")
    
    @pytest.mark.asyncio
    async def test_calculate_dynamic_weights_async(self, mock_doc_service, mock_graph_service):
        """测试异步动态权重计算"""
        with patch('src.services.hybrid_retrieval_service.DynamicWeightManager') as mock_weight_manager:
            mock_manager_instance = Mock()
            mock_manager_instance.calculate_weights_async = AsyncMock(return_value={'doc_weight': 0.8, 'graph_weight': 0.2})
            mock_manager_instance.set_strategy = Mock()
            mock_weight_manager.return_value = mock_manager_instance
            
            service = HybridRetrievalService(
                document_retrieval_service=mock_doc_service,
                graph_retrieval_service=mock_graph_service,
                enable_dynamic_weights=True
            )
            
            weights = await service._calculate_dynamic_weights_async("测试查询")
            
            assert weights['doc_weight'] == 0.8
            assert weights['graph_weight'] == 0.2
            mock_manager_instance.calculate_weights_async.assert_called_once_with("测试查询")
    
    def test_get_query_stats(self, hybrid_service):
        """测试查询统计获取"""
        stats = hybrid_service.get_query_stats()
        
        assert 'total_queries' in stats
        assert 'concurrent_queries' in stats
        assert 'avg_response_time' in stats
        assert 'doc_query_time' in stats
        assert 'graph_query_time' in stats
        assert stats['total_queries'] == 0
        assert stats['concurrent_queries'] == 0
    
    def test_reset_query_stats(self, hybrid_service):
        """测试查询统计重置"""
        # 先设置一些统计数据
        hybrid_service.query_stats['total_queries'] = 10
        hybrid_service.query_stats['concurrent_queries'] = 5
        
        hybrid_service.reset_query_stats()
        
        assert hybrid_service.query_stats['total_queries'] == 0
        assert hybrid_service.query_stats['concurrent_queries'] == 0
        assert hybrid_service.query_stats['avg_response_time'] == 0.0
    
    def test_update_weights(self, hybrid_service):
        """测试权重更新"""
        hybrid_service.update_weights(doc_weight=0.8, graph_weight=0.2)
        
        assert hybrid_service.current_doc_weight == 0.8
        assert hybrid_service.current_graph_weight == 0.2
    
    def test_get_current_weights(self, hybrid_service):
        """测试当前权重获取"""
        weights = hybrid_service.get_current_weights()
        
        assert weights['doc_weight'] == 0.6
        assert weights['graph_weight'] == 0.4
        assert weights['strategy'] == WeightStrategy.STATIC
        assert weights['dynamic_enabled'] is False
    
    def test_cleanup(self, hybrid_service):
        """测试资源清理"""
        # 模拟有线程池执行器
        hybrid_service.executor = Mock()
        hybrid_service.executor.shutdown = Mock()
        
        hybrid_service.cleanup()
        
        hybrid_service.executor.shutdown.assert_called_once_with(wait=True)
    
    @pytest.mark.asyncio
    async def test_search_hybrid_performance_monitoring(self, mock_doc_service, mock_graph_service, mock_hybrid_result):
        """测试性能监控功能"""
        with patch('src.services.hybrid_retrieval_service.performance_monitor') as mock_monitor:
            mock_monitor.start_query = Mock(return_value="query_123")
            mock_monitor.end_query = Mock()
            mock_monitor.record_component_start = Mock()
            mock_monitor.record_component_end = Mock()
            
            service = HybridRetrievalService(
                document_retrieval_service=mock_doc_service,
                graph_retrieval_service=mock_graph_service,
                enable_performance_monitoring=True,
                enable_concurrent_queries=False
            )
            
            # 设置模拟返回值
            mock_doc_service.search_hybrid_async.return_value = mock_hybrid_result
            mock_graph_service.recognize_intent_async.return_value = Mock(
                intent_type="查询", confidence=0.8, entities=[], relations=[]
            )
            mock_graph_service.search_entities_by_name_async.return_value = []
            mock_graph_service.search_relationships_by_query_async.return_value = []
            
            # 执行测试
            result = await service.search_hybrid("测试查询", top_k=10, user_id="user123")
            
            # 验证性能监控调用
            mock_monitor.start_query.assert_called_once_with("测试查询", "user123")
            mock_monitor.end_query.assert_called_once()
            mock_monitor.record_component_start.assert_called()
            mock_monitor.record_component_end.assert_called()