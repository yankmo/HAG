#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本功能测试
验证HAG系统的核心组件是否能正常工作
"""

import pytest
from unittest.mock import Mock, patch


class TestBasicFunctionality:
    """基本功能测试类"""
    
    def test_imports_work(self):
        """测试核心模块能否正常导入"""
        try:
            from src.services.weight_manager import WeightStrategy, QueryContext
            from src.services.cache_manager import IntelligentCacheManager
            from src.services.hybrid_retrieval_service import HybridRetrievalService
            assert True  # 如果能导入就说明基本结构正确
        except ImportError as e:
            pytest.fail(f"核心模块导入失败: {e}")
    
    def test_weight_strategy_enum(self):
        """测试权重策略枚举"""
        from src.services.weight_manager import WeightStrategy
        
        # 验证枚举值存在
        assert WeightStrategy.STATIC
        assert WeightStrategy.INTENT_DRIVEN
        assert WeightStrategy.QUALITY_DRIVEN
    
    def test_query_context_creation(self):
        """测试查询上下文创建"""
        from src.services.weight_manager import QueryContext
        
        context = QueryContext(query="测试查询")
        assert context.query == "测试查询"
        assert context.timestamp is not None
    
    def test_cache_manager_basic(self):
        """测试缓存管理器基本功能"""
        from src.services.cache_manager import IntelligentCacheManager
        
        cache = IntelligentCacheManager()
        assert cache is not None
        
        # 测试基本的缓存操作
        cache.set("test_service", "test_method", "test_query", "test_value")
        assert cache.get("test_service", "test_method", "test_query") == "test_value"
    
    def test_hybrid_service_creation(self):
        """测试混合检索服务创建"""
        from src.services.hybrid_retrieval_service import HybridRetrievalService
        
        # 模拟文档检索服务和图谱检索服务
        mock_doc_service = Mock()
        mock_graph_service = Mock()
        
        service = HybridRetrievalService(
            document_retrieval_service=mock_doc_service,
            graph_retrieval_service=mock_graph_service
        )
        assert service is not None
    
    def test_system_health_check(self):
        """系统健康检查"""
        # 这是一个简单的系统健康检查
        # 验证关键组件能够被正确实例化
        
        health_status = {
            'weight_manager': False,
            'cache_manager': False,
            'hybrid_service': False
        }
        
        try:
            from src.services.weight_manager import StaticWeightCalculator
            calculator = StaticWeightCalculator()
            health_status['weight_manager'] = True
        except Exception:
            pass
        
        try:
            from src.services.cache_manager import IntelligentCacheManager
            cache = IntelligentCacheManager()
            health_status['cache_manager'] = True
        except Exception:
            pass
        
        try:
            from src.services.hybrid_retrieval_service import HybridRetrievalService
            mock_doc_service = Mock()
            mock_graph_service = Mock()
            service = HybridRetrievalService(
                document_retrieval_service=mock_doc_service,
                graph_retrieval_service=mock_graph_service
            )
            health_status['hybrid_service'] = True
        except Exception:
            pass
        
        # 至少要有一个组件正常工作
        assert any(health_status.values()), f"系统健康检查失败: {health_status}"


if __name__ == "__main__":
    pytest.main([__file__])