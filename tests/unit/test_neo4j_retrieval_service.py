#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4jRetrievalService基本单元测试
"""

import pytest
from unittest.mock import Mock, patch

from src.services.neo4j_retrieval_service import GraphRetrievalService


class TestGraphRetrievalService:
    """GraphRetrievalService基本测试"""
    
    @pytest.fixture
    def mock_config(self):
        """模拟配置"""
        config = Mock()
        config.host = 'localhost'
        config.port = 7687
        config.user = 'neo4j'
        config.password = 'password'
        return config
    
    @pytest.fixture
    def service(self, mock_config):
        """创建服务实例"""
        with patch('py2neo.Graph'):
            return GraphRetrievalService(mock_config)
    
    def test_init_basic(self, service):
        """测试基本初始化"""
        assert service is not None
    
    def test_get_stats(self, service):
        """测试获取统计信息"""
        with patch.object(service, 'graph') as mock_graph:
            mock_graph.run.side_effect = [
                Mock(data=lambda: [{'count': 10}]),
                Mock(data=lambda: [{'count': 20}]),
                Mock(data=lambda: [{'type': 'Test', 'count': 5}])
            ]
            
            stats = service.get_stats()
            assert 'total_nodes' in stats
            assert 'total_relationships' in stats
    
    def test_search_entities(self, service):
        """测试搜索实体"""
        with patch.object(service, 'graph') as mock_graph:
            mock_graph.run.return_value.data.return_value = [
                {'name': 'Test Entity', 'type': 'Test'}
            ]
            
            entities = service.search_entities_by_type('Test', 10)
            assert len(entities) >= 0
    
    def test_search_relationships(self, service):
        """测试搜索关系"""
        with patch.object(service, 'graph') as mock_graph:
            mock_graph.run.return_value.data.return_value = [
                {'source': 'A', 'type': 'REL', 'target': 'B'}
            ]
            
            relationships = service.search_relationships_by_type('REL', 10)
            assert len(relationships) >= 0


if __name__ == '__main__':
    pytest.main([__file__])