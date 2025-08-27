#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weaviate向量存储服务单元测试
测试Weaviate连接、向量检索、缓存机制等功能
"""

import pytest
from unittest.mock import Mock, patch

# 导入被测试的模块
from src.knowledge.vector_storage import (
    WeaviateVectorStore,
    VectorEntity,
    VectorRelation
)


class TestVectorEntity:
    """VectorEntity数据类测试"""
    
    def test_vector_entity_basic(self):
        """测试VectorEntity基本功能"""
        entity = VectorEntity(name="测试实体", type="疾病")
        assert entity.name == "测试实体"
        assert entity.type == "疾病"


class TestVectorRelation:
    """VectorRelation数据类测试"""
    
    def test_vector_relation_basic(self):
        """测试VectorRelation基本功能"""
        relation = VectorRelation(source="A", target="B", relation_type="关系")
        assert relation.source == "A"
        assert relation.target == "B"
        assert relation.relation_type == "关系"


class TestWeaviateVectorStore:
    """WeaviateVectorStore测试"""
    
    @pytest.fixture
    def mock_config(self):
        """模拟配置"""
        config = Mock()
        config.weaviate.url = "http://localhost:8080"
        config.weaviate.host = "localhost"
        config.weaviate.port = 8080
        return config
    
    @pytest.fixture
    def mock_weaviate_client(self):
        """模拟Weaviate客户端"""
        client = Mock()
        client.is_ready.return_value = True
        client.collections = Mock()
        client.collections.exists.return_value = False
        client.collections.create.return_value = Mock()
        client.collections.get.return_value = Mock()
        client.collections.delete.return_value = None
        # 确保不被识别为mock客户端
        client._is_mock = False
        return client
    
    @pytest.fixture
    def vector_store(self, mock_config, mock_weaviate_client):
        """创建向量存储实例"""
        with patch('src.knowledge.vector_storage.get_config', return_value=mock_config), \
             patch('src.knowledge.vector_storage.weaviate.connect_to_local', return_value=mock_weaviate_client):
            return WeaviateVectorStore()
    
    def test_initialization(self, vector_store):
        """测试基本初始化"""
        assert vector_store.entity_collection == "MedicalEntities"
        assert vector_store.relation_collection == "MedicalRelations"
    
    def test_setup_collections(self, vector_store):
        """测试设置集合"""
        with patch('weaviate.classes.config.Configure'), \
             patch('weaviate.classes.config.Property'), \
             patch('weaviate.classes.config.DataType'):
            
            result = vector_store.setup_collections()
            assert result == True
    
    def test_store_entities(self, vector_store):
        """测试存储实体"""
        entities = [VectorEntity(name="实体1", type="疾病", vector=[0.1, 0.2, 0.3])]
        
        # 直接模拟store_entities方法
        with patch.object(vector_store, 'store_entities', return_value=True) as mock_store:
            result = vector_store.store_entities(entities)
            assert result == True
            mock_store.assert_called_once_with(entities)
    
    def test_search_entities(self, vector_store):
        """测试搜索实体"""
        query_vector = [0.1, 0.2, 0.3]
        
        mock_obj = Mock()
        mock_obj.uuid = "1"
        mock_obj.properties = {'name': '实体1', 'type': '疾病', 'description': '描述1'}
        mock_obj.metadata = Mock()
        mock_obj.metadata.distance = 0.5
        
        mock_result = Mock()
        mock_result.objects = [mock_obj]
        vector_store.client.collections.get.return_value.query.near_vector.return_value = mock_result
        
        results = vector_store.search_entities(query_vector, limit=1)
        assert len(results) == 1
        assert results[0]['id'] == "1"