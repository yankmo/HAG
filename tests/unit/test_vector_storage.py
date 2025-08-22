#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weaviate向量存储服务单元测试
测试Weaviate连接、向量检索、缓存机制等功能
"""

import pytest
import unittest.mock as mock
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import List, Dict, Any
import logging
from datetime import datetime

# 导入被测试的模块
from src.knowledge.vector_storage import (
    WeaviateVectorStore,
    VectorEntity,
    VectorRelation
)


class TestVectorEntity:
    """VectorEntity数据类测试"""
    
    def test_vector_entity_initialization(self):
        """测试VectorEntity初始化"""
        entity = VectorEntity(
            name="测试实体",
            type="疾病",
            properties={"description": "测试描述"},
            vector=[0.1, 0.2, 0.3],
            source_text="原始文本"
        )
        
        assert entity.name == "测试实体"
        assert entity.type == "疾病"
        assert entity.properties["description"] == "测试描述"
        assert entity.vector == [0.1, 0.2, 0.3]
        assert entity.source_text == "原始文本"
    
    def test_vector_entity_default_values(self):
        """测试VectorEntity默认值"""
        entity = VectorEntity(name="实体", type="类型")
        
        assert entity.properties == {}
        assert entity.vector == []
        assert entity.source_text == ""


class TestVectorRelation:
    """VectorRelation数据类测试"""
    
    def test_vector_relation_initialization(self):
        """测试VectorRelation初始化"""
        relation = VectorRelation(
            source="实体A",
            target="实体B",
            relation_type="治疗",
            description="治疗关系",
            vector=[0.4, 0.5, 0.6],
            source_text="关系文本"
        )
        
        assert relation.source == "实体A"
        assert relation.target == "实体B"
        assert relation.relation_type == "治疗"
        assert relation.description == "治疗关系"
        assert relation.vector == [0.4, 0.5, 0.6]
        assert relation.source_text == "关系文本"
    
    def test_vector_relation_default_values(self):
        """测试VectorRelation默认值"""
        relation = VectorRelation(
            source="A",
            target="B",
            relation_type="关系"
        )
        
        assert relation.description == ""
        assert relation.vector == []
        assert relation.source_text == ""


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
        return client
    
    @pytest.fixture
    def vector_store(self, mock_config, mock_weaviate_client):
        """创建向量存储实例"""
        with patch('src.knowledge.vector_storage.get_config', return_value=mock_config), \
             patch('src.knowledge.vector_storage.weaviate.connect_to_local', return_value=mock_weaviate_client):
            return WeaviateVectorStore()
    
    def test_initialization_success(self, mock_config, mock_weaviate_client):
        """测试成功初始化"""
        with patch('src.knowledge.vector_storage.get_config', return_value=mock_config), \
             patch('src.knowledge.vector_storage.weaviate.connect_to_local', return_value=mock_weaviate_client):
            
            store = WeaviateVectorStore()
            
            assert store.entity_collection == "MedicalEntities"
            assert store.relation_collection == "MedicalRelations"
            assert store.client == mock_weaviate_client
    
    def test_initialization_with_custom_url(self, mock_config, mock_weaviate_client):
        """测试使用自定义URL初始化"""
        custom_url = "http://custom:8080"
        
        with patch('src.knowledge.vector_storage.get_config', return_value=mock_config), \
             patch('src.knowledge.vector_storage.weaviate.connect_to_local', return_value=mock_weaviate_client):
            
            store = WeaviateVectorStore(url=custom_url)
            assert store.client == mock_weaviate_client
    
    def test_connection_failure_fallback_to_mock(self, mock_config):
        """测试连接失败时回退到模拟客户端"""
        with patch('src.knowledge.vector_storage.get_config', return_value=mock_config), \
             patch('src.knowledge.vector_storage.weaviate.connect_to_local', side_effect=Exception("连接失败")):
            
            store = WeaviateVectorStore()
            
            assert store.is_mock_client() == True
            assert store.is_available() == False
    
    def test_is_mock_client(self, vector_store):
        """测试模拟客户端检测"""
        # 正常客户端
        assert vector_store.is_mock_client() == False
        
        # 模拟客户端
        vector_store.client._is_mock = True
        assert vector_store.is_mock_client() == True
    
    def test_is_available(self, vector_store):
        """测试服务可用性检查"""
        # 正常情况
        vector_store.client.is_ready.return_value = True
        assert vector_store.is_available() == True
        
        # 服务不可用
        vector_store.client.is_ready.return_value = False
        assert vector_store.is_available() == False
        
        # 异常情况
        vector_store.client.is_ready.side_effect = Exception("连接错误")
        assert vector_store.is_available() == False
    
    def test_setup_collections_success(self, vector_store):
        """测试成功设置集合"""
        # 模拟集合不存在
        vector_store.client.collections.exists.return_value = False
        
        with patch('src.knowledge.vector_storage.Configure'), \
             patch('src.knowledge.vector_storage.Property'), \
             patch('src.knowledge.vector_storage.DataType'):
            
            result = vector_store.setup_collections()
            
            assert result == True
            assert vector_store.client.collections.create.call_count == 2
    
    def test_setup_collections_with_existing(self, vector_store):
        """测试设置集合时删除已存在的集合"""
        # 模拟集合已存在
        vector_store.client.collections.exists.return_value = True
        
        with patch('src.knowledge.vector_storage.Configure'), \
             patch('src.knowledge.vector_storage.Property'), \
             patch('src.knowledge.vector_storage.DataType'):
            
            result = vector_store.setup_collections()
            
            assert result == True
            assert vector_store.client.collections.delete.call_count == 2
            assert vector_store.client.collections.create.call_count == 2
    
    def test_setup_collections_mock_client(self, vector_store):
        """测试模拟客户端设置集合"""
        vector_store.client._is_mock = True
        
        result = vector_store.setup_collections()
        
        assert result == True
    
    def test_store_entities_success(self, vector_store):
        """测试成功存储实体"""
        entities = [
            VectorEntity(
                name="实体1",
                type="疾病",
                properties={"description": "描述1"},
                vector=[0.1, 0.2, 0.3],
                source_text="文本1"
            ),
            VectorEntity(
                name="实体2",
                type="症状",
                vector=[0.4, 0.5, 0.6]
            )
        ]
        
        mock_collection = Mock()
        vector_store.client.collections.get.return_value = mock_collection
        
        with patch('src.knowledge.vector_storage.DataObject') as mock_data_object:
            result = vector_store.store_entities(entities)
            
            assert result == True
            mock_collection.data.insert_many.assert_called_once()
    
    def test_store_entities_skip_no_vector(self, vector_store):
        """测试跳过没有向量的实体"""
        entities = [
            VectorEntity(name="实体1", type="疾病"),  # 没有向量
            VectorEntity(name="实体2", type="症状", vector=[0.1, 0.2])
        ]
        
        mock_collection = Mock()
        vector_store.client.collections.get.return_value = mock_collection
        
        with patch('src.knowledge.vector_storage.DataObject'):
            result = vector_store.store_entities(entities)
            
            assert result == True
            # 只有一个实体有向量，所以只调用一次
            mock_collection.data.insert_many.assert_called_once()
    
    def test_store_entities_mock_client(self, vector_store):
        """测试模拟客户端存储实体"""
        vector_store.client._is_mock = True
        entities = [VectorEntity(name="实体", type="类型", vector=[0.1])]
        
        result = vector_store.store_entities(entities)
        
        assert result == True
    
    def test_store_relations_success(self, vector_store):
        """测试成功存储关系"""
        relations = [
            VectorRelation(
                source="实体A",
                target="实体B",
                relation_type="治疗",
                description="治疗关系",
                vector=[0.1, 0.2, 0.3]
            )
        ]
        
        mock_collection = Mock()
        vector_store.client.collections.get.return_value = mock_collection
        
        with patch('src.knowledge.vector_storage.DataObject'):
            result = vector_store.store_relations(relations)
            
            assert result == True
            mock_collection.data.insert_many.assert_called_once()
    
    def test_search_entities_euclidean(self, vector_store):
        """测试使用欧氏距离搜索实体"""
        query_vector = [0.1, 0.2, 0.3]
        
        # 模拟搜索结果
        mock_obj = Mock()
        mock_obj.uuid = "test-uuid"
        mock_obj.properties = {
            "name": "测试实体",
            "type": "疾病",
            "description": "测试描述",
            "source_text": "原始文本",
            "neo4j_id": "neo4j-123"
        }
        mock_obj.metadata = Mock()
        mock_obj.metadata.distance = 0.5
        
        mock_response = Mock()
        mock_response.objects = [mock_obj]
        
        mock_collection = Mock()
        mock_collection.query.near_vector.return_value = mock_response
        vector_store.client.collections.get.return_value = mock_collection
        
        results = vector_store.search_entities(query_vector, limit=10, distance_metric="euclidean")
        
        assert len(results) == 1
        assert results[0]["name"] == "测试实体"
        assert results[0]["distance"] == 0.5
        assert results[0]["distance_metric"] == "euclidean"
    
    def test_search_entities_cosine(self, vector_store):
        """测试使用余弦相似度搜索实体"""
        query_vector = [0.1, 0.2, 0.3]
        
        # 模拟搜索结果
        mock_obj = Mock()
        mock_obj.uuid = "test-uuid"
        mock_obj.properties = {"name": "测试实体", "type": "疾病"}
        mock_obj.metadata = Mock()
        mock_obj.metadata.distance = 0.3
        mock_obj.metadata.certainty = 0.8
        
        mock_response = Mock()
        mock_response.objects = [mock_obj]
        
        mock_collection = Mock()
        mock_collection.query.near_vector.return_value = mock_response
        vector_store.client.collections.get.return_value = mock_collection
        
        results = vector_store.search_entities(query_vector, distance_metric="cosine")
        
        assert len(results) == 1
        assert results[0]["distance_metric"] == "cosine"
        assert results[0]["certainty"] == 0.8
        assert results[0]["cosine_similarity"] == 0.8
    
    def test_search_entities_mock_client(self, vector_store):
        """测试模拟客户端搜索实体"""
        vector_store.client._is_mock = True
        
        results = vector_store.search_entities([0.1, 0.2, 0.3])
        
        assert results == []
    
    def test_search_entities_hybrid(self, vector_store):
        """测试混合检索"""
        query_vector = [0.1, 0.2, 0.3]
        
        # 模拟余弦相似度结果
        cosine_results = [
            {"id": "1", "name": "实体1", "distance_metric": "cosine"},
            {"id": "2", "name": "实体2", "distance_metric": "cosine"}
        ]
        
        # 模拟欧氏距离结果
        euclidean_results = [
            {"id": "2", "name": "实体2", "distance_metric": "euclidean", "distance": 0.5},
            {"id": "3", "name": "实体3", "distance_metric": "euclidean"}
        ]
        
        with patch.object(vector_store, 'search_entities') as mock_search:
            mock_search.side_effect = [cosine_results, euclidean_results]
            
            results = vector_store.search_entities_hybrid(query_vector, limit=5)
            
            assert "cosine_results" in results
            assert "euclidean_results" in results
            assert "hybrid_results" in results
            assert "total_unique" in results
            assert results["total_unique"] == 3  # 去重后的总数
    
    def test_search_relations(self, vector_store):
        """测试搜索关系"""
        query_vector = [0.1, 0.2, 0.3]
        
        # 模拟搜索结果
        mock_obj = Mock()
        mock_obj.uuid = "relation-uuid"
        mock_obj.properties = {
            "source": "实体A",
            "target": "实体B",
            "relation_type": "治疗",
            "description": "治疗关系",
            "source_text": "关系文本",
            "neo4j_id": "rel-123"
        }
        mock_obj.metadata = Mock()
        mock_obj.metadata.distance = 0.4
        
        mock_response = Mock()
        mock_response.objects = [mock_obj]
        
        mock_collection = Mock()
        mock_collection.query.near_vector.return_value = mock_response
        vector_store.client.collections.get.return_value = mock_collection
        
        results = vector_store.search_relations(query_vector)
        
        assert len(results) == 1
        assert results[0]["source"] == "实体A"
        assert results[0]["target"] == "实体B"
        assert results[0]["relation_type"] == "治疗"
        assert results[0]["distance"] == 0.4
    
    def test_get_stats(self, vector_store):
        """测试获取统计信息"""
        # 模拟实体集合统计
        mock_entity_collection = Mock()
        mock_entity_aggregate = Mock()
        mock_entity_aggregate.total_count = 100
        mock_entity_collection.aggregate.over_all.return_value = mock_entity_aggregate
        
        # 模拟关系集合统计
        mock_relation_collection = Mock()
        mock_relation_aggregate = Mock()
        mock_relation_aggregate.total_count = 50
        mock_relation_collection.aggregate.over_all.return_value = mock_relation_aggregate
        
        vector_store.client.collections.get.side_effect = [
            mock_entity_collection,
            mock_relation_collection
        ]
        
        stats = vector_store.get_stats()
        
        assert stats["entities"] == 100
        assert stats["relations"] == 50
        assert stats["total"] == 150
    
    def test_get_stats_error(self, vector_store):
        """测试获取统计信息时的错误处理"""
        vector_store.client.collections.get.side_effect = Exception("统计错误")
        
        stats = vector_store.get_stats()
        
        assert stats == {"entities": 0, "relations": 0, "total": 0}
    
    def test_store_entities_error(self, vector_store):
        """测试存储实体时的错误处理"""
        entities = [VectorEntity(name="实体", type="类型", vector=[0.1])]
        
        vector_store.client.collections.get.side_effect = Exception("存储错误")
        
        result = vector_store.store_entities(entities)
        
        assert result == False
    
    def test_store_relations_error(self, vector_store):
        """测试存储关系时的错误处理"""
        relations = [VectorRelation(source="A", target="B", relation_type="关系", vector=[0.1])]
        
        vector_store.client.collections.get.side_effect = Exception("存储错误")
        
        result = vector_store.store_relations(relations)
        
        assert result == False
    
    def test_search_entities_error(self, vector_store):
        """测试搜索实体时的错误处理"""
        vector_store.client.collections.get.side_effect = Exception("搜索错误")
        
        results = vector_store.search_entities([0.1, 0.2, 0.3])
        
        assert results == []
    
    def test_search_relations_error(self, vector_store):
        """测试搜索关系时的错误处理"""
        vector_store.client.collections.get.side_effect = Exception("搜索错误")
        
        results = vector_store.search_relations([0.1, 0.2, 0.3])
        
        assert results == []
    
    def test_search_entities_hybrid_error(self, vector_store):
        """测试混合检索时的错误处理"""
        with patch.object(vector_store, 'search_entities', side_effect=Exception("搜索错误")):
            results = vector_store.search_entities_hybrid([0.1, 0.2, 0.3])
            
            assert results == {
                "cosine_results": [],
                "euclidean_results": [],
                "hybrid_results": [],
                "total_unique": 0
            }