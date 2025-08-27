#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量化图谱存储模块
在Weaviate中存储Neo4j图谱的向量化节点和关系，实现真正的向量化图谱检索
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import os

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """图谱节点"""
    name: str
    type: str
    description: str = ""
    properties: Dict[str, Any] = None
    source_text: str = ""
    neo4j_id: str = ""
    vector: List[float] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.vector is None:
            self.vector = []

@dataclass
class GraphRelation:
    """图谱关系"""
    source_node: str
    target_node: str
    relation_type: str
    description: str = ""
    properties: Dict[str, Any] = None
    source_text: str = ""
    neo4j_id: str = ""
    vector: List[float] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.vector is None:
            self.vector = []

@dataclass
class KnowledgeChain:
    """知识链路"""
    nodes: List[GraphNode]
    relations: List[GraphRelation]
    chain_score: float = 0.0
    chain_description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [{
                "name": node.name,
                "type": node.type,
                "description": node.description,
                "neo4j_id": node.neo4j_id
            } for node in self.nodes],
            "relations": [{
                "source": rel.source_node,
                "target": rel.target_node,
                "type": rel.relation_type,
                "description": rel.description,
                "neo4j_id": rel.neo4j_id
            } for rel in self.relations],
            "chain_score": self.chain_score,
            "chain_description": self.chain_description
        }

class VectorizedGraphStorage:
    """向量化图谱存储服务"""
    
    def __init__(self, weaviate_url: str = None):
        # 从配置文件或环境变量获取设置
        self.weaviate_config = self._load_weaviate_config()
        self.weaviate_url = weaviate_url or self.weaviate_config.get('url', 'http://localhost:8080')
        self.weaviate_host = self.weaviate_config.get('host', 'localhost')
        self.weaviate_port = self.weaviate_config.get('port', 8080)
        self.weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
        self.client = self._init_weaviate_client()
        
        # 集合名称
        self.graph_nodes_collection = "GraphNodes"
        self.graph_relations_collection = "GraphRelations"
        
        # 延迟初始化嵌入服务
        self.embedding_service = None
        
        # 设置集合
        self.setup_collections()
        
        logger.info("向量化图谱存储服务初始化完成")
    
    def _get_embedding_service(self):
        """延迟初始化嵌入服务"""
        if self.embedding_service is None:
            try:
                from ..services.embedding_service import OllamaEmbeddingService
                self.embedding_service = OllamaEmbeddingService()
            except ImportError as e:
                logger.error(f"无法导入嵌入服务: {e}")
                # 创建一个模拟的嵌入服务
                class MockEmbeddingService:
                    def embed_text(self, text):
                        return [0.1] * 384  # 返回固定维度的模拟向量
                self.embedding_service = MockEmbeddingService()
        return self.embedding_service
    
    def _load_weaviate_config(self) -> Dict[str, Any]:
        """加载Weaviate配置"""
        try:
            import yaml
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'config.yaml')
            
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    return config.get('weaviate', {})
            else:
                logger.warning(f"配置文件不存在: {config_path}")
                return {}
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}")
            return {}
    
    def _init_weaviate_client(self):
        """初始化Weaviate客户端"""
        try:
            import weaviate
            from weaviate.classes.init import Auth
            
            # 确保端口是整数类型
            port = int(self.weaviate_port) if isinstance(self.weaviate_port, str) else self.weaviate_port
            
            client = weaviate.connect_to_local(
                host=self.weaviate_host,
                port=port,
                grpc_port=50051,
                headers={"X-OpenAI-Api-Key": "dummy"},
                skip_init_checks=True,
                additional_config=weaviate.classes.init.AdditionalConfig(
                    timeout=weaviate.classes.init.Timeout(init=30, query=60, insert=120)
                )
            )
            
            logger.info(f"Weaviate客户端连接成功 - {self.weaviate_host}:{port}")
            return client
            
        except Exception as e:
            logger.warning(f"Weaviate连接失败，使用模拟客户端: {e}")
            return self._create_mock_client()
    
    def _create_mock_client(self):
        """创建模拟客户端"""
        class MockWeaviateClient:
            def __init__(self):
                self._is_mock = True
                self.collections = MockCollections()
            
            def is_ready(self):
                return False
        
        class MockCollections:
            def exists(self, name):
                return False
            
            def delete(self, name):
                logger.warning(f"模拟删除集合: {name}")
            
            def create(self, **kwargs):
                logger.warning(f"模拟创建集合: {kwargs.get('name', 'unknown')}")
                return MockCollection()
            
            def get(self, name):
                return MockCollection()
        
        class MockCollection:
            def __init__(self):
                self.data = MockData()
                self.query = MockQuery()
                self.aggregate = MockAggregate()
        
        class MockData:
            def insert_many(self, objects):
                logger.warning(f"模拟插入 {len(objects)} 个对象")
                return []
        
        class MockQuery:
            def near_vector(self, *args, **kwargs):
                logger.warning("模拟向量搜索")
                return MockResponse()
        
        class MockAggregate:
            def over_all(self, total_count=False):
                return MockAggregateResult()
        
        class MockAggregateResult:
            def __init__(self):
                self.total_count = 0
        
        class MockResponse:
            def __init__(self):
                self.objects = []
        
        return MockWeaviateClient()
    
    def is_mock_client(self) -> bool:
        """检查是否为模拟客户端"""
        return hasattr(self.client, '_is_mock') and self.client._is_mock
    
    def setup_collections(self) -> bool:
        """设置Weaviate集合"""
        if self.is_mock_client():
            logger.warning("使用模拟客户端，跳过集合设置")
            return True
        
        try:
            from weaviate.classes.config import Configure, Property, DataType
            
            # 检查集合是否已存在，如果存在则直接返回
            nodes_exists = self.client.collections.exists(self.graph_nodes_collection)
            relations_exists = self.client.collections.exists(self.graph_relations_collection)
            
            if nodes_exists and relations_exists:
                logger.info(f"图谱集合已存在，跳过创建: {self.graph_nodes_collection}, {self.graph_relations_collection}")
                return True
            
            # 创建图谱节点集合（如果不存在）
            if not nodes_exists:
                nodes_collection = self.client.collections.create(
                    name=self.graph_nodes_collection,
                    description="Knowledge graph nodes with vector embeddings",
                    vectorizer_config=Configure.Vectorizer.none(),  # 使用自定义向量
                    properties=[
                        Property(name="name", data_type=DataType.TEXT, description="Node name"),
                        Property(name="type", data_type=DataType.TEXT, description="Node type"),
                        Property(name="description", data_type=DataType.TEXT, description="Node description"),
                        Property(name="source_text", data_type=DataType.TEXT, description="Original source text"),
                        Property(name="neo4j_id", data_type=DataType.TEXT, description="Neo4j node ID"),
                        Property(name="properties", data_type=DataType.TEXT, description="Additional properties as JSON"),
                        Property(name="created_at", data_type=DataType.TEXT, description="Creation timestamp")
                    ]
                )
                logger.info(f"创建图谱节点集合: {self.graph_nodes_collection}")
            
            # 创建图谱关系集合（如果不存在）
            if not relations_exists:
                relations_collection = self.client.collections.create(
                    name=self.graph_relations_collection,
                    description="Knowledge graph relations with vector embeddings",
                    vectorizer_config=Configure.Vectorizer.none(),  # 使用自定义向量
                    properties=[
                        Property(name="source_node", data_type=DataType.TEXT, description="Source node name"),
                        Property(name="target_node", data_type=DataType.TEXT, description="Target node name"),
                        Property(name="relation_type", data_type=DataType.TEXT, description="Relation type"),
                        Property(name="description", data_type=DataType.TEXT, description="Relation description"),
                        Property(name="source_text", data_type=DataType.TEXT, description="Original source text"),
                        Property(name="neo4j_id", data_type=DataType.TEXT, description="Neo4j relation ID"),
                        Property(name="properties", data_type=DataType.TEXT, description="Additional properties as JSON"),
                        Property(name="created_at", data_type=DataType.TEXT, description="Creation timestamp")
                    ]
                )
                logger.info(f"创建图谱关系集合: {self.graph_relations_collection}")
            
            return True
            
        except Exception as e:
            logger.error(f"设置集合失败: {e}")
            return False
    
    def vectorize_and_store_nodes(self, nodes: List[GraphNode]) -> bool:
        """向量化并存储图谱节点"""
        if self.is_mock_client():
            logger.warning(f"使用模拟客户端，模拟存储 {len(nodes)} 个节点")
            return True
        
        try:
            nodes_collection = self.client.collections.get(self.graph_nodes_collection)
            
            data_objects = []
            for node in nodes:
                # 生成节点的文本表示用于向量化
                node_text = self._generate_node_text(node)
                
                # 生成向量
                if not node.vector:
                    node.vector = self._get_embedding_service().embed_text(node_text)
                
                if not node.vector:
                    logger.warning(f"节点 {node.name} 向量生成失败，跳过")
                    continue
                
                properties = {
                    "name": node.name,
                    "type": node.type,
                    "description": node.description,
                    "source_text": node.source_text,
                    "neo4j_id": node.neo4j_id,
                    "properties": str(node.properties) if node.properties else "",
                    "created_at": datetime.now().isoformat()
                }
                
                data_objects.append({
                    "properties": properties,
                    "vector": node.vector
                })
            
            if data_objects:
                from weaviate.classes.data import DataObject
                nodes_collection.data.insert_many([
                    DataObject(properties=obj["properties"], vector=obj["vector"])
                    for obj in data_objects
                ])
            
            logger.info(f"成功存储 {len(data_objects)} 个图谱节点向量")
            return True
            
        except Exception as e:
            logger.error(f"存储图谱节点向量失败: {e}")
            return False
    
    def vectorize_and_store_relations(self, relations: List[GraphRelation]) -> bool:
        """向量化并存储图谱关系"""
        if self.is_mock_client():
            logger.warning(f"使用模拟客户端，模拟存储 {len(relations)} 个关系")
            return True
        
        try:
            relations_collection = self.client.collections.get(self.graph_relations_collection)
            
            data_objects = []
            for relation in relations:
                # 生成关系的文本表示用于向量化
                relation_text = self._generate_relation_text(relation)
                
                # 生成向量
                if not relation.vector:
                    relation.vector = self._get_embedding_service().embed_text(relation_text)
                
                if not relation.vector:
                    logger.warning(f"关系 {relation.source_node}-{relation.relation_type}-{relation.target_node} 向量生成失败，跳过")
                    continue
                
                properties = {
                    "source_node": relation.source_node,
                    "target_node": relation.target_node,
                    "relation_type": relation.relation_type,
                    "description": relation.description,
                    "source_text": relation.source_text,
                    "neo4j_id": relation.neo4j_id,
                    "properties": str(relation.properties) if relation.properties else "",
                    "created_at": datetime.now().isoformat()
                }
                
                data_objects.append({
                    "properties": properties,
                    "vector": relation.vector
                })
            
            if data_objects:
                from weaviate.classes.data import DataObject
                relations_collection.data.insert_many([
                    DataObject(properties=obj["properties"], vector=obj["vector"])
                    for obj in data_objects
                ])
            
            logger.info(f"成功存储 {len(data_objects)} 个图谱关系向量")
            return True
            
        except Exception as e:
            logger.error(f"存储图谱关系向量失败: {e}")
            return False
    
    def _generate_node_text(self, node: GraphNode) -> str:
        """生成节点的文本表示"""
        parts = []
        
        # 节点名称和类型
        parts.append(f"节点: {node.name}")
        parts.append(f"类型: {node.type}")
        
        # 描述
        if node.description:
            parts.append(f"描述: {node.description}")
        
        # 属性
        if node.properties:
            for key, value in node.properties.items():
                parts.append(f"{key}: {value}")
        
        # 原始文本
        if node.source_text:
            parts.append(f"来源: {node.source_text}")
        
        return " ".join(parts)
    
    def _generate_relation_text(self, relation: GraphRelation) -> str:
        """生成关系的文本表示"""
        parts = []
        
        # 关系三元组
        parts.append(f"关系: {relation.source_node} {relation.relation_type} {relation.target_node}")
        
        # 描述
        if relation.description:
            parts.append(f"描述: {relation.description}")
        
        # 属性
        if relation.properties:
            for key, value in relation.properties.items():
                parts.append(f"{key}: {value}")
        
        # 原始文本
        if relation.source_text:
            parts.append(f"来源: {relation.source_text}")
        
        return " ".join(parts)
    
    def search_similar_nodes(self, query_vector: List[float] = None, query: str = None, limit: int = 10, min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """搜索相似节点"""
        if self.is_mock_client():
            logger.warning("使用模拟客户端，返回空搜索结果")
            return []
        
        # 如果传入的是查询文本，先转换为向量
        if query and not query_vector:
            query_vector = self._get_embedding_service().embed_text(query)
        
        if not query_vector:
            logger.error("需要提供查询向量或查询文本")
            return []
        
        try:
            nodes_collection = self.client.collections.get(self.graph_nodes_collection)
            
            response = nodes_collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                return_metadata=['distance', 'certainty']
            )
            
            results = []
            for obj in response.objects:
                similarity = obj.metadata.certainty if obj.metadata else 0.0
                
                # 应用最小相似度过滤
                if similarity >= min_similarity:
                    result = {
                        "id": str(obj.uuid),
                        "name": obj.properties.get("name"),
                        "type": obj.properties.get("type"),
                        "description": obj.properties.get("description"),
                        "source_text": obj.properties.get("source_text"),
                        "neo4j_id": obj.properties.get("neo4j_id"),
                        "distance": obj.metadata.distance if obj.metadata else None,
                        "similarity": similarity
                    }
                    results.append(result)
            
            logger.info(f"搜索到 {len(results)} 个相似节点")
            return results
            
        except Exception as e:
            logger.error(f"搜索相似节点失败: {e}")
            return []
    
    def search_similar_relations(self, query_vector: List[float], limit: int = 10, min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """搜索相似关系"""
        if self.is_mock_client():
            logger.warning("使用模拟客户端，返回空搜索结果")
            return []
        
        try:
            relations_collection = self.client.collections.get(self.graph_relations_collection)
            
            response = relations_collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                return_metadata=['distance', 'certainty']
            )
            
            results = []
            for obj in response.objects:
                similarity = obj.metadata.certainty if obj.metadata else 0.0
                
                # 应用最小相似度过滤
                if similarity >= min_similarity:
                    result = {
                        "id": str(obj.uuid),
                        "source_node": obj.properties.get("source_node"),
                        "target_node": obj.properties.get("target_node"),
                        "relation_type": obj.properties.get("relation_type"),
                        "description": obj.properties.get("description"),
                        "source_text": obj.properties.get("source_text"),
                        "neo4j_id": obj.properties.get("neo4j_id"),
                        "distance": obj.metadata.distance if obj.metadata else None,
                        "similarity": similarity
                    }
                    results.append(result)
            
            logger.info(f"搜索到 {len(results)} 个相似关系")
            return results
            
        except Exception as e:
            logger.error(f"搜索相似关系失败: {e}")
            return []
    
    def get_stats(self) -> Dict[str, int]:
        """获取存储统计信息"""
        if self.is_mock_client():
            return {"nodes": 0, "relations": 0, "total": 0}
        
        try:
            nodes_collection = self.client.collections.get(self.graph_nodes_collection)
            relations_collection = self.client.collections.get(self.graph_relations_collection)
            
            nodes_count = nodes_collection.aggregate.over_all(total_count=True).total_count
            relations_count = relations_collection.aggregate.over_all(total_count=True).total_count
            
            stats = {
                "nodes": nodes_count,
                "relations": relations_count,
                "total": nodes_count + relations_count
            }
            
            logger.info(f"向量化图谱存储统计: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"nodes": 0, "relations": 0, "total": 0}