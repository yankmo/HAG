#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量存储模块
提供Weaviate向量存储的核心功能
"""

import sys
import os
import weaviate
from typing import List, Dict, Any
from dataclasses import dataclass
import logging
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import get_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VectorEntity:
    """向量化实体"""
    name: str
    type: str
    properties: Dict[str, Any] = None
    vector: List[float] = None
    source_text: str = ""
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.vector is None:
            self.vector = []
    
@dataclass
class VectorRelation:
    """向量化关系"""
    source: str
    target: str
    relation_type: str
    description: str = ""
    vector: List[float] = None
    source_text: str = ""
    
    def __post_init__(self):
        if self.vector is None:
            self.vector = []

class WeaviateVectorStore:
    """Weaviate向量存储"""
    
    def __init__(self, url: str = None):
        config = get_config()
        
        url = url or config.weaviate.url
        self.client = weaviate.connect_to_local(host=config.weaviate.host, port=config.weaviate.port)
        self.entity_collection = "MedicalEntities"
        self.relation_collection = "MedicalRelations"
        
    def setup_collections(self) -> bool:
        """设置Weaviate集合"""
        try:
            # 删除已存在的集合（如果存在）
            if self.client.collections.exists(self.entity_collection):
                self.client.collections.delete(self.entity_collection)
                logger.info(f"删除已存在的实体集合: {self.entity_collection}")
                
            if self.client.collections.exists(self.relation_collection):
                self.client.collections.delete(self.relation_collection)
                logger.info(f"删除已存在的关系集合: {self.relation_collection}")
            
            # 创建实体集合
            from weaviate.classes.config import Configure, Property, DataType
            
            entity_collection = self.client.collections.create(
                name=self.entity_collection,
                description="Medical entities with embeddings",
                properties=[
                    Property(name="name", data_type=DataType.TEXT, description="Entity name"),
                    Property(name="type", data_type=DataType.TEXT, description="Entity type"),
                    Property(name="description", data_type=DataType.TEXT, description="Entity description"),
                    Property(name="source_text", data_type=DataType.TEXT, description="Original source text"),
                    Property(name="neo4j_id", data_type=DataType.TEXT, description="Neo4j node ID"),
                    Property(name="created_at", data_type=DataType.TEXT, description="Creation timestamp")
                ]
            )
            logger.info(f"创建实体集合: {self.entity_collection}")
            
            # 创建关系集合
            relation_collection = self.client.collections.create(
                name=self.relation_collection,
                description="Medical relations with embeddings",
                properties=[
                    Property(name="source", data_type=DataType.TEXT, description="Source entity"),
                    Property(name="target", data_type=DataType.TEXT, description="Target entity"),
                    Property(name="relation_type", data_type=DataType.TEXT, description="Relation type"),
                    Property(name="description", data_type=DataType.TEXT, description="Relation description"),
                    Property(name="source_text", data_type=DataType.TEXT, description="Original source text"),
                    Property(name="neo4j_id", data_type=DataType.TEXT, description="Neo4j relation ID"),
                    Property(name="created_at", data_type=DataType.TEXT, description="Creation timestamp")
                ]
            )
            logger.info(f"创建关系集合: {self.relation_collection}")
            
            return True
            
        except Exception as e:
            logger.error(f"设置集合失败: {e}")
            return False
    
    def store_entities(self, entities: List[VectorEntity]) -> bool:
        """存储实体向量"""
        try:
            entity_collection = self.client.collections.get(self.entity_collection)
            
            data_objects = []
            for entity in entities:
                if not entity.vector:
                    logger.warning(f"实体 {entity.name} 没有向量，跳过")
                    continue
                    
                properties = {
                    "name": entity.name,
                    "type": entity.type,
                    "description": entity.properties.get('description', '') if entity.properties else '',
                    "source_text": entity.source_text,
                    "neo4j_id": entity.neo4j_id or "",  # Neo4j节点ID
                    "created_at": datetime.now().isoformat()
                }
                
                data_objects.append({
                    "properties": properties,
                    "vector": entity.vector
                })
            
            if data_objects:
                from weaviate.classes.data import DataObject
                entity_collection.data.insert_many([
                    DataObject(properties=obj["properties"], vector=obj["vector"])
                    for obj in data_objects
                ])
            
            logger.info(f"成功存储 {len(data_objects)} 个实体向量")
            return True
            
        except Exception as e:
            logger.error(f"存储实体向量失败: {e}")
            return False
    
    def store_relations(self, relations: List[VectorRelation]) -> bool:
        """存储关系向量"""
        try:
            relation_collection = self.client.collections.get(self.relation_collection)
            
            data_objects = []
            for relation in relations:
                if not relation.vector:
                    logger.warning(f"关系 {relation.source}-{relation.relation_type}-{relation.target} 没有向量，跳过")
                    continue
                    
                properties = {
                    "source": relation.source,
                    "target": relation.target,
                    "relation_type": relation.relation_type,
                    "description": relation.description,
                    "source_text": relation.source_text,
                    "neo4j_id": relation.neo4j_id or "",  # Neo4j关系ID
                    "created_at": datetime.now().isoformat()
                }
                
                data_objects.append({
                    "properties": properties,
                    "vector": relation.vector
                })
            
            if data_objects:
                from weaviate.classes.data import DataObject
                relation_collection.data.insert_many([
                    DataObject(properties=obj["properties"], vector=obj["vector"])
                    for obj in data_objects
                ])
            
            logger.info(f"成功存储 {len(data_objects)} 个关系向量")
            return True
            
        except Exception as e:
            logger.error(f"存储关系向量失败: {e}")
            return False
    
    def search_entities(self, query_vector: List[float], limit: int = 10, distance_metric: str = "euclidean") -> List[Dict]:
        """搜索相似实体
        
        Args:
            query_vector: 查询向量
            limit: 返回结果数量限制
            distance_metric: 距离度量方式 ("euclidean" 或 "cosine")
        """
        try:
            entity_collection = self.client.collections.get(self.entity_collection)
            
            # 根据距离度量选择查询方式
            if distance_metric == "cosine":
                response = entity_collection.query.near_vector(
                    near_vector=query_vector,
                    limit=limit,
                    return_metadata=['distance', 'certainty']
                )
            else:  # euclidean
                response = entity_collection.query.near_vector(
                    near_vector=query_vector,
                    limit=limit,
                    return_metadata=['distance']
                )
            
            results = []
            for obj in response.objects:
                result = {
                    "id": str(obj.uuid),
                    "name": obj.properties.get("name"),
                    "type": obj.properties.get("type"),
                    "description": obj.properties.get("description"),
                    "source_text": obj.properties.get("source_text"),
                    "neo4j_id": obj.properties.get("neo4j_id"),  # Neo4j节点ID
                    "distance": obj.metadata.distance if obj.metadata else None,
                    "distance_metric": distance_metric
                }
                
                # 如果是余弦相似度，添加certainty信息
                if distance_metric == "cosine" and obj.metadata:
                    result["certainty"] = obj.metadata.certainty
                    result["cosine_similarity"] = obj.metadata.certainty  # certainty就是余弦相似度
                
                results.append(result)
            
            logger.info(f"搜索到 {len(results)} 个相似实体 (距离度量: {distance_metric})")
            return results
            
        except Exception as e:
            logger.error(f"搜索实体失败: {e}")
            return []
    
    def search_entities_hybrid(self, query_vector: List[float], limit: int = 5) -> Dict[str, List[Dict]]:
        """使用余弦相似度和欧氏距离进行混合检索
        
        Args:
            query_vector: 查询向量
            limit: 每种度量方式的返回结果数量
            
        Returns:
            包含两种度量结果的字典
        """
        try:
            # 使用余弦相似度检索
            cosine_results = self.search_entities(query_vector, limit, "cosine")
            
            # 使用欧氏距离检索
            euclidean_results = self.search_entities(query_vector, limit, "euclidean")
            
            # 合并并去重（基于ID）
            all_results = []
            seen_ids = set()
            
            # 先添加余弦相似度结果
            for result in cosine_results:
                if result["id"] not in seen_ids:
                    result["rank_cosine"] = len(all_results) + 1
                    all_results.append(result)
                    seen_ids.add(result["id"])
            
            # 再添加欧氏距离结果
            for result in euclidean_results:
                if result["id"] not in seen_ids:
                    result["rank_euclidean"] = len(all_results) + 1
                    all_results.append(result)
                    seen_ids.add(result["id"])
                else:
                    # 如果已存在，添加欧氏距离排名信息
                    for existing in all_results:
                        if existing["id"] == result["id"]:
                            existing["rank_euclidean"] = euclidean_results.index(result) + 1
                            existing["euclidean_distance"] = result["distance"]
                            break
            
            # 按综合评分排序（余弦相似度权重更高）
            def calculate_score(item):
                cosine_rank = item.get("rank_cosine", limit + 1)
                euclidean_rank = item.get("rank_euclidean", limit + 1)
                # 余弦相似度权重0.7，欧氏距离权重0.3
                return 0.7 * (1.0 / cosine_rank) + 0.3 * (1.0 / euclidean_rank)
            
            all_results.sort(key=calculate_score, reverse=True)
            
            return {
                "cosine_results": cosine_results,
                "euclidean_results": euclidean_results,
                "hybrid_results": all_results[:limit],
                "total_unique": len(all_results)
            }
            
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            return {
                "cosine_results": [],
                "euclidean_results": [],
                "hybrid_results": [],
                "total_unique": 0
            }
    
    def search_relations(self, query_vector: List[float], limit: int = 10) -> List[Dict]:
        """搜索相似关系"""
        try:
            relation_collection = self.client.collections.get(self.relation_collection)
            
            response = relation_collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                return_metadata=['distance']
            )
            
            results = []
            for obj in response.objects:
                result = {
                    "id": str(obj.uuid),
                    "source": obj.properties.get("source"),
                    "target": obj.properties.get("target"),
                    "relation_type": obj.properties.get("relation_type"),
                    "description": obj.properties.get("description"),
                    "source_text": obj.properties.get("source_text"),
                    "neo4j_id": obj.properties.get("neo4j_id"),  # Neo4j关系ID
                    "distance": obj.metadata.distance if obj.metadata else None
                }
                results.append(result)
            
            logger.info(f"搜索到 {len(results)} 个相似关系")
            return results
            
        except Exception as e:
            logger.error(f"搜索关系失败: {e}")
            return []
    
    def get_stats(self) -> Dict[str, int]:
        """获取存储统计信息"""
        try:
            entity_collection = self.client.collections.get(self.entity_collection)
            relation_collection = self.client.collections.get(self.relation_collection)
            
            entity_count = entity_collection.aggregate.over_all(total_count=True).total_count
            relation_count = relation_collection.aggregate.over_all(total_count=True).total_count
            
            stats = {
                "entities": entity_count,
                "relations": relation_count,
                "total": entity_count + relation_count
            }
            
            logger.info(f"存储统计: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"entities": 0, "relations": 0, "total": 0}