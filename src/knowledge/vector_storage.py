#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量存储模块
使用Ollama的bgm-m3:latest模型进行向量化，并存储到Weaviate
"""

import json
import requests
import weaviate
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

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
    neo4j_id: str = None  # Neo4j节点ID，用于联动
    
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
    neo4j_id: str = None  # Neo4j关系ID，用于联动
    
    def __post_init__(self):
        if self.vector is None:
            self.vector = []

class OllamaEmbeddingClient:
    """Ollama向量化客户端"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "bge-m3:latest"):
        self.base_url = base_url
        self.model = model
        
    def embed_text(self, text: str, timeout: int = 60, max_retries: int = 3) -> List[float]:
        """将文本转换为向量"""
        if not text or not text.strip():
            logger.warning("文本为空，无法向量化")
            return []
        
        url = f"{self.base_url}/api/embeddings"
        
        data = {
            "model": self.model,
            "prompt": text.strip()
        }
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"向量化请求 (尝试 {attempt + 1}/{max_retries}): {text[:50]}...")
                response = requests.post(url, json=data, timeout=timeout)
                response.raise_for_status()
                result = response.json()
                
                embedding = result.get("embedding", [])
                if embedding:
                    logger.debug(f"向量化成功，维度: {len(embedding)}")
                    return embedding
                else:
                    logger.warning("向量化响应中没有embedding字段")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"向量化请求超时 (尝试 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(1)  # 等待1秒后重试
            except requests.exceptions.ConnectionError:
                logger.warning(f"向量化连接失败 (尝试 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2)  # 等待2秒后重试
            except Exception as e:
                logger.error(f"向量化请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        logger.error(f"向量化最终失败，已尝试 {max_retries} 次")
        return []
    
    def embed_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """批量向量化文本"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"正在处理批次 {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            for text in batch:
                embedding = self.embed_text(text)
                embeddings.append(embedding)
                
        return embeddings

class WeaviateVectorStore:
    """Weaviate向量存储"""
    
    def __init__(self, url: str = "http://localhost:8080"):
        self.client = weaviate.connect_to_local(host="localhost", port=8080)
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
                    Property(name="properties", data_type=DataType.TEXT, description="Entity properties as JSON string"),
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
                    Property(name="properties", data_type=DataType.TEXT, description="Relation properties as JSON string"),
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

class VectorKnowledgeProcessor:
    """向量知识处理器"""
    
    def __init__(self, embedding_client: OllamaEmbeddingClient, vector_store: WeaviateVectorStore):
        self.embedding_client = embedding_client
        self.vector_store = vector_store
    
    def process_entities_and_relations(self, entities: List[Any], relations: List[Any], source_text: str = "") -> Tuple[List[VectorEntity], List[VectorRelation]]:
        """处理实体和关系，生成向量"""
        logger.info(f"开始向量化处理: {len(entities)} 个实体, {len(relations)} 个关系")
        
        # 处理实体
        vector_entities = []
        entity_texts = []
        
        for entity in entities:
            # 构建实体的文本表示
            entity_text = f"实体: {entity.name}, 类型: {entity.type}"
            if hasattr(entity, 'properties') and entity.properties and entity.properties.get("description"):
                entity_text += f", 描述: {entity.properties['description']}"
            entity_texts.append(entity_text)
        
        # 批量向量化实体
        if entity_texts:
            logger.info("正在向量化实体...")
            entity_embeddings = self.embedding_client.embed_batch(entity_texts)
            
            for i, entity in enumerate(entities):
                if i < len(entity_embeddings) and entity_embeddings[i]:
                    vector_entity = VectorEntity(
                        name=entity.name,
                        type=entity.type,
                        description=getattr(entity, 'properties', {}).get('description', '') if hasattr(entity, 'properties') else '',
                        vector=entity_embeddings[i],
                        source_text=source_text,
                        neo4j_id=getattr(entity, 'neo4j_id', None)  # 从实体获取Neo4j ID
                    )
                    vector_entities.append(vector_entity)
        
        # 处理关系
        vector_relations = []
        relation_texts = []
        
        for relation in relations:
            # 构建关系的文本表示
            relation_text = f"关系: {relation.source} {relation.relation_type} {relation.target}"
            if hasattr(relation, 'properties') and relation.properties and relation.properties.get("description"):
                relation_text += f", 描述: {relation.properties['description']}"
            relation_texts.append(relation_text)
        
        # 批量向量化关系
        if relation_texts:
            logger.info("正在向量化关系...")
            relation_embeddings = self.embedding_client.embed_batch(relation_texts)
            
            for i, relation in enumerate(relations):
                if i < len(relation_embeddings) and relation_embeddings[i]:
                    vector_relation = VectorRelation(
                        source=relation.source,
                        target=relation.target,
                        relation_type=relation.relation_type,
                        description=getattr(relation, 'properties', {}).get('description', '') if hasattr(relation, 'properties') else '',
                        vector=relation_embeddings[i],
                        source_text=source_text,
                        neo4j_id=getattr(relation, 'neo4j_id', None)  # 从关系获取Neo4j ID
                    )
                    vector_relations.append(vector_relation)
        
        logger.info(f"向量化完成: {len(vector_entities)} 个实体向量, {len(vector_relations)} 个关系向量")
        return vector_entities, vector_relations
    
    def store_vectors(self, vector_entities: List[VectorEntity], vector_relations: List[VectorRelation]) -> bool:
        """存储向量到Weaviate"""
        success = True
        
        if vector_entities:
            success &= self.vector_store.store_entities(vector_entities)
        
        if vector_relations:
            success &= self.vector_store.store_relations(vector_relations)
        
        return success
    
    def process_and_store_entities(self, entities: List[Dict], source_text: str = "") -> bool:
        """处理并存储实体"""
        try:
            # 构建实体文本
            entity_texts = []
            for entity in entities:
                entity_text = f"实体: {entity['name']}, 类型: {entity['type']}"
                if entity.get('description'):
                    entity_text += f", 描述: {entity['description']}"
                entity_texts.append(entity_text)
            
            # 向量化
            if entity_texts:
                embeddings = self.embedding_client.embed_batch(entity_texts)
                
                # 创建向量实体
                vector_entities = []
                for i, entity in enumerate(entities):
                    if i < len(embeddings) and embeddings[i]:
                        vector_entity = VectorEntity(
                            name=entity['name'],
                            type=entity['type'],
                            description=entity.get('description', ''),
                            vector=embeddings[i],
                            source_text=source_text,
                            neo4j_id=entity.get('neo4j_id')  # 从实体获取Neo4j ID
                        )
                        vector_entities.append(vector_entity)
                
                # 存储
                return self.vector_store.store_entities(vector_entities)
            
            return True
            
        except Exception as e:
            logger.error(f"处理并存储实体失败: {e}")
            return False
    
    def process_and_store_relations(self, relations: List[Dict], source_text: str = "") -> bool:
        """处理并存储关系"""
        try:
            # 构建关系文本
            relation_texts = []
            for relation in relations:
                relation_text = f"关系: {relation['source']} {relation['relation_type']} {relation['target']}"
                if relation.get('description'):
                    relation_text += f", 描述: {relation['description']}"
                relation_texts.append(relation_text)
            
            # 向量化
            if relation_texts:
                embeddings = self.embedding_client.embed_batch(relation_texts)
                
                # 创建向量关系
                vector_relations = []
                for i, relation in enumerate(relations):
                    if i < len(embeddings) and embeddings[i]:
                        vector_relation = VectorRelation(
                            source=relation['source'],
                            target=relation['target'],
                            relation_type=relation['relation_type'],
                            description=relation.get('description', ''),
                            vector=embeddings[i],
                            source_text=source_text,
                            neo4j_id=relation.get('neo4j_id')  # 从关系获取Neo4j ID
                        )
                        vector_relations.append(vector_relation)
                
                # 存储
                return self.vector_store.store_relations(vector_relations)
            
            return True
            
        except Exception as e:
            logger.error(f"处理并存储关系失败: {e}")
            return False

    def search_knowledge(self, query: str, limit: int = 10) -> List[Dict]:
        """搜索知识，返回合并的结果"""
        try:
            # 向量化查询
            query_vector = self.embedding_client.embed_text(query)
            if not query_vector:
                logger.error("查询向量化失败")
                return []
            
            # 搜索相似实体和关系
            similar_entities = self.vector_store.search_entities(query_vector, limit//2)
            similar_relations = self.vector_store.search_relations(query_vector, limit//2)
            
            # 合并结果
            results = []
            results.extend(similar_entities)
            results.extend(similar_relations)
            
            # 按距离排序
            results.sort(key=lambda x: x.get('distance', float('inf')))
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"知识搜索失败: {e}")
            return []

    def search_knowledge_detailed(self, query: str, entity_limit: int = 5, relation_limit: int = 5) -> Dict[str, List]:
        """搜索知识，返回详细分类结果"""
        try:
            # 检查查询有效性
            if not query or not query.strip():
                logger.warning("查询为空")
                return {"entities": [], "relations": []}
            
            # 向量化查询
            query_vector = self.embedding_client.embed_text(query)
            if not query_vector:
                logger.error("查询向量化失败")
                return {"entities": [], "relations": []}
            
            # 检查向量维度
            if len(query_vector) == 0:
                logger.error("查询向量为空")
                return {"entities": [], "relations": []}
            
            logger.info(f"查询向量化成功，维度: {len(query_vector)}")
            
            # 搜索相似实体和关系
            similar_entities = []
            similar_relations = []
            
            try:
                similar_entities = self.vector_store.search_entities(query_vector, entity_limit)
                logger.info(f"实体搜索完成，找到 {len(similar_entities)} 个结果")
            except Exception as e:
                logger.error(f"实体搜索失败: {e}")
            
            try:
                similar_relations = self.vector_store.search_relations(query_vector, relation_limit)
                logger.info(f"关系搜索完成，找到 {len(similar_relations)} 个结果")
            except Exception as e:
                logger.error(f"关系搜索失败: {e}")
            
            # 确保返回的是列表
            if not isinstance(similar_entities, list):
                similar_entities = []
            if not isinstance(similar_relations, list):
                similar_relations = []
            
            result = {
                "entities": similar_entities,
                "relations": similar_relations
            }
            
            return result
            
        except Exception as e:
            logger.error(f"详细知识搜索失败: {e}")
            return {"entities": [], "relations": []}
            
    def search_knowledge_hybrid(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """使用余弦相似度和欧氏距离进行混合知识检索
        
        Args:
            query: 查询文本
            limit: 返回结果数量限制
            
        Returns:
            包含混合检索结果的字典
        """
        try:
            # 检查查询有效性
            if not query or not query.strip():
                logger.warning("查询为空")
                return self._empty_hybrid_result()
            
            # 向量化查询
            query_vector = self.embedding_client.embed_text(query)
            if not query_vector:
                logger.error("查询向量化失败")
                return self._empty_hybrid_result()
            
            logger.info(f"查询向量化成功，维度: {len(query_vector)}")
            
            # 执行混合检索
            hybrid_results = self.vector_store.search_entities_hybrid(query_vector, limit)
            
            # 格式化结果，提取纯文本文档内容
            formatted_results = self._format_hybrid_results(hybrid_results, query)
            
            logger.info(f"混合检索完成: 找到 {len(formatted_results['top5_knowledge'])} 个知识片段")
            return formatted_results
            
        except Exception as e:
            logger.error(f"混合知识检索失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return self._empty_hybrid_result()
    
    def _empty_hybrid_result(self) -> Dict[str, Any]:
        """返回空的混合检索结果"""
        return {
            "top5_knowledge": [],
            "cosine_results": [],
            "euclidean_results": [],
            "hybrid_results": [],
            "retrieval_stats": {
                "total_found": 0,
                "cosine_count": 0,
                "euclidean_count": 0,
                "hybrid_count": 0
            }
        }
    
    def _format_hybrid_results(self, hybrid_results: Dict[str, List[Dict]], query: str) -> Dict[str, Any]:
        """格式化混合检索结果"""
        try:
            # 提取top5知识片段（纯文本文档内容）
            top5_knowledge = []
            for i, result in enumerate(hybrid_results.get("hybrid_results", [])[:5], 1):
                knowledge_item = {
                    "rank": i,
                    "content": result.get("description", ""),  # 纯文本文档内容
                    "source": result.get("name", f"文档片段_{i}"),
                    "cosine_similarity": result.get("cosine_similarity", None),
                    "euclidean_distance": result.get("distance", None),
                    "rank_cosine": result.get("rank_cosine", None),
                    "rank_euclidean": result.get("rank_euclidean", None),
                    "source_text": result.get("source_text", "")
                }
                
                # 确保内容不为空
                if knowledge_item["content"]:
                    top5_knowledge.append(knowledge_item)
            
            # 构建完整结果
            formatted_result = {
                "top5_knowledge": top5_knowledge,
                "cosine_results": hybrid_results.get("cosine_results", []),
                "euclidean_results": hybrid_results.get("euclidean_results", []),
                "hybrid_results": hybrid_results.get("hybrid_results", []),
                "retrieval_stats": {
                    "total_found": hybrid_results.get("total_unique", 0),
                    "cosine_count": len(hybrid_results.get("cosine_results", [])),
                    "euclidean_count": len(hybrid_results.get("euclidean_results", [])),
                    "hybrid_count": len(hybrid_results.get("hybrid_results", []))
                }
            }
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"格式化混合检索结果失败: {e}")
            return self._empty_hybrid_result()
    
    def get_knowledge_for_prompt(self, query: str, limit: int = 5) -> str:
        """获取用于提示词模板的知识内容
        
        Args:
            query: 查询文本
            limit: 知识片段数量限制
            
        Returns:
            格式化的知识内容字符串
        """
        try:
            # 执行混合检索
            hybrid_results = self.search_knowledge_hybrid(query, limit)
            
            # 提取top5知识
            top5_knowledge = hybrid_results.get("top5_knowledge", [])
            
            if not top5_knowledge:
                return "未找到相关知识内容。"
            
            # 格式化知识内容
            knowledge_text = "相关知识内容：\n\n"
            for i, knowledge in enumerate(top5_knowledge, 1):
                content = knowledge.get("content", "").strip()
                source = knowledge.get("source", f"文档片段_{i}")
                
                if content:
                    knowledge_text += f"{i}. 【{source}】\n{content}\n\n"
            
            # 添加检索统计信息
            stats = hybrid_results.get("retrieval_stats", {})
            knowledge_text += f"（检索统计：共找到 {stats.get('total_found', 0)} 个相关片段，"
            knowledge_text += f"余弦相似度 {stats.get('cosine_count', 0)} 个，"
            knowledge_text += f"欧氏距离 {stats.get('euclidean_count', 0)} 个）"
            
            return knowledge_text
            
        except Exception as e:
            logger.error(f"获取提示词知识内容失败: {e}")
            return "知识检索过程中发生错误。"