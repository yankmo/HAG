#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j向量存储模块
提供基于Neo4j的向量存储和检索功能，支持意图识别
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import logging
from datetime import datetime
from py2neo import Graph, Node
from py2neo.matching import NodeMatcher, RelationshipMatcher

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import get_config
from src.knowledge.intent_recognition_neo4j import Entity, Relation, IntentRecognizer, OllamaClient

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Neo4jVectorEntity:
    """Neo4j向量化实体"""
    name: str
    type: str
    properties: Dict[str, Any] = None
    vector: List[float] = None
    source_text: str = ""
    neo4j_id: str = ""
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.vector is None:
            self.vector = []

@dataclass
class Neo4jVectorRelation:
    """Neo4j向量化关系"""
    source: str
    target: str
    relation_type: str
    description: str = ""
    vector: List[float] = None
    source_text: str = ""
    neo4j_id: str = ""
    
    def __post_init__(self):
        if self.vector is None:
            self.vector = []

@dataclass
class IntentResult:
    """意图识别结果"""
    intent_type: str
    confidence: float
    entities: List[str]
    relations: List[str]
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

class Neo4jVectorStore:
    """Neo4j向量存储"""
    
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        config = get_config()
        
        uri = uri or config.neo4j.uri
        username = username or config.neo4j.username
        password = password or config.neo4j.password
        
        try:
            self.graph = Graph(uri, auth=(username, password))
            self.node_matcher = NodeMatcher(self.graph)
            self.rel_matcher = RelationshipMatcher(self.graph)
            logger.info("Neo4j连接成功")
            
            # 初始化向量索引
            self._setup_vector_indexes()
            
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            raise
    
    def _setup_vector_indexes(self):
        """设置向量索引"""
        try:
            # 创建实体向量索引
            self.graph.run("""
                CREATE VECTOR INDEX entity_vector_index IF NOT EXISTS
                FOR (n:Entity) ON (n.vector)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1024,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
            
            # 创建关系向量索引
            self.graph.run("""
                CREATE VECTOR INDEX relation_vector_index IF NOT EXISTS
                FOR ()-[r:RELATION]-() ON (r.vector)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1024,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
            
            logger.info("向量索引设置完成")
            
        except Exception as e:
            logger.warning(f"设置向量索引失败: {e}")
    
    def clear_database(self):
        """清空数据库"""
        try:
            self.graph.delete_all()
            logger.info("数据库已清空")
            # 重新设置索引
            self._setup_vector_indexes()
        except Exception as e:
            logger.error(f"清空数据库失败: {e}")
    
    def store_entities(self, entities: List[Neo4jVectorEntity]) -> bool:
        """存储实体向量"""
        try:
            stored_count = 0
            for entity in entities:
                if not entity.vector:
                    logger.warning(f"实体 {entity.name} 没有向量，跳过")
                    continue
                
                # 创建或更新实体节点
                # 避免description重复，从properties中移除
                properties = entity.properties.copy() if entity.properties else {}
                description = properties.pop('description', '')
                
                node = Node(
                    "Entity", entity.type,
                    name=entity.name,
                    description=description,
                    source_text=entity.source_text,
                    vector=entity.vector,
                    created_at=datetime.now().isoformat(),
                    **properties
                )
                
                # 使用MERGE避免重复
                result = self.graph.run("""
                    MERGE (n:Entity {name: $name, type: $type})
                    SET n.description = $description,
                        n.source_text = $source_text,
                        n.vector = $vector,
                        n.created_at = $created_at
                    RETURN id(n) as node_id
                """, {
                    'name': entity.name,
                    'type': entity.type,
                    'description': description,
                    'source_text': entity.source_text,
                    'vector': entity.vector,
                    'created_at': datetime.now().isoformat()
                }).data()
                
                if result and len(result) > 0:
                    entity.neo4j_id = str(result[0]['node_id'])
                    stored_count += 1
            
            logger.info(f"成功存储 {stored_count} 个实体向量")
            return True
            
        except Exception as e:
            logger.error(f"存储实体向量失败: {e}")
            return False
    
    def store_relations(self, relations: List[Neo4jVectorRelation]) -> bool:
        """存储关系向量"""
        try:
            stored_count = 0
            for relation in relations:
                if not relation.vector:
                    logger.warning(f"关系 {relation.source}-{relation.relation_type}-{relation.target} 没有向量，跳过")
                    continue
                
                # 查找源节点和目标节点
                result = self.graph.run("""
                    MATCH (source:Entity {name: $source_name})
                    MATCH (target:Entity {name: $target_name})
                    MERGE (source)-[r:RELATION {type: $relation_type}]->(target)
                    SET r.description = $description,
                        r.source_text = $source_text,
                        r.vector = $vector,
                        r.created_at = $created_at
                    RETURN id(r) as rel_id
                """, {
                    'source_name': relation.source,
                    'target_name': relation.target,
                    'relation_type': relation.relation_type,
                    'description': relation.description,
                    'source_text': relation.source_text,
                    'vector': relation.vector,
                    'created_at': datetime.now().isoformat()
                }).data()
                
                if result and len(result) > 0:
                    relation.neo4j_id = str(result[0]['rel_id'])
                    stored_count += 1
            
            logger.info(f"成功存储 {stored_count} 个关系向量")
            return True
            
        except Exception as e:
            logger.error(f"存储关系向量失败: {e}")
            return False
    
    def search_entities(self, query_vector: List[float], limit: int = 10, 
                       distance_metric: str = "cosine") -> List[Dict]:
        """搜索相似实体"""
        try:
            if distance_metric == "cosine":
                # 使用Neo4j向量索引进行余弦相似度搜索
                results = self.graph.run("""
                    CALL db.index.vector.queryNodes('entity_vector_index', $limit, $query_vector)
                    YIELD node, score
                    RETURN id(node) as id, 
                           node.name as name,
                           node.type as type,
                           node.description as description,
                           node.source_text as source_text,
                           score as certainty,
                           (1.0 - score) as distance
                    ORDER BY score DESC
                """, {
                    'query_vector': query_vector,
                    'limit': limit
                }).data()
                
            else:  # euclidean
                # 手动计算欧氏距离
                all_entities = self.graph.run("""
                    MATCH (n:Entity)
                    WHERE n.vector IS NOT NULL
                    RETURN id(n) as id,
                           n.name as name,
                           n.type as type,
                           n.description as description,
                           n.source_text as source_text,
                           n.vector as vector
                """).data()
                
                # 计算欧氏距离并排序
                results = []
                query_vec = np.array(query_vector)
                
                for entity in all_entities:
                    entity_vec = np.array(entity['vector'])
                    distance = float(np.linalg.norm(query_vec - entity_vec))
                    
                    results.append({
                        'id': entity['id'],
                        'name': entity['name'],
                        'type': entity['type'],
                        'description': entity['description'],
                        'source_text': entity['source_text'],
                        'distance': distance,
                        'certainty': 1.0 / (1.0 + distance)
                    })
                
                # 按距离排序并限制结果数量
                results = sorted(results, key=lambda x: x['distance'])[:limit]
            
            logger.info(f"实体搜索完成: {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索实体失败: {e}")
            return []
    
    def search_entities_hybrid(self, query_vector: List[float], limit: int = 10) -> Dict:
        """混合搜索实体（余弦相似度 + 欧氏距离）"""
        try:
            # 获取余弦相似度结果
            cosine_results = self.search_entities(query_vector, limit, "cosine")
            
            # 获取欧氏距离结果
            euclidean_results = self.search_entities(query_vector, limit, "euclidean")
            
            # 合并结果并计算混合分数
            all_results = {}
            
            # 处理余弦相似度结果
            for i, result in enumerate(cosine_results):
                entity_id = result['id']
                all_results[entity_id] = {
                    **result,
                    'rank_cosine': i + 1,
                    'rank_euclidean': limit + 1  # 默认值
                }
            
            # 处理欧氏距离结果
            for i, result in enumerate(euclidean_results):
                entity_id = result['id']
                if entity_id in all_results:
                    all_results[entity_id]['rank_euclidean'] = i + 1
                    all_results[entity_id]['euclidean_distance'] = result['distance']
                else:
                    all_results[entity_id] = {
                        **result,
                        'rank_cosine': limit + 1,
                        'rank_euclidean': i + 1,
                        'euclidean_distance': result['distance']
                    }
            
            # 计算混合分数并排序
            hybrid_results = []
            cosine_weight = 0.7
            euclidean_weight = 0.3
            
            for entity_id, result in all_results.items():
                cosine_rank = result.get('rank_cosine', limit + 1)
                euclidean_rank = result.get('rank_euclidean', limit + 1)
                
                # 计算混合分数
                hybrid_score = (cosine_weight * (1.0 / cosine_rank) + 
                              euclidean_weight * (1.0 / euclidean_rank))
                
                result['hybrid_score'] = hybrid_score
                hybrid_results.append(result)
            
            # 按混合分数排序
            hybrid_results = sorted(hybrid_results, key=lambda x: x['hybrid_score'], reverse=True)[:limit]
            
            return {
                'cosine_results': cosine_results,
                'euclidean_results': euclidean_results,
                'hybrid_results': hybrid_results,
                'total_unique': len(all_results)
            }
            
        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            return {
                'cosine_results': [],
                'euclidean_results': [],
                'hybrid_results': [],
                'total_unique': 0
            }
    
    def search_relations(self, query_vector: List[float], limit: int = 10) -> List[Dict]:
        """搜索相似关系"""
        try:
            # 使用Neo4j向量索引进行关系搜索
            results = self.graph.run("""
                CALL db.index.vector.queryRelationships('relation_vector_index', $limit, $query_vector)
                YIELD relationship, score
                MATCH (source)-[relationship]->(target)
                RETURN id(relationship) as id,
                       source.name as source,
                       target.name as target,
                       relationship.type as relation_type,
                       relationship.description as description,
                       relationship.source_text as source_text,
                       score as certainty,
                       (1.0 - score) as distance
                ORDER BY score DESC
            """, {
                'query_vector': query_vector,
                'limit': limit
            }).data()
            
            logger.info(f"关系搜索完成: {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索关系失败: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            # 获取实体数量
            entity_result = self.graph.run("MATCH (n:Entity) RETURN count(n) as count").data()
            entity_count = entity_result[0]['count'] if entity_result else 0
            
            # 获取关系数量
            relation_result = self.graph.run("MATCH ()-[r:RELATION]-() RETURN count(r) as count").data()
            relation_count = relation_result[0]['count'] if relation_result else 0
            
            # 获取有向量的实体数量
            vector_entity_result = self.graph.run("""
                MATCH (n:Entity) 
                WHERE n.vector IS NOT NULL 
                RETURN count(n) as count
            """).data()
            vector_entity_count = vector_entity_result[0]['count'] if vector_entity_result else 0
            
            # 获取有向量的关系数量
            vector_relation_result = self.graph.run("""
                MATCH ()-[r:RELATION]-() 
                WHERE r.vector IS NOT NULL 
                RETURN count(r) as count
            """).data()
            vector_relation_count = vector_relation_result[0]['count'] if vector_relation_result else 0
            
            return {
                'entities': entity_count,
                'relations': relation_count,
                'vector_entities': vector_entity_count,
                'vector_relations': vector_relation_count,
                'total': entity_count + relation_count,
                'entity_count': entity_count,  # 兼容性
                'relation_count': relation_count  # 兼容性
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {
                'entities': 0,
                'relations': 0,
                'vector_entities': 0,
                'vector_relations': 0,
                'total': 0,
                'entity_count': 0,
                'relation_count': 0
            }

class Neo4jIntentRecognizer:
    """基于Neo4j的意图识别器"""
    
    def __init__(self, vector_store: Neo4jVectorStore = None, 
                 ollama_client: OllamaClient = None):
        self.vector_store = vector_store or Neo4jVectorStore()
        self.ollama_client = ollama_client or OllamaClient()
        self.intent_recognizer = IntentRecognizer(self.ollama_client)
    
    def recognize_intent(self, query: str, context: Dict[str, Any] = None) -> IntentResult:
        """识别查询意图"""
        try:
            # 基础意图分类
            intent_type = self._classify_intent(query)
            
            # 提取相关实体和关系
            entities, relations = self.intent_recognizer.extract_entities_and_relations(query)
            
            # 计算置信度
            confidence = self._calculate_confidence(query, entities, relations)
            
            # 构建上下文
            intent_context = {
                'query_length': len(query),
                'entity_count': len(entities),
                'relation_count': len(relations),
                'extracted_entities': [e.name for e in entities],
                'extracted_relations': [f"{r.source}-{r.relation_type}-{r.target}" for r in relations]
            }
            
            if context:
                intent_context.update(context)
            
            return IntentResult(
                intent_type=intent_type,
                confidence=confidence,
                entities=[e.name for e in entities],
                relations=[f"{r.source}-{r.relation_type}-{r.target}" for r in relations],
                context=intent_context
            )
            
        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            return IntentResult(
                intent_type="unknown",
                confidence=0.0,
                entities=[],
                relations=[],
                context={'error': str(e)}
            )
    
    def _classify_intent(self, query: str) -> str:
        """分类查询意图"""
        query_lower = query.lower()
        
        # 定义意图关键词
        intent_keywords = {
            'treatment': ['治疗', '疗法', '怎么治', '如何治', '治愈', '医治'],
            'symptom': ['症状', '表现', '征象', '现象', '特征'],
            'cause': ['原因', '病因', '为什么', '怎么引起', '导致'],
            'prevention': ['预防', '避免', '防止', '预防措施'],
            'diagnosis': ['诊断', '检查', '确诊', '判断'],
            'prognosis': ['预后', '恢复', '康复', '好转'],
            'drug': ['药物', '药品', '用药', '吃药', '服药'],
            'general': ['是什么', '什么是', '介绍', '了解']
        }
        
        # 匹配意图
        for intent, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        
        return 'general'
    
    def _calculate_confidence(self, query: str, entities: List[Entity], 
                            relations: List[Relation]) -> float:
        """计算意图识别置信度"""
        try:
            base_confidence = 0.5
            
            # 根据提取的实体数量调整置信度
            entity_bonus = min(len(entities) * 0.1, 0.3)
            
            # 根据提取的关系数量调整置信度
            relation_bonus = min(len(relations) * 0.15, 0.2)
            
            # 根据查询长度调整置信度
            length_factor = min(len(query) / 50.0, 1.0) * 0.1
            
            confidence = base_confidence + entity_bonus + relation_bonus + length_factor
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"计算置信度失败: {e}")
            return 0.5
    
    def get_related_knowledge(self, intent_result: IntentResult, 
                            limit: int = 10) -> Dict[str, Any]:
        """根据意图获取相关知识"""
        try:
            # 构建查询向量（这里简化处理，实际应该使用embedding服务）
            from src.services.embedding_service import OllamaEmbeddingService
            embedding_service = OllamaEmbeddingService()
            
            # 构建查询文本
            query_text = " ".join(intent_result.entities + intent_result.relations)
            if not query_text.strip():
                query_text = intent_result.context.get('original_query', '')
            
            query_vector = embedding_service.embed_text(query_text)
            if not query_vector:
                return {'entities': [], 'relations': [], 'intent': intent_result}
            
            # 搜索相关实体和关系
            related_entities = self.vector_store.search_entities(query_vector, limit)
            related_relations = self.vector_store.search_relations(query_vector, limit)
            
            return {
                'entities': related_entities,
                'relations': related_relations,
                'intent': intent_result,
                'statistics': {
                    'entity_count': len(related_entities),
                    'relation_count': len(related_relations),
                    'intent_confidence': intent_result.confidence
                }
            }
            
        except Exception as e:
            logger.error(f"获取相关知识失败: {e}")
            return {'entities': [], 'relations': [], 'intent': intent_result}