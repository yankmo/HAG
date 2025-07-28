#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j检索服务模块
提供基于Neo4j的检索功能，兼容现有接口
"""

import numpy as np
import sys
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import get_config
from src.services.embedding_service import OllamaEmbeddingService
from src.knowledge.neo4j_vector_storage import Neo4jVectorStore, Neo4jIntentRecognizer, IntentResult

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DistanceMetric(Enum):
    """距离度量类型"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    DOT_PRODUCT = "dot_product"

@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    content: str
    score: float
    distance: float
    metadata: Dict[str, Any] = None
    distance_metric: str = ""
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class HybridSearchResult:
    """混合搜索结果"""
    cosine_results: List[SearchResult]
    euclidean_results: List[SearchResult]
    hybrid_results: List[SearchResult]
    statistics: Dict[str, Any]

@dataclass
class IntentAwareSearchResult:
    """意图感知搜索结果"""
    intent: IntentResult
    search_results: HybridSearchResult
    related_knowledge: Dict[str, Any]
    recommendations: List[str]

class SimilarityCalculator:
    """相似度计算器"""
    
    @staticmethod
    def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
        """计算余弦相似度"""
        try:
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            
            similarity = dot_product / (norm_v1 * norm_v2)
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"计算余弦相似度失败: {e}")
            return 0.0
    
    @staticmethod
    def euclidean_distance(vector1: List[float], vector2: List[float]) -> float:
        """计算欧氏距离"""
        try:
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            distance = np.linalg.norm(v1 - v2)
            return float(distance)
            
        except Exception as e:
            logger.error(f"计算欧氏距离失败: {e}")
            return float('inf')

class Neo4jRetrievalService:
    """基于Neo4j的检索服务"""
    
    def __init__(self, embedding_service: OllamaEmbeddingService = None, 
                 vector_store: Neo4jVectorStore = None,
                 intent_recognizer: Neo4jIntentRecognizer = None):
        """初始化检索服务"""
        self.embedding_service = embedding_service or OllamaEmbeddingService()
        self.vector_store = vector_store or Neo4jVectorStore()
        self.intent_recognizer = intent_recognizer or Neo4jIntentRecognizer(self.vector_store)
        self.similarity_calculator = SimilarityCalculator()
        self.config = get_config()
        
        # 混合检索权重配置
        self.cosine_weight = 0.7
        self.euclidean_weight = 0.3
        
        logger.info("Neo4j检索服务初始化完成")
    
    def get_stats(self):
        """获取系统统计信息"""
        try:
            # 获取Neo4j统计信息
            neo4j_stats = self.vector_store.get_stats()
            
            return {
                "neo4j_nodes": neo4j_stats.get("entities", 0),
                "neo4j_relationships": neo4j_stats.get("relations", 0),
                "neo4j_vector_entities": neo4j_stats.get("vector_entities", 0),
                "neo4j_vector_relations": neo4j_stats.get("vector_relations", 0),
                "weaviate_entities": 0,  # 新系统不使用Weaviate
                "weaviate_relations": 0,
                "status": "Neo4j检索服务",
                "cosine_weight": self.cosine_weight,
                "euclidean_weight": self.euclidean_weight,
                "total_entities": neo4j_stats.get("entities", 0),
                "total_relations": neo4j_stats.get("relations", 0)
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {
                "neo4j_nodes": 0,
                "neo4j_relationships": 0,
                "neo4j_vector_entities": 0,
                "neo4j_vector_relations": 0,
                "weaviate_entities": 0,
                "weaviate_relations": 0,
                "status": f"错误: {e}",
                "cosine_weight": self.cosine_weight,
                "euclidean_weight": self.euclidean_weight,
                "total_entities": 0,
                "total_relations": 0
            }
    
    def search_by_cosine(self, query: str, limit: int = 10) -> List[SearchResult]:
        """使用余弦相似度搜索"""
        try:
            # 向量化查询
            query_vector = self.embedding_service.embed_text(query)
            if not query_vector:
                logger.error("查询向量化失败")
                return []
            
            # 使用Neo4j进行余弦相似度搜索
            raw_results = self.vector_store.search_entities(
                query_vector, limit, "cosine"
            )
            
            # 转换为SearchResult格式
            results = []
            for result in raw_results:
                search_result = SearchResult(
                    id=str(result.get("id", "")),
                    content=result.get("source_text", ""),
                    score=result.get("certainty", 0.0),
                    distance=result.get("distance", 0.0),
                    metadata={
                        "name": result.get("name", ""),
                        "type": result.get("type", ""),
                        "description": result.get("description", ""),
                        "neo4j_id": str(result.get("id", ""))
                    },
                    distance_metric="cosine"
                )
                results.append(search_result)
            
            logger.info(f"余弦相似度搜索完成: {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"余弦相似度搜索失败: {e}")
            return []
    
    def search_by_euclidean(self, query: str, limit: int = 10) -> List[SearchResult]:
        """使用欧氏距离搜索"""
        try:
            # 向量化查询
            query_vector = self.embedding_service.embed_text(query)
            if not query_vector:
                logger.error("查询向量化失败")
                return []
            
            # 使用Neo4j进行欧氏距离搜索
            raw_results = self.vector_store.search_entities(
                query_vector, limit, "euclidean"
            )
            
            # 转换为SearchResult格式
            results = []
            for result in raw_results:
                # 欧氏距离越小越相似，转换为相似度分数
                distance = result.get("distance", float('inf'))
                score = 1.0 / (1.0 + distance) if distance != float('inf') else 0.0
                
                search_result = SearchResult(
                    id=str(result.get("id", "")),
                    content=result.get("source_text", ""),
                    score=score,
                    distance=distance,
                    metadata={
                        "name": result.get("name", ""),
                        "type": result.get("type", ""),
                        "description": result.get("description", ""),
                        "neo4j_id": str(result.get("id", ""))
                    },
                    distance_metric="euclidean"
                )
                results.append(search_result)
            
            logger.info(f"欧氏距离搜索完成: {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"欧氏距离搜索失败: {e}")
            return []
    
    def search_hybrid(self, query: str, limit: int = 10) -> HybridSearchResult:
        """混合检索（余弦相似度 + 欧氏距离）"""
        try:
            logger.info(f"开始混合检索: {query}")
            
            # 向量化查询
            query_vector = self.embedding_service.embed_text(query)
            if not query_vector:
                logger.error("查询向量化失败")
                return HybridSearchResult([], [], [], {})
            
            # 使用Neo4j的混合检索功能
            raw_results = self.vector_store.search_entities_hybrid(query_vector, limit)
            
            # 转换余弦相似度结果
            cosine_results = []
            for result in raw_results.get("cosine_results", []):
                search_result = SearchResult(
                    id=str(result.get("id", "")),
                    content=result.get("source_text", ""),
                    score=result.get("certainty", 0.0),
                    distance=result.get("distance", 0.0),
                    metadata={
                        "name": result.get("name", ""),
                        "type": result.get("type", ""),
                        "description": result.get("description", ""),
                        "neo4j_id": str(result.get("id", ""))
                    },
                    distance_metric="cosine"
                )
                cosine_results.append(search_result)
            
            # 转换欧氏距离结果
            euclidean_results = []
            for result in raw_results.get("euclidean_results", []):
                distance = result.get("distance", float('inf'))
                score = 1.0 / (1.0 + distance) if distance != float('inf') else 0.0
                
                search_result = SearchResult(
                    id=str(result.get("id", "")),
                    content=result.get("source_text", ""),
                    score=score,
                    distance=distance,
                    metadata={
                        "name": result.get("name", ""),
                        "type": result.get("type", ""),
                        "description": result.get("description", ""),
                        "neo4j_id": str(result.get("id", ""))
                    },
                    distance_metric="euclidean"
                )
                euclidean_results.append(search_result)
            
            # 转换混合结果
            hybrid_results = []
            for result in raw_results.get("hybrid_results", []):
                # 使用Neo4j计算的混合分数
                hybrid_score = result.get("hybrid_score", 0.0)
                
                search_result = SearchResult(
                    id=str(result.get("id", "")),
                    content=result.get("source_text", ""),
                    score=hybrid_score,
                    distance=result.get("distance", 0.0),
                    metadata={
                        "name": result.get("name", ""),
                        "type": result.get("type", ""),
                        "description": result.get("description", ""),
                        "neo4j_id": str(result.get("id", "")),
                        "cosine_rank": result.get("rank_cosine", limit + 1),
                        "euclidean_rank": result.get("rank_euclidean", limit + 1),
                        "euclidean_distance": result.get("euclidean_distance", 0.0)
                    },
                    distance_metric="hybrid"
                )
                hybrid_results.append(search_result)
            
            # 统计信息
            statistics = {
                "query": query,
                "total_cosine": len(cosine_results),
                "total_euclidean": len(euclidean_results),
                "total_hybrid": len(hybrid_results),
                "total_unique": raw_results.get("total_unique", 0),
                "cosine_weight": self.cosine_weight,
                "euclidean_weight": self.euclidean_weight
            }
            
            logger.info(f"混合检索完成: {statistics}")
            
            return HybridSearchResult(
                cosine_results=cosine_results,
                euclidean_results=euclidean_results,
                hybrid_results=hybrid_results,
                statistics=statistics
            )
            
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            return HybridSearchResult([], [], [], {})
    
    def search_with_intent(self, query: str, limit: int = 10) -> IntentAwareSearchResult:
        """意图感知搜索"""
        try:
            logger.info(f"开始意图感知搜索: {query}")
            
            # 识别意图
            intent_result = self.intent_recognizer.recognize_intent(query)
            logger.info(f"识别意图: {intent_result.intent_type}, 置信度: {intent_result.confidence:.2f}")
            
            # 执行混合检索
            search_results = self.search_hybrid(query, limit)
            
            # 获取相关知识
            related_knowledge = self.intent_recognizer.get_related_knowledge(intent_result, limit)
            
            # 生成推荐
            recommendations = self._generate_recommendations(intent_result, search_results)
            
            return IntentAwareSearchResult(
                intent=intent_result,
                search_results=search_results,
                related_knowledge=related_knowledge,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"意图感知搜索失败: {e}")
            return IntentAwareSearchResult(
                intent=IntentResult("unknown", 0.0, [], [], {"error": str(e)}),
                search_results=HybridSearchResult([], [], [], {}),
                related_knowledge={},
                recommendations=[]
            )
    
    def _generate_recommendations(self, intent: IntentResult, 
                                search_results: HybridSearchResult) -> List[str]:
        """生成搜索推荐"""
        recommendations = []
        
        try:
            # 基于意图类型生成推荐
            if intent.intent_type == "treatment":
                recommendations.extend([
                    "查看相关治疗方法",
                    "了解药物治疗选项",
                    "探索非药物治疗方案"
                ])
            elif intent.intent_type == "symptom":
                recommendations.extend([
                    "查看相关症状描述",
                    "了解症状发展过程",
                    "探索症状管理方法"
                ])
            elif intent.intent_type == "cause":
                recommendations.extend([
                    "了解病因机制",
                    "查看风险因素",
                    "探索预防措施"
                ])
            elif intent.intent_type == "diagnosis":
                recommendations.extend([
                    "了解诊断标准",
                    "查看检查方法",
                    "探索鉴别诊断"
                ])
            
            # 基于搜索结果生成推荐
            if search_results.hybrid_results:
                top_result = search_results.hybrid_results[0]
                entity_type = top_result.metadata.get("type", "")
                
                if entity_type == "Disease":
                    recommendations.append(f"深入了解{top_result.metadata.get('name', '')}疾病")
                elif entity_type == "Treatment":
                    recommendations.append(f"详细了解{top_result.metadata.get('name', '')}治疗方法")
                elif entity_type == "Drug":
                    recommendations.append(f"查看{top_result.metadata.get('name', '')}药物信息")
            
            # 限制推荐数量
            return recommendations[:5]
            
        except Exception as e:
            logger.error(f"生成推荐失败: {e}")
            return ["探索相关医学知识"]
    
    def search_with_custom_weights(self, query: str, cosine_weight: float = 0.7, 
                                  euclidean_weight: float = 0.3, 
                                  limit: int = 10) -> HybridSearchResult:
        """使用自定义权重进行混合检索"""
        # 临时设置权重
        original_cosine_weight = self.cosine_weight
        original_euclidean_weight = self.euclidean_weight
        
        self.cosine_weight = cosine_weight
        self.euclidean_weight = euclidean_weight
        
        try:
            result = self.search_hybrid(query, limit)
            return result
        finally:
            # 恢复原始权重
            self.cosine_weight = original_cosine_weight
            self.euclidean_weight = original_euclidean_weight
    
    def compare_distance_metrics(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """比较不同距离度量的检索结果"""
        logger.info(f"开始比较距离度量: {query}")
        
        # 获取不同度量的结果
        cosine_results = self.search_by_cosine(query, limit)
        euclidean_results = self.search_by_euclidean(query, limit)
        hybrid_result = self.search_hybrid(query, limit)
        
        # 分析结果差异
        cosine_ids = set(r.id for r in cosine_results)
        euclidean_ids = set(r.id for r in euclidean_results)
        hybrid_ids = set(r.id for r in hybrid_result.hybrid_results)
        
        comparison = {
            "query": query,
            "cosine_results": cosine_results,
            "euclidean_results": euclidean_results,
            "hybrid_results": hybrid_result.hybrid_results,
            "analysis": {
                "cosine_only": list(cosine_ids - euclidean_ids),
                "euclidean_only": list(euclidean_ids - cosine_ids),
                "common_results": list(cosine_ids & euclidean_ids),
                "hybrid_unique": list(hybrid_ids - (cosine_ids | euclidean_ids)),
                "overlap_rate": len(cosine_ids & euclidean_ids) / max(len(cosine_ids | euclidean_ids), 1)
            },
            "statistics": hybrid_result.statistics
        }
        
        logger.info(f"距离度量比较完成: 重叠率 {comparison['analysis']['overlap_rate']:.2%}")
        
        return comparison
    
    def get_entity_relationships(self, entity_name: str, limit: int = 10) -> Dict[str, Any]:
        """获取实体的关系网络"""
        try:
            # 查询实体的所有关系
            relationships = self.vector_store.graph.run("""
                MATCH (e:Entity {name: $entity_name})-[r:RELATION]-(related:Entity)
                RETURN e.name as entity,
                       type(r) as relation_type,
                       r.description as relation_description,
                       related.name as related_entity,
                       related.type as related_type,
                       related.description as related_description
                LIMIT $limit
            """, {
                'entity_name': entity_name,
                'limit': limit
            }).data()
            
            return {
                'entity': entity_name,
                'relationships': relationships,
                'count': len(relationships)
            }
            
        except Exception as e:
            logger.error(f"获取实体关系失败: {e}")
            return {'entity': entity_name, 'relationships': [], 'count': 0}
    
    def get_knowledge_graph_summary(self) -> Dict[str, Any]:
        """获取知识图谱摘要"""
        try:
            # 获取实体类型分布
            entity_types = self.vector_store.graph.run("""
                MATCH (n:Entity)
                RETURN n.type as type, count(n) as count
                ORDER BY count DESC
            """).data()
            
            # 获取关系类型分布
            relation_types = self.vector_store.graph.run("""
                MATCH ()-[r:RELATION]-()
                RETURN r.type as type, count(r) as count
                ORDER BY count DESC
            """).data()
            
            # 获取连接度最高的实体
            top_connected = self.vector_store.graph.run("""
                MATCH (n:Entity)-[r:RELATION]-()
                RETURN n.name as entity, n.type as type, count(r) as connections
                ORDER BY connections DESC
                LIMIT 10
            """).data()
            
            return {
                'entity_types': entity_types,
                'relation_types': relation_types,
                'top_connected_entities': top_connected,
                'summary': {
                    'total_entity_types': len(entity_types),
                    'total_relation_types': len(relation_types),
                    'most_common_entity_type': entity_types[0]['type'] if entity_types else None,
                    'most_common_relation_type': relation_types[0]['type'] if relation_types else None
                }
            }
            
        except Exception as e:
            logger.error(f"获取知识图谱摘要失败: {e}")
            return {
                'entity_types': [],
                'relation_types': [],
                'top_connected_entities': [],
                'summary': {}
            }