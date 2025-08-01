#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索服务模块
提供余弦相似度和欧氏距离的混合检索功能
"""

import sys
import os
import logging
from typing import List, Dict, Any, Optional, Tuple

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import get_config
from src.services.embedding_service import OllamaEmbeddingService
from src.knowledge.vector_storage import WeaviateVectorStore
from src.services.common_types import (
    DistanceMetric, SearchResult, HybridSearchResult, SimilarityCalculator
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetrievalService:
    """检索服务"""
    
    def __init__(self, embedding_service: OllamaEmbeddingService = None, 
                 vector_store: WeaviateVectorStore = None):
        """初始化检索服务
        
        Args:
            embedding_service: 向量化服务实例
            vector_store: 向量存储实例
        """
        self.embedding_service = embedding_service or OllamaEmbeddingService()
        self.vector_store = vector_store or WeaviateVectorStore()
        self.similarity_calculator = SimilarityCalculator()
        self.config = get_config()
        
        # 混合检索权重配置
        self.cosine_weight = 0.7  # 余弦相似度权重
        self.euclidean_weight = 0.3  # 欧氏距离权重
    
    def get_stats(self):
        """获取系统统计信息"""
        try:
            # 获取向量存储统计信息
            vector_stats = self.vector_store.get_stats() if hasattr(self.vector_store, 'get_stats') else {}
            
            return {
                "neo4j_nodes": 0,  # 新系统暂不使用Neo4j
                "neo4j_relationships": 0,
                "weaviate_entities": vector_stats.get("entity_count", 0),
                "weaviate_relations": vector_stats.get("relation_count", 0),
                "status": "新检索服务",
                "cosine_weight": self.cosine_weight,
                "euclidean_weight": self.euclidean_weight
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {
                "neo4j_nodes": 0,
                "neo4j_relationships": 0,
                "weaviate_entities": 0,
                "weaviate_relations": 0,
                "status": f"错误: {e}",
                "cosine_weight": self.cosine_weight,
                "euclidean_weight": self.euclidean_weight
            }
    
    def search_by_cosine(self, query: str, limit: int = 10) -> List[SearchResult]:
        """使用余弦相似度搜索
        
        Args:
            query: 查询文本
            limit: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        try:
            # 向量化查询
            query_vector = self.embedding_service.embed_text(query)
            if not query_vector:
                logger.error("查询向量化失败")
                return []
            
            # 使用Weaviate进行余弦相似度搜索
            raw_results = self.vector_store.search_entities(
                query_vector, limit, "cosine"
            )
            
            # 转换为SearchResult格式
            results = []
            for result in raw_results:
                search_result = SearchResult(
                    id=result.get("id", ""),
                    content=result.get("source_text", ""),
                    score=result.get("certainty", 0.0),
                    distance=result.get("distance", 0.0),
                    metadata={
                        "name": result.get("name", ""),
                        "type": result.get("type", ""),
                        "description": result.get("description", ""),
                        "neo4j_id": result.get("neo4j_id", "")
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
        """使用欧氏距离搜索
        
        Args:
            query: 查询文本
            limit: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        try:
            # 向量化查询
            query_vector = self.embedding_service.embed_text(query)
            if not query_vector:
                logger.error("查询向量化失败")
                return []
            
            # 使用Weaviate进行欧氏距离搜索
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
                    id=result.get("id", ""),
                    content=result.get("source_text", ""),
                    score=score,
                    distance=distance,
                    metadata={
                        "name": result.get("name", ""),
                        "type": result.get("type", ""),
                        "description": result.get("description", ""),
                        "neo4j_id": result.get("neo4j_id", "")
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
        """混合检索（余弦相似度 + 欧氏距离）
        
        Args:
            query: 查询文本
            limit: 返回结果数量
            
        Returns:
            混合搜索结果
        """
        try:
            logger.info(f"开始混合检索: {query}")
            
            # 向量化查询
            query_vector = self.embedding_service.embed_text(query)
            if not query_vector:
                logger.error("查询向量化失败")
                return HybridSearchResult([], [], [], {})
            
            # 使用Weaviate的混合检索功能
            raw_results = self.vector_store.search_entities_hybrid(query_vector, limit)
            
            # 转换余弦相似度结果
            cosine_results = []
            for result in raw_results.get("cosine_results", []):
                search_result = SearchResult(
                    id=result.get("id", ""),
                    content=result.get("source_text", ""),
                    score=result.get("certainty", 0.0),
                    distance=result.get("distance", 0.0),
                    metadata={
                        "name": result.get("name", ""),
                        "type": result.get("type", ""),
                        "description": result.get("description", ""),
                        "neo4j_id": result.get("neo4j_id", "")
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
                    id=result.get("id", ""),
                    content=result.get("source_text", ""),
                    score=score,
                    distance=distance,
                    metadata={
                        "name": result.get("name", ""),
                        "type": result.get("type", ""),
                        "description": result.get("description", ""),
                        "neo4j_id": result.get("neo4j_id", "")
                    },
                    distance_metric="euclidean"
                )
                euclidean_results.append(search_result)
            
            # 转换混合结果
            hybrid_results = []
            for result in raw_results.get("hybrid_results", []):
                # 计算综合分数
                cosine_rank = result.get("rank_cosine", limit + 1)
                euclidean_rank = result.get("rank_euclidean", limit + 1)
                hybrid_score = (self.cosine_weight * (1.0 / cosine_rank) + 
                              self.euclidean_weight * (1.0 / euclidean_rank))
                
                search_result = SearchResult(
                    id=result.get("id", ""),
                    content=result.get("source_text", ""),
                    score=hybrid_score,
                    distance=result.get("distance", 0.0),
                    metadata={
                        "name": result.get("name", ""),
                        "type": result.get("type", ""),
                        "description": result.get("description", ""),
                        "neo4j_id": result.get("neo4j_id", ""),
                        "cosine_rank": cosine_rank,
                        "euclidean_rank": euclidean_rank,
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
    
    def search_with_custom_weights(self, query: str, cosine_weight: float = 0.7, 
                                  euclidean_weight: float = 0.3, 
                                  limit: int = 10) -> HybridSearchResult:
        """使用自定义权重进行混合检索
        
        Args:
            query: 查询文本
            cosine_weight: 余弦相似度权重
            euclidean_weight: 欧氏距离权重
            limit: 返回结果数量
            
        Returns:
            混合搜索结果
        """
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
        """比较不同距离度量的检索结果
        
        Args:
            query: 查询文本
            limit: 返回结果数量
            
        Returns:
            比较结果
        """
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