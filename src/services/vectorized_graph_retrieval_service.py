#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量化图谱检索服务
完全基于向量相似度的图谱检索，替代传统的关键词匹配
"""

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass
from typing import List, Dict, Any

from ..knowledge.vectorized_graph_storage import VectorizedGraphStorage, GraphNode, GraphRelation
from ..knowledge.vectorized_graph_retrieval import VectorizedGraphRetrieval, KnowledgeChain, KnowledgeChainConfig
from ..services.embedding_service import OllamaEmbeddingService

logger = logging.getLogger(__name__)

@dataclass
class VectorSearchResult:
    """向量搜索结果"""
    nodes: List[GraphNode]
    relations: List[GraphRelation]
    knowledge_chains: List[KnowledgeChain]
    query_vector: List[float]
    search_time: float
    total_results: int
    
    def __len__(self) -> int:
        """返回总结果数量"""
        return self.total_results
    
    def __iter__(self):
        """支持迭代，返回所有结果项"""
        # 将节点转换为字典格式
        for node in self.nodes:
            yield {
                'type': 'node',
                'name': node.name,
                'description': node.description,
                'similarity': getattr(node, 'similarity', 0.0),
                'score': getattr(node, 'similarity', 0.0)
            }
        
        # 将关系转换为字典格式
        for relation in self.relations:
            yield {
                'type': 'relation',
                'description': relation.description,
                'source': relation.source_node,
                'target': relation.target_node,
                'similarity': getattr(relation, 'similarity', 0.0),
                'score': getattr(relation, 'similarity', 0.0)
            }
        
        # 将知识链转换为字典格式
        for chain in self.knowledge_chains:
            yield {
                'type': 'knowledge_chain',
                'description': chain.chain_description,
                'score': chain.chain_score,
                'similarity': chain.chain_score
            }

@dataclass
class GraphSearchConfig:
    """图谱搜索配置"""
    max_nodes: int = 20
    max_relations: int = 30
    max_chains: int = 10
    min_similarity: float = 0.7
    enable_chain_building: bool = True
    enable_async_search: bool = True
    search_timeout: float = 30.0

class VectorizedGraphRetrievalService:
    """向量化图谱检索服务 - 完全基于向量相似度的图谱检索"""
    
    def __init__(self, 
                 storage: VectorizedGraphStorage = None,
                 config: GraphSearchConfig = None,
                 chain_config: KnowledgeChainConfig = None):
        """
        初始化向量化图谱检索服务
        
        Args:
            storage: 向量化图谱存储
            config: 搜索配置
            chain_config: 知识链路配置
        """
        self.storage = storage or VectorizedGraphStorage()
        self.config = config or GraphSearchConfig()
        self.embedding_service = OllamaEmbeddingService()
        
        # 初始化向量化图谱检索器
        self.graph_retrieval = VectorizedGraphRetrieval(
            storage=self.storage,
            config=chain_config
        )
        
        # 线程池用于异步搜索
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("向量化图谱检索服务初始化完成")
    
    def search_by_query(self, query: str, limit: int = 10) -> VectorSearchResult:
        """基于查询进行向量化图谱搜索"""
        start_time = time.time()
        
        try:
            logger.info(f"开始向量化图谱搜索: {query}")
            
            # 1. 将查询向量化
            query_vector = self.embedding_service.get_embedding(query)
            if not query_vector:
                logger.error("查询向量化失败")
                return self._empty_result(start_time)
            
            # 2. 搜索相似节点
            similar_nodes = self.storage.search_similar_nodes(
                query_vector=query_vector,
                limit=min(limit * 2, self.config.max_nodes),
                min_similarity=self.config.min_similarity
            )
            
            logger.info(f"找到 {len(similar_nodes)} 个相似节点")
            
            # 3. 搜索相似关系
            similar_relations = self.storage.search_similar_relations(
                query_vector=query_vector,
                limit=min(limit * 3, self.config.max_relations),
                min_similarity=self.config.min_similarity
            )
            
            logger.info(f"找到 {len(similar_relations)} 个相似关系")
            
            # 4. 构建知识链路
            knowledge_chains = []
            if self.config.enable_chain_building and similar_nodes:
                try:
                    knowledge_chains = self.graph_retrieval.build_knowledge_chains(
                        query=query,
                        max_chains=min(limit, self.config.max_chains)
                    )
                    logger.info(f"构建了 {len(knowledge_chains)} 条知识链路")
                except Exception as e:
                    logger.warning(f"知识链路构建失败: {e}")
            
            # 5. 限制结果数量
            final_nodes = similar_nodes[:limit]
            final_relations = similar_relations[:limit]
            final_chains = knowledge_chains[:self.config.max_chains]
            
            search_time = time.time() - start_time
            total_results = len(final_nodes) + len(final_relations) + len(final_chains)
            
            logger.info(f"向量化图谱搜索完成: {total_results} 个结果，耗时 {search_time:.2f}s")
            
            return VectorSearchResult(
                nodes=final_nodes,
                relations=final_relations,
                knowledge_chains=final_chains,
                query_vector=query_vector,
                search_time=search_time,
                total_results=total_results
            )
            
        except Exception as e:
            logger.error(f"向量化图谱搜索失败: {e}")
            return self._empty_result(start_time)
    
    async def search_by_query_async(self, query: str, limit: int = 10) -> VectorSearchResult:
        """异步向量化图谱搜索"""
        if not self.config.enable_async_search:
            return self.search_by_query(query, limit)
        
        try:
            # 使用asyncio.wait_for设置超时
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.executor, 
                    self.search_by_query, 
                    query, 
                    limit
                ),
                timeout=self.config.search_timeout
            )
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"异步搜索超时: {query}")
            return self._empty_result(time.time())
        except Exception as e:
            logger.error(f"异步搜索失败: {e}")
            return self._empty_result(time.time())
    
    def search_entities_by_name(self, name_pattern: str, limit: int = 10) -> List[Dict[str, Any]]:
        """基于向量相似度搜索实体（兼容原有接口）"""
        try:
            # 将名称模式向量化
            query_vector = self.embedding_service.get_embedding(name_pattern)
            if not query_vector:
                return []
            
            # 搜索相似节点
            similar_nodes = self.storage.search_similar_nodes(
                query_vector=query_vector,
                limit=limit,
                min_similarity=self.config.min_similarity
            )
            
            # 转换为兼容格式
            entities = []
            for node in similar_nodes:
                entity = {
                    'name': node.get('name'),
                    'type': node.get('type'),
                    'description': node.get('description'),
                    'source_text': node.get('source_text'),
                    'similarity': node.get('similarity', 0.0),
                    'neo4j_id': node.get('neo4j_id')
                }
                entities.append(entity)
            
            logger.info(f"向量化实体搜索完成: {len(entities)} 个结果")
            return entities
            
        except Exception as e:
            logger.error(f"向量化实体搜索失败: {e}")
            return []
    
    def search_relationships_by_query(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """基于向量相似度搜索关系（兼容原有接口）"""
        try:
            # 将查询向量化
            query_vector = self.embedding_service.get_embedding(query)
            if not query_vector:
                return []
            
            # 搜索相似关系
            similar_relations = self.storage.search_similar_relations(
                query_vector=query_vector,
                limit=limit,
                min_similarity=self.config.min_similarity
            )
            
            # 转换为兼容格式
            relationships = []
            for relation in similar_relations:
                relationship = {
                    'source': relation.get('source_node'),
                    'target': relation.get('target_node'),
                    'type': relation.get('relation_type'),
                    'description': relation.get('description'),
                    'source_text': relation.get('source_text'),
                    'similarity': relation.get('similarity', 0.0),
                    'neo4j_id': relation.get('neo4j_id'),
                    'relevance_score': relation.get('similarity', 0.0)  # 兼容原有字段
                }
                relationships.append(relationship)
            
            logger.info(f"向量化关系搜索完成: {len(relationships)} 个结果")
            return relationships
            
        except Exception as e:
            logger.error(f"向量化关系搜索失败: {e}")
            return []
    
    def search_relationships_by_type(self, relation_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """基于关系类型的向量化搜索（兼容原有接口）"""
        try:
            # 构建类型查询
            type_query = f"关系类型: {relation_type}"
            return self.search_relationships_by_query(type_query, limit)
            
        except Exception as e:
            logger.error(f"按类型搜索关系失败: {e}")
            return []
    
    def search_entities_by_type(self, entity_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """基于实体类型的向量化搜索（兼容原有接口）"""
        try:
            # 构建类型查询
            type_query = f"实体类型: {entity_type}"
            return self.search_entities_by_name(type_query, limit)
            
        except Exception as e:
            logger.error(f"按类型搜索实体失败: {e}")
            return []
    
    def get_entity_relationships(self, entity_name: str, limit: int = 10) -> Dict[str, Any]:
        """获取实体的向量化关系网络（兼容原有接口）"""
        try:
            # 基于实体名称搜索相关关系
            entity_query = f"实体: {entity_name}"
            relationships = self.search_relationships_by_query(entity_query, limit)
            
            return {
                'entity': entity_name,
                'relationships': relationships,
                'count': len(relationships)
            }
            
        except Exception as e:
            logger.error(f"获取实体关系失败: {e}")
            return {'entity': entity_name, 'relationships': [], 'count': 0}
    
    def search_with_intent(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """基于向量相似度的意图感知搜索（兼容原有接口）"""
        try:
            logger.info(f"开始向量化意图感知搜索: {query}")
            
            # 执行向量化搜索
            search_result = self.search_by_query(query, limit)
            
            # 转换节点为实体格式
            entities = []
            for node in search_result.nodes:
                entity = {
                    'name': node.get('name'),
                    'type': node.get('type'),
                    'description': node.get('description'),
                    'source_text': node.get('source_text'),
                    'similarity': node.get('similarity', 0.0)
                }
                entities.append(entity)
            
            # 转换关系格式
            relationships = []
            for relation in search_result.relations:
                relationship = {
                    'source': relation.get('source_node'),
                    'target': relation.get('target_node'),
                    'type': relation.get('relation_type'),
                    'description': relation.get('description'),
                    'source_text': relation.get('source_text'),
                    'similarity': relation.get('similarity', 0.0)
                }
                relationships.append(relationship)
            
            # 构建意图结果（模拟）
            intent_result = {
                'intent_type': 'vector_search',
                'confidence': 1.0,
                'entities': [node.get('name') for node in search_result.nodes[:5] if node.get('name')],
                'relations': [rel.get('relation_type') for rel in search_result.relations[:5] if rel.get('relation_type')],
                'metadata': {
                    'search_method': 'vectorized',
                    'total_results': search_result.total_results,
                    'search_time': search_result.search_time,
                    'knowledge_chains': len(search_result.knowledge_chains)
                }
            }
            
            return {
                'intent': intent_result,
                'entities': entities,
                'relationships': relationships,
                'knowledge_chains': search_result.knowledge_chains,
                'query': query,
                'search_time': search_result.search_time
            }
            
        except Exception as e:
            logger.error(f"向量化意图感知搜索失败: {e}")
            return {
                'intent': None,
                'entities': [],
                'relationships': [],
                'knowledge_chains': [],
                'query': query,
                'error': str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取向量化图谱统计信息"""
        try:
            storage_stats = self.storage.get_stats()
            retrieval_stats = self.graph_retrieval.get_retrieval_stats()
            
            return {
                'storage_stats': storage_stats,
                'retrieval_stats': retrieval_stats,
                'config': {
                    'max_nodes': self.config.max_nodes,
                    'max_relations': self.config.max_relations,
                    'max_chains': self.config.max_chains,
                    'min_similarity': self.config.min_similarity,
                    'enable_chain_building': self.config.enable_chain_building,
                    'enable_async_search': self.config.enable_async_search
                },
                'service_type': 'vectorized_graph_retrieval',
                'embedding_service': type(self.embedding_service).__name__
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {
                'storage_stats': {},
                'retrieval_stats': {},
                'config': {},
                'service_type': 'vectorized_graph_retrieval',
                'error': str(e)
            }
    
    def get_knowledge_graph_summary(self) -> Dict[str, Any]:
        """获取向量化知识图谱摘要"""
        try:
            stats = self.storage.get_stats()
            
            # 基于向量存储统计构建摘要
            return {
                'entity_types': [],  # Weaviate中可以通过聚合查询获取
                'relation_types': [],  # Weaviate中可以通过聚合查询获取
                'top_connected_entities': [],  # 需要额外实现
                'summary': {
                    'total_nodes': stats.get('total_nodes', 0),
                    'total_relations': stats.get('total_relations', 0),
                    'storage_type': 'vectorized',
                    'search_method': 'vector_similarity'
                },
                'vector_stats': stats
            }
            
        except Exception as e:
            logger.error(f"获取知识图谱摘要失败: {e}")
            return {
                'entity_types': [],
                'relation_types': [],
                'top_connected_entities': [],
                'summary': {},
                'error': str(e)
            }
    
    def _empty_result(self, start_time: float) -> VectorSearchResult:
        """创建空的搜索结果"""
        return VectorSearchResult(
            nodes=[],
            relations=[],
            knowledge_chains=[],
            query_vector=[],
            search_time=time.time() - start_time,
            total_results=0
        )
    
    def close(self):
        """关闭服务"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
            logger.info("向量化图谱检索服务已关闭")
        except Exception as e:
            logger.error(f"关闭服务失败: {e}")

# 兼容性别名
GraphRetrievalService = VectorizedGraphRetrievalService

def create_vectorized_graph_service(config: GraphSearchConfig = None) -> VectorizedGraphRetrievalService:
    """创建向量化图谱检索服务的工厂函数"""
    return VectorizedGraphRetrievalService(config=config)

def main():
    """主函数 - 用于测试"""
    import sys
    
    try:
        # 创建向量化图谱检索服务
        service = VectorizedGraphRetrievalService()
        
        # 获取统计信息
        stats = service.get_stats()
        logger.info(f"服务统计: {stats}")
        
        # 测试查询
        if len(sys.argv) > 1:
            query = sys.argv[1]
            logger.info(f"测试查询: {query}")
            
            # 执行搜索
            result = service.search_by_query(query, limit=5)
            
            logger.info(f"搜索结果:")
            logger.info(f"  节点数: {len(result.nodes)}")
            logger.info(f"  关系数: {len(result.relations)}")
            logger.info(f"  知识链数: {len(result.knowledge_chains)}")
            logger.info(f"  搜索时间: {result.search_time:.2f}s")
            
            # 打印前几个结果
            for i, node in enumerate(result.nodes[:3]):
                logger.info(f"  节点{i+1}: {node.name} ({node.type})")
            
            for i, relation in enumerate(result.relations[:3]):
                logger.info(f"  关系{i+1}: {relation.source_node} -> {relation.target_node} ({relation.relation_type})")
        
        logger.info("向量化图谱检索服务测试完成")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
    finally:
        if 'service' in locals():
            service.close()

if __name__ == "__main__":
    main()