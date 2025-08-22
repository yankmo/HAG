#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量化图谱检索服务
基于向量相似度实现知识图谱检索和知识链路构建
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from .vectorized_graph_storage import VectorizedGraphStorage, GraphNode, GraphRelation, KnowledgeChain
# 延迟导入embedding_service避免循环依赖
# from ..config.settings import get_config  # 移除config依赖

logger = logging.getLogger(__name__)

@dataclass
class VectorSearchResult:
    """向量搜索结果"""
    nodes: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    query_vector: List[float]
    search_time: float = 0.0

@dataclass
class KnowledgeChainConfig:
    """知识链路配置"""
    max_chain_length: int = 5  # 最大链路长度
    min_similarity_threshold: float = 0.7  # 最小相似度阈值
    max_nodes_per_query: int = 20  # 每次查询最大节点数
    max_relations_per_query: int = 30  # 每次查询最大关系数
    chain_weight_decay: float = 0.8  # 链路权重衰减因子
    relation_boost_factor: float = 1.2  # 关系权重提升因子

class VectorizedGraphRetrieval:
    """向量化图谱检索服务"""
    
    def __init__(self, storage: VectorizedGraphStorage = None, config: KnowledgeChainConfig = None):
        self.storage = storage or VectorizedGraphStorage()
        self.embedding_service = None  # 延迟初始化
        self.config = config or KnowledgeChainConfig()
        
        logger.info("向量化图谱检索服务初始化完成")
    
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
    
    def search_knowledge_chains(self, query: str, max_chains: int = 5) -> List[KnowledgeChain]:
        """搜索知识链路"""
        try:
            # 1. 将查询向量化
            query_vector = self._get_embedding_service().embed_text(query)
            if not query_vector:
                logger.error("查询向量化失败")
                return []
            
            # 2. 搜索相似节点和关系
            search_result = self._vector_search(query_vector)
            
            # 3. 构建知识链路
            knowledge_chains = self._build_knowledge_chains(
                search_result, query, max_chains
            )
            
            # 4. 按相似度排序
            knowledge_chains.sort(key=lambda x: x.chain_score, reverse=True)
            
            logger.info(f"为查询 '{query}' 构建了 {len(knowledge_chains)} 条知识链路")
            return knowledge_chains[:max_chains]
            
        except Exception as e:
            logger.error(f"搜索知识链路失败: {e}")
            return []
    
    def _vector_search(self, query_vector: List[float]) -> VectorSearchResult:
        """执行向量搜索"""
        import time
        start_time = time.time()
        
        # 搜索相似节点
        similar_nodes = self.storage.search_similar_nodes(
            query_vector, 
            limit=self.config.max_nodes_per_query
        )
        
        # 搜索相似关系
        similar_relations = self.storage.search_similar_relations(
            query_vector,
            limit=self.config.max_relations_per_query
        )
        
        search_time = time.time() - start_time
        
        return VectorSearchResult(
            nodes=similar_nodes,
            relations=similar_relations,
            query_vector=query_vector,
            search_time=search_time
        )
    
    def _build_knowledge_chains(self, search_result: VectorSearchResult, 
                               query: str, max_chains: int) -> List[KnowledgeChain]:
        """构建知识链路"""
        chains = []
        
        # 过滤高相似度的节点和关系
        filtered_nodes = self._filter_by_similarity(
            search_result.nodes, self.config.min_similarity_threshold
        )
        filtered_relations = self._filter_by_similarity(
            search_result.relations, self.config.min_similarity_threshold
        )
        
        if not filtered_nodes:
            logger.warning("没有找到满足相似度阈值的节点")
            return chains
        
        # 构建节点索引
        node_index = {node["name"]: node for node in filtered_nodes}
        
        # 构建关系图
        relation_graph = self._build_relation_graph(filtered_relations)
        
        # 为每个高相似度节点构建知识链路
        for start_node_data in filtered_nodes[:max_chains * 2]:  # 多取一些候选节点
            start_node_name = start_node_data["name"]
            
            # 使用深度优先搜索构建链路
            chain_paths = self._dfs_build_chains(
                start_node_name, 
                relation_graph, 
                node_index,
                max_depth=self.config.max_chain_length
            )
            
            # 为每条路径创建知识链路对象
            for path in chain_paths:
                chain = self._create_knowledge_chain(
                    path, node_index, filtered_relations, query
                )
                if chain:
                    chains.append(chain)
        
        # 去重和排序
        chains = self._deduplicate_chains(chains)
        chains.sort(key=lambda x: x.chain_score, reverse=True)
        
        return chains[:max_chains]
    
    def _filter_by_similarity(self, items: List[Dict[str, Any]], 
                             threshold: float) -> List[Dict[str, Any]]:
        """根据相似度过滤项目"""
        filtered = []
        for item in items:
            similarity = item.get("similarity", 0.0)
            if similarity and similarity >= threshold:
                filtered.append(item)
        return filtered
    
    def _build_relation_graph(self, relations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """构建关系图"""
        graph = defaultdict(list)
        
        for relation in relations:
            source = relation.get("source_node")
            target = relation.get("target_node")
            
            if source and target:
                # 添加正向关系
                graph[source].append({
                    "target": target,
                    "relation": relation
                })
                
                # 添加反向关系（无向图）
                graph[target].append({
                    "target": source,
                    "relation": relation
                })
        
        return graph
    
    def _dfs_build_chains(self, start_node: str, relation_graph: Dict[str, List[Dict[str, Any]]], 
                         node_index: Dict[str, Dict[str, Any]], max_depth: int) -> List[List[str]]:
        """使用深度优先搜索构建链路"""
        chains = []
        visited = set()
        
        def dfs(current_node: str, path: List[str], depth: int):
            if depth >= max_depth:
                return
            
            if current_node in visited:
                return
            
            visited.add(current_node)
            path.append(current_node)
            
            # 如果路径长度大于1，添加到结果中
            if len(path) > 1:
                chains.append(path.copy())
            
            # 继续探索邻居节点
            for neighbor_info in relation_graph.get(current_node, []):
                neighbor = neighbor_info["target"]
                if neighbor in node_index and neighbor not in visited:
                    dfs(neighbor, path, depth + 1)
            
            path.pop()
            visited.remove(current_node)
        
        dfs(start_node, [], 0)
        return chains
    
    def _create_knowledge_chain(self, path: List[str], node_index: Dict[str, Dict[str, Any]], 
                               relations: List[Dict[str, Any]], query: str) -> Optional[KnowledgeChain]:
        """创建知识链路对象"""
        if len(path) < 2:
            return None
        
        try:
            # 构建节点列表
            chain_nodes = []
            for node_name in path:
                if node_name in node_index:
                    node_data = node_index[node_name]
                    graph_node = GraphNode(
                        name=node_data["name"],
                        type=node_data["type"],
                        description=node_data.get("description", ""),
                        source_text=node_data.get("source_text", ""),
                        neo4j_id=node_data.get("neo4j_id", "")
                    )
                    chain_nodes.append(graph_node)
            
            # 构建关系列表
            chain_relations = []
            for i in range(len(path) - 1):
                source_node = path[i]
                target_node = path[i + 1]
                
                # 查找连接这两个节点的关系
                connecting_relation = self._find_connecting_relation(
                    source_node, target_node, relations
                )
                
                if connecting_relation:
                    graph_relation = GraphRelation(
                        source_node=connecting_relation["source_node"],
                        target_node=connecting_relation["target_node"],
                        relation_type=connecting_relation["relation_type"],
                        description=connecting_relation.get("description", ""),
                        source_text=connecting_relation.get("source_text", ""),
                        neo4j_id=connecting_relation.get("neo4j_id", "")
                    )
                    chain_relations.append(graph_relation)
            
            # 计算链路得分
            chain_score = self._calculate_chain_score(chain_nodes, chain_relations, query)
            
            # 生成链路描述
            chain_description = self._generate_chain_description(chain_nodes, chain_relations)
            
            return KnowledgeChain(
                nodes=chain_nodes,
                relations=chain_relations,
                chain_score=chain_score,
                chain_description=chain_description
            )
            
        except Exception as e:
            logger.error(f"创建知识链路失败: {e}")
            return None
    
    def _find_connecting_relation(self, source: str, target: str, 
                                 relations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """查找连接两个节点的关系"""
        for relation in relations:
            rel_source = relation.get("source_node")
            rel_target = relation.get("target_node")
            
            # 检查正向和反向连接
            if (rel_source == source and rel_target == target) or \
               (rel_source == target and rel_target == source):
                return relation
        
        return None
    
    def _calculate_chain_score(self, nodes: List[GraphNode], 
                              relations: List[GraphRelation], query: str) -> float:
        """计算知识链路得分"""
        if not nodes:
            return 0.0
        
        try:
            # 查询向量
            query_vector = self._get_embedding_service().embed_text(query)
            if not query_vector:
                return 0.0
            
            total_score = 0.0
            weight = 1.0
            
            # 计算节点相似度得分
            for i, node in enumerate(nodes):
                node_text = f"{node.name} {node.type} {node.description}"
                node_vector = self._get_embedding_service().embed_text(node_text)
                
                if node_vector:
                    similarity = self._calculate_cosine_similarity(query_vector, node_vector)
                    total_score += similarity * weight
                
                # 权重衰减
                weight *= self.config.chain_weight_decay
            
            # 计算关系相似度得分（给予额外权重）
            relation_weight = self.config.relation_boost_factor
            for relation in relations:
                relation_text = f"{relation.source_node} {relation.relation_type} {relation.target_node} {relation.description}"
                relation_vector = self._get_embedding_service().embed_text(relation_text)
                
                if relation_vector:
                    similarity = self._calculate_cosine_similarity(query_vector, relation_vector)
                    total_score += similarity * relation_weight
                
                relation_weight *= self.config.chain_weight_decay
            
            # 归一化得分
            total_elements = len(nodes) + len(relations)
            if total_elements > 0:
                total_score /= total_elements
            
            return min(total_score, 1.0)  # 确保得分不超过1
            
        except Exception as e:
            logger.error(f"计算链路得分失败: {e}")
            return 0.0
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)  # 确保相似度非负
            
        except Exception as e:
            logger.error(f"计算余弦相似度失败: {e}")
            return 0.0
    
    def _generate_chain_description(self, nodes: List[GraphNode], 
                                   relations: List[GraphRelation]) -> str:
        """生成知识链路描述"""
        if not nodes:
            return "空知识链路"
        
        if len(nodes) == 1:
            return f"单节点: {nodes[0].name} ({nodes[0].type})"
        
        # 构建链路描述
        description_parts = []
        
        for i in range(len(nodes)):
            node = nodes[i]
            description_parts.append(f"{node.name}({node.type})")
            
            # 添加关系描述
            if i < len(relations):
                relation = relations[i]
                description_parts.append(f" --[{relation.relation_type}]--> ")
        
        return "".join(description_parts)
    
    def retrieve(self, query: str, max_chains: int = 10, max_depth: int = 3) -> List[KnowledgeChain]:
        """检索知识链路"""
        try:
            # 1. 向量化查询
            query_vector = self._get_embedding_service().embed_text(query)
            if not query_vector:
                logger.error(f"查询 '{query}' 向量化失败")
                return []
            
            # 2. 搜索相关节点
            similar_nodes = self.storage.search_similar_nodes(
                query_vector=query_vector, 
                limit=self.config.max_nodes_per_query
            )
            
            if not similar_nodes:
                logger.info(f"未找到与查询 '{query}' 相关的节点")
                return []
            
            # 3. 搜索相关关系
            similar_relations = self.storage.search_similar_relations(
                query_vector=query_vector,
                limit=self.config.max_relations_per_query
            )
            
            # 4. 构建知识链路
            search_result = VectorSearchResult(
                nodes=similar_nodes,
                relations=similar_relations,
                query_vector=query_vector,
                search_time=0.0
            )
            chains = self._build_knowledge_chains(
                search_result, query, max_chains
            )
            
            # 5. 去重和排序
            chains = self._deduplicate_chains(chains)
            chains.sort(key=lambda x: x.chain_score, reverse=True)
            
            return chains[:max_chains]
            
        except Exception as e:
            logger.error(f"检索知识链路失败: {e}")
            return []
    
    def build_knowledge_chains(self, query: str, max_chains: int = 10, max_depth: int = 3, max_chain_length: int = None) -> List[KnowledgeChain]:
        """构建知识链路"""
        try:
            # 使用现有的retrieve方法获取知识链路
            chains = self.retrieve(query, max_chains, max_depth)
            
            # 如果指定了最大链路长度，进行过滤
            if max_chain_length is not None:
                filtered_chains = []
                for chain in chains:
                    if len(chain.nodes) <= max_chain_length:
                        filtered_chains.append(chain)
                chains = filtered_chains
            
            return chains
        except Exception as e:
            logger.error(f"构建知识链路失败: {e}")
            return []
    
    def _deduplicate_chains(self, chains: List[KnowledgeChain]) -> List[KnowledgeChain]:
        """去除重复的知识链路"""
        seen_chains = set()
        unique_chains = []
        
        for chain in chains:
            # 创建链路的唯一标识
            node_names = [node.name for node in chain.nodes]
            relation_types = [rel.relation_type for rel in chain.relations]
            
            chain_signature = tuple(node_names + relation_types)
            
            if chain_signature not in seen_chains:
                seen_chains.add(chain_signature)
                unique_chains.append(chain)
        
        return unique_chains
    
    def search_by_node_type(self, node_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """根据节点类型搜索"""
        try:
            # 生成类型查询向量
            type_query = f"节点类型 {node_type}"
            query_vector = self._get_embedding_service().embed_text(type_query)
            
            if not query_vector:
                logger.error(f"类型查询 '{node_type}' 向量化失败")
                return []
            
            # 搜索相似节点
            similar_nodes = self.storage.search_similar_nodes(query_vector, limit=limit)
            
            # 过滤指定类型的节点
            filtered_nodes = [
                node for node in similar_nodes 
                if node.get("type", "").lower() == node_type.lower()
            ]
            
            logger.info(f"找到 {len(filtered_nodes)} 个类型为 '{node_type}' 的节点")
            return filtered_nodes
            
        except Exception as e:
            logger.error(f"按节点类型搜索失败: {e}")
            return []
    
    def search_by_relation_type(self, relation_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """根据关系类型搜索"""
        try:
            # 生成关系类型查询向量
            type_query = f"关系类型 {relation_type}"
            query_vector = self._get_embedding_service().embed_text(type_query)
            
            if not query_vector:
                logger.error(f"关系类型查询 '{relation_type}' 向量化失败")
                return []
            
            # 搜索相似关系
            similar_relations = self.storage.search_similar_relations(query_vector, limit=limit)
            
            # 过滤指定类型的关系
            filtered_relations = [
                relation for relation in similar_relations 
                if relation.get("relation_type", "").lower() == relation_type.lower()
            ]
            
            logger.info(f"找到 {len(filtered_relations)} 个类型为 '{relation_type}' 的关系")
            return filtered_relations
            
        except Exception as e:
            logger.error(f"按关系类型搜索失败: {e}")
            return []
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        storage_stats = self.storage.get_stats()
        
        return {
            "storage_stats": storage_stats,
            "config": {
                "max_chain_length": self.config.max_chain_length,
                "min_similarity_threshold": self.config.min_similarity_threshold,
                "max_nodes_per_query": self.config.max_nodes_per_query,
                "max_relations_per_query": self.config.max_relations_per_query
            },
            "embedding_service": type(self._get_embedding_service()).__name__
        }


class VectorizedGraphRetrievalService:
    """向量化图谱检索服务"""
    
    def __init__(self, storage: VectorizedGraphStorage, config: Dict[str, Any], chain_config: Dict[str, Any]):
        """初始化向量化图谱检索服务"""
        self.storage = storage
        self.config = config
        self.chain_config = chain_config
        
        # 创建配置对象
        chain_config_obj = KnowledgeChainConfig(
            max_chain_length=config.get('max_chain_length', 5),
            min_similarity_threshold=config.get('min_similarity_threshold', 0.3),
            max_nodes_per_query=config.get('max_nodes_per_query', 20),
            max_relations_per_query=config.get('max_relations_per_query', 20),
            chain_weight_decay=chain_config.get('chain_weight_decay', 0.8),
            relation_boost_factor=chain_config.get('relation_boost_factor', 1.2)
        )
        
        self.retrieval = VectorizedGraphRetrieval(storage, chain_config_obj)
    
    async def search_async(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """异步搜索图谱节点"""
        try:
            # 使用向量相似度搜索节点
            results = self.storage.search_similar_nodes(query=query, limit=max_results)
            return results
        except Exception as e:
            logger.error(f"异步搜索失败: {e}")
            return []
    
    async def search_entities_by_name_async(self, entity_name: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """根据实体名称异步搜索"""
        try:
            # 构建实体名称查询
            query = f"实体名称 {entity_name}"
            results = self.storage.search_similar_nodes(query=query, limit=max_results)
            
            # 过滤名称匹配的结果
            filtered_results = []
            for result in results:
                if entity_name.lower() in result.get("name", "").lower():
                    filtered_results.append(result)
            
            return filtered_results
        except Exception as e:
            logger.error(f"根据实体名称异步搜索失败: {e}")
            return []
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """同步搜索图谱节点"""
        try:
            results = self.storage.search_similar_nodes(query=query, limit=max_results)
            return results
        except Exception as e:
            logger.error(f"同步搜索失败: {e}")
            return []
    
    def get_knowledge_chains(self, query: str, max_chains: int = 10, max_depth: int = 3) -> List[KnowledgeChain]:
        """获取知识链路"""
        return self.retrieval.build_knowledge_chains(query, max_chains, max_depth)