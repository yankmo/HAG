#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块化RAG系统 - 将存储和检索功能模块化设计
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.knowledge.intent_recognition_neo4j import KnowledgeGraphBuilder, Entity, Relation
from src.knowledge.vector_storage import (
    WeaviateVectorStore,
    VectorKnowledgeProcessor,
    VectorEntity,
    VectorRelation
)
from src.services.embedding_service import OllamaEmbeddingService as OllamaEmbeddingClient
from config import get_config
from py2neo import Graph, Node, Relationship
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KnowledgeStorageManager:
    """知识存储管理器 - 负责实体和关系的存储"""
    
    def __init__(self, neo4j_uri: str = None, 
                 neo4j_auth: Tuple[str, str] = None):
        """初始化存储管理器"""
        config = get_config()
        
        # 使用配置文件中的设置，如果没有传入参数的话
        neo4j_uri = neo4j_uri or config.neo4j.uri
        neo4j_auth = neo4j_auth or config.neo4j.to_auth_tuple()
        
        self.embedding_client = OllamaEmbeddingClient()
        self.vector_store = WeaviateVectorStore()
        self.vector_processor = VectorKnowledgeProcessor(self.embedding_client, self.vector_store)
        self.neo4j_graph = Graph(neo4j_uri, auth=neo4j_auth)
        
        logger.info("知识存储管理器初始化完成")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保资源清理"""
        self.close()
    
    def close(self):
        """关闭所有连接"""
        try:
            # 关闭Weaviate连接
            if hasattr(self.vector_store, 'client') and self.vector_store.client:
                self.vector_store.client.close()
                logger.debug("Weaviate连接已关闭")
            
            # Neo4j连接会自动管理，但我们可以显式清理
            if hasattr(self.neo4j_graph, '_connector') and self.neo4j_graph._connector:
                self.neo4j_graph._connector.close()
                logger.debug("Neo4j连接已关闭")
                
        except Exception as e:
            logger.warning(f"关闭连接时出现警告: {e}")
    
    def __del__(self):
        """析构函数，确保资源清理"""
        self.close()
    
    def setup_storage(self) -> bool:
        """设置存储系统"""
        try:
            logger.info("设置存储系统...")
            
            # 清空Neo4j数据库
            self.neo4j_graph.delete_all()
            logger.info("Neo4j数据库已清空")
            
            # 设置Weaviate向量存储
            success = self.vector_store.setup_collections()
            if success:
                logger.info("Weaviate向量存储设置完成")
                return True
            else:
                logger.error("Weaviate向量存储设置失败")
                return False
                
        except Exception as e:
            logger.error(f"设置存储系统失败: {e}")
            return False
    
    def store_entity(self, entity: Entity, source_text: str = "") -> Optional[str]:
        """存储单个实体到Neo4j和Weaviate"""
        try:
            # 1. 存储到Neo4j
            node_properties = {"name": entity.name}
            if entity.properties:
                for key, value in entity.properties.items():
                    node_properties[key] = value
            
            neo4j_node = Node(entity.type, **node_properties)
            self.neo4j_graph.create(neo4j_node)
            neo4j_id = str(neo4j_node.identity)
            
            # 2. 创建向量实体
            entity_text = f"实体: {entity.name}, 类型: {entity.type}"
            if entity.properties and entity.properties.get("description"):
                entity_text += f", 描述: {entity.properties['description']}"
            
            # 向量化
            vector = self.embedding_client.embed_text(entity_text)
            if vector:
                entity_properties = entity.properties.copy() if entity.properties else {}
                
                vector_entity = VectorEntity(
                    name=entity.name,
                    type=entity.type,
                    properties=entity_properties,
                    vector=vector,
                    source_text=source_text,
                    neo4j_id=neo4j_id
                )
                
                # 3. 存储向量到Weaviate
                success = self.vector_store.store_entities([vector_entity])
                if success:
                    logger.debug(f"实体 {entity.name} 存储完成，Neo4j ID: {neo4j_id}")
                    return neo4j_id
                else:
                    logger.error(f"实体 {entity.name} 向量存储失败")
            else:
                logger.error(f"实体 {entity.name} 向量化失败")
            
            return neo4j_id
            
        except Exception as e:
            logger.error(f"存储实体失败: {e}")
            return None
    
    def store_relation(self, relation: Relation, source_text: str = "") -> Optional[str]:
        """存储单个关系到Neo4j和Weaviate"""
        try:
            # 1. 获取Neo4j节点
            source_node = self.neo4j_graph.nodes.match(name=relation.source).first()
            target_node = self.neo4j_graph.nodes.match(name=relation.target).first()
            
            if not source_node or not target_node:
                logger.error(f"找不到关系的源节点或目标节点: {relation.source} -> {relation.target}")
                return None
            
            # 2. 创建关系
            rel_properties = {}
            if relation.properties:
                for key, value in relation.properties.items():
                    rel_properties[key] = value
            
            neo4j_rel = Relationship(source_node, relation.relation_type, target_node, **rel_properties)
            self.neo4j_graph.create(neo4j_rel)
            neo4j_id = str(neo4j_rel.identity)
            
            # 3. 创建向量关系
            relation_text = f"关系: {relation.source} {relation.relation_type} {relation.target}"
            if relation.properties and relation.properties.get("description"):
                relation_text += f", 描述: {relation.properties['description']}"
            
            # 向量化
            vector = self.embedding_client.embed_text(relation_text)
            if vector:
                vector_relation = VectorRelation(
                    source=relation.source,
                    target=relation.target,
                    relation_type=relation.relation_type,
                    description=relation.properties.get('description', '') if relation.properties else '',
                    vector=vector,
                    source_text=source_text,
                    neo4j_id=neo4j_id
                )
                
                # 4. 存储向量到Weaviate
                success = self.vector_store.store_relations([vector_relation])
                if success:
                    logger.debug(f"关系 {relation.source}-{relation.relation_type}-{relation.target} 存储完成，Neo4j ID: {neo4j_id}")
                    return neo4j_id
                else:
                    logger.error(f"关系向量存储失败")
            else:
                logger.error(f"关系向量化失败")
            
            return neo4j_id
            
        except Exception as e:
            logger.error(f"存储关系失败: {e}")
            return None
    
    def batch_store_entities(self, entities: List[Entity], source_text: str = "") -> Dict[str, str]:
        """批量存储实体"""
        entity_id_map = {}
        
        for entity in entities:
            if entity.name not in entity_id_map:  # 避免重复存储
                neo4j_id = self.store_entity(entity, source_text)
                if neo4j_id:
                    entity_id_map[entity.name] = neo4j_id
        
        logger.info(f"批量存储完成: {len(entity_id_map)} 个实体")
        return entity_id_map
    
    def batch_store_relations(self, relations: List[Relation], entity_id_map: Dict[str, str], source_text: str = "") -> List[str]:
        """批量存储关系"""
        relation_ids = []
        
        for relation in relations:
            # 确保源和目标实体都存在
            if relation.source in entity_id_map and relation.target in entity_id_map:
                neo4j_id = self.store_relation(relation, source_text)
                if neo4j_id:
                    relation_ids.append(neo4j_id)
            else:
                logger.warning(f"关系 {relation.source}-{relation.relation_type}-{relation.target} 的实体不存在，跳过")
        
        logger.info(f"批量存储完成: {len(relation_ids)} 个关系")
        return relation_ids
    
    def get_storage_stats(self) -> Dict[str, int]:
        """获取存储统计信息"""
        try:
            # Neo4j统计
            neo4j_nodes = self.neo4j_graph.run("MATCH (n) RETURN count(n) as count").data()[0]['count']
            neo4j_rels = self.neo4j_graph.run("MATCH ()-[r]->() RETURN count(r) as count").data()[0]['count']
            
            # Weaviate统计
            vector_stats = self.vector_store.get_stats()
            
            return {
                "neo4j_nodes": neo4j_nodes,
                "neo4j_relationships": neo4j_rels,
                "vector_entities": vector_stats.get('entities', 0),
                "vector_relations": vector_stats.get('relations', 0)
            }
        except Exception as e:
            logger.error(f"获取存储统计失败: {e}")
            return {}


class KnowledgeRetrievalManager:
    """知识检索管理器 - 负责向量搜索和图谱搜索"""
    
    def __init__(self, neo4j_uri: str = None, 
                 neo4j_auth: Tuple[str, str] = None):
        """初始化检索管理器"""
        config = get_config()
        
        # 使用配置文件中的设置，如果没有传入参数的话
        neo4j_uri = neo4j_uri or config.neo4j.uri
        neo4j_auth = neo4j_auth or config.neo4j.to_auth_tuple()
        
        self.embedding_client = OllamaEmbeddingClient()
        self.vector_store = WeaviateVectorStore()
        self.vector_processor = VectorKnowledgeProcessor(self.embedding_client, self.vector_store)
        self.neo4j_graph = Graph(neo4j_uri, auth=neo4j_auth)
        
        logger.info("知识检索管理器初始化完成")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保资源清理"""
        self.close()
    
    def close(self):
        """关闭所有连接"""
        try:
            # 关闭Weaviate连接
            if hasattr(self.vector_store, 'client') and self.vector_store.client:
                self.vector_store.client.close()
                logger.debug("检索管理器：Weaviate连接已关闭")
            
            # Neo4j连接会自动管理，但我们可以显式清理
            if hasattr(self.neo4j_graph, '_connector') and self.neo4j_graph._connector:
                self.neo4j_graph._connector.close()
                logger.debug("检索管理器：Neo4j连接已关闭")
                
        except Exception as e:
            logger.warning(f"检索管理器关闭连接时出现警告: {e}")
    
    def __del__(self):
        """析构函数，确保资源清理"""
        self.close()
    
    def vector_search(self, query: str, entity_limit: int = 5, relation_limit: int = 5) -> Dict[str, List]:
        """向量搜索"""
        try:
            if not query or not query.strip():
                logger.warning("查询为空，返回空结果")
                return {"entities": [], "relations": []}
            
            # 检查向量存储连接
            try:
                stats = self.vector_store.get_stats()
                if stats.get('total', 0) == 0:
                    logger.warning("向量存储为空，无法进行检索")
                    return {"entities": [], "relations": []}
            except Exception as e:
                logger.error(f"检查向量存储状态失败: {e}")
                return {"entities": [], "relations": []}
            
            # 执行搜索
            results = self.vector_processor.search_knowledge_detailed(query, entity_limit, relation_limit)
            
            # 验证结果格式
            if not isinstance(results, dict):
                logger.error("向量搜索返回格式错误")
                return {"entities": [], "relations": []}
            
            # 确保返回正确的键
            entities = results.get("entities", [])
            relations = results.get("relations", [])
            
            logger.info(f"向量搜索完成: 找到 {len(entities)} 个实体, {len(relations)} 个关系")
            return {"entities": entities, "relations": relations}
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return {"entities": [], "relations": []}
    
    def graph_search_topk_nodes(self, query: str, top_k: int = 10, include_relations: bool = True) -> Dict[str, Any]:
        """图谱搜索：检索topk个最相关的节点及其关系"""
        try:
            # 1. 先通过向量搜索找到相关实体
            vector_results = self.vector_search(query, entity_limit=top_k*2)  # 获取更多候选
            
            # 2. 提取实体名称
            entity_names = [entity['name'] for entity in vector_results['entities'][:top_k]]
            
            if not entity_names:
                return {"nodes": [], "relationships": [], "total_nodes": 0, "total_relationships": 0}
            
            # 3. 构建Cypher查询获取topk节点
            entity_list = "', '".join(entity_names)
            
            if include_relations:
                # 获取节点及其直接关系
                cypher_query = f"""
                MATCH (n) WHERE n.name IN ['{entity_list}']
                OPTIONAL MATCH (n)-[r]-(m)
                RETURN n, r, m
                ORDER BY n.name
                LIMIT {top_k * 10}
                """
            else:
                # 只获取节点
                cypher_query = f"""
                MATCH (n) WHERE n.name IN ['{entity_list}']
                RETURN n
                ORDER BY n.name
                LIMIT {top_k}
                """
            
            result = self.neo4j_graph.run(cypher_query)
            
            nodes = []
            relationships = []
            node_ids = set()
            rel_ids = set()
            
            for record in result:
                # 处理主节点
                if record.get('n'):
                    node = record['n']
                    node_id = str(node.identity)
                    if node_id not in node_ids:
                        nodes.append({
                            "id": node_id,
                            "name": dict(node).get("name", ""),
                            "labels": list(node.labels),
                            "properties": dict(node)
                        })
                        node_ids.add(node_id)
                
                # 处理相关节点
                if record.get('m') and include_relations:
                    node = record['m']
                    node_id = str(node.identity)
                    if node_id not in node_ids:
                        nodes.append({
                            "id": node_id,
                            "name": dict(node).get("name", ""),
                            "labels": list(node.labels),
                            "properties": dict(node)
                        })
                        node_ids.add(node_id)
                
                # 处理关系
                if record.get('r') and include_relations:
                    rel = record['r']
                    rel_id = str(rel.identity)
                    if rel_id not in rel_ids:
                        relationships.append({
                            "id": rel_id,
                            "type": type(rel).__name__,
                            "properties": dict(rel),
                            "start_node_id": str(rel.start_node.identity),
                            "end_node_id": str(rel.end_node.identity)
                        })
                        rel_ids.add(rel_id)
            
            # 限制返回的节点数量
            nodes = nodes[:top_k]
            
            return {
                "nodes": nodes,
                "relationships": relationships,
                "total_nodes": len(nodes),
                "total_relationships": len(relationships),
                "query": query
            }
            
        except Exception as e:
            logger.error(f"图谱搜索失败: {e}")
            return {"nodes": [], "relationships": [], "total_nodes": 0, "total_relationships": 0, "error": str(e)}
    
    def graph_expand_from_nodes(self, node_ids: List[str], depth: int = 2, max_nodes: int = 50) -> Dict[str, Any]:
        """从指定节点扩展子图"""
        try:
            if not node_ids:
                return {"nodes": [], "relationships": [], "paths": []}
            
            # 构建Cypher查询
            node_ids_str = ', '.join(node_ids)
            cypher_query = f"""
            MATCH (n) WHERE ID(n) IN [{node_ids_str}]
            OPTIONAL MATCH path = (n)-[*1..{depth}]-(m)
            WITH n, path, relationships(path) as rels, nodes(path) as path_nodes
            RETURN n, path, rels, path_nodes
            LIMIT {max_nodes}
            """
            
            result = self.neo4j_graph.run(cypher_query)
            
            all_nodes = []
            all_relationships = []
            all_paths = []
            node_ids_seen = set()
            rel_ids_seen = set()
            
            for record in result:
                # 处理起始节点
                if record.get('n'):
                    node = record['n']
                    node_id = str(node.identity)
                    if node_id not in node_ids_seen:
                        all_nodes.append({
                            "id": node_id,
                            "labels": list(node.labels),
                            "properties": dict(node)
                        })
                        node_ids_seen.add(node_id)
                
                # 处理路径中的节点
                if record.get('path_nodes'):
                    for node in record['path_nodes']:
                        if node:
                            node_id = str(node.identity)
                            if node_id not in node_ids_seen:
                                all_nodes.append({
                                    "id": node_id,
                                    "labels": list(node.labels),
                                    "properties": dict(node)
                                })
                                node_ids_seen.add(node_id)
                
                # 处理路径中的关系
                if record.get('rels'):
                    for rel in record['rels']:
                        if rel:
                            rel_id = str(rel.identity)
                            if rel_id not in rel_ids_seen:
                                all_relationships.append({
                                    "id": rel_id,
                                    "type": type(rel).__name__,
                                    "properties": dict(rel),
                                    "start_node_id": str(rel.start_node.identity),
                                    "end_node_id": str(rel.end_node.identity)
                                })
                                rel_ids_seen.add(rel_id)
                
                # 处理路径
                if record.get('path'):
                    path = record['path']
                    all_paths.append({
                        "length": len(path),
                        "nodes": [str(node.identity) for node in path.nodes],
                        "relationships": [str(rel.identity) for rel in path.relationships]
                    })
            
            return {
                "nodes": all_nodes,
                "relationships": all_relationships,
                "paths": all_paths,
                "total_nodes": len(all_nodes),
                "total_relationships": len(all_relationships),
                "total_paths": len(all_paths)
            }
            
        except Exception as e:
            logger.error(f"图谱扩展失败: {e}")
            return {"nodes": [], "relationships": [], "paths": [], "error": str(e)}


class HybridSearchManager:
    """混合搜索管理器 - 整合向量搜索和图谱搜索"""
    
    def __init__(self, neo4j_uri: str = None, 
                 neo4j_auth: Tuple[str, str] = None):
        """初始化混合搜索管理器"""
        config = get_config()
        
        # 使用配置文件中的设置，如果没有传入参数的话
        neo4j_uri = neo4j_uri or config.neo4j.uri
        neo4j_auth = neo4j_auth or config.neo4j.to_auth_tuple()
        
        self.retrieval_manager = KnowledgeRetrievalManager(neo4j_uri, neo4j_auth)
        self.kg_builder = KnowledgeGraphBuilder()
        
        logger.info("混合搜索管理器初始化完成")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保资源清理"""
        self.close()
    
    def close(self):
        """关闭所有连接"""
        try:
            if hasattr(self.retrieval_manager, 'close'):
                self.retrieval_manager.close()
                logger.debug("混合搜索管理器：检索管理器连接已关闭")
        except Exception as e:
            logger.warning(f"混合搜索管理器关闭连接时出现警告: {e}")
    
    def __del__(self):
        """析构函数，确保资源清理"""
        self.close()
    
    def hybrid_search(self, query: str, 
                     vector_entity_limit: int = 5, 
                     vector_relation_limit: int = 5,
                     graph_top_k: int = 10,
                     expand_depth: int = 2,
                     max_expand_nodes: int = 50) -> Dict[str, Any]:
        """执行混合搜索"""
        try:
            logger.info(f"执行混合搜索: {query}")
            
            # 1. 向量搜索
            vector_results = self.retrieval_manager.vector_search(query, vector_entity_limit, vector_relation_limit)
            
            # 2. 图谱搜索topk节点
            graph_results = self.retrieval_manager.graph_search_topk_nodes(query, graph_top_k, include_relations=True)
            
            # 3. 提取Neo4j ID进行图谱扩展
            neo4j_entity_ids = []
            for entity in vector_results["entities"]:
                if entity.get("neo4j_id"):
                    neo4j_entity_ids.append(entity["neo4j_id"])
            
            # 4. 图谱扩展
            expanded_subgraph = {}
            if neo4j_entity_ids:
                expanded_subgraph = self.retrieval_manager.graph_expand_from_nodes(
                    neo4j_entity_ids, expand_depth, max_expand_nodes
                )
            
            return {
                "query": query,
                "vector_search": vector_results,
                "graph_search": graph_results,
                "expanded_subgraph": expanded_subgraph,
                "search_stats": {
                    "vector_entities": len(vector_results["entities"]),
                    "vector_relations": len(vector_results["relations"]),
                    "graph_nodes": graph_results["total_nodes"],
                    "graph_relationships": graph_results["total_relationships"],
                    "expanded_nodes": expanded_subgraph.get("total_nodes", 0),
                    "expanded_relationships": expanded_subgraph.get("total_relationships", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            return {"error": str(e)}
    
    def generate_answer(self, query: str, search_results: Dict[str, Any]) -> str:
        """基于搜索结果生成答案"""
        try:
            # 构建上下文
            context_parts = []
            
            # 添加向量搜索的实体信息
            vector_search = search_results.get("vector_search", {})
            if vector_search.get("entities"):
                context_parts.append("🔍 相关实体:")
                for entity in vector_search["entities"][:3]:
                    similarity = 1 - entity.get('distance', 0)
                    context_parts.append(f"- {entity['name']} ({entity['type']}) - 相似度: {similarity:.3f}")
                    if entity.get('description'):
                        context_parts.append(f"  描述: {entity['description']}")
            
            # 添加向量搜索的关系信息
            if vector_search.get("relations"):
                context_parts.append("\n🔗 相关关系:")
                for relation in vector_search["relations"][:3]:
                    similarity = 1 - relation.get('distance', 0)
                    context_parts.append(f"- {relation['source']} → {relation['relation_type']} → {relation['target']} - 相似度: {similarity:.3f}")
                    if relation.get('description'):
                        context_parts.append(f"  描述: {relation['description']}")
            
            # 添加图谱搜索信息
            graph_search = search_results.get("graph_search", {})
            if graph_search.get("nodes"):
                context_parts.append(f"\n📊 知识图谱: 发现 {graph_search['total_nodes']} 个相关节点, {graph_search['total_relationships']} 个关系")
                
                for node in graph_search["nodes"][:5]:
                    labels = ", ".join(node["labels"]) if node["labels"] else "未知类型"
                    name = node["properties"].get("name", "未知名称")
                    context_parts.append(f"  • {name} ({labels})")
            
            context = "\n".join(context_parts)
            
            # 构建提示词
            prompt = f"""基于以下混合知识图谱信息回答问题：

问题: {query}

知识来源:
{context}

请基于上述向量搜索和知识图谱的信息，提供准确、详细、结构化的回答。
如果信息不足，请说明需要更多哪方面的信息。"""

            # 使用Ollama生成回答
            response = self.kg_builder.recognizer.ollama.generate(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return f"抱歉，生成答案时出现错误: {e}"
    
    def chat(self, query: str, **search_params) -> Dict[str, Any]:
        """完整的对话流程"""
        try:
            # 1. 混合搜索
            search_results = self.hybrid_search(query, **search_params)
            
            # 2. 生成答案
            answer = self.generate_answer(query, search_results)
            
            return {
                "query": query,
                "answer": answer,
                "search_results": search_results,
                "timestamp": str(datetime.now())
            }
            
        except Exception as e:
            logger.error(f"对话处理失败: {e}")
            return {
                "query": query,
                "answer": f"抱歉，处理您的问题时出现错误: {e}",
                "error": str(e)
            }


class ModularRAGSystem:
    """模块化RAG系统 - 整合所有功能模块"""
    
    def __init__(self, neo4j_uri: str = None, 
                 neo4j_auth: Tuple[str, str] = None):
        """初始化模块化RAG系统"""
        config = get_config()
        
        # 使用配置文件中的设置，如果没有传入参数的话
        neo4j_uri = neo4j_uri or config.neo4j.uri
        neo4j_auth = neo4j_auth or config.neo4j.to_auth_tuple()
        
        self.storage_manager = KnowledgeStorageManager(neo4j_uri, neo4j_auth)
        self.retrieval_manager = KnowledgeRetrievalManager(neo4j_uri, neo4j_auth)
        self.search_manager = HybridSearchManager(neo4j_uri, neo4j_auth)
        self.kg_builder = KnowledgeGraphBuilder()
        
        logger.info("模块化RAG系统初始化完成")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保资源清理"""
        self.close()
    
    def close(self):
        """关闭所有连接"""
        try:
            if hasattr(self.storage_manager, 'close'):
                self.storage_manager.close()
            if hasattr(self.retrieval_manager, 'close'):
                self.retrieval_manager.close()
            if hasattr(self.search_manager, 'close'):
                self.search_manager.close()
            logger.info("模块化RAG系统：所有连接已关闭")
        except Exception as e:
            logger.warning(f"模块化RAG系统关闭连接时出现警告: {e}")
    
    def __del__(self):
        """析构函数，确保资源清理"""
        self.close()
    
    def build_knowledge_base(self, file_path: str, chunk_size: int = 500) -> Dict[str, Any]:
        """构建知识库"""
        try:
            logger.info(f"开始构建知识库: {file_path}")
            
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"文件大小: {len(content)} 字符")
            
            # 设置存储系统
            if not self.storage_manager.setup_storage():
                return {"error": "存储系统设置失败"}
            
            # 分块处理文本
            chunks = self.kg_builder._split_text(content, chunk_size)
            logger.info(f"文本分为 {len(chunks)} 个块")
            
            total_entities = 0
            total_relations = 0
            entity_id_map = {}
            
            for i, chunk in enumerate(chunks):
                logger.info(f"处理第 {i+1}/{len(chunks)} 块")
                
                try:
                    # 提取实体和关系
                    entities, relations = self.kg_builder.recognizer.extract_entities_and_relations(chunk)
                    
                    if entities:
                        # 存储实体
                        chunk_entity_map = self.storage_manager.batch_store_entities(entities, chunk)
                        entity_id_map.update(chunk_entity_map)
                        total_entities += len(chunk_entity_map)
                    
                    if relations:
                        # 存储关系
                        relation_ids = self.storage_manager.batch_store_relations(relations, entity_id_map, chunk)
                        total_relations += len(relation_ids)
                    
                    logger.info(f"块 {i+1}: 处理了 {len(entities)} 个实体, {len(relations)} 个关系")
                        
                except Exception as e:
                    logger.error(f"处理块 {i+1} 时出错: {e}")
            
            # 获取最终统计
            stats = self.storage_manager.get_storage_stats()
            
            logger.info("知识库构建完成")
            
            return {
                "success": True,
                "processed_chunks": len(chunks),
                "total_entities_processed": total_entities,
                "total_relations_processed": total_relations,
                "final_stats": stats,
                "entity_id_map": entity_id_map
            }
            
        except Exception as e:
            logger.error(f"构建知识库失败: {e}")
            return {"error": str(e)}
    
    def search(self, query: str, **search_params) -> Dict[str, Any]:
        """搜索知识库"""
        return self.search_manager.chat(query, **search_params)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return self.storage_manager.get_storage_stats()