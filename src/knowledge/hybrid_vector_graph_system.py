#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合向量图谱系统 - 实现向量存储与Neo4j图谱的完整联动
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
from py2neo import Graph, Node, Relationship
import logging
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import get_config
from typing import List, Dict, Any, Tuple
from datetime import datetime
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridVectorGraphSystem:
    """混合向量图谱系统"""
    
    def __init__(self):
        # 获取配置
        config = get_config()
        
        # 初始化各个组件
        self.kg_builder = KnowledgeGraphBuilder()
        self.embedding_client = OllamaEmbeddingClient()  # 使用配置中的默认模型
        self.vector_store = WeaviateVectorStore()  # 使用配置中的默认URL
        self.vector_processor = VectorKnowledgeProcessor(self.embedding_client, self.vector_store)
        self.neo4j_graph = Graph(config.neo4j.uri, auth=config.neo4j.to_auth_tuple())
        
        logger.info("混合向量图谱系统初始化完成")
    
    def setup_storage(self):
        """设置存储系统"""
        logger.info("设置存储系统...")
        
        # 清空Neo4j数据库
        self.neo4j_graph.delete_all()
        logger.info("Neo4j数据库已清空")
        
        # 设置Weaviate向量存储
        self.vector_store.setup_collections()
        logger.info("Weaviate向量存储设置完成")
    
    def store_entity_with_vector(self, entity: Entity, source_text: str = "") -> str:
        """存储实体到Neo4j并获取ID，然后存储向量到Weaviate"""
        try:
            # 1. 存储到Neo4j并获取ID
            # 准备节点属性，避免重复的description
            node_properties = {"name": entity.name}
            if entity.properties:
                # 复制properties，确保不会有重复的键
                for key, value in entity.properties.items():
                    node_properties[key] = value
            
            neo4j_node = Node(entity.type, **node_properties)
            
            self.neo4j_graph.create(neo4j_node)
            neo4j_id = str(neo4j_node.identity)
            
            # 2. 创建向量实体（包含Neo4j ID）
            entity_text = f"实体: {entity.name}, 类型: {entity.type}"
            if entity.properties and entity.properties.get("description"):
                entity_text += f", 描述: {entity.properties['description']}"
            
            # 向量化
            vector = self.embedding_client.embed_text(entity_text)
            if vector:
                # 准备properties字典
                entity_properties = {}
                if entity.properties:
                    entity_properties = entity.properties.copy()
                
                vector_entity = VectorEntity(
                    name=entity.name,
                    type=entity.type,
                    properties=entity_properties,  # 使用properties字段
                    vector=vector,
                    source_text=source_text,
                    neo4j_id=neo4j_id  # 关键：存储Neo4j ID
                )
                
                # 3. 存储向量到Weaviate
                success = self.vector_store.store_entities([vector_entity])
                if success:
                    logger.info(f"实体 {entity.name} 存储完成，Neo4j ID: {neo4j_id}")
                    return neo4j_id
                else:
                    logger.error(f"实体 {entity.name} 向量存储失败")
            else:
                logger.error(f"实体 {entity.name} 向量化失败")
            
            return neo4j_id
            
        except Exception as e:
            logger.error(f"存储实体失败: {e}")
            return None
    
    def store_relation_with_vector(self, relation: Relation, source_node_id: str, target_node_id: str, source_text: str = "") -> str:
        """存储关系到Neo4j并获取ID，然后存储向量到Weaviate"""
        try:
            # 1. 获取Neo4j节点
            source_node = self.neo4j_graph.nodes.match(name=relation.source).first()
            target_node = self.neo4j_graph.nodes.match(name=relation.target).first()
            
            if not source_node or not target_node:
                logger.error(f"找不到关系的源节点或目标节点: {relation.source} -> {relation.target}")
                return None
            
            # 2. 创建关系
            # 准备关系属性，避免重复的description
            rel_properties = {}
            if relation.properties:
                for key, value in relation.properties.items():
                    rel_properties[key] = value
            
            neo4j_rel = Relationship(source_node, relation.relation_type, target_node, **rel_properties)
            
            self.neo4j_graph.create(neo4j_rel)
            neo4j_id = str(neo4j_rel.identity)
            
            # 3. 创建向量关系（包含Neo4j ID）
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
                    neo4j_id=neo4j_id  # 关键：存储Neo4j ID
                )
                
                # 4. 存储向量到Weaviate
                success = self.vector_store.store_relations([vector_relation])
                if success:
                    logger.info(f"关系 {relation.source}-{relation.relation_type}-{relation.target} 存储完成，Neo4j ID: {neo4j_id}")
                    return neo4j_id
                else:
                    logger.error(f"关系向量存储失败")
            else:
                logger.error(f"关系向量化失败")
            
            return neo4j_id
            
        except Exception as e:
            logger.error(f"存储关系失败: {e}")
            return None
    
    def build_hybrid_knowledge_graph(self, file_path: str, chunk_size: int = 500):
        """构建混合知识图谱"""
        try:
            logger.info(f"开始构建混合知识图谱: {file_path}")
            
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"文件大小: {len(content)} 字符")
            
            # 设置存储系统
            self.setup_storage()
            
            # 分块处理文本
            chunks = self.kg_builder._split_text(content, chunk_size)
            logger.info(f"文本分为 {len(chunks)} 个块")
            
            entity_count = 0
            relation_count = 0
            entity_id_map = {}  # 实体名称到Neo4j ID的映射
            
            for i, chunk in enumerate(chunks):
                logger.info(f"处理第 {i+1}/{len(chunks)} 块")
                
                try:
                    # 1. 使用Ollama提取实体和关系
                    entities, relations = self.kg_builder.recognizer.extract_entities_and_relations(chunk)
                    
                    if entities:
                        # 2. 存储实体（Neo4j + Weaviate）
                        for entity in entities:
                            if entity.name not in entity_id_map:  # 避免重复存储
                                neo4j_id = self.store_entity_with_vector(entity, chunk)
                                if neo4j_id:
                                    entity_id_map[entity.name] = neo4j_id
                                    entity_count += 1
                    
                    if relations:
                        # 3. 存储关系（Neo4j + Weaviate）
                        for relation in relations:
                            # 确保源和目标实体都存在
                            if relation.source in entity_id_map and relation.target in entity_id_map:
                                neo4j_id = self.store_relation_with_vector(
                                    relation, 
                                    entity_id_map[relation.source],
                                    entity_id_map[relation.target],
                                    chunk
                                )
                                if neo4j_id:
                                    relation_count += 1
                            else:
                                logger.warning(f"关系 {relation.source}-{relation.relation_type}-{relation.target} 的实体不存在，跳过")
                    
                    logger.info(f"块 {i+1}: 处理了 {len(entities)} 个实体, {len(relations)} 个关系")
                        
                except Exception as e:
                    logger.error(f"处理块 {i+1} 时出错: {e}")
                    import traceback
                    logger.error(f"错误详情: {traceback.format_exc()}")
            
            # 获取存储统计
            vector_stats = self.vector_store.get_stats()
            
            logger.info("混合知识图谱构建完成")
            
            return {
                "neo4j_entities": entity_count,
                "neo4j_relations": relation_count,
                "vector_entities": vector_stats['entities'],
                "vector_relations": vector_stats['relations'],
                "entity_id_map": entity_id_map
            }
            
        except Exception as e:
            logger.error(f"构建混合知识图谱失败: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            return None
    
    def hybrid_search_with_graph_expansion(self, query: str, entity_limit: int = 5, relation_limit: int = 5, expand_depth: int = 2) -> Dict[str, Any]:
        """混合搜索：向量检索 + 图谱扩展"""
        try:
            logger.info(f"执行混合搜索: {query}")
            
            # 1. 向量搜索
            search_results = self.vector_processor.search_knowledge_detailed(query, entity_limit, relation_limit)
            
            # 2. 提取Neo4j ID
            neo4j_entity_ids = []
            neo4j_relation_ids = []
            
            for entity in search_results["entities"]:
                if entity.get("neo4j_id"):
                    neo4j_entity_ids.append(entity["neo4j_id"])
            
            for relation in search_results["relations"]:
                if relation.get("neo4j_id"):
                    neo4j_relation_ids.append(relation["neo4j_id"])
            
            # 3. 从Neo4j扩展子图
            expanded_subgraph = self._expand_subgraph_by_ids(neo4j_entity_ids, neo4j_relation_ids, expand_depth)
            
            return {
                "query": query,
                "vector_search": search_results,
                "expanded_subgraph": expanded_subgraph,
                "neo4j_entity_ids": neo4j_entity_ids,
                "neo4j_relation_ids": neo4j_relation_ids
            }
            
        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            return {"error": str(e)}
    
    def _expand_subgraph_by_ids(self, entity_ids: List[str], relation_ids: List[str], depth: int = 2) -> Dict[str, Any]:
        """根据Neo4j ID扩展子图"""
        try:
            if not entity_ids and not relation_ids:
                return {"nodes": [], "relationships": [], "paths": []}
            
            # 构建Cypher查询 - 根据ID查找节点和关系
            cypher_parts = []
            
            if entity_ids:
                entity_ids_str = ', '.join(entity_ids)
                cypher_parts.append(f"""
                MATCH (n) WHERE ID(n) IN [{entity_ids_str}]
                OPTIONAL MATCH path = (n)-[*1..{depth}]-(m)
                RETURN n, relationships(path) as rels, nodes(path) as nodes, path
                LIMIT 100
                """)
            
            if relation_ids:
                relation_ids_str = ', '.join(relation_ids)
                cypher_parts.append(f"""
                MATCH ()-[r]-() WHERE ID(r) IN [{relation_ids_str}]
                MATCH (start)-[r]->(end)
                OPTIONAL MATCH path = (start)-[*1..{depth}]-(m)
                RETURN start, end, r, relationships(path) as rels, nodes(path) as nodes, path
                LIMIT 100
                """)
            
            all_nodes = []
            all_relationships = []
            all_paths = []
            
            for cypher_query in cypher_parts:
                result = self.neo4j_graph.run(cypher_query)
                
                for record in result:
                    # 处理节点
                    for node_key in ['n', 'start', 'end', 'm']:
                        if record.get(node_key):
                            node_data = {
                                "id": str(record[node_key].identity),
                                "labels": list(record[node_key].labels),
                                "properties": dict(record[node_key])
                            }
                            if node_data not in all_nodes:
                                all_nodes.append(node_data)
                    
                    # 处理路径中的节点
                    if record.get('nodes'):
                        for node in record['nodes']:
                            if node:
                                node_data = {
                                    "id": str(node.identity),
                                    "labels": list(node.labels),
                                    "properties": dict(node)
                                }
                                if node_data not in all_nodes:
                                    all_nodes.append(node_data)
                    
                    # 处理关系
                    if record.get('r'):
                        rel_data = {
                            "id": str(record['r'].identity),
                            "type": type(record['r']).__name__,
                            "properties": dict(record['r']),
                            "start_node_id": str(record['r'].start_node.identity),
                            "end_node_id": str(record['r'].end_node.identity)
                        }
                        if rel_data not in all_relationships:
                            all_relationships.append(rel_data)
                    
                    # 处理路径中的关系
                    if record.get('rels'):
                        for rel in record['rels']:
                            if rel:
                                rel_data = {
                                    "id": str(rel.identity),
                                    "type": type(rel).__name__,
                                    "properties": dict(rel),
                                    "start_node_id": str(rel.start_node.identity),
                                    "end_node_id": str(rel.end_node.identity)
                                }
                                if rel_data not in all_relationships:
                                    all_relationships.append(rel_data)
                    
                    # 处理路径
                    if record.get('path'):
                        path_data = {
                            "length": len(record['path']),
                            "nodes": [str(node.identity) for node in record['path'].nodes],
                            "relationships": [str(rel.identity) for rel in record['path'].relationships]
                        }
                        all_paths.append(path_data)
            
            return {
                "nodes": all_nodes,
                "relationships": all_relationships,
                "paths": all_paths,
                "total_nodes": len(all_nodes),
                "total_relationships": len(all_relationships),
                "total_paths": len(all_paths)
            }
            
        except Exception as e:
            logger.error(f"扩展子图失败: {e}")
            return {"nodes": [], "relationships": [], "paths": [], "error": str(e)}
    
    def generate_comprehensive_answer(self, query: str, search_results: Dict[str, Any]) -> str:
        """基于向量搜索和图谱扩展结果生成综合答案"""
        try:
            # 构建上下文
            context_parts = []
            
            # 添加向量搜索的实体信息
            if search_results.get("vector_search", {}).get("entities"):
                context_parts.append("🔍 相关实体:")
                for entity in search_results["vector_search"]["entities"][:3]:
                    similarity = 1 - entity.get('distance', 0)  # 转换为相似度
                    context_parts.append(f"- {entity['name']} ({entity['type']}) - 相似度: {similarity:.3f}")
                    if entity.get('description'):
                        context_parts.append(f"  描述: {entity['description']}")
            
            # 添加向量搜索的关系信息
            if search_results.get("vector_search", {}).get("relations"):
                context_parts.append("\n🔗 相关关系:")
                for relation in search_results["vector_search"]["relations"][:3]:
                    similarity = 1 - relation.get('distance', 0)
                    context_parts.append(f"- {relation['source']} → {relation['relation_type']} → {relation['target']} - 相似度: {similarity:.3f}")
                    if relation.get('description'):
                        context_parts.append(f"  描述: {relation['description']}")
            
            # 添加扩展子图信息
            subgraph = search_results.get("expanded_subgraph", {})
            if subgraph.get("nodes"):
                context_parts.append(f"\n📊 知识图谱扩展: 发现 {subgraph['total_nodes']} 个相关节点, {subgraph['total_relationships']} 个关系")
                
                # 添加一些关键节点信息
                for node in subgraph["nodes"][:5]:  # 只显示前5个节点
                    labels = ", ".join(node["labels"]) if node["labels"] else "未知类型"
                    name = node["properties"].get("name", "未知名称")
                    context_parts.append(f"  • {name} ({labels})")
            
            context = "\n".join(context_parts)
            
            # 构建提示词
            prompt = f"""基于以下混合知识图谱信息回答问题：

问题: {query}

知识来源:
{context}

请基于上述向量搜索和知识图谱扩展的信息，提供准确、详细、结构化的回答。
如果信息不足，请说明需要更多哪方面的信息。"""

            # 使用Ollama生成回答
            response = self.kg_builder.recognizer.ollama.generate(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return f"抱歉，生成答案时出现错误: {e}"
    
    def chat(self, query: str) -> Dict[str, Any]:
        """完整的对话流程"""
        try:
            # 1. 混合搜索
            search_results = self.hybrid_search_with_graph_expansion(query)
            
            # 2. 生成答案
            answer = self.generate_comprehensive_answer(query, search_results)
            
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

def main():
    """主函数"""
    try:
        # 创建混合向量图谱系统
        hybrid_system = HybridVectorGraphSystem()
        
        # 构建混合知识图谱
        file_path = "e:/Program/Project/rag-first/data/pajinsen.txt"
        logger.info("开始构建混合向量图谱...")
        
        stats = hybrid_system.build_hybrid_knowledge_graph(file_path)
        
        if stats:
            print("✅ 混合向量图谱构建完成！")
            print(f"📊 Neo4j: {stats['neo4j_entities']} 个实体, {stats['neo4j_relations']} 个关系")
            print(f"🔍 Weaviate: {stats['vector_entities']} 个实体向量, {stats['vector_relations']} 个关系向量")
            print("🔗 Neo4j Browser: http://localhost:7474")
            print("🔗 Weaviate: http://localhost:8080")
            
            # 测试混合搜索功能
            print("\n🔍 测试混合搜索...")
            test_queries = [
                "帕金森病的症状有哪些？",
                "多巴胺的作用机制",
                "神经退行性疾病的治疗方法"
            ]
            
            for query in test_queries:
                print(f"\n🔎 查询: '{query}'")
                result = hybrid_system.chat(query)
                print(f"📋 答案: {result['answer'][:200]}...")
                
                # 显示搜索统计
                search_stats = result.get('search_results', {})
                vector_entities = len(search_stats.get('vector_search', {}).get('entities', []))
                vector_relations = len(search_stats.get('vector_search', {}).get('relations', []))
                graph_nodes = search_stats.get('expanded_subgraph', {}).get('total_nodes', 0)
                graph_rels = search_stats.get('expanded_subgraph', {}).get('total_relationships', 0)
                
                print(f"📈 搜索统计: 向量({vector_entities}实体+{vector_relations}关系) + 图谱({graph_nodes}节点+{graph_rels}关系)")
        else:
            print("❌ 混合向量图谱构建失败")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        print(f"❌ 执行失败: {e}")

if __name__ == "__main__":
    main()