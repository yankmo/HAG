#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合RAG系统 - 集成意图识别、Neo4j知识图谱和向量存储
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knowledge.intent_recognition_neo4j import KnowledgeGraphBuilder, Entity, Relation
from src.knowledge.vector_storage import (
    OllamaEmbeddingClient, 
    WeaviateVectorStore, 
    VectorKnowledgeProcessor,
    VectorEntity,
    VectorRelation
)
import logging
from typing import List, Dict, Any, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridRAGSystem:
    """混合RAG系统"""
    
    def __init__(self):
        # 初始化各个组件
        self.kg_builder = KnowledgeGraphBuilder()
        self.embedding_client = OllamaEmbeddingClient(model="bgm-m3:latest")
        self.vector_store = WeaviateVectorStore()
        self.vector_processor = VectorKnowledgeProcessor(self.embedding_client, self.vector_store)
        
        logger.info("混合RAG系统初始化完成")
    
    def setup_vector_storage(self):
        """设置向量存储"""
        logger.info("设置Weaviate向量存储...")
        self.vector_store.setup_collections()
        logger.info("向量存储设置完成")
    
    def build_knowledge_graph(self, file_path: str, chunk_size: int = 500):
        """构建知识图谱（包含Neo4j和向量存储）"""
        try:
            logger.info(f"开始构建混合知识图谱: {file_path}")
            
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"文件大小: {len(content)} 字符")
            
            # 清空Neo4j数据库
            self.kg_builder.importer.clear_database()
            
            # 设置向量存储
            self.setup_vector_storage()
            
            # 分块处理文本
            chunks = self.kg_builder._split_text(content, chunk_size)
            logger.info(f"文本分为 {len(chunks)} 个块")
            
            all_entities = []
            all_relations = []
            all_vector_entities = []
            all_vector_relations = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"处理第 {i+1}/{len(chunks)} 块")
                
                try:
                    # 1. 使用Ollama提取实体和关系
                    entities, relations = self.kg_builder.recognizer.extract_entities_and_relations(chunk)
                    
                    if entities or relations:
                        # 2. 向量化实体和关系
                        vector_entities, vector_relations = self.vector_processor.process_entities_and_relations(
                            entities, relations, chunk
                        )
                        
                        # 收集所有数据
                        all_entities.extend(entities)
                        all_relations.extend(relations)
                        all_vector_entities.extend(vector_entities)
                        all_vector_relations.extend(vector_relations)
                        
                        logger.info(f"块 {i+1}: 提取到 {len(entities)} 个实体, {len(relations)} 个关系")
                        logger.info(f"块 {i+1}: 向量化 {len(vector_entities)} 个实体, {len(vector_relations)} 个关系")
                    else:
                        logger.warning(f"块 {i+1}: 未提取到任何实体或关系")
                        
                except Exception as e:
                    logger.error(f"处理块 {i+1} 时出错: {e}")
                    import traceback
                    logger.error(f"错误详情: {traceback.format_exc()}")
            
            # 去重
            unique_entities = self.kg_builder._deduplicate_entities(all_entities)
            unique_relations = self.kg_builder._deduplicate_relations(all_relations)
            
            logger.info(f"去重后: {len(unique_entities)} 个实体, {len(unique_relations)} 个关系")
            
            # 3. 存储到Neo4j
            logger.info("正在存储到Neo4j...")
            self.kg_builder.importer.import_entities_and_relations(unique_entities, unique_relations)
            
            # 4. 存储向量到Weaviate
            logger.info("正在存储向量到Weaviate...")
            success = self.vector_processor.store_vectors(all_vector_entities, all_vector_relations)
            
            if success:
                logger.info("向量存储成功")
            else:
                logger.error("向量存储失败")
            
            # 获取存储统计
            stats = self.vector_store.get_collection_stats()
            logger.info(f"Weaviate存储统计: {stats['entities']} 个实体向量, {stats['relations']} 个关系向量")
            
            logger.info("混合知识图谱构建完成")
            
            return {
                "neo4j_entities": len(unique_entities),
                "neo4j_relations": len(unique_relations),
                "vector_entities": stats['entities'],
                "vector_relations": stats['relations']
            }
            
        except Exception as e:
            logger.error(f"构建知识图谱失败: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            return None
    
    def hybrid_search(self, query: str, entity_limit: int = 5, relation_limit: int = 5) -> Dict[str, Any]:
        """混合搜索"""
        try:
            logger.info(f"执行混合搜索: {query}")
            
            # 1. 向量搜索
            search_results = self.vector_processor.search_knowledge(
                query, entity_limit, relation_limit
            )
            
            # 2. 提取相关实体名称用于Neo4j图遍历
            relevant_entity_names = []
            for entity in search_results["entities"]:
                relevant_entity_names.append(entity["name"])
            
            # 3. 在Neo4j中扩展子图（这里简化实现，实际可以更复杂）
            subgraph_info = self._expand_subgraph_from_neo4j(relevant_entity_names)
            
            return {
                "query": query,
                "vector_search": search_results,
                "subgraph": subgraph_info,
                "relevant_entities": relevant_entity_names
            }
            
        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            return {"error": str(e)}
    
    def _expand_subgraph_from_neo4j(self, entity_names: List[str]) -> Dict[str, Any]:
        """从Neo4j扩展子图"""
        try:
            if not entity_names:
                return {"nodes": [], "relationships": []}
            
            # 构建Cypher查询
            entity_list = "', '".join(entity_names)
            cypher_query = f"""
            MATCH (n)
            WHERE n.name IN ['{entity_list}']
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n, r, m
            LIMIT 50
            """
            
            result = self.kg_builder.importer.graph.run(cypher_query)
            
            nodes = []
            relationships = []
            
            for record in result:
                # 添加节点
                if record["n"]:
                    node_data = dict(record["n"])
                    node_data["labels"] = list(record["n"].labels)
                    if node_data not in nodes:
                        nodes.append(node_data)
                
                if record["m"]:
                    node_data = dict(record["m"])
                    node_data["labels"] = list(record["m"].labels)
                    if node_data not in nodes:
                        nodes.append(node_data)
                
                # 添加关系
                if record["r"]:
                    rel_data = {
                        "type": type(record["r"]).__name__,
                        "properties": dict(record["r"]),
                        "start_node": dict(record["n"]) if record["n"] else None,
                        "end_node": dict(record["m"]) if record["m"] else None
                    }
                    relationships.append(rel_data)
            
            return {
                "nodes": nodes,
                "relationships": relationships,
                "total_nodes": len(nodes),
                "total_relationships": len(relationships)
            }
            
        except Exception as e:
            logger.error(f"扩展子图失败: {e}")
            return {"nodes": [], "relationships": [], "error": str(e)}
    
    def generate_answer(self, query: str, search_results: Dict[str, Any]) -> str:
        """基于搜索结果生成答案"""
        try:
            # 构建上下文
            context_parts = []
            
            # 添加向量搜索的实体信息
            if search_results.get("vector_search", {}).get("entities"):
                context_parts.append("相关实体:")
                for entity in search_results["vector_search"]["entities"][:3]:
                    context_parts.append(f"- {entity['name']} ({entity['type']})")
            
            # 添加向量搜索的关系信息
            if search_results.get("vector_search", {}).get("relations"):
                context_parts.append("相关关系:")
                for relation in search_results["vector_search"]["relations"][:3]:
                    context_parts.append(f"- {relation['source']} {relation['relation_type']} {relation['target']}")
            
            # 添加子图信息
            if search_results.get("subgraph", {}).get("nodes"):
                context_parts.append(f"知识图谱中找到 {len(search_results['subgraph']['nodes'])} 个相关节点")
            
            context = "\n".join(context_parts)
            
            # 构建提示词
            prompt = f"""基于以下知识图谱信息回答问题：

问题: {query}

相关知识:
{context}

请基于上述信息提供准确、详细的回答。如果信息不足，请说明。"""

            # 使用Ollama生成回答
            response = self.kg_builder.ollama.generate(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return f"抱歉，生成答案时出现错误: {e}"
    
    def chat(self, query: str) -> Dict[str, Any]:
        """完整的对话流程"""
        try:
            # 1. 混合搜索
            search_results = self.hybrid_search(query)
            
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

def main():
    """主函数"""
    try:
        # 创建混合RAG系统
        rag_system = HybridRAGSystem()
        
        # 构建知识图谱
        file_path = "e:/Program/Project/rag-first/data/pajinsen.txt"
        logger.info("开始构建混合知识图谱...")
        
        stats = rag_system.build_knowledge_graph(file_path)
        
        if stats:
            print("✅ 混合知识图谱构建完成！")
            print(f"📊 Neo4j: {stats['neo4j_entities']} 个实体, {stats['neo4j_relations']} 个关系")
            print(f"🔍 Weaviate: {stats['vector_entities']} 个实体向量, {stats['vector_relations']} 个关系向量")
            print("🔗 Neo4j Browser: http://localhost:7474")
            print("🔗 Weaviate: http://localhost:8080")
            
            # 测试搜索功能
            print("\n🔍 测试混合搜索...")
            test_query = "帕金森病的症状有哪些？"
            search_results = rag_system.hybrid_search(test_query)
            print(f"搜索查询: {test_query}")
            print(f"找到 {len(search_results.get('vector_search', {}).get('entities', []))} 个相关实体")
            print(f"找到 {len(search_results.get('vector_search', {}).get('relations', []))} 个相关关系")
            
        else:
            print("❌ 知识图谱构建失败")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        print(f"❌ 执行失败: {e}")

if __name__ == "__main__":
    from datetime import datetime
    main()