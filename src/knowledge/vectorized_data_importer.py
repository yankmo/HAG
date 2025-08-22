#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量化数据导入器
将Neo4j图谱数据向量化后存储到Weaviate，实现真正的向量化图谱检索
"""

import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import json
from py2neo import Graph

from src.knowledge.intent_recognition_neo4j import KnowledgeGraphBuilder, Entity, Relation
from src.knowledge.vectorized_graph_storage import VectorizedGraphStorage, GraphNode, GraphRelation
# from ..services.embedding_service import OllamaEmbeddingService  # 延迟导入
import os

logger = logging.getLogger(__name__)

@dataclass
class ImportStats:
    """导入统计信息"""
    total_entities: int = 0
    total_relations: int = 0
    vectorized_nodes: int = 0
    vectorized_relations: int = 0
    failed_nodes: int = 0
    failed_relations: int = 0
    processing_time: float = 0.0

class VectorizedDataImporter:
    """向量化数据导入器"""
    
    def __init__(self):
        self.kg_builder = KnowledgeGraphBuilder()
        self.vector_storage = VectorizedGraphStorage()
        self.embedding_service = None  # 延迟初始化
        
        # Neo4j连接
        try:
            neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
            neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
            neo4j_password = os.getenv('NEO4J_PASSWORD', 'hrx274700')
            
            self.neo4j_graph = Graph(
                neo4j_uri,
                auth=(neo4j_username, neo4j_password)
            )
            logger.info("Neo4j连接成功")
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            self.neo4j_graph = None
        
        logger.info("向量化数据导入器初始化完成")
    
    def _get_embedding_service(self):
        """延迟初始化嵌入服务"""
        if self.embedding_service is None:
            try:
                from src.services.embedding_service import OllamaEmbeddingService
                self.embedding_service = OllamaEmbeddingService()
            except ImportError as e:
                logger.error(f"无法导入嵌入服务: {e}")
                # 创建一个模拟的嵌入服务
                class MockEmbeddingService:
                    def embed_text(self, text):
                        return [0.1] * 384  # 返回固定维度的模拟向量
                self.embedding_service = MockEmbeddingService()
        return self.embedding_service
    
    def process_and_vectorize_file(self, file_path: str, chunk_size: int = 500) -> ImportStats:
        """处理文件并向量化存储"""
        import time
        start_time = time.time()
        
        stats = ImportStats()
        
        try:
            # 1. 使用原有逻辑提取实体和关系
            logger.info(f"开始处理文件: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"文件大小: {len(content)} 字符")
            
            # 分块处理文本
            chunks = self._split_text(content, chunk_size)
            logger.info(f"文本分为 {len(chunks)} 个块")
            
            all_entities = []
            all_relations = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"处理第 {i+1}/{len(chunks)} 块")
                
                try:
                    entities, relations = self.kg_builder.recognizer.extract_entities_and_relations(chunk)
                    
                    # 为实体和关系添加源文本信息
                    for entity in entities:
                        entity.properties = entity.properties or {}
                        entity.properties['source_text'] = chunk[:200]  # 保存前200字符作为源文本
                    
                    for relation in relations:
                        relation.properties = relation.properties or {}
                        relation.properties['source_text'] = chunk[:200]
                    
                    all_entities.extend(entities)
                    all_relations.extend(relations)
                    
                    logger.info(f"块 {i+1}: 提取到 {len(entities)} 个实体, {len(relations)} 个关系")
                            
                except Exception as e:
                    logger.error(f"处理块 {i+1} 时出错: {e}")
            
            # 2. 去重
            unique_entities = self._deduplicate_entities(all_entities)
            unique_relations = self._deduplicate_relations(all_relations)
            
            stats.total_entities = len(unique_entities)
            stats.total_relations = len(unique_relations)
            
            logger.info(f"去重后: {stats.total_entities} 个实体, {stats.total_relations} 个关系")
            
            # 3. 导入到Neo4j（保持原有功能）
            if self.neo4j_graph:
                logger.info("导入数据到Neo4j...")
                self.kg_builder.importer.clear_database()
                self.kg_builder.importer.import_entities_and_relations(unique_entities, unique_relations)
            
            # 4. 向量化并存储到Weaviate
            logger.info("开始向量化存储...")
            
            # 转换实体为GraphNode并向量化存储
            graph_nodes = self._convert_entities_to_graph_nodes(unique_entities)
            vectorized_nodes = self._vectorize_and_store_nodes(graph_nodes)
            stats.vectorized_nodes = vectorized_nodes
            stats.failed_nodes = len(graph_nodes) - vectorized_nodes
            
            # 转换关系为GraphRelation并向量化存储
            graph_relations = self._convert_relations_to_graph_relations(unique_relations)
            vectorized_relations = self._vectorize_and_store_relations(graph_relations)
            stats.vectorized_relations = vectorized_relations
            stats.failed_relations = len(graph_relations) - vectorized_relations
            
            stats.processing_time = time.time() - start_time
            
            logger.info(f"向量化导入完成: {stats.vectorized_nodes}/{stats.total_entities} 节点, {stats.vectorized_relations}/{stats.total_relations} 关系")
            logger.info(f"处理时间: {stats.processing_time:.2f} 秒")
            
            return stats
            
        except Exception as e:
            logger.error(f"向量化导入失败: {e}")
            stats.processing_time = time.time() - start_time
            return stats
    
    def import_from_neo4j(self) -> ImportStats:
        """从现有Neo4j数据库导入并向量化"""
        import time
        start_time = time.time()
        
        stats = ImportStats()
        
        if not self.neo4j_graph:
            logger.error("Neo4j连接不可用")
            return stats
        
        try:
            # 1. 从Neo4j读取所有节点
            logger.info("从Neo4j读取节点...")
            nodes_query = "MATCH (n) RETURN n"
            node_results = self.neo4j_graph.run(nodes_query)
            
            graph_nodes = []
            for record in node_results:
                node = record['n']
                
                # 获取节点标签（类型）
                node_labels = list(node.labels)
                node_type = node_labels[0] if node_labels else "Unknown"
                
                # 获取节点属性
                node_props = dict(node)
                node_name = node_props.get('name', str(node.identity))
                
                # 创建GraphNode
                graph_node = GraphNode(
                    name=node_name,
                    type=node_type,
                    description=node_props.get('description', ''),
                    properties=node_props,
                    source_text=node_props.get('source_text', ''),
                    neo4j_id=str(node.identity)
                )
                graph_nodes.append(graph_node)
            
            stats.total_entities = len(graph_nodes)
            logger.info(f"读取到 {stats.total_entities} 个节点")
            
            # 2. 从Neo4j读取所有关系
            logger.info("从Neo4j读取关系...")
            relations_query = "MATCH (a)-[r]->(b) RETURN a.name as source, type(r) as rel_type, b.name as target, r, id(r) as rel_id"
            relation_results = self.neo4j_graph.run(relations_query)
            
            graph_relations = []
            for record in relation_results:
                source_name = record['source'] or f"node_{record.get('a_id', 'unknown')}"
                target_name = record['target'] or f"node_{record.get('b_id', 'unknown')}"
                rel_type = record['rel_type']
                rel_props = dict(record['r']) if record['r'] else {}
                rel_id = record['rel_id']
                
                # 创建GraphRelation
                graph_relation = GraphRelation(
                    source_node=source_name,
                    target_node=target_name,
                    relation_type=rel_type,
                    description=rel_props.get('description', ''),
                    properties=rel_props,
                    source_text=rel_props.get('source_text', ''),
                    neo4j_id=str(rel_id)
                )
                graph_relations.append(graph_relation)
            
            stats.total_relations = len(graph_relations)
            logger.info(f"读取到 {stats.total_relations} 个关系")
            
            # 3. 向量化并存储节点
            vectorized_nodes = self._vectorize_and_store_nodes(graph_nodes)
            stats.vectorized_nodes = vectorized_nodes
            stats.failed_nodes = len(graph_nodes) - vectorized_nodes
            
            # 4. 向量化并存储关系
            vectorized_relations = self._vectorize_and_store_relations(graph_relations)
            stats.vectorized_relations = vectorized_relations
            stats.failed_relations = len(graph_relations) - vectorized_relations
            
            stats.processing_time = time.time() - start_time
            
            logger.info(f"Neo4j数据向量化完成: {stats.vectorized_nodes}/{stats.total_entities} 节点, {stats.vectorized_relations}/{stats.total_relations} 关系")
            logger.info(f"处理时间: {stats.processing_time:.2f} 秒")
            
            return stats
            
        except Exception as e:
            logger.error(f"从Neo4j导入失败: {e}")
            stats.processing_time = time.time() - start_time
            return stats
    
    def _convert_entities_to_graph_nodes(self, entities: List[Entity]) -> List[GraphNode]:
        """将Entity转换为GraphNode"""
        graph_nodes = []
        
        for entity in entities:
            # 生成描述
            description = entity.properties.get('description', '') if entity.properties else ''
            source_text = entity.properties.get('source_text', '') if entity.properties else ''
            
            graph_node = GraphNode(
                name=entity.name,
                type=entity.type,
                description=description,
                properties=entity.properties or {},
                source_text=source_text,
                neo4j_id=""  # 将在Neo4j导入后更新
            )
            graph_nodes.append(graph_node)
        
        return graph_nodes
    
    def _convert_relations_to_graph_relations(self, relations: List[Relation]) -> List[GraphRelation]:
        """将Relation转换为GraphRelation"""
        graph_relations = []
        
        for relation in relations:
            # 生成描述
            description = relation.properties.get('description', '') if relation.properties else ''
            source_text = relation.properties.get('source_text', '') if relation.properties else ''
            
            graph_relation = GraphRelation(
                source_node=relation.source,
                target_node=relation.target,
                relation_type=relation.relation_type,
                description=description,
                properties=relation.properties or {},
                source_text=source_text,
                neo4j_id=""  # 将在Neo4j导入后更新
            )
            graph_relations.append(graph_relation)
        
        return graph_relations
    
    def _vectorize_and_store_nodes(self, graph_nodes: List[GraphNode]) -> int:
        """向量化并存储节点"""
        try:
            # 批量向量化
            for node in graph_nodes:
                if not node.vector:
                    node_text = self._generate_node_text(node)
                    node.vector = self._get_embedding_service().embed_text(node_text)
            
            # 存储到Weaviate
            success = self.vector_storage.vectorize_and_store_nodes(graph_nodes)
            
            if success:
                return len([node for node in graph_nodes if node.vector])
            else:
                return 0
                
        except Exception as e:
            logger.error(f"节点向量化存储失败: {e}")
            return 0
    
    def _vectorize_and_store_relations(self, graph_relations: List[GraphRelation]) -> int:
        """向量化并存储关系"""
        try:
            # 批量向量化
            for relation in graph_relations:
                if not relation.vector:
                    relation_text = self._generate_relation_text(relation)
                    relation.vector = self._get_embedding_service().embed_text(relation_text)
            
            # 存储到Weaviate
            success = self.vector_storage.vectorize_and_store_relations(graph_relations)
            
            if success:
                return len([relation for relation in graph_relations if relation.vector])
            else:
                return 0
                
        except Exception as e:
            logger.error(f"关系向量化存储失败: {e}")
            return 0
    
    def _generate_node_text(self, node: GraphNode) -> str:
        """生成节点的文本表示"""
        parts = []
        
        # 节点名称和类型
        parts.append(f"节点: {node.name}")
        parts.append(f"类型: {node.type}")
        
        # 描述
        if node.description:
            parts.append(f"描述: {node.description}")
        
        # 属性
        if node.properties:
            for key, value in node.properties.items():
                if key not in ['source_text']:  # 排除源文本避免重复
                    parts.append(f"{key}: {value}")
        
        # 原始文本
        if node.source_text:
            parts.append(f"来源: {node.source_text}")
        
        return " ".join(parts)
    
    def _generate_relation_text(self, relation: GraphRelation) -> str:
        """生成关系的文本表示"""
        parts = []
        
        # 关系三元组
        parts.append(f"关系: {relation.source_node} {relation.relation_type} {relation.target_node}")
        
        # 描述
        if relation.description:
            parts.append(f"描述: {relation.description}")
        
        # 属性
        if relation.properties:
            for key, value in relation.properties.items():
                if key not in ['source_text']:  # 排除源文本避免重复
                    parts.append(f"{key}: {value}")
        
        # 原始文本
        if relation.source_text:
            parts.append(f"来源: {relation.source_text}")
        
        return " ".join(parts)
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """分割文本"""
        # 复用原有的文本分割逻辑
        return self.kg_builder._split_text(text, chunk_size)
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """去重实体"""
        return self.kg_builder._deduplicate_entities(entities)
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """去重关系"""
        return self.kg_builder._deduplicate_relations(relations)
    
    def import_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中导入数据到向量化图谱"""
        try:
            # 使用现有的KnowledgeGraphBuilder提取实体和关系
            entities, relations = self.kg_builder.recognizer.extract_entities_and_relations(text)
            
            # 为实体和关系添加源文本信息
            for entity in entities:
                entity.properties = entity.properties or {}
                entity.properties['source_text'] = text[:200]  # 保存前200字符作为源文本
            
            for relation in relations:
                relation.properties = relation.properties or {}
                relation.properties['source_text'] = text[:200]
            
            # 去重处理
            unique_entities = self._deduplicate_entities(entities)
            unique_relations = self._deduplicate_relations(relations)
            
            # 转换为图节点和关系
            graph_nodes = self._convert_entities_to_graph_nodes(unique_entities)
            graph_relations = self._convert_relations_to_graph_relations(unique_relations)
            
            # 向量化并存储
            stored_nodes = self._vectorize_and_store_nodes(graph_nodes)
            stored_relations = self._vectorize_and_store_relations(graph_relations)
            
            logger.info(f"成功导入 {stored_nodes} 个节点和 {stored_relations} 个关系")
            
            return {
                "success": True,
                "nodes_imported": stored_nodes,
                "relations_imported": stored_relations
            }
            
        except Exception as e:
            logger.error(f"从文本导入数据失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "nodes_imported": 0,
                "relations_imported": 0
            }
    
    def get_import_stats(self) -> Dict[str, Any]:
        """获取导入统计信息"""
        storage_stats = self.vector_storage.get_stats()
        
        return {
            "vector_storage_stats": storage_stats,
            "neo4j_connected": self.neo4j_graph is not None,
            "embedding_service": type(self._get_embedding_service()).__name__
        }

def main():
    """主函数"""
    import sys
    import os
    
    try:
        # 创建向量化数据导入器
        importer = VectorizedDataImporter()
        
        # 从命令行参数获取操作类型和文件路径
        if len(sys.argv) < 2:
            logger.info("使用方法:")
            logger.info("  python vectorized_data_importer.py file [文件路径]  # 处理文件并向量化")
            logger.info("  python vectorized_data_importer.py neo4j           # 从Neo4j导入并向量化")
            return
        
        operation = sys.argv[1].lower()
        
        if operation == "file":
            # 处理文件
            if len(sys.argv) < 3:
                file_path = os.path.join("data", "pajinsen.txt")
            else:
                file_path = sys.argv[2]
            
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return
            
            logger.info(f"开始处理文件: {file_path}")
            stats = importer.process_and_vectorize_file(file_path)
            
        elif operation == "neo4j":
            # 从Neo4j导入
            logger.info("开始从Neo4j导入数据")
            stats = importer.import_from_neo4j()
            
        else:
            logger.error(f"未知操作: {operation}")
            return
        
        # 打印统计信息
        logger.info("=== 导入统计 ===")
        logger.info(f"总实体数: {stats.total_entities}")
        logger.info(f"总关系数: {stats.total_relations}")
        logger.info(f"成功向量化节点: {stats.vectorized_nodes}")
        logger.info(f"成功向量化关系: {stats.vectorized_relations}")
        logger.info(f"失败节点: {stats.failed_nodes}")
        logger.info(f"失败关系: {stats.failed_relations}")
        logger.info(f"处理时间: {stats.processing_time:.2f} 秒")
        
        logger.info("向量化数据导入完成！")
        logger.info("现在可以使用向量化图谱检索服务进行查询")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    main()