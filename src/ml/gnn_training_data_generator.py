#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN训练数据生成器
从Neo4j知识图谱中提取训练样本，用于训练图神经网络权重学习模型
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import random

# Neo4j连接
try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

logger = logging.getLogger(__name__)

@dataclass
class TrainingNode:
    """训练节点数据"""
    id: str
    label: str
    properties: Dict[str, Any]
    embedding: Optional[List[float]] = None
    degree: int = 0
    centrality: float = 0.0

@dataclass
class TrainingEdge:
    """训练边数据"""
    source: str
    target: str
    relation_type: str
    properties: Dict[str, Any]
    weight: float = 1.0
    importance: float = 0.0

@dataclass
class TrainingQuery:
    """训练查询数据"""
    query_text: str
    intent: str
    entities: List[str]
    relations: List[str]
    expected_nodes: List[str]
    expected_edges: List[str]
    relevance_scores: Dict[str, float]
    timestamp: datetime

@dataclass
class TrainingDataset:
    """训练数据集"""
    nodes: List[TrainingNode]
    edges: List[TrainingEdge]
    queries: List[TrainingQuery]
    node_features: np.ndarray
    edge_features: np.ndarray
    adjacency_matrix: np.ndarray
    labels: np.ndarray
    metadata: Dict[str, Any]

class GNNTrainingDataGenerator:
    """GNN训练数据生成器"""
    
    def __init__(self, 
                 neo4j_uri: str,
                 neo4j_user: str,
                 neo4j_password: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化训练数据生成器
        
        Args:
            neo4j_uri: Neo4j数据库URI
            neo4j_user: 用户名
            neo4j_password: 密码
            config: 配置参数
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.config = config or {}
        
        # 配置参数
        self.max_nodes = self.config.get('max_nodes', 10000)
        self.max_edges = self.config.get('max_edges', 50000)
        self.min_degree = self.config.get('min_degree', 1)
        self.sample_ratio = self.config.get('sample_ratio', 0.1)
        self.feature_dim = self.config.get('feature_dim', 128)
        
        # Neo4j驱动
        self.driver = None
        if GraphDatabase:
            try:
                self.driver = GraphDatabase.driver(
                    neo4j_uri, 
                    auth=(neo4j_user, neo4j_password)
                )
                logger.info("Neo4j连接已建立")
            except Exception as e:
                logger.error(f"Neo4j连接失败: {e}")
        else:
            logger.warning("Neo4j驱动未安装，使用模拟数据")
        
        # 缓存
        self.node_cache = {}
        self.edge_cache = {}
        self.query_cache = []
    
    def __del__(self):
        """析构函数"""
        if self.driver:
            self.driver.close()
    
    def extract_graph_structure(self) -> Tuple[List[TrainingNode], List[TrainingEdge]]:
        """从Neo4j提取图结构"""
        try:
            if not self.driver:
                return self._generate_mock_graph_structure()
            
            nodes = []
            edges = []
            
            with self.driver.session() as session:
                # 提取节点
                node_query = """
                MATCH (n)
                WITH n, size((n)--()) as degree
                WHERE degree >= $min_degree
                RETURN id(n) as node_id, labels(n) as labels, properties(n) as props, degree
                ORDER BY degree DESC
                LIMIT $max_nodes
                """
                
                result = session.run(node_query, 
                                   min_degree=self.min_degree, 
                                   max_nodes=self.max_nodes)
                
                for record in result:
                    node = TrainingNode(
                        id=str(record['node_id']),
                        label=record['labels'][0] if record['labels'] else 'Unknown',
                        properties=dict(record['props']),
                        degree=record['degree']
                    )
                    nodes.append(node)
                    self.node_cache[node.id] = node
                
                # 提取边
                edge_query = """
                MATCH (a)-[r]->(b)
                WHERE id(a) IN $node_ids AND id(b) IN $node_ids
                RETURN id(a) as source, id(b) as target, type(r) as rel_type, properties(r) as props
                LIMIT $max_edges
                """
                
                node_ids = [int(node.id) for node in nodes]
                result = session.run(edge_query, 
                                   node_ids=node_ids, 
                                   max_edges=self.max_edges)
                
                for record in result:
                    edge = TrainingEdge(
                        source=str(record['source']),
                        target=str(record['target']),
                        relation_type=record['rel_type'],
                        properties=dict(record['props'])
                    )
                    edges.append(edge)
                    self.edge_cache[f"{edge.source}-{edge.target}"] = edge
            
            logger.info(f"提取图结构完成: {len(nodes)}个节点, {len(edges)}条边")
            return nodes, edges
            
        except Exception as e:
            logger.error(f"提取图结构失败: {e}")
            return self._generate_mock_graph_structure()
    
    def _generate_mock_graph_structure(self) -> Tuple[List[TrainingNode], List[TrainingEdge]]:
        """生成模拟图结构数据"""
        logger.info("生成模拟图结构数据")
        
        # 生成节点
        nodes = []
        node_types = ['Person', 'Organization', 'Location', 'Concept', 'Event']
        
        for i in range(min(1000, self.max_nodes)):
            node = TrainingNode(
                id=str(i),
                label=random.choice(node_types),
                properties={
                    'name': f'Node_{i}',
                    'created_at': datetime.now().isoformat(),
                    'importance': random.uniform(0, 1)
                },
                degree=random.randint(1, 20)
            )
            nodes.append(node)
            self.node_cache[node.id] = node
        
        # 生成边
        edges = []
        relation_types = ['RELATED_TO', 'PART_OF', 'LOCATED_IN', 'WORKS_FOR', 'KNOWS']
        
        for i in range(min(5000, self.max_edges)):
            source = random.choice(nodes)
            target = random.choice(nodes)
            
            if source.id != target.id:
                edge = TrainingEdge(
                    source=source.id,
                    target=target.id,
                    relation_type=random.choice(relation_types),
                    properties={
                        'strength': random.uniform(0, 1),
                        'created_at': datetime.now().isoformat()
                    },
                    weight=random.uniform(0.1, 1.0)
                )
                edges.append(edge)
                self.edge_cache[f"{edge.source}-{edge.target}"] = edge
        
        return nodes, edges
    
    def extract_query_patterns(self, days: int = 30) -> List[TrainingQuery]:
        """提取查询模式数据"""
        try:
            if not self.driver:
                return self._generate_mock_query_patterns()
            
            queries = []
            
            # 这里应该从实际的查询日志中提取
            # 由于没有查询日志表，我们生成一些模拟查询
            queries = self._generate_mock_query_patterns()
            
            logger.info(f"提取查询模式完成: {len(queries)}个查询")
            return queries
            
        except Exception as e:
            logger.error(f"提取查询模式失败: {e}")
            return self._generate_mock_query_patterns()
    
    def _generate_mock_query_patterns(self) -> List[TrainingQuery]:
        """生成模拟查询模式"""
        logger.info("生成模拟查询模式")
        
        queries = []
        query_templates = [
            "查找与{entity}相关的信息",
            "{entity}的详细资料",
            "{entity}和{entity2}的关系",
            "关于{concept}的所有内容",
            "{location}发生的{event}"
        ]
        
        intents = ['search', 'detail', 'relation', 'concept', 'event']
        entities = ['张三', '北京', '人工智能', '公司', '会议']
        
        for i in range(100):
            template = random.choice(query_templates)
            intent = random.choice(intents)
            
            # 填充模板
            if '{entity2}' in template:
                query_text = template.format(
                    entity=random.choice(entities),
                    entity2=random.choice(entities)
                )
                query_entities = random.sample(entities, 2)
            elif '{entity}' in template:
                entity = random.choice(entities)
                query_text = template.format(entity=entity)
                query_entities = [entity]
            else:
                query_text = template.format(
                    concept=random.choice(['技术', '管理', '创新']),
                    location=random.choice(['上海', '深圳', '广州']),
                    event=random.choice(['会议', '展览', '培训'])
                )
                query_entities = random.sample(entities, 1)
            
            query = TrainingQuery(
                query_text=query_text,
                intent=intent,
                entities=query_entities,
                relations=['RELATED_TO', 'PART_OF'],
                expected_nodes=random.sample(list(self.node_cache.keys()), 
                                           min(5, len(self.node_cache))),
                expected_edges=random.sample(list(self.edge_cache.keys()), 
                                           min(3, len(self.edge_cache))),
                relevance_scores={
                    node_id: random.uniform(0, 1) 
                    for node_id in random.sample(list(self.node_cache.keys()), 
                                                min(10, len(self.node_cache)))
                },
                timestamp=datetime.now() - timedelta(days=random.randint(0, 30))
            )
            queries.append(query)
        
        return queries
    
    def calculate_node_features(self, nodes: List[TrainingNode]) -> np.ndarray:
        """计算节点特征"""
        try:
            features = []
            
            for node in nodes:
                # 基础特征
                feature_vector = [
                    node.degree,  # 度数
                    len(node.properties),  # 属性数量
                    hash(node.label) % 1000 / 1000.0,  # 标签哈希
                ]
                
                # 属性特征
                importance = node.properties.get('importance', 0.0)
                if isinstance(importance, (int, float)):
                    feature_vector.append(importance)
                else:
                    feature_vector.append(0.0)
                
                # 填充到固定维度
                while len(feature_vector) < self.feature_dim:
                    feature_vector.append(0.0)
                
                # 截断到固定维度
                feature_vector = feature_vector[:self.feature_dim]
                
                features.append(feature_vector)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"计算节点特征失败: {e}")
            return np.zeros((len(nodes), self.feature_dim), dtype=np.float32)
    
    def calculate_edge_features(self, edges: List[TrainingEdge]) -> np.ndarray:
        """计算边特征"""
        try:
            features = []
            
            for edge in edges:
                # 基础特征
                feature_vector = [
                    edge.weight,  # 权重
                    edge.importance,  # 重要性
                    hash(edge.relation_type) % 1000 / 1000.0,  # 关系类型哈希
                    len(edge.properties),  # 属性数量
                ]
                
                # 属性特征
                strength = edge.properties.get('strength', 0.0)
                if isinstance(strength, (int, float)):
                    feature_vector.append(strength)
                else:
                    feature_vector.append(0.0)
                
                # 填充到固定维度
                while len(feature_vector) < 32:  # 边特征维度较小
                    feature_vector.append(0.0)
                
                # 截断到固定维度
                feature_vector = feature_vector[:32]
                
                features.append(feature_vector)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"计算边特征失败: {e}")
            return np.zeros((len(edges), 32), dtype=np.float32)
    
    def build_adjacency_matrix(self, nodes: List[TrainingNode], edges: List[TrainingEdge]) -> np.ndarray:
        """构建邻接矩阵"""
        try:
            n_nodes = len(nodes)
            node_id_to_idx = {node.id: i for i, node in enumerate(nodes)}
            
            # 初始化邻接矩阵
            adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
            
            # 填充邻接矩阵
            for edge in edges:
                if edge.source in node_id_to_idx and edge.target in node_id_to_idx:
                    src_idx = node_id_to_idx[edge.source]
                    tgt_idx = node_id_to_idx[edge.target]
                    adj_matrix[src_idx, tgt_idx] = edge.weight
                    # 如果是无向图，也设置反向边
                    adj_matrix[tgt_idx, src_idx] = edge.weight
            
            return adj_matrix
            
        except Exception as e:
            logger.error(f"构建邻接矩阵失败: {e}")
            return np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    
    def generate_labels(self, nodes: List[TrainingNode], queries: List[TrainingQuery]) -> np.ndarray:
        """生成训练标签"""
        try:
            n_nodes = len(nodes)
            node_id_to_idx = {node.id: i for i, node in enumerate(nodes)}
            
            # 基于查询相关性生成标签
            labels = np.zeros(n_nodes, dtype=np.float32)
            
            for query in queries:
                for node_id, relevance in query.relevance_scores.items():
                    if node_id in node_id_to_idx:
                        idx = node_id_to_idx[node_id]
                        labels[idx] = max(labels[idx], relevance)
            
            return labels
            
        except Exception as e:
            logger.error(f"生成标签失败: {e}")
            return np.zeros(len(nodes), dtype=np.float32)
    
    def generate_training_dataset(self, 
                                output_path: Optional[str] = None) -> TrainingDataset:
        """生成完整的训练数据集"""
        try:
            logger.info("开始生成训练数据集")
            
            # 1. 提取图结构
            nodes, edges = self.extract_graph_structure()
            
            # 2. 提取查询模式
            queries = self.extract_query_patterns()
            
            # 3. 计算特征
            node_features = self.calculate_node_features(nodes)
            edge_features = self.calculate_edge_features(edges)
            
            # 4. 构建邻接矩阵
            adjacency_matrix = self.build_adjacency_matrix(nodes, edges)
            
            # 5. 生成标签
            labels = self.generate_labels(nodes, queries)
            
            # 6. 构建数据集
            dataset = TrainingDataset(
                nodes=nodes,
                edges=edges,
                queries=queries,
                node_features=node_features,
                edge_features=edge_features,
                adjacency_matrix=adjacency_matrix,
                labels=labels,
                metadata={
                    'n_nodes': len(nodes),
                    'n_edges': len(edges),
                    'n_queries': len(queries),
                    'feature_dim': self.feature_dim,
                    'generated_at': datetime.now().isoformat(),
                    'config': self.config
                }
            )
            
            # 7. 保存数据集
            if output_path:
                self.save_dataset(dataset, output_path)
            
            logger.info(f"训练数据集生成完成: {len(nodes)}个节点, {len(edges)}条边, {len(queries)}个查询")
            return dataset
            
        except Exception as e:
            logger.error(f"生成训练数据集失败: {e}")
            raise
    
    def save_dataset(self, dataset: TrainingDataset, output_path: str):
        """保存训练数据集"""
        try:
            import pickle
            import os
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存数据集
            with open(output_path, 'wb') as f:
                pickle.dump(dataset, f)
            
            # 保存元数据
            metadata_path = output_path.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(dataset.metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"训练数据集已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存训练数据集失败: {e}")
            raise
    
    def load_dataset(self, input_path: str) -> TrainingDataset:
        """加载训练数据集"""
        try:
            import pickle
            
            with open(input_path, 'rb') as f:
                dataset = pickle.load(f)
            
            logger.info(f"训练数据集已加载: {input_path}")
            return dataset
            
        except Exception as e:
            logger.error(f"加载训练数据集失败: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据生成器统计信息"""
        return {
            'cached_nodes': len(self.node_cache),
            'cached_edges': len(self.edge_cache),
            'cached_queries': len(self.query_cache),
            'config': self.config,
            'neo4j_connected': self.driver is not None
        }


def main():
    """主函数 - 用于测试"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建数据生成器
    generator = GNNTrainingDataGenerator(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        config={
            'max_nodes': 1000,
            'max_edges': 5000,
            'feature_dim': 128
        }
    )
    
    # 生成训练数据集
    dataset = generator.generate_training_dataset(
        output_path="training_data/gnn_dataset.pkl"
    )
    
    # 打印统计信息
    print(f"数据集统计:")
    print(f"  节点数: {len(dataset.nodes)}")
    print(f"  边数: {len(dataset.edges)}")
    print(f"  查询数: {len(dataset.queries)}")
    print(f"  节点特征维度: {dataset.node_features.shape}")
    print(f"  边特征维度: {dataset.edge_features.shape}")
    print(f"  邻接矩阵维度: {dataset.adjacency_matrix.shape}")
    print(f"  标签维度: {dataset.labels.shape}")


if __name__ == "__main__":
    main()