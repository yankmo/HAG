import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class NodeFeatures:
    """节点特征"""
    entity_id: str
    entity_type: str
    embedding: np.ndarray
    degree: int = 0
    centrality: float = 0.0
    frequency: int = 0
    
@dataclass
class EdgeFeatures:
    """边特征"""
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    frequency: int = 1
    confidence: float = 1.0

@dataclass
class GraphData:
    """图数据结构"""
    nodes: Dict[str, NodeFeatures]
    edges: List[EdgeFeatures]
    node_to_idx: Dict[str, int]
    idx_to_node: Dict[int, str]
    
    def __post_init__(self):
        if not self.node_to_idx:
            self.node_to_idx = {node_id: idx for idx, node_id in enumerate(self.nodes.keys())}
        if not self.idx_to_node:
            self.idx_to_node = {idx: node_id for node_id, idx in self.node_to_idx.items()}

class GraphAttentionNetwork(nn.Module):
    """图注意力网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32, 
                 num_heads: int = 4, dropout: float = 0.1):
        super(GraphAttentionNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # 图注意力层
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, dropout=dropout)
        
        # 权重预测层
        self.weight_predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出0-1之间的权重
        )
        
        # 关系权重预测层
        self.relation_weight_predictor = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),  # 源节点和目标节点特征拼接
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index, batch=None):
        """前向传播"""
        # 第一层GAT
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        
        # 第二层GAT
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        
        # 节点权重预测
        node_weights = self.weight_predictor(x)
        
        # 如果有batch信息，进行图级别的池化
        if batch is not None:
            graph_embedding = global_mean_pool(x, batch)
            return x, node_weights, graph_embedding
        
        return x, node_weights
    
    def predict_edge_weights(self, node_embeddings, edge_index):
        """预测边权重"""
        source_embeddings = node_embeddings[edge_index[0]]
        target_embeddings = node_embeddings[edge_index[1]]
        
        # 拼接源节点和目标节点的嵌入
        edge_features = torch.cat([source_embeddings, target_embeddings], dim=1)
        edge_weights = self.relation_weight_predictor(edge_features)
        
        return edge_weights

class GraphConvolutionalNetwork(nn.Module):
    """图卷积网络（备选方案）"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32, 
                 num_layers: int = 2, dropout: float = 0.1):
        super(GraphConvolutionalNetwork, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # 权重预测层
        self.weight_predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, batch=None):
        """前向传播"""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 节点权重预测
        node_weights = self.weight_predictor(x)
        
        if batch is not None:
            graph_embedding = global_mean_pool(x, batch)
            return x, node_weights, graph_embedding
        
        return x, node_weights

class GNNWeightLearner:
    """图神经网络权重学习器"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, output_dim: int = 32,
                 model_type: str = "gat", device: str = "cpu", learning_rate: float = 0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model_type = model_type
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        
        # 初始化模型
        if model_type == "gat":
            self.model = GraphAttentionNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            ).to(self.device)
        elif model_type == "gcn":
            self.model = GraphConvolutionalNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # 训练历史
        self.training_history = {
            "losses": [],
            "epochs": [],
            "learning_rates": []
        }
        
        logger.info(f"GNNWeightLearner initialized with {model_type} model on {device}")
    
    def prepare_graph_data(self, graph_data: GraphData) -> Data:
        """准备图数据用于训练"""
        # 构建节点特征矩阵
        node_features = []
        for node_id in graph_data.node_to_idx.keys():
            node = graph_data.nodes[node_id]
            # 组合多种特征
            features = np.concatenate([
                node.embedding,
                [node.degree, node.centrality, node.frequency]
            ])
            node_features.append(features)
        
        x = torch.FloatTensor(node_features).to(self.device)
        
        # 构建边索引
        edge_indices = []
        edge_weights = []
        
        for edge in graph_data.edges:
            source_idx = graph_data.node_to_idx[edge.source_id]
            target_idx = graph_data.node_to_idx[edge.target_id]
            
            edge_indices.append([source_idx, target_idx])
            edge_indices.append([target_idx, source_idx])  # 无向图
            
            edge_weights.append(edge.weight)
            edge_weights.append(edge.weight)
        
        edge_index = torch.LongTensor(edge_indices).t().contiguous().to(self.device)
        edge_attr = torch.FloatTensor(edge_weights).to(self.device)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def train_epoch(self, data_loader: DataLoader, criterion) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in data_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # 前向传播
            if hasattr(batch, 'batch'):
                node_embeddings, node_weights, graph_embedding = self.model(batch.x, batch.edge_index, batch.batch)
            else:
                node_embeddings, node_weights = self.model(batch.x, batch.edge_index)
            
            # 计算损失（这里需要根据具体任务定义损失函数）
            # 示例：使用节点权重的方差作为损失，鼓励权重分布的多样性
            loss = criterion(node_weights, batch.y if hasattr(batch, 'y') else None)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train(self, graph_data: GraphData, num_epochs: int = 100, 
              validation_data: Optional[GraphData] = None):
        """训练模型"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # 准备训练数据
        train_data = self.prepare_graph_data(graph_data)
        train_loader = DataLoader([train_data], batch_size=1, shuffle=False)
        
        # 定义损失函数
        criterion = self._get_loss_function()
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader, criterion)
            
            # 验证
            val_loss = train_loss  # 简化版本，实际应该用验证集
            if validation_data:
                val_loss = self.validate(validation_data)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 记录训练历史
            self.training_history["losses"].append(train_loss)
            self.training_history["epochs"].append(epoch)
            self.training_history["learning_rates"].append(self.optimizer.param_groups[0]['lr'])
            
            # 早停检查
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                self.save_model("best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= 20:  # 早停
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        logger.info("Training completed")
    
    def _get_loss_function(self):
        """获取损失函数"""
        def custom_loss(node_weights, targets=None):
            # 多目标损失函数
            losses = []
            
            # 1. 权重分布的多样性损失
            diversity_loss = -torch.var(node_weights)  # 鼓励权重分布的多样性
            losses.append(diversity_loss)
            
            # 2. 权重平滑性损失（相邻节点的权重应该相似）
            # 这里简化处理，实际应该基于图结构
            smoothness_loss = torch.mean(torch.abs(node_weights[:-1] - node_weights[1:]))
            losses.append(smoothness_loss * 0.1)
            
            # 3. 如果有目标权重，添加监督损失
            if targets is not None:
                supervised_loss = F.mse_loss(node_weights.squeeze(), targets)
                losses.append(supervised_loss)
            
            return sum(losses)
        
        return custom_loss
    
    def validate(self, validation_data: GraphData) -> float:
        """验证模型"""
        self.model.eval()
        val_data = self.prepare_graph_data(validation_data)
        
        with torch.no_grad():
            node_embeddings, node_weights = self.model(val_data.x, val_data.edge_index)
            criterion = self._get_loss_function()
            loss = criterion(node_weights)
        
        return loss.item()
    
    def predict_weights(self, graph_data: GraphData) -> Dict[str, float]:
        """预测节点权重"""
        self.model.eval()
        data = self.prepare_graph_data(graph_data)
        
        with torch.no_grad():
            node_embeddings, node_weights = self.model(data.x, data.edge_index)
            
            # 预测边权重
            edge_weights = None
            if hasattr(self.model, 'predict_edge_weights'):
                edge_weights = self.model.predict_edge_weights(node_embeddings, data.edge_index)
        
        # 转换为字典格式
        node_weight_dict = {}
        for node_id, idx in graph_data.node_to_idx.items():
            node_weight_dict[node_id] = float(node_weights[idx].cpu().numpy())
        
        result = {"node_weights": node_weight_dict}
        
        if edge_weights is not None:
            edge_weight_dict = {}
            for i, edge in enumerate(graph_data.edges):
                if i < len(edge_weights):
                    edge_key = f"{edge.source_id}-{edge.target_id}"
                    edge_weight_dict[edge_key] = float(edge_weights[i].cpu().numpy())
            result["edge_weights"] = edge_weight_dict
        
        return result
    
    def calculate_query_weights(self, graph_data: GraphData, query_entities: List[str]) -> Tuple[float, float]:
        """基于查询实体计算文档和图谱权重"""
        weights = self.predict_weights(graph_data)
        node_weights = weights["node_weights"]
        
        # 计算查询相关实体的平均权重
        relevant_weights = []
        for entity in query_entities:
            if entity in node_weights:
                relevant_weights.append(node_weights[entity])
        
        if not relevant_weights:
            return 0.5, 0.5  # 默认平衡权重
        
        # 图谱权重基于相关实体的平均权重
        graph_weight = np.mean(relevant_weights)
        doc_weight = 1.0 - graph_weight
        
        # 确保权重在合理范围内
        graph_weight = max(0.1, min(0.9, graph_weight))
        doc_weight = 1.0 - graph_weight
        
        return doc_weight, graph_weight
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'model_config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'model_type': self.model_type
            }
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        if not os.path.exists(filepath):
            logger.warning(f"Model file {filepath} not found")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        
        logger.info(f"Model loaded from {filepath}")
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "device": str(self.device),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "training_history": self.training_history
        }