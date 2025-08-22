from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# 尝试导入GNN模块
try:
    from ..ml.gnn_weight_learner import GNNWeightLearner, GraphData, NodeFeatures, EdgeFeatures
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    # 创建占位符类
    class GNNWeightLearner:
        pass
    class GraphData:
        pass
    class NodeFeatures:
        pass
    class EdgeFeatures:
        pass

logger = logging.getLogger(__name__)

class WeightStrategy(Enum):
    """权重分配策略枚举"""
    STATIC = "static"  # 静态权重
    INTENT_DRIVEN = "intent_driven"  # 意图驱动
    QUALITY_DRIVEN = "quality_driven"  # 质量驱动
    GNN_DRIVEN = "gnn_driven"  # 图神经网络驱动
    HYBRID = "hybrid"  # 混合策略
    ADAPTIVE = "adaptive"  # 自适应策略
    ENSEMBLE = "ensemble"  # 集成策略

@dataclass
class QueryContext:
    """查询上下文信息"""
    query: str
    intent: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = None
    domain: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class SearchResult:
    """搜索结果信息"""
    content: str
    score: float
    source: str  # 'document' or 'graph'
    metadata: Dict[str, Any] = None
    relevance_score: Optional[float] = None
    confidence_score: Optional[float] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class WeightResult:
    """权重分配结果"""
    doc_weight: float
    graph_weight: float
    strategy_used: WeightStrategy
    confidence: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # 确保权重和为1
        total = self.doc_weight + self.graph_weight
        if total > 0:
            self.doc_weight /= total
            self.graph_weight /= total

class WeightCalculator(ABC):
    """权重计算器抽象基类"""
    
    @abstractmethod
    def calculate_weights(self, context: QueryContext, 
                         doc_results: List[SearchResult] = None,
                         graph_results: List[SearchResult] = None) -> WeightResult:
        """计算权重"""
        pass
    
    @abstractmethod
    def get_strategy(self) -> WeightStrategy:
        """获取策略类型"""
        pass

class StaticWeightCalculator(WeightCalculator):
    """静态权重计算器"""
    
    def __init__(self, doc_weight: float = 0.6, graph_weight: float = 0.4):
        self.doc_weight = doc_weight
        self.graph_weight = graph_weight
    
    def calculate_weights(self, context: QueryContext, 
                         doc_results: List[SearchResult] = None,
                         graph_results: List[SearchResult] = None) -> WeightResult:
        return WeightResult(
            doc_weight=self.doc_weight,
            graph_weight=self.graph_weight,
            strategy_used=WeightStrategy.STATIC,
            confidence=1.0,
            metadata={"calculator": "static"}
        )
    
    def get_strategy(self) -> WeightStrategy:
        return WeightStrategy.STATIC

class IntentDrivenWeightCalculator(WeightCalculator):
    """基于意图的权重计算器"""
    
    def __init__(self):
        # 不同意图对应的权重偏好
        self.intent_weights = {
            "factual": {"doc_weight": 0.7, "graph_weight": 0.3},  # 事实查询偏向文档
            "relational": {"doc_weight": 0.3, "graph_weight": 0.7},  # 关系查询偏向图谱
            "conceptual": {"doc_weight": 0.5, "graph_weight": 0.5},  # 概念查询平衡
            "analytical": {"doc_weight": 0.4, "graph_weight": 0.6},  # 分析查询偏向图谱
            "exploratory": {"doc_weight": 0.6, "graph_weight": 0.4},  # 探索查询偏向文档
            "default": {"doc_weight": 0.5, "graph_weight": 0.5}  # 默认平衡
        }
    
    def calculate_weights(self, context: QueryContext, 
                         doc_results: List[SearchResult] = None,
                         graph_results: List[SearchResult] = None) -> WeightResult:
        intent = context.intent or "default"
        weights = self.intent_weights.get(intent, self.intent_weights["default"])
        
        # 根据意图识别的置信度调整权重
        confidence = self._calculate_intent_confidence(context)
        
        return WeightResult(
            doc_weight=weights["doc_weight"],
            graph_weight=weights["graph_weight"],
            strategy_used=WeightStrategy.INTENT_DRIVEN,
            confidence=confidence,
            metadata={"intent": intent, "calculator": "intent_driven"}
        )
    
    def _calculate_intent_confidence(self, context: QueryContext) -> float:
        """计算意图识别的置信度"""
        if not context.intent or context.intent == "default":
            return 0.5
        
        # 简单的置信度计算，可以根据实际情况优化
        query_length = len(context.query.split())
        if query_length < 3:
            return 0.6
        elif query_length < 6:
            return 0.8
        else:
            return 0.9
    
    def get_strategy(self) -> WeightStrategy:
        return WeightStrategy.INTENT_DRIVEN

class QualityDrivenWeightCalculator(WeightCalculator):
    """基于结果质量的权重计算器"""
    
    def calculate_weights(self, context: QueryContext, 
                         doc_results: List[SearchResult] = None,
                         graph_results: List[SearchResult] = None) -> WeightResult:
        doc_quality = self._calculate_quality_score(doc_results or [])
        graph_quality = self._calculate_quality_score(graph_results or [])
        
        # 根据质量分数动态调整权重
        total_quality = doc_quality + graph_quality
        if total_quality > 0:
            doc_weight = doc_quality / total_quality
            graph_weight = graph_quality / total_quality
        else:
            doc_weight = graph_weight = 0.5
        
        # 计算置信度
        confidence = min(doc_quality, graph_quality) / max(doc_quality, graph_quality, 0.1)
        
        return WeightResult(
            doc_weight=doc_weight,
            graph_weight=graph_weight,
            strategy_used=WeightStrategy.QUALITY_DRIVEN,
            confidence=confidence,
            metadata={
                "doc_quality": doc_quality,
                "graph_quality": graph_quality,
                "calculator": "quality_driven"
            }
        )
    
    def _calculate_quality_score(self, results: List[SearchResult]) -> float:
        """计算结果质量分数"""
        if not results:
            return 0.0
        
        # 综合考虑相关性分数、置信度分数和结果数量
        relevance_scores = [r.relevance_score or r.score for r in results]
        confidence_scores = [r.confidence_score or 0.5 for r in results]
        
        avg_relevance = np.mean(relevance_scores)
        avg_confidence = np.mean(confidence_scores)
        result_count_factor = min(len(results) / 5.0, 1.0)  # 结果数量因子
        
        return (avg_relevance * 0.5 + avg_confidence * 0.3 + result_count_factor * 0.2)
    
    def get_strategy(self) -> WeightStrategy:
        return WeightStrategy.QUALITY_DRIVEN

class AdaptiveWeightCalculator(WeightCalculator):
    """自适应权重计算器"""
    
    def __init__(self):
        self.intent_calculator = IntentDrivenWeightCalculator()
        self.quality_calculator = QualityDrivenWeightCalculator()
        self.static_calculator = StaticWeightCalculator()
        
        # 历史性能记录
        self.performance_history = {}
    
    def calculate_weights(self, context: QueryContext, 
                         doc_results: List[SearchResult] = None,
                         graph_results: List[SearchResult] = None) -> WeightResult:
        # 获取多种策略的权重建议
        intent_result = self.intent_calculator.calculate_weights(context, doc_results, graph_results)
        quality_result = self.quality_calculator.calculate_weights(context, doc_results, graph_results)
        static_result = self.static_calculator.calculate_weights(context, doc_results, graph_results)
        
        # 根据历史性能和当前情况选择最佳策略
        best_strategy = self._select_best_strategy(context, [intent_result, quality_result, static_result])
        
        return WeightResult(
            doc_weight=best_strategy.doc_weight,
            graph_weight=best_strategy.graph_weight,
            strategy_used=WeightStrategy.ADAPTIVE,
            confidence=best_strategy.confidence,
            metadata={
                "selected_strategy": best_strategy.strategy_used.value,
                "calculator": "adaptive",
                "alternatives": {
                    "intent": {"doc": intent_result.doc_weight, "graph": intent_result.graph_weight},
                    "quality": {"doc": quality_result.doc_weight, "graph": quality_result.graph_weight},
                    "static": {"doc": static_result.doc_weight, "graph": static_result.graph_weight}
                }
            }
        )
    
    def _select_best_strategy(self, context: QueryContext, 
                             strategy_results: List[WeightResult]) -> WeightResult:
        """选择最佳策略"""
        # 简单的策略选择逻辑，可以根据实际情况优化
        if context.intent and context.intent != "default":
            return strategy_results[0]  # intent_result
        elif len(strategy_results) > 1:
            return strategy_results[1]  # quality_result
        else:
            return strategy_results[2]  # static_result
    
    def update_performance(self, context: QueryContext, result: WeightResult, 
                          feedback_score: float):
        """更新性能记录"""
        strategy_key = result.strategy_used.value
        if strategy_key not in self.performance_history:
            self.performance_history[strategy_key] = []
        
        self.performance_history[strategy_key].append({
            "timestamp": context.timestamp,
            "feedback_score": feedback_score,
            "confidence": result.confidence
        })
        
        # 保持历史记录在合理范围内
        if len(self.performance_history[strategy_key]) > 1000:
            self.performance_history[strategy_key] = self.performance_history[strategy_key][-500:]
    
    def get_strategy(self) -> WeightStrategy:
        return WeightStrategy.ADAPTIVE

class GNNDrivenWeightCalculator(WeightCalculator):
    """基于图神经网络的权重计算器"""
    
    def __init__(self, neo4j_service=None, model_path: Optional[str] = None):
        if not GNN_AVAILABLE:
            raise ImportError("GNN module is not available. Please install required dependencies.")
        
        self.neo4j_service = neo4j_service
        self.gnn_learner = None
        self.model_path = model_path
        self.graph_cache = {}
        self.cache_timeout = 300  # 5分钟缓存
        
        # 初始化GNN模型
        self._initialize_gnn_model()
    
    def _initialize_gnn_model(self):
        """初始化GNN模型"""
        try:
            self.gnn_learner = GNNWeightLearner(
                input_dim=128,  # 可配置
                hidden_dim=64,
                output_dim=32,
                model_type="gat"  # 使用图注意力网络
            )
            
            # 如果有预训练模型，加载它
            if self.model_path and self.gnn_learner.load_model(self.model_path):
                logger.info(f"Loaded pre-trained GNN model from {self.model_path}")
            else:
                logger.info("Using untrained GNN model. Consider training the model for better performance.")
                
        except Exception as e:
            logger.error(f"Failed to initialize GNN model: {e}")
            self.gnn_learner = None
    
    def calculate_weights(self, context: QueryContext, 
                         doc_results: List[SearchResult] = None,
                         graph_results: List[SearchResult] = None) -> WeightResult:
        if not self.gnn_learner:
            logger.warning("GNN model not available, falling back to static weights")
            return WeightResult(
                doc_weight=0.5,
                graph_weight=0.5,
                strategy_used=WeightStrategy.GNN_DRIVEN,
                confidence=0.1,
                metadata={"error": "GNN model not available", "calculator": "gnn_driven"}
            )
        
        try:
            # 从查询中提取实体
            query_entities = self._extract_entities_from_query(context.query)
            
            # 获取图数据
            graph_data = self._get_graph_data(query_entities)
            
            if not graph_data or not graph_data.nodes:
                logger.warning("No graph data available for GNN weight calculation")
                return WeightResult(
                    doc_weight=0.6,
                    graph_weight=0.4,
                    strategy_used=WeightStrategy.GNN_DRIVEN,
                    confidence=0.3,
                    metadata={"warning": "No graph data", "calculator": "gnn_driven"}
                )
            
            # 使用GNN计算权重
            doc_weight, graph_weight = self.gnn_learner.calculate_query_weights(
                graph_data, query_entities
            )
            
            # 计算置信度
            confidence = self._calculate_gnn_confidence(graph_data, query_entities)
            
            return WeightResult(
                doc_weight=doc_weight,
                graph_weight=graph_weight,
                strategy_used=WeightStrategy.GNN_DRIVEN,
                confidence=confidence,
                metadata={
                    "query_entities": query_entities,
                    "graph_nodes": len(graph_data.nodes),
                    "graph_edges": len(graph_data.edges),
                    "calculator": "gnn_driven"
                }
            )
            
        except Exception as e:
            logger.error(f"Error in GNN weight calculation: {e}")
            return WeightResult(
                doc_weight=0.5,
                graph_weight=0.5,
                strategy_used=WeightStrategy.GNN_DRIVEN,
                confidence=0.1,
                metadata={"error": str(e), "calculator": "gnn_driven"}
            )
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """从查询中提取实体（简化版本）"""
        # 这里应该使用更复杂的NER或实体链接
        # 简化版本：分词并过滤
        words = query.lower().split()
        # 过滤停用词和短词
        stop_words = {'的', '是', '在', '有', '和', '与', '或', '但', '如果', 'the', 'is', 'in', 'and', 'or', 'but', 'if'}
        entities = [word for word in words if len(word) > 2 and word not in stop_words]
        return entities[:10]  # 限制实体数量
    
    def _get_graph_data(self, query_entities: List[str]) -> Optional[GraphData]:
        """获取图数据"""
        if not self.neo4j_service:
            return None
        
        try:
            # 从Neo4j获取相关的子图
            # 这里需要实现具体的图数据提取逻辑
            nodes = {}
            edges = []
            
            # 简化版本：创建模拟数据
            for i, entity in enumerate(query_entities):
                nodes[entity] = NodeFeatures(
                    entity_id=entity,
                    entity_type="concept",
                    embedding=np.random.randn(125),  # 125 + 3 = 128
                    degree=np.random.randint(1, 10),
                    centrality=np.random.random(),
                    frequency=np.random.randint(1, 100)
                )
            
            # 创建一些边
            entity_list = list(query_entities)
            for i in range(len(entity_list) - 1):
                edges.append(EdgeFeatures(
                    source_id=entity_list[i],
                    target_id=entity_list[i + 1],
                    relation_type="related_to",
                    weight=np.random.random(),
                    frequency=np.random.randint(1, 50)
                ))
            
            return GraphData(nodes=nodes, edges=edges, node_to_idx={}, idx_to_node={})
            
        except Exception as e:
            logger.error(f"Error getting graph data: {e}")
            return None
    
    def _calculate_gnn_confidence(self, graph_data: GraphData, query_entities: List[str]) -> float:
        """计算GNN预测的置信度"""
        # 基于图的连通性和实体覆盖率计算置信度
        total_entities = len(query_entities)
        covered_entities = len([e for e in query_entities if e in graph_data.nodes])
        
        coverage_ratio = covered_entities / total_entities if total_entities > 0 else 0
        connectivity_score = len(graph_data.edges) / max(len(graph_data.nodes), 1)
        
        confidence = (coverage_ratio * 0.7 + min(connectivity_score, 1.0) * 0.3)
        return max(0.1, min(0.9, confidence))
    
    def get_strategy(self) -> WeightStrategy:
        return WeightStrategy.GNN_DRIVEN

class HybridWeightCalculator(WeightCalculator):
    """混合权重计算器"""
    
    def __init__(self, calculators: Dict[WeightStrategy, WeightCalculator] = None):
        self.calculators = calculators or {}
        self.strategy_weights = {
            WeightStrategy.INTENT_DRIVEN: 0.3,
            WeightStrategy.QUALITY_DRIVEN: 0.3,
            WeightStrategy.GNN_DRIVEN: 0.4
        }
    
    def calculate_weights(self, context: QueryContext, 
                         doc_results: List[SearchResult] = None,
                         graph_results: List[SearchResult] = None) -> WeightResult:
        results = {}
        total_confidence = 0
        
        # 收集各策略的结果
        for strategy, calculator in self.calculators.items():
            if strategy in self.strategy_weights:
                try:
                    result = calculator.calculate_weights(context, doc_results, graph_results)
                    results[strategy] = result
                    total_confidence += result.confidence * self.strategy_weights[strategy]
                except Exception as e:
                    logger.warning(f"Error in {strategy.value} calculator: {e}")
        
        if not results:
            # 回退到静态权重
            return WeightResult(
                doc_weight=0.5,
                graph_weight=0.5,
                strategy_used=WeightStrategy.HYBRID,
                confidence=0.5,
                metadata={"error": "No calculators available", "calculator": "hybrid"}
            )
        
        # 加权平均计算最终权重
        final_doc_weight = 0
        final_graph_weight = 0
        
        for strategy, result in results.items():
            weight = self.strategy_weights.get(strategy, 0)
            final_doc_weight += result.doc_weight * weight
            final_graph_weight += result.graph_weight * weight
        
        # 归一化
        total_weight = final_doc_weight + final_graph_weight
        if total_weight > 0:
            final_doc_weight /= total_weight
            final_graph_weight /= total_weight
        
        return WeightResult(
            doc_weight=final_doc_weight,
            graph_weight=final_graph_weight,
            strategy_used=WeightStrategy.HYBRID,
            confidence=total_confidence,
            metadata={
                "component_results": {k.value: {"doc": v.doc_weight, "graph": v.graph_weight, "conf": v.confidence} 
                                    for k, v in results.items()},
                "strategy_weights": self.strategy_weights,
                "calculator": "hybrid"
            }
        )
    
    def update_strategy_weights(self, new_weights: Dict[WeightStrategy, float]):
        """更新策略权重"""
        self.strategy_weights.update(new_weights)
        # 归一化权重
        total = sum(self.strategy_weights.values())
        if total > 0:
            self.strategy_weights = {k: v/total for k, v in self.strategy_weights.items()}
    
    def get_strategy(self) -> WeightStrategy:
        return WeightStrategy.HYBRID

class EnsembleWeightCalculator(WeightCalculator):
    """集成权重计算器"""
    
    def __init__(self, calculators: List[WeightCalculator] = None):
        self.calculators = calculators or []
        self.voting_method = "weighted_average"  # 可选: "majority_vote", "weighted_average", "stacking"
    
    def calculate_weights(self, context: QueryContext, 
                         doc_results: List[SearchResult] = None,
                         graph_results: List[SearchResult] = None) -> WeightResult:
        if not self.calculators:
            return WeightResult(
                doc_weight=0.5,
                graph_weight=0.5,
                strategy_used=WeightStrategy.ENSEMBLE,
                confidence=0.5,
                metadata={"error": "No calculators in ensemble", "calculator": "ensemble"}
            )
        
        results = []
        for calculator in self.calculators:
            try:
                result = calculator.calculate_weights(context, doc_results, graph_results)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error in ensemble calculator {calculator.__class__.__name__}: {e}")
        
        if not results:
            return WeightResult(
                doc_weight=0.5,
                graph_weight=0.5,
                strategy_used=WeightStrategy.ENSEMBLE,
                confidence=0.1,
                metadata={"error": "All calculators failed", "calculator": "ensemble"}
            )
        
        # 集成结果
        if self.voting_method == "weighted_average":
            return self._weighted_average_ensemble(results)
        elif self.voting_method == "majority_vote":
            return self._majority_vote_ensemble(results)
        else:
            return self._weighted_average_ensemble(results)
    
    def _weighted_average_ensemble(self, results: List[WeightResult]) -> WeightResult:
        """加权平均集成"""
        total_weight = sum(r.confidence for r in results)
        if total_weight == 0:
            weights = [1.0 / len(results)] * len(results)
        else:
            weights = [r.confidence / total_weight for r in results]
        
        doc_weight = sum(r.doc_weight * w for r, w in zip(results, weights))
        graph_weight = sum(r.graph_weight * w for r, w in zip(results, weights))
        confidence = sum(r.confidence * w for r, w in zip(results, weights))
        
        return WeightResult(
            doc_weight=doc_weight,
            graph_weight=graph_weight,
            strategy_used=WeightStrategy.ENSEMBLE,
            confidence=confidence,
            metadata={
                "ensemble_method": "weighted_average",
                "component_count": len(results),
                "component_strategies": [r.strategy_used.value for r in results],
                "calculator": "ensemble"
            }
        )
    
    def _majority_vote_ensemble(self, results: List[WeightResult]) -> WeightResult:
        """多数投票集成"""
        # 简化版本：选择置信度最高的结果
        best_result = max(results, key=lambda r: r.confidence)
        
        return WeightResult(
            doc_weight=best_result.doc_weight,
            graph_weight=best_result.graph_weight,
            strategy_used=WeightStrategy.ENSEMBLE,
            confidence=best_result.confidence,
            metadata={
                "ensemble_method": "majority_vote",
                "selected_strategy": best_result.strategy_used.value,
                "component_count": len(results),
                "calculator": "ensemble"
            }
        )
    
    def get_strategy(self) -> WeightStrategy:
        return WeightStrategy.ENSEMBLE

class DynamicWeightManager:
    """动态权重管理器"""
    
    def __init__(self, neo4j_service=None, doc_service=None, config: Dict = None):
        self.neo4j_service = neo4j_service
        self.doc_service = doc_service
        self.config = config or {}
        
        # 权重计算器
        self.calculators = {
            WeightStrategy.STATIC: StaticWeightCalculator(),
            WeightStrategy.INTENT_DRIVEN: IntentDrivenWeightCalculator(),
            WeightStrategy.QUALITY_DRIVEN: QualityDrivenWeightCalculator(),
            WeightStrategy.ADAPTIVE: AdaptiveWeightCalculator()
        }
        
        # 初始化GNN计算器（如果可用）
        self._initialize_gnn_calculator()
        
        # 初始化混合和集成计算器
        self._initialize_advanced_calculators()
        
        # 当前策略
        self.current_strategy = WeightStrategy.ADAPTIVE
        
        # 历史记录
        self.weight_history = []
        
        # 缓存
        self.weight_cache = {}
        self.cache_ttl = 300  # 5分钟
        
        # 异步执行器
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # 性能监控
        self.performance_metrics = {
            "total_requests": 0,
            "strategy_usage": {strategy.value: 0 for strategy in WeightStrategy},
            "average_confidence": 0.0
        }
        
        logger.info(f"DynamicWeightManager initialized with strategy: {self.current_strategy.value}")
    
    def _initialize_gnn_calculator(self):
        """初始化GNN计算器"""
        if GNN_AVAILABLE:
            try:
                gnn_model_path = self.config.get('gnn_model_path')
                self.calculators[WeightStrategy.GNN_DRIVEN] = GNNDrivenWeightCalculator(
                    neo4j_service=self.neo4j_service,
                    model_path=gnn_model_path
                )
                logger.info("GNN weight calculator initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize GNN calculator: {e}")
        else:
            logger.info("GNN module not available, skipping GNN calculator initialization")
    
    def _initialize_advanced_calculators(self):
        """初始化高级计算器"""
        # 混合计算器
        hybrid_calculators = {
            WeightStrategy.INTENT_DRIVEN: self.calculators[WeightStrategy.INTENT_DRIVEN],
            WeightStrategy.QUALITY_DRIVEN: self.calculators[WeightStrategy.QUALITY_DRIVEN]
        }
        
        # 如果GNN可用，添加到混合计算器中
        if WeightStrategy.GNN_DRIVEN in self.calculators:
            hybrid_calculators[WeightStrategy.GNN_DRIVEN] = self.calculators[WeightStrategy.GNN_DRIVEN]
        
        self.calculators[WeightStrategy.HYBRID] = HybridWeightCalculator(hybrid_calculators)
        
        # 集成计算器
        ensemble_calculators = [
            self.calculators[WeightStrategy.INTENT_DRIVEN],
            self.calculators[WeightStrategy.QUALITY_DRIVEN],
            self.calculators[WeightStrategy.ADAPTIVE]
        ]
        
        if WeightStrategy.GNN_DRIVEN in self.calculators:
            ensemble_calculators.append(self.calculators[WeightStrategy.GNN_DRIVEN])
        
        self.calculators[WeightStrategy.ENSEMBLE] = EnsembleWeightCalculator(ensemble_calculators)
    
    async def calculate_weights_async(self, context: QueryContext, 
                                    doc_results: List[SearchResult] = None,
                                    graph_results: List[SearchResult] = None) -> WeightResult:
        """异步计算权重"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.calculate_weights, 
            context, doc_results, graph_results
        )
    
    def calculate_weights(self, context: QueryContext, 
                         strategy: Optional[WeightStrategy] = None,
                         doc_results: List[SearchResult] = None,
                         graph_results: List[SearchResult] = None) -> WeightResult:
        """计算权重"""
        try:
            # 选择策略
            selected_strategy = strategy or self.current_strategy
            
            # 如果请求的策略不可用，回退到默认策略
            if selected_strategy not in self.calculators:
                logger.warning(f"Strategy {selected_strategy.value} not available, using {self.current_strategy.value}")
                selected_strategy = self.current_strategy
            
            # 计算权重
            calculator = self.calculators[selected_strategy]
            result = calculator.calculate_weights(context, doc_results, graph_results)
            
            # 更新性能指标
            self._update_metrics(result)
            
            logger.debug(f"Weight calculation completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in weight calculation: {e}")
            # 回退到静态权重
            return self.calculators[WeightStrategy.STATIC].calculate_weights(context)
    
    def register_calculator(self, strategy: WeightStrategy, calculator: WeightCalculator):
        """注册新的权重计算器"""
        self.calculators[strategy] = calculator
        self.performance_metrics["strategy_usage"][strategy.value] = 0
        logger.info(f"Registered calculator for strategy: {strategy.value}")
    
    def set_strategy(self, strategy: WeightStrategy):
        """设置权重策略"""
        if strategy in self.calculators:
            old_strategy = self.current_strategy
            self.current_strategy = strategy
            logger.info(f"Weight strategy changed from {old_strategy.value} to {strategy.value}")
        else:
            logger.warning(f"Strategy {strategy.value} not available")
    
    def get_available_strategies(self) -> List[WeightStrategy]:
        """获取可用的策略列表"""
        return list(self.calculators.keys())
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return self.performance_metrics.copy()
    
    def update_hybrid_weights(self, strategy_weights: Dict[WeightStrategy, float]):
        """更新混合策略的权重"""
        if WeightStrategy.HYBRID in self.calculators:
            hybrid_calc = self.calculators[WeightStrategy.HYBRID]
            if hasattr(hybrid_calc, 'update_strategy_weights'):
                hybrid_calc.update_strategy_weights(strategy_weights)
                logger.info(f"Updated hybrid strategy weights: {strategy_weights}")
    
    def get_weight_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取权重历史记录"""
        return self.weight_history[-limit:] if self.weight_history else []
    
    def clear_cache(self):
        """清空缓存"""
        self.weight_cache.clear()
        logger.info("Weight cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        current_time = time.time()
        valid_entries = sum(
            1 for data in self.weight_cache.values()
            if current_time - data["timestamp"] < self.cache_ttl
        )
        
        return {
            "total_entries": len(self.weight_cache),
            "valid_entries": valid_entries,
            "cache_ttl": self.cache_ttl,
            "hit_rate": self.performance_metrics.get("cache_hit_rate", 0.0)
        }
    
    def _update_metrics(self, result: WeightResult):
        """更新性能指标"""
        self.performance_metrics["total_requests"] += 1
        self.performance_metrics["strategy_usage"][result.strategy_used.value] += 1
        
        # 更新平均置信度
        total_requests = self.performance_metrics["total_requests"]
        current_avg = self.performance_metrics["average_confidence"]
        self.performance_metrics["average_confidence"] = (
            (current_avg * (total_requests - 1) + result.confidence) / total_requests
        )
    
    def reset_metrics(self):
        """重置性能指标"""
        self.performance_metrics = {
            "total_requests": 0,
            "strategy_usage": {strategy.value: 0 for strategy in WeightStrategy},
            "average_confidence": 0.0
        }
        logger.info("Performance metrics reset")
    
    def _record_weight_history(self, context: QueryContext, result: WeightResult):
        """记录权重历史"""
        history_entry = {
            "timestamp": time.time(),
            "query": context.query,
            "strategy": result.strategy_used.value,
            "doc_weight": result.doc_weight,
            "graph_weight": result.graph_weight,
            "confidence": result.confidence,
            "metadata": result.metadata
        }
        
        self.weight_history.append(history_entry)
        
        # 保持历史记录在合理范围内
        if len(self.weight_history) > 1000:
            self.weight_history = self.weight_history[-500:]
        
        # 更新性能指标
        self._update_performance_metrics(result)
    
    def _update_performance_metrics(self, result: WeightResult):
        """更新性能指标"""
        strategy = result.strategy_used.value
        
        if strategy not in self.performance_metrics:
            self.performance_metrics[strategy] = {
                "total_calls": 0,
                "avg_confidence": 0.0,
                "avg_doc_weight": 0.0,
                "avg_graph_weight": 0.0
            }
        
        metrics = self.performance_metrics[strategy]
        metrics["total_calls"] += 1
        
        # 计算移动平均
        alpha = 0.1  # 学习率
        metrics["avg_confidence"] = (1 - alpha) * metrics["avg_confidence"] + alpha * result.confidence
        metrics["avg_doc_weight"] = (1 - alpha) * metrics["avg_doc_weight"] + alpha * result.doc_weight
        metrics["avg_graph_weight"] = (1 - alpha) * metrics["avg_graph_weight"] + alpha * result.graph_weight
    
    def _generate_cache_key(self, context: QueryContext, 
                           doc_results: List[SearchResult] = None,
                           graph_results: List[SearchResult] = None) -> str:
        """生成缓存键"""
        key_data = {
            "query": context.query,
            "strategy": self.current_strategy.value,
            "doc_count": len(doc_results) if doc_results else 0,
            "graph_count": len(graph_results) if graph_results else 0
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[WeightResult]:
        """获取缓存结果"""
        if cache_key in self.weight_cache:
            cached_data = self.weight_cache[cache_key]
            if time.time() - cached_data["timestamp"] < self.cache_ttl:
                # 记录缓存命中
                self.performance_metrics["cache_hits"] = self.performance_metrics.get("cache_hits", 0) + 1
                return cached_data["result"]
            else:
                # 清理过期缓存
                del self.weight_cache[cache_key]
        
        # 记录缓存未命中
        self.performance_metrics["cache_misses"] = self.performance_metrics.get("cache_misses", 0) + 1
        return None
    
    def _cache_result(self, cache_key: str, result: WeightResult):
        """缓存结果"""
        self.weight_cache[cache_key] = {
            "timestamp": time.time(),
            "result": result
        }
        
        # 清理过期缓存
        self._cleanup_expired_cache()
    
    def _cleanup_expired_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [
            key for key, data in self.weight_cache.items()
            if current_time - data["timestamp"] > self.cache_ttl
        ]
        for key in expired_keys:
            del self.weight_cache[key]
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)