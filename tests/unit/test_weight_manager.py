#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
权重管理器单元测试
测试动态权重计算、策略切换、性能监控等功能
"""

import pytest
from unittest.mock import patch
from datetime import datetime

# 导入被测试的模块
from src.services.weight_manager import (
    WeightStrategy,
    QueryContext,
    SearchResult,
    WeightResult,
    StaticWeightCalculator,
    IntentDrivenWeightCalculator,
    QualityDrivenWeightCalculator,
    AdaptiveWeightCalculator,
    HybridWeightCalculator,
    EnsembleWeightCalculator,
    DynamicWeightManager
)

class TestQueryContext:
    """测试QueryContext数据类"""
    
    def test_query_context_initialization(self):
        """测试QueryContext初始化"""
        context = QueryContext(
            query="测试查询",
            intent="factual",
            user_id="user123",
            session_id="session456"
        )
        
        assert context.query == "测试查询"
        assert context.intent == "factual"
        assert context.user_id == "user123"
        assert context.session_id == "session456"
        assert isinstance(context.timestamp, datetime)
    
    def test_query_context_default_timestamp(self):
        """测试QueryContext默认时间戳"""
        context = QueryContext(query="测试查询")
        assert context.timestamp is not None
        assert isinstance(context.timestamp, datetime)

class TestSearchResult:
    """测试SearchResult数据类"""
    
    def test_search_result_initialization(self):
        """测试SearchResult初始化"""
        result = SearchResult(
            content="测试内容",
            score=0.8,
            source="document",
            metadata={"type": "text"},
            relevance_score=0.9,
            confidence_score=0.7
        )
        
        assert result.content == "测试内容"
        assert result.score == 0.8
        assert result.source == "document"
        assert result.metadata == {"type": "text"}
        assert result.relevance_score == 0.9
        assert result.confidence_score == 0.7
    
    def test_search_result_default_metadata(self):
        """测试SearchResult默认元数据"""
        result = SearchResult(
            content="测试内容",
            score=0.8,
            source="graph"
        )
        assert result.metadata == {}

class TestWeightResult:
    """测试WeightResult数据类"""
    
    def test_weight_result_initialization(self):
        """测试WeightResult初始化"""
        result = WeightResult(
            doc_weight=0.6,
            graph_weight=0.4,
            strategy_used=WeightStrategy.STATIC,
            confidence=0.9
        )
        
        assert result.doc_weight == 0.6
        assert result.graph_weight == 0.4
        assert result.strategy_used == WeightStrategy.STATIC
        assert result.confidence == 0.9
        assert result.metadata == {}
    
    def test_weight_result_normalization(self):
        """测试权重归一化"""
        result = WeightResult(
            doc_weight=0.8,
            graph_weight=0.6,  # 总和为1.4
            strategy_used=WeightStrategy.STATIC,
            confidence=0.9
        )
        
        # 权重应该被归一化为总和为1
        assert abs(result.doc_weight + result.graph_weight - 1.0) < 1e-6
        assert abs(result.doc_weight - 0.8/1.4) < 1e-6
        assert abs(result.graph_weight - 0.6/1.4) < 1e-6

class TestStaticWeightCalculator:
    """测试静态权重计算器"""
    
    def test_static_calculator_initialization(self):
        """测试静态计算器初始化"""
        calculator = StaticWeightCalculator(doc_weight=0.7, graph_weight=0.3)
        assert calculator.doc_weight == 0.7
        assert calculator.graph_weight == 0.3
    
    def test_static_calculator_default_weights(self):
        """测试静态计算器默认权重"""
        calculator = StaticWeightCalculator()
        assert calculator.doc_weight == 0.6
        assert calculator.graph_weight == 0.4
    
    def test_static_calculator_calculate_weights(self):
        """测试静态权重计算"""
        calculator = StaticWeightCalculator(doc_weight=0.7, graph_weight=0.3)
        context = QueryContext(query="测试查询")
        
        result = calculator.calculate_weights(context)
        
        assert result.doc_weight == 0.7
        assert result.graph_weight == 0.3
        assert result.strategy_used == WeightStrategy.STATIC
        assert result.confidence == 1.0
        assert result.metadata["calculator"] == "static"
    
    def test_static_calculator_get_strategy(self):
        """测试获取策略类型"""
        calculator = StaticWeightCalculator()
        assert calculator.get_strategy() == WeightStrategy.STATIC

class TestIntentDrivenWeightCalculator:
    """测试意图驱动权重计算器"""
    
    def test_intent_calculator_initialization(self):
        """测试意图计算器初始化"""
        calculator = IntentDrivenWeightCalculator()
        assert "factual" in calculator.intent_weights
        assert "relational" in calculator.intent_weights
        assert "default" in calculator.intent_weights
    
    def test_intent_calculator_factual_query(self):
        """测试事实查询权重计算"""
        calculator = IntentDrivenWeightCalculator()
        context = QueryContext(query="什么是人工智能？", intent="factual")
        
        result = calculator.calculate_weights(context)
        
        assert result.doc_weight == 0.7  # 事实查询偏向文档
        assert result.graph_weight == 0.3
        assert result.strategy_used == WeightStrategy.INTENT_DRIVEN
        assert result.metadata["intent"] == "factual"
    
    def test_intent_calculator_relational_query(self):
        """测试关系查询权重计算"""
        calculator = IntentDrivenWeightCalculator()
        context = QueryContext(query="A和B的关系是什么？", intent="relational")
        
        result = calculator.calculate_weights(context)
        
        assert result.doc_weight == 0.3  # 关系查询偏向图谱
        assert result.graph_weight == 0.7
        assert result.strategy_used == WeightStrategy.INTENT_DRIVEN
        assert result.metadata["intent"] == "relational"
    
    def test_intent_calculator_default_intent(self):
        """测试默认意图权重计算"""
        calculator = IntentDrivenWeightCalculator()
        context = QueryContext(query="测试查询")  # 无意图
        
        result = calculator.calculate_weights(context)
        
        assert result.doc_weight == 0.5  # 默认平衡
        assert result.graph_weight == 0.5
        assert result.metadata["intent"] == "default"
    
    def test_intent_calculator_confidence_calculation(self):
        """测试置信度计算"""
        calculator = IntentDrivenWeightCalculator()
        
        # 短查询
        short_context = QueryContext(query="AI", intent="factual")
        short_result = calculator.calculate_weights(short_context)
        
        # 中等长度查询
        medium_context = QueryContext(query="什么是人工智能", intent="factual")
        medium_result = calculator.calculate_weights(medium_context)
        
        # 长查询
        long_context = QueryContext(query="什么是人工智能以及它的应用领域", intent="factual")
        long_result = calculator.calculate_weights(long_context)
        
        assert short_result.confidence < medium_result.confidence < long_result.confidence

class TestQualityDrivenWeightCalculator:
    """测试质量驱动权重计算器"""
    
    def test_quality_calculator_initialization(self):
        """测试质量计算器初始化"""
        calculator = QualityDrivenWeightCalculator()
        assert calculator.get_strategy() == WeightStrategy.QUALITY_DRIVEN
    
    def test_quality_calculator_with_results(self):
        """测试基于结果质量的权重计算"""
        calculator = QualityDrivenWeightCalculator()
        context = QueryContext(query="测试查询")
        
        # 高质量文档结果
        doc_results = [
            SearchResult("文档1", 0.9, "document", relevance_score=0.9, confidence_score=0.8),
            SearchResult("文档2", 0.8, "document", relevance_score=0.8, confidence_score=0.7)
        ]
        
        # 低质量图谱结果
        graph_results = [
            SearchResult("图谱1", 0.5, "graph", relevance_score=0.5, confidence_score=0.4)
        ]
        
        result = calculator.calculate_weights(context, doc_results, graph_results)
        
        # 文档质量更高，应该获得更大权重
        assert result.doc_weight > result.graph_weight
        assert result.strategy_used == WeightStrategy.QUALITY_DRIVEN
        assert "doc_quality" in result.metadata
        assert "graph_quality" in result.metadata
    
    def test_quality_calculator_no_results(self):
        """测试无结果时的权重计算"""
        calculator = QualityDrivenWeightCalculator()
        context = QueryContext(query="测试查询")
        
        result = calculator.calculate_weights(context)
        
        # 无结果时应该平衡分配
        assert result.doc_weight == 0.5
        assert result.graph_weight == 0.5
    
    def test_quality_score_calculation(self):
        """测试质量分数计算"""
        calculator = QualityDrivenWeightCalculator()
        
        # 高质量结果
        high_quality_results = [
            SearchResult("内容1", 0.9, "document", relevance_score=0.9, confidence_score=0.8),
            SearchResult("内容2", 0.8, "document", relevance_score=0.8, confidence_score=0.7),
            SearchResult("内容3", 0.7, "document", relevance_score=0.7, confidence_score=0.6)
        ]
        
        # 低质量结果
        low_quality_results = [
            SearchResult("内容1", 0.3, "document", relevance_score=0.3, confidence_score=0.2)
        ]
        
        high_score = calculator._calculate_quality_score(high_quality_results)
        low_score = calculator._calculate_quality_score(low_quality_results)
        
        assert high_score > low_score
        assert high_score > 0.5
        assert low_score < 0.5

class TestAdaptiveWeightCalculator:
    """测试自适应权重计算器"""
    
    def test_adaptive_calculator_initialization(self):
        """测试自适应计算器初始化"""
        calculator = AdaptiveWeightCalculator()
        assert hasattr(calculator, 'intent_calculator')
        assert hasattr(calculator, 'quality_calculator')
        assert hasattr(calculator, 'static_calculator')
        assert hasattr(calculator, 'performance_history')
    
    def test_adaptive_calculator_with_intent(self):
        """测试有意图时的自适应计算"""
        calculator = AdaptiveWeightCalculator()
        context = QueryContext(query="测试查询", intent="factual")
        
        result = calculator.calculate_weights(context)
        
        assert result.strategy_used == WeightStrategy.ADAPTIVE
        assert "selected_strategy" in result.metadata
        assert "alternatives" in result.metadata
        assert result.metadata["selected_strategy"] == "intent_driven"
    
    def test_adaptive_calculator_without_intent(self):
        """测试无意图时的自适应计算"""
        calculator = AdaptiveWeightCalculator()
        context = QueryContext(query="测试查询")
        
        doc_results = [SearchResult("文档", 0.8, "document")]
        graph_results = [SearchResult("图谱", 0.6, "graph")]
        
        result = calculator.calculate_weights(context, doc_results, graph_results)
        
        assert result.strategy_used == WeightStrategy.ADAPTIVE
        assert result.metadata["selected_strategy"] == "quality_driven"
    
    def test_adaptive_calculator_performance_update(self):
        """测试性能记录更新"""
        calculator = AdaptiveWeightCalculator()
        context = QueryContext(query="测试查询")
        
        result = WeightResult(
            doc_weight=0.6,
            graph_weight=0.4,
            strategy_used=WeightStrategy.INTENT_DRIVEN,
            confidence=0.8
        )
        
        calculator.update_performance(context, result, feedback_score=0.9)
        
        assert "intent_driven" in calculator.performance_history
        assert len(calculator.performance_history["intent_driven"]) == 1
        assert calculator.performance_history["intent_driven"][0]["feedback_score"] == 0.9

class TestDynamicWeightManager:
    """测试动态权重管理器"""
    
    def test_manager_initialization(self):
        """测试管理器初始化"""
        manager = DynamicWeightManager()
        
        assert WeightStrategy.STATIC in manager.calculators
        assert WeightStrategy.INTENT_DRIVEN in manager.calculators
        assert WeightStrategy.QUALITY_DRIVEN in manager.calculators
        assert WeightStrategy.ADAPTIVE in manager.calculators
        assert manager.current_strategy == WeightStrategy.ADAPTIVE
        assert manager.performance_metrics["total_requests"] == 0
    
    def test_manager_calculate_weights(self):
        """测试权重计算"""
        manager = DynamicWeightManager()
        context = QueryContext(query="测试查询")
        
        result = manager.calculate_weights(context)
        
        assert isinstance(result, WeightResult)
        assert result.strategy_used == WeightStrategy.ADAPTIVE
        assert manager.performance_metrics["total_requests"] == 1
    
    def test_manager_set_strategy(self):
        """测试设置策略"""
        manager = DynamicWeightManager()
        
        manager.set_strategy(WeightStrategy.STATIC)
        assert manager.current_strategy == WeightStrategy.STATIC
        
        # 测试设置不可用策略
        manager.set_strategy(WeightStrategy.GNN_DRIVEN)  # 可能不可用
        # 策略应该保持不变或有适当的警告处理
    
    def test_manager_register_calculator(self):
        """测试注册新计算器"""
        manager = DynamicWeightManager()
        custom_calculator = StaticWeightCalculator(0.8, 0.2)
        
        manager.register_calculator(WeightStrategy.STATIC, custom_calculator)
        
        assert manager.calculators[WeightStrategy.STATIC] == custom_calculator
        assert WeightStrategy.STATIC.value in manager.performance_metrics["strategy_usage"]
    
    def test_manager_get_available_strategies(self):
        """测试获取可用策略"""
        manager = DynamicWeightManager()
        strategies = manager.get_available_strategies()
        
        assert WeightStrategy.STATIC in strategies
        assert WeightStrategy.INTENT_DRIVEN in strategies
        assert WeightStrategy.QUALITY_DRIVEN in strategies
        assert WeightStrategy.ADAPTIVE in strategies
    
    def test_manager_performance_metrics(self):
        """测试性能指标"""
        manager = DynamicWeightManager()
        context = QueryContext(query="测试查询")
        
        # 执行多次计算
        for i in range(5):
            manager.calculate_weights(context)
        
        metrics = manager.get_performance_metrics()
        assert metrics["total_requests"] == 5
        assert metrics["strategy_usage"][WeightStrategy.ADAPTIVE.value] == 5
        assert metrics["average_confidence"] > 0
    
    def test_manager_reset_metrics(self):
        """测试重置性能指标"""
        manager = DynamicWeightManager()
        context = QueryContext(query="测试查询")
        
        # 执行一些计算
        manager.calculate_weights(context)
        assert manager.performance_metrics["total_requests"] == 1
        
        # 重置指标
        manager.reset_metrics()
        assert manager.performance_metrics["total_requests"] == 0
        assert manager.performance_metrics["average_confidence"] == 0.0
    
    def test_manager_cache_functionality(self):
        """测试缓存功能"""
        manager = DynamicWeightManager()
        
        # 测试缓存统计
        cache_stats = manager.get_cache_stats()
        assert "total_entries" in cache_stats
        assert "valid_entries" in cache_stats
        assert "cache_ttl" in cache_stats
        
        # 测试清空缓存
        manager.clear_cache()
        cache_stats_after = manager.get_cache_stats()
        assert cache_stats_after["total_entries"] == 0
    
    def test_manager_weight_history(self):
        """测试权重历史记录"""
        manager = DynamicWeightManager()
        
        # 初始历史应该为空
        history = manager.get_weight_history()
        assert len(history) == 0
        
        # 执行一些计算后检查历史
        context = QueryContext(query="测试查询")
        manager.calculate_weights(context)
        
        # 注意：这里可能需要调用内部方法来记录历史
        # 具体实现取决于DynamicWeightManager的实际代码
    
    @pytest.mark.asyncio
    async def test_manager_async_calculate_weights(self):
        """测试异步权重计算"""
        manager = DynamicWeightManager()
        context = QueryContext(query="测试查询")
        
        result = await manager.calculate_weights_async(context)
        
        assert isinstance(result, WeightResult)
        assert result.strategy_used == WeightStrategy.ADAPTIVE
    
    def test_manager_error_handling(self):
        """测试错误处理"""
        manager = DynamicWeightManager()
        
        # 模拟计算器抛出异常
        with patch.object(manager.calculators[WeightStrategy.ADAPTIVE], 'calculate_weights', 
                         side_effect=Exception("测试异常")):
            context = QueryContext(query="测试查询")
            result = manager.calculate_weights(context)
            
            # 应该回退到静态权重
            assert result.strategy_used == WeightStrategy.STATIC
    
    def test_manager_update_hybrid_weights(self):
        """测试更新混合权重"""
        manager = DynamicWeightManager()
        
        # 如果混合计算器可用
        if WeightStrategy.HYBRID in manager.calculators:
            strategy_weights = {
                WeightStrategy.INTENT_DRIVEN: 0.4,
                WeightStrategy.QUALITY_DRIVEN: 0.6
            }
            
            manager.update_hybrid_weights(strategy_weights)
            # 验证权重是否更新（具体验证方式取决于实现）

class TestHybridWeightCalculator:
    """测试混合权重计算器"""
    
    def test_hybrid_calculator_initialization(self):
        """测试混合计算器初始化"""
        calculators = {
            WeightStrategy.INTENT_DRIVEN: IntentDrivenWeightCalculator(),
            WeightStrategy.QUALITY_DRIVEN: QualityDrivenWeightCalculator()
        }
        
        hybrid_calc = HybridWeightCalculator(calculators)
        assert len(hybrid_calc.calculators) == 2
        assert WeightStrategy.INTENT_DRIVEN in hybrid_calc.calculators
        assert WeightStrategy.QUALITY_DRIVEN in hybrid_calc.calculators
    
    def test_hybrid_calculator_calculate_weights(self):
        """测试混合权重计算"""
        calculators = {
            WeightStrategy.INTENT_DRIVEN: IntentDrivenWeightCalculator(),
            WeightStrategy.QUALITY_DRIVEN: QualityDrivenWeightCalculator()
        }
        
        hybrid_calc = HybridWeightCalculator(calculators)
        context = QueryContext(query="测试查询", intent="factual")
        
        result = hybrid_calc.calculate_weights(context)
        
        assert result.strategy_used == WeightStrategy.HYBRID
        assert 0 <= result.doc_weight <= 1
        assert 0 <= result.graph_weight <= 1
        assert abs(result.doc_weight + result.graph_weight - 1.0) < 1e-6

class TestEnsembleWeightCalculator:
    """测试集成权重计算器"""
    
    def test_ensemble_calculator_initialization(self):
        """测试集成计算器初始化"""
        calculators = [
            IntentDrivenWeightCalculator(),
            QualityDrivenWeightCalculator(),
            StaticWeightCalculator()
        ]
        
        ensemble_calc = EnsembleWeightCalculator(calculators)
        assert len(ensemble_calc.calculators) == 3
    
    def test_ensemble_calculator_calculate_weights(self):
        """测试集成权重计算"""
        calculators = [
            IntentDrivenWeightCalculator(),
            QualityDrivenWeightCalculator(),
            StaticWeightCalculator()
        ]
        
        ensemble_calc = EnsembleWeightCalculator(calculators)
        context = QueryContext(query="测试查询", intent="factual")
        
        result = ensemble_calc.calculate_weights(context)
        
        assert result.strategy_used == WeightStrategy.ENSEMBLE
        assert 0 <= result.doc_weight <= 1
        assert 0 <= result.graph_weight <= 1
        assert "ensemble_method" in result.metadata
        assert "selected_strategy" in result.metadata

if __name__ == "__main__":
    pytest.main([__file__])