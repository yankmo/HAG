#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
权重管理器单元测试
测试基本权重计算功能
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

# 导入被测试的模块
from src.services.weight_manager import (
    WeightStrategy,
    QueryContext,
    WeightResult,
    StaticWeightCalculator,
    DynamicWeightManager
)

class TestQueryContext:
    """测试QueryContext数据类"""
    
    def test_query_context_initialization(self):
        """测试QueryContext初始化"""
        context = QueryContext(query="测试查询")
        assert context.query == "测试查询"
        assert isinstance(context.timestamp, datetime)

class TestWeightResult:
    """测试WeightResult数据类"""
    
    def test_weight_result_initialization(self):
        """测试WeightResult初始化"""
        result = WeightResult(
            doc_weight=0.6,
            graph_weight=0.4,
            strategy_used=WeightStrategy.STATIC,
            confidence=1.0
        )
        
        assert result.doc_weight == 0.6
        assert result.graph_weight == 0.4
        assert result.strategy_used == WeightStrategy.STATIC
    


class TestStaticWeightCalculator:
    """测试静态权重计算器"""
    
    def test_static_calculator_initialization(self):
        """测试静态权重计算器初始化"""
        calculator = StaticWeightCalculator(doc_weight=0.7, graph_weight=0.3)
        assert calculator.doc_weight == 0.7
        assert calculator.graph_weight == 0.3
    
    def test_static_calculator_weight_calculation(self):
        """测试静态权重计算"""
        calculator = StaticWeightCalculator(doc_weight=0.6, graph_weight=0.4)
        context = QueryContext(query="测试查询")
        
        result = calculator.calculate_weights(context)
        
        assert result.doc_weight == 0.6
        assert result.graph_weight == 0.4







class TestDynamicWeightManager:
    """测试动态权重管理器"""
    
    def test_manager_initialization(self):
        """测试管理器初始化"""
        manager = DynamicWeightManager()
        assert hasattr(manager, 'calculators')
        assert hasattr(manager, 'current_strategy')
    
    def test_manager_calculate_weights(self):
        """测试权重计算"""
        manager = DynamicWeightManager()
        context = QueryContext(query="测试查询")
        
        result = manager.calculate_weights(context)
        
        assert isinstance(result, WeightResult)
        assert result.doc_weight is not None
        assert result.graph_weight is not None

if __name__ == "__main__":
    pytest.main([__file__])