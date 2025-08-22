#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试工具包
提供测试辅助工具、模拟服务、数据生成器等
"""

from .test_helpers import (
    # 测试指标
    MetricsData,
    
    # 异步测试辅助
    AsyncTestHelper,
    
    # 模拟服务工厂
    MockServiceFactory,
    
    # 数据生成器
    DataGenerator,
    
    # 性能监控
    PerformanceMonitor,
    
    # 并发测试辅助
    ConcurrencyTestHelper,
    
    # 数据库测试辅助
    DatabaseTestHelper,
    
    # 文件测试辅助
    FileTestHelper,
    
    # 配置测试辅助
    ConfigTestHelper,
    
    # 装饰器
    timeout,
    retry,
    measure_performance,
    skip_if_no_service,
    
    # 断言辅助函数
    assert_response_structure,
    assert_performance_threshold,
    assert_cache_hit_rate,
    assert_concurrent_success_rate,
    
    # 数据验证函数
    validate_document_structure,
    validate_entity_structure,
    validate_relationship_structure,
    validate_search_result_structure,
    
    # 测试报告生成器
    ReportGenerator,
)

__all__ = [
    # 测试指标
    'MetricsData',
    
    # 异步测试辅助
    'AsyncTestHelper',
    
    # 模拟服务工厂
    'MockServiceFactory',
    
    # 数据生成器
    'DataGenerator',
    
    # 性能监控
    'PerformanceMonitor',
    
    # 并发测试辅助
    'ConcurrencyTestHelper',
    
    # 数据库测试辅助
    'DatabaseTestHelper',
    
    # 文件测试辅助
    'FileTestHelper',
    
    # 配置测试辅助
    'ConfigTestHelper',
    
    # 装饰器
    'timeout',
    'retry',
    'measure_performance',
    'skip_if_no_service',
    
    # 断言辅助函数
    'assert_response_structure',
    'assert_performance_threshold',
    'assert_cache_hit_rate',
    'assert_concurrent_success_rate',
    
    # 数据验证函数
    'validate_document_structure',
    'validate_entity_structure',
    'validate_relationship_structure',
    'validate_search_result_structure',
    
    # 测试报告生成器
    'ReportGenerator',
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'HAG Test Team'
__description__ = 'HAG系统测试工具包'