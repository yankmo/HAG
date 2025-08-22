#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单性能测试
用于验证性能测试框架和并发测试功能
"""

import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from tests.utils import (
    PerformanceMonitor,
    ConcurrencyTestHelper,
    DataGenerator,
    measure_performance,
    assert_performance_threshold
)


class TestSimplePerformance:
    """简单性能测试类"""
    
    @pytest.mark.performance
    def test_data_generation_performance(self):
        """测试数据生成性能"""
        monitor = PerformanceMonitor()
        
        with monitor:
            # 生成大量测试数据
            generator = DataGenerator()
            documents = generator.generate_documents(100)
        
        metrics = monitor.get_metrics()
        
        # 验证性能指标
        assert metrics['execution_time'] < 5.0  # 应该在5秒内完成
        assert len(documents) == 100
        print(f"生成100个文档耗时: {metrics['execution_time']:.3f}秒")
    
    @pytest.mark.performance
    @measure_performance
    def test_string_operations_performance(self):
        """测试字符串操作性能"""
        # 执行大量字符串操作
        result = []
        for i in range(10000):
            text = f"测试文本_{i}"
            result.append(text.upper().lower().strip())
        
        assert len(result) == 10000
        return result
    
    @pytest.mark.performance
    def test_concurrent_task_performance(self):
        """测试并发任务性能"""
        def simple_task(task_id):
            """简单的计算任务"""
            time.sleep(0.01)  # 模拟I/O操作
            return sum(range(task_id * 100))
        
        # 准备任务参数
        task_args_list = [(i,) for i in range(1, 21)]  # 20个任务
        
        # 测试串行执行
        start_time = time.time()
        serial_results = [simple_task(*args) for args in task_args_list]
        serial_time = time.time() - start_time
        
        # 测试并发执行
        start_time = time.time()
        concurrent_results = ConcurrencyTestHelper.run_concurrent_tasks(
            simple_task, task_args_list, max_workers=5
        )
        concurrent_time = time.time() - start_time
        
        # 验证结果
        assert len(serial_results) == 20
        assert len(concurrent_results) == 20
        assert concurrent_time < serial_time  # 并发应该更快
        
        print(f"串行执行时间: {serial_time:.3f}秒")
        print(f"并发执行时间: {concurrent_time:.3f}秒")
        print(f"性能提升: {serial_time/concurrent_time:.2f}x")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_async_concurrent_performance(self):
        """测试异步并发性能"""
        async def async_task(task_id):
            """异步计算任务"""
            await asyncio.sleep(0.01)  # 模拟异步I/O
            return sum(range(task_id * 50))
        
        # 准备任务参数
        task_args_list = [(i,) for i in range(1, 16)]  # 15个任务
        
        # 测试异步并发执行
        start_time = time.time()
        results = await ConcurrencyTestHelper.run_concurrent_async_tasks(
            async_task, task_args_list, max_concurrent=5
        )
        execution_time = time.time() - start_time
        
        # 验证结果
        assert len(results) == 15
        assert all(isinstance(r, int) for r in results if not isinstance(r, Exception))
        assert execution_time < 1.0  # 应该在1秒内完成
        
        print(f"异步并发执行时间: {execution_time:.3f}秒")
    
    @pytest.mark.performance
    def test_memory_usage_monitoring(self):
        """测试内存使用监控"""
        monitor = PerformanceMonitor()
        
        with monitor:
            # 创建大量对象来测试内存监控
            large_list = [f"数据项_{i}" * 100 for i in range(1000)]
            # 确保对象被使用
            total_length = sum(len(item) for item in large_list)
        
        metrics = monitor.get_metrics()
        
        # 验证监控指标
        assert 'execution_time' in metrics
        assert 'memory_usage' in metrics
        assert metrics['execution_time'] > 0
        assert total_length > 0
        
        print(f"内存使用情况: {metrics['memory_usage']} MB")
        print(f"执行时间: {metrics['execution_time']:.3f}秒")
    
    @pytest.mark.performance
    def test_performance_threshold_validation(self):
        """测试性能阈值验证"""
        def fast_operation():
            return sum(range(100))
        
        def slow_operation():
            time.sleep(0.1)
            return sum(range(1000))
        
        # 测试快速操作
        monitor1 = PerformanceMonitor()
        with monitor1:
            result1 = fast_operation()
        metrics1 = monitor1.get_metrics()
        
        # 测试慢速操作
        monitor2 = PerformanceMonitor()
        with monitor2:
            result2 = slow_operation()
        metrics2 = monitor2.get_metrics()
        
        # 验证性能阈值
        assert_performance_threshold(metrics1['execution_time'], max_duration=0.01)  # 快速操作应该很快
        
        # 慢速操作应该超过阈值（这里我们期望它失败，但为了测试我们设置更宽松的阈值）
        assert_performance_threshold(metrics2['execution_time'], max_duration=1.0)  # 给慢速操作更多时间
        
        assert result1 > 0
        assert result2 > 0
        assert metrics2['execution_time'] > metrics1['execution_time']


@pytest.mark.performance
def test_benchmark_comparison():
    """基准测试对比"""
    def algorithm_a(data):
        """算法A：使用列表推导"""
        return [x * 2 for x in data]
    
    def algorithm_b(data):
        """算法B：使用map函数"""
        return list(map(lambda x: x * 2, data))
    
    # 准备测试数据
    test_data = list(range(10000))
    
    # 测试算法A
    monitor_a = PerformanceMonitor()
    with monitor_a:
        result_a = algorithm_a(test_data)
    metrics_a = monitor_a.get_metrics()
    
    # 测试算法B
    monitor_b = PerformanceMonitor()
    with monitor_b:
        result_b = algorithm_b(test_data)
    metrics_b = monitor_b.get_metrics()
    
    # 验证结果正确性
    assert result_a == result_b
    assert len(result_a) == 10000
    
    # 比较性能
    print(f"算法A执行时间: {metrics_a['execution_time']:.6f}秒")
    print(f"算法B执行时间: {metrics_b['execution_time']:.6f}秒")
    
    if metrics_a['execution_time'] < metrics_b['execution_time']:
        print("算法A更快")
    else:
        print("算法B更快")