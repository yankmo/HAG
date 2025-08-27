#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试套件 - 并发查询性能基准测试
测试HAG系统的性能表现，包括并发处理、权重策略效果和资源使用情况
"""

import pytest
import time
import threading
import psutil
import statistics
import concurrent.futures
from typing import List
from unittest.mock import Mock, patch
from dataclasses import dataclass, field
import gc
import tracemalloc

# 导入被测试的模块
try:
    from api import HAGIntegratedAPI
    from src.services import HybridRetrievalService, RetrievalService
    from src.utils.weight_manager import WeightManager
    from src.utils.cache_manager import IntelligentCacheManager
except ImportError as e:
    pytest.skip(f"无法导入必要模块: {e}", allow_module_level=True)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    response_times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    cpu_usage: List[float] = field(default_factory=list)
    cache_hit_rates: List[float] = field(default_factory=list)
    error_count: int = 0
    success_count: int = 0
    
    @property
    def avg_response_time(self) -> float:
        return statistics.mean(self.response_times) if self.response_times else 0
    
    @property
    def p95_response_time(self) -> float:
        return statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) >= 20 else 0
    
    @property
    def p99_response_time(self) -> float:
        return statistics.quantiles(self.response_times, n=100)[98] if len(self.response_times) >= 100 else 0
    
    @property
    def avg_memory_usage(self) -> float:
        return statistics.mean(self.memory_usage) if self.memory_usage else 0
    
    @property
    def avg_cpu_usage(self) -> float:
        return statistics.mean(self.cpu_usage) if self.cpu_usage else 0
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 0


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.metrics = PerformanceMetrics()
        self._monitor_thread = None
    
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _monitor_resources(self):
        """监控系统资源"""
        while self.monitoring:
            try:
                # 获取内存使用情况（MB）
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.metrics.memory_usage.append(memory_mb)
                
                # 获取CPU使用率
                cpu_percent = self.process.cpu_percent()
                self.metrics.cpu_usage.append(cpu_percent)
                
                time.sleep(0.1)  # 每100ms采样一次
            except Exception:
                pass
    
    def record_response_time(self, response_time: float):
        """记录响应时间"""
        self.metrics.response_times.append(response_time)
    
    def record_success(self):
        """记录成功请求"""
        self.metrics.success_count += 1
    
    def record_error(self):
        """记录错误请求"""
        self.metrics.error_count += 1
    
    def record_cache_hit_rate(self, hit_rate: float):
        """记录缓存命中率"""
        self.metrics.cache_hit_rates.append(hit_rate)


class TestConcurrentPerformance:
    """并发性能测试"""
    
    @pytest.fixture
    def mock_services(self):
        """模拟服务依赖"""
        # 模拟向量化服务
        embedding_service = Mock()
        embedding_service.embed_text.return_value = [0.1] * 384
        
        # 模拟向量存储
        vector_store = Mock()
        vector_store.search.return_value = [
            {'content': f'测试文档{i}', 'score': 0.9 - i*0.1, 'metadata': {'source': f'doc{i}.txt'}}
            for i in range(3)
        ]
        
        # 模拟图谱服务
        graph_service = Mock()
        graph_service.search_entities_by_name.return_value = [
            {'name': f'实体{i}', 'type': 'Test', 'properties': {}}
            for i in range(2)
        ]
        graph_service.search_relationships_by_query.return_value = [
            {'source': '实体1', 'target': '实体2', 'type': 'RELATED', 'description': '测试关系'}
        ]
        
        # 模拟LLM服务
        llm_service = Mock()
        llm_service.generate_response.return_value = "这是一个测试回答。"
        
        return {
            'embedding_service': embedding_service,
            'vector_store': vector_store,
            'graph_service': graph_service,
            'llm_service': llm_service
        }
    
    @pytest.fixture
    def hag_api(self, mock_services):
        """创建HAG API实例"""
        with patch('api.get_config'), \
             patch('api.OllamaEmbeddingService', return_value=mock_services['embedding_service']), \
             patch('api.WeaviateVectorStore', return_value=mock_services['vector_store']), \
             patch('api.GraphRetrievalService', return_value=mock_services['graph_service']), \
             patch('api.OllamaLLMService', return_value=mock_services['llm_service']), \
             patch('api.RetrievalService'), \
             patch('api.HybridRetrievalService'), \
             patch('api.RAGPipeline'):
            
            api = HAGIntegratedAPI()
            return api
    
    def test_single_query_baseline(self, hag_api):
        """测试单查询基准性能"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        question = "什么是人工智能？"
        
        # 执行单次查询
        start_time = time.time()
        result = hag_api.query(question)
        end_time = time.time()
        
        response_time = end_time - start_time
        monitor.record_response_time(response_time)
        monitor.record_success()
        
        monitor.stop_monitoring()
        
        # 验证基准性能
        assert response_time < 5.0  # 单查询应在5秒内完成
        assert result.answer is not None
        assert len(result.answer) > 0
        
        print(f"单查询基准响应时间: {response_time:.3f}秒")
        print(f"平均内存使用: {monitor.metrics.avg_memory_usage:.2f}MB")
        print(f"平均CPU使用: {monitor.metrics.avg_cpu_usage:.2f}%")
    
    @pytest.mark.parametrize("concurrent_users", [5, 10, 20])
    def test_concurrent_queries(self, hag_api, concurrent_users):
        """测试并发查询性能"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        questions = [
            "什么是人工智能？",
            "机器学习的基本概念是什么？",
            "深度学习和传统机器学习的区别？",
            "自然语言处理的应用领域有哪些？",
            "计算机视觉技术的发展趋势？"
        ]
        
        def execute_query(question_id):
            """执行单个查询"""
            question = questions[question_id % len(questions)]
            try:
                start_time = time.time()
                result = hag_api.query(question)
                end_time = time.time()
                
                response_time = end_time - start_time
                monitor.record_response_time(response_time)
                monitor.record_success()
                
                return {'success': True, 'response_time': response_time, 'result': result}
            except Exception as e:
                monitor.record_error()
                return {'success': False, 'error': str(e)}
        
        # 并发执行查询
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(execute_query, i) for i in range(concurrent_users * 2)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        monitor.stop_monitoring()
        
        # 分析结果
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        # 性能断言
        assert len(successful_results) > 0, "至少应有一个成功的查询"
        assert monitor.metrics.success_rate >= 0.8, f"成功率应大于80%，实际: {monitor.metrics.success_rate:.2%}"
        assert monitor.metrics.avg_response_time < 10.0, f"平均响应时间应小于10秒，实际: {monitor.metrics.avg_response_time:.3f}秒"
        
        print(f"\n并发用户数: {concurrent_users}")
        print(f"总查询数: {len(results)}")
        print(f"成功查询数: {len(successful_results)}")
        print(f"失败查询数: {len(failed_results)}")
        print(f"成功率: {monitor.metrics.success_rate:.2%}")
        print(f"平均响应时间: {monitor.metrics.avg_response_time:.3f}秒")
        print(f"P95响应时间: {monitor.metrics.p95_response_time:.3f}秒")
        print(f"最大内存使用: {max(monitor.metrics.memory_usage):.2f}MB")
        print(f"平均CPU使用: {monitor.metrics.avg_cpu_usage:.2f}%")
    
    def test_sustained_load(self, hag_api):
        """测试持续负载性能"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        duration = 30  # 持续30秒
        query_interval = 0.5  # 每0.5秒一个查询
        
        questions = [
            "什么是人工智能？",
            "机器学习的应用场景？",
            "深度学习的优势？"
        ]
        
        start_time = time.time()
        query_count = 0
        
        while time.time() - start_time < duration:
            question = questions[query_count % len(questions)]
            
            try:
                query_start = time.time()
                result = hag_api.query(question)
                query_end = time.time()
                
                response_time = query_end - query_start
                monitor.record_response_time(response_time)
                monitor.record_success()
                
                query_count += 1
                
                # 控制查询间隔
                elapsed = time.time() - query_start
                if elapsed < query_interval:
                    time.sleep(query_interval - elapsed)
                    
            except Exception as e:
                monitor.record_error()
                print(f"查询失败: {e}")
        
        monitor.stop_monitoring()
        
        # 性能分析
        total_time = time.time() - start_time
        qps = query_count / total_time
        
        print(f"\n持续负载测试结果:")
        print(f"测试时长: {total_time:.1f}秒")
        print(f"总查询数: {query_count}")
        print(f"QPS: {qps:.2f}")
        print(f"成功率: {monitor.metrics.success_rate:.2%}")
        print(f"平均响应时间: {monitor.metrics.avg_response_time:.3f}秒")
        
        # 性能断言
        assert monitor.metrics.success_rate >= 0.9, "持续负载下成功率应大于90%"
        assert monitor.metrics.avg_response_time < 5.0, "持续负载下平均响应时间应小于5秒"
        assert qps >= 1.0, "QPS应大于1.0"


class TestWeightStrategyPerformance:
    """权重策略性能对比测试"""
    
    @pytest.fixture
    def weight_manager(self):
        """创建权重管理器"""
        config = {
            'strategies': {
                'fixed': {'doc_weight': 0.6, 'graph_weight': 0.4},
                'adaptive': {'base_doc_weight': 0.5, 'base_graph_weight': 0.5},
                'performance_based': {'initial_doc_weight': 0.6, 'initial_graph_weight': 0.4}
            },
            'ab_test': {'enabled': True, 'test_ratio': 0.5}
        }
        return WeightManager(config)
    
    def test_strategy_performance_comparison(self, weight_manager):
        """测试不同权重策略的性能对比"""
        strategies = ['fixed', 'adaptive', 'performance_based']
        strategy_metrics = {}
        
        for strategy in strategies:
            print(f"\n测试策略: {strategy}")
            
            # 设置策略
            weight_manager.set_strategy(strategy)
            
            # 性能监控
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            # 模拟查询
            query_count = 50
            for i in range(query_count):
                start_time = time.time()
                
                # 模拟权重计算
                weights = weight_manager.get_weights(
                    query=f"测试查询{i}",
                    doc_results=[{'score': 0.8}, {'score': 0.7}],
                    graph_results=[{'score': 0.9}, {'score': 0.6}]
                )
                
                # 模拟处理时间
                time.sleep(0.01)  # 10ms处理时间
                
                end_time = time.time()
                response_time = end_time - start_time
                monitor.record_response_time(response_time)
                monitor.record_success()
            
            monitor.stop_monitoring()
            
            # 记录策略性能
            strategy_metrics[strategy] = {
                'avg_response_time': monitor.metrics.avg_response_time,
                'memory_usage': monitor.metrics.avg_memory_usage,
                'cpu_usage': monitor.metrics.avg_cpu_usage
            }
            
            print(f"平均响应时间: {monitor.metrics.avg_response_time:.4f}秒")
            print(f"平均内存使用: {monitor.metrics.avg_memory_usage:.2f}MB")
            print(f"平均CPU使用: {monitor.metrics.avg_cpu_usage:.2f}%")
        
        # 性能对比分析
        print("\n策略性能对比:")
        for strategy, metrics in strategy_metrics.items():
            print(f"{strategy}: 响应时间={metrics['avg_response_time']:.4f}s, "
                  f"内存={metrics['memory_usage']:.2f}MB, CPU={metrics['cpu_usage']:.2f}%")
        
        # 验证所有策略都能正常工作
        for strategy, metrics in strategy_metrics.items():
            assert metrics['avg_response_time'] > 0, f"{strategy}策略响应时间异常"
            assert metrics['avg_response_time'] < 1.0, f"{strategy}策略响应时间过长"
    
    def test_ab_test_performance_impact(self, weight_manager):
        """测试A/B测试对性能的影响"""
        # 启用A/B测试
        weight_manager.config['ab_test']['enabled'] = True
        
        monitor_ab = PerformanceMonitor()
        monitor_ab.start_monitoring()
        
        # 执行带A/B测试的查询
        for i in range(100):
            start_time = time.time()
            weights = weight_manager.get_weights(
                query=f"A/B测试查询{i}",
                doc_results=[{'score': 0.8}],
                graph_results=[{'score': 0.9}]
            )
            end_time = time.time()
            monitor_ab.record_response_time(end_time - start_time)
        
        monitor_ab.stop_monitoring()
        
        # 禁用A/B测试
        weight_manager.config['ab_test']['enabled'] = False
        
        monitor_no_ab = PerformanceMonitor()
        monitor_no_ab.start_monitoring()
        
        # 执行不带A/B测试的查询
        for i in range(100):
            start_time = time.time()
            weights = weight_manager.get_weights(
                query=f"普通查询{i}",
                doc_results=[{'score': 0.8}],
                graph_results=[{'score': 0.9}]
            )
            end_time = time.time()
            monitor_no_ab.record_response_time(end_time - start_time)
        
        monitor_no_ab.stop_monitoring()
        
        # 性能对比
        ab_avg_time = monitor_ab.metrics.avg_response_time
        no_ab_avg_time = monitor_no_ab.metrics.avg_response_time
        performance_overhead = (ab_avg_time - no_ab_avg_time) / no_ab_avg_time * 100
        
        print(f"\nA/B测试性能影响:")
        print(f"启用A/B测试平均响应时间: {ab_avg_time:.4f}秒")
        print(f"禁用A/B测试平均响应时间: {no_ab_avg_time:.4f}秒")
        print(f"性能开销: {performance_overhead:.2f}%")
        
        # 验证A/B测试开销在可接受范围内
        assert performance_overhead < 20, f"A/B测试性能开销过大: {performance_overhead:.2f}%"


class TestMemoryAndResourceMonitoring:
    """内存和资源监控测试"""
    
    def test_memory_leak_detection(self, hag_api):
        """测试内存泄漏检测"""
        # 启动内存跟踪
        tracemalloc.start()
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # 执行大量查询
        for i in range(100):
            question = f"测试查询{i}：什么是人工智能？"
            try:
                result = hag_api.query(question)
                # 强制垃圾回收
                if i % 10 == 0:
                    gc.collect()
            except Exception:
                pass
        
        # 最终内存检查
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # 获取内存快照
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:10]
        
        print(f"\n内存使用情况:")
        print(f"初始内存: {initial_memory:.2f}MB")
        print(f"最终内存: {final_memory:.2f}MB")
        print(f"内存增长: {memory_growth:.2f}MB")
        
        print("\n内存使用热点:")
        for stat in top_stats[:5]:
            print(f"{stat.traceback.format()[-1]}: {stat.size / 1024 / 1024:.2f}MB")
        
        tracemalloc.stop()
        
        # 验证内存增长在合理范围内
        assert memory_growth < 100, f"内存增长过大: {memory_growth:.2f}MB"
    
    def test_cache_performance_impact(self):
        """测试缓存对性能的影响"""
        # 创建缓存管理器
        cache_config = {
            'type': 'lru',
            'max_size': 1000,
            'ttl': 3600
        }
        cache_manager = IntelligentCacheManager(cache_config)
        
        # 测试无缓存性能
        monitor_no_cache = PerformanceMonitor()
        monitor_no_cache.start_monitoring()
        
        for i in range(50):
            key = f"test_key_{i % 10}"  # 重复键以测试缓存效果
            value = f"test_value_{i}"
            
            start_time = time.time()
            # 模拟计算密集型操作
            time.sleep(0.01)
            end_time = time.time()
            
            monitor_no_cache.record_response_time(end_time - start_time)
        
        monitor_no_cache.stop_monitoring()
        
        # 测试有缓存性能
        monitor_with_cache = PerformanceMonitor()
        monitor_with_cache.start_monitoring()
        
        cache_hits = 0
        cache_misses = 0
        
        for i in range(50):
            key = f"test_key_{i % 10}"
            value = f"test_value_{i}"
            
            start_time = time.time()
            
            # 尝试从缓存获取
            cached_value = cache_manager.get(key)
            if cached_value is None:
                # 缓存未命中，模拟计算
                time.sleep(0.01)
                cache_manager.set(key, value)
                cache_misses += 1
            else:
                cache_hits += 1
            
            end_time = time.time()
            monitor_with_cache.record_response_time(end_time - start_time)
        
        monitor_with_cache.stop_monitoring()
        
        # 计算缓存命中率
        hit_rate = cache_hits / (cache_hits + cache_misses)
        
        # 性能对比
        no_cache_avg = monitor_no_cache.metrics.avg_response_time
        with_cache_avg = monitor_with_cache.metrics.avg_response_time
        performance_improvement = (no_cache_avg - with_cache_avg) / no_cache_avg * 100
        
        print(f"\n缓存性能测试结果:")
        print(f"缓存命中率: {hit_rate:.2%}")
        print(f"无缓存平均响应时间: {no_cache_avg:.4f}秒")
        print(f"有缓存平均响应时间: {with_cache_avg:.4f}秒")
        print(f"性能提升: {performance_improvement:.2f}%")
        
        # 验证缓存效果
        assert hit_rate > 0.5, f"缓存命中率过低: {hit_rate:.2%}"
        assert performance_improvement > 0, f"缓存未带来性能提升: {performance_improvement:.2f}%"
    
    def test_resource_cleanup(self):
        """测试资源清理"""
        initial_threads = threading.active_count()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # 创建多个服务实例
        services = []
        for i in range(10):
            cache_config = {'type': 'lru', 'max_size': 100}
            cache_manager = IntelligentCacheManager(cache_config)
            services.append(cache_manager)
        
        # 使用服务
        for i, service in enumerate(services):
            service.set(f"key_{i}", f"value_{i}")
            service.get(f"key_{i}")
        
        # 清理服务
        for service in services:
            if hasattr(service, 'cleanup'):
                service.cleanup()
        
        # 强制垃圾回收
        services.clear()
        gc.collect()
        
        # 等待清理完成
        time.sleep(1)
        
        final_threads = threading.active_count()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        print(f"\n资源清理测试:")
        print(f"初始线程数: {initial_threads}")
        print(f"最终线程数: {final_threads}")
        print(f"线程增长: {final_threads - initial_threads}")
        print(f"内存变化: {final_memory - initial_memory:.2f}MB")
        
        # 验证资源清理效果
        assert final_threads - initial_threads <= 2, "线程泄漏检测"
        assert abs(final_memory - initial_memory) < 50, "内存泄漏检测"


class TestPerformanceRegression:
    """性能回归测试"""
    
    def test_performance_baseline(self, hag_api):
        """建立性能基准"""
        baseline_metrics = {
            'single_query_max_time': 5.0,
            'concurrent_success_rate': 0.8,
            'memory_growth_limit': 100.0,
            'cache_hit_rate_min': 0.5
        }
        
        # 单查询性能测试
        start_time = time.time()
        result = hag_api.query("性能基准测试查询")
        single_query_time = time.time() - start_time
        
        # 验证基准
        assert single_query_time < baseline_metrics['single_query_max_time'], \
            f"单查询时间超出基准: {single_query_time:.3f}s > {baseline_metrics['single_query_max_time']}s"
        
        print(f"\n性能基准测试通过:")
        print(f"单查询时间: {single_query_time:.3f}s (基准: <{baseline_metrics['single_query_max_time']}s)")
        
        # 保存基准数据
        baseline_data = {
            'timestamp': time.time(),
            'single_query_time': single_query_time,
            'baseline_metrics': baseline_metrics
        }
        
        return baseline_data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])