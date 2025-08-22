#!/usr/bin/env python3
"""
性能监控模块 - 监控查询延迟、缓存命中率和并发查询效果
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import statistics
import json
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    """查询指标"""
    query_id: str
    query_text: str
    start_time: float
    end_time: float
    total_duration: float
    doc_search_duration: float = 0.0
    graph_search_duration: float = 0.0
    cache_hit: bool = False
    concurrent_mode: bool = False
    result_count: int = 0
    error: Optional[str] = None
    user_id: Optional[str] = None
    
    @property
    def parallel_efficiency(self) -> float:
        """并行效率：理论最大时间 / 实际总时间"""
        if not self.concurrent_mode or self.total_duration == 0:
            return 1.0
        
        max_component_time = max(self.doc_search_duration, self.graph_search_duration)
        if max_component_time == 0:
            return 1.0
        
        return max_component_time / self.total_duration

@dataclass
class CacheMetrics:
    """缓存指标"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_errors: int = 0
    avg_hit_time: float = 0.0
    avg_miss_time: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """缓存未命中率"""
        return 1.0 - self.hit_rate

@dataclass
class ConnectionPoolMetrics:
    """连接池指标"""
    pool_name: str
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    created_connections: int = 0
    closed_connections: int = 0
    connection_errors: int = 0
    avg_connection_time: float = 0.0
    max_connections_reached: int = 0
    
    @property
    def utilization_rate(self) -> float:
        """连接池利用率"""
        if self.total_connections == 0:
            return 0.0
        return self.active_connections / self.total_connections

@dataclass
class PerformanceStats:
    """性能统计"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    concurrent_queries: int = 0
    sequential_queries: int = 0
    avg_query_time: float = 0.0
    avg_doc_search_time: float = 0.0
    avg_graph_search_time: float = 0.0
    avg_parallel_efficiency: float = 0.0
    p95_query_time: float = 0.0
    p99_query_time: float = 0.0
    cache_metrics: CacheMetrics = field(default_factory=CacheMetrics)
    connection_pools: Dict[str, ConnectionPoolMetrics] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """查询成功率"""
        if self.total_queries == 0:
            return 0.0
        return self.successful_queries / self.total_queries
    
    @property
    def concurrent_usage_rate(self) -> float:
        """并发查询使用率"""
        if self.total_queries == 0:
            return 0.0
        return self.concurrent_queries / self.total_queries

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, 
                 max_history_size: int = 10000,
                 stats_window_minutes: int = 60,
                 enable_detailed_logging: bool = True):
        
        self.max_history_size = max_history_size
        self.stats_window_minutes = stats_window_minutes
        self.enable_detailed_logging = enable_detailed_logging
        
        # 查询历史记录
        self.query_history: deque[QueryMetrics] = deque(maxlen=max_history_size)
        self.recent_queries: deque[QueryMetrics] = deque()
        
        # 实时统计
        self.stats = PerformanceStats()
        
        # 线程安全锁
        self.lock = threading.RLock()
        
        # 启动清理线程
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_data, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"性能监控器初始化完成 - 历史记录上限: {max_history_size}, 统计窗口: {stats_window_minutes}分钟")
    
    def start_query(self, query_text: str, user_id: Optional[str] = None) -> str:
        """开始查询监控"""
        query_id = f"query_{int(time.time() * 1000000)}_{threading.get_ident()}"
        
        # 创建查询指标对象（暂时存储在线程本地）
        if not hasattr(threading.current_thread(), 'query_metrics'):
            threading.current_thread().query_metrics = {}
        
        threading.current_thread().query_metrics[query_id] = QueryMetrics(
            query_id=query_id,
            query_text=query_text,
            start_time=time.time(),
            end_time=0.0,
            total_duration=0.0,
            user_id=user_id
        )
        
        if self.enable_detailed_logging:
            logger.debug(f"开始查询监控 - ID: {query_id}, 查询: {query_text[:100]}...")
        
        return query_id
    
    def record_component_start(self, query_id: str, component: str):
        """记录组件开始时间"""
        try:
            if hasattr(threading.current_thread(), 'query_metrics'):
                metrics = threading.current_thread().query_metrics.get(query_id)
                if metrics:
                    # 在metrics对象中存储组件开始时间
                    if not hasattr(metrics, 'component_start_times'):
                        metrics.component_start_times = {}
                    metrics.component_start_times[component] = time.time()
                    
                    if self.enable_detailed_logging:
                        logger.debug(f"开始组件监控 - 查询: {query_id}, 组件: {component}")
        except Exception as e:
            logger.warning(f"记录组件开始时间失败: {e}")
    
    def record_component_end(self, query_id: str, component: str, result_count: int = 0, error: Optional[str] = None):
        """记录组件结束时间和结果"""
        try:
            if hasattr(threading.current_thread(), 'query_metrics'):
                metrics = threading.current_thread().query_metrics.get(query_id)
                if metrics and hasattr(metrics, 'component_start_times'):
                    start_time = metrics.component_start_times.get(component)
                    if start_time:
                        duration = time.time() - start_time
                        
                        if component == 'document_search':
                            metrics.doc_search_duration = duration
                        elif component == 'graph_search':
                            metrics.graph_search_duration = duration
                        
                        if self.enable_detailed_logging:
                            logger.debug(f"组件完成 - 查询: {query_id}, 组件: {component}, 耗时: {duration:.3f}s, 结果数: {result_count}, 错误: {error}")
        except Exception as e:
            logger.warning(f"记录组件结束时间失败: {e}")
    
    def record_component_time(self, query_id: str, component: str, duration: float):
        """记录组件执行时间（向后兼容方法）"""
        try:
            if hasattr(threading.current_thread(), 'query_metrics'):
                metrics = threading.current_thread().query_metrics.get(query_id)
                if metrics:
                    if component == 'document_search':
                        metrics.doc_search_duration = duration
                    elif component == 'graph_search':
                        metrics.graph_search_duration = duration
                    
                    if self.enable_detailed_logging:
                        logger.debug(f"记录组件时间 - 查询: {query_id}, 组件: {component}, 耗时: {duration:.3f}s")
        except Exception as e:
            logger.warning(f"记录组件时间失败: {e}")
    
    def record_cache_hit(self, query_id: str, hit: bool, duration: float = 0.0):
        """记录缓存命中情况"""
        try:
            if hasattr(threading.current_thread(), 'query_metrics'):
                metrics = threading.current_thread().query_metrics.get(query_id)
                if metrics:
                    metrics.cache_hit = hit
            
            # 更新缓存统计
            with self.lock:
                self.stats.cache_metrics.total_requests += 1
                if hit:
                    self.stats.cache_metrics.cache_hits += 1
                    # 更新平均命中时间
                    if self.stats.cache_metrics.cache_hits == 1:
                        self.stats.cache_metrics.avg_hit_time = duration
                    else:
                        self.stats.cache_metrics.avg_hit_time = (
                            (self.stats.cache_metrics.avg_hit_time * (self.stats.cache_metrics.cache_hits - 1) + duration) /
                            self.stats.cache_metrics.cache_hits
                        )
                else:
                    self.stats.cache_metrics.cache_misses += 1
                    # 更新平均未命中时间
                    if self.stats.cache_metrics.cache_misses == 1:
                        self.stats.cache_metrics.avg_miss_time = duration
                    else:
                        self.stats.cache_metrics.avg_miss_time = (
                            (self.stats.cache_metrics.avg_miss_time * (self.stats.cache_metrics.cache_misses - 1) + duration) /
                            self.stats.cache_metrics.cache_misses
                        )
            
            if self.enable_detailed_logging:
                logger.debug(f"记录缓存状态 - 查询: {query_id}, 命中: {hit}, 耗时: {duration:.3f}s")
                
        except Exception as e:
            logger.warning(f"记录缓存状态失败: {e}")
    
    def end_query(self, query_id: str, result_count: int = 0, error: Optional[str] = None, concurrent_mode: bool = False):
        """结束查询监控"""
        try:
            if not hasattr(threading.current_thread(), 'query_metrics'):
                return
            
            metrics = threading.current_thread().query_metrics.get(query_id)
            if not metrics:
                return
            
            # 完成查询指标
            metrics.end_time = time.time()
            metrics.total_duration = metrics.end_time - metrics.start_time
            metrics.result_count = result_count
            metrics.error = error
            metrics.concurrent_mode = concurrent_mode
            
            # 添加到历史记录
            with self.lock:
                self.query_history.append(metrics)
                self.recent_queries.append(metrics)
                
                # 更新统计信息
                self._update_stats(metrics)
            
            # 清理线程本地存储
            del threading.current_thread().query_metrics[query_id]
            
            if self.enable_detailed_logging:
                logger.info(f"查询完成 - ID: {query_id}, 耗时: {metrics.total_duration:.3f}s, 结果数: {result_count}, 并发: {concurrent_mode}, 错误: {error}")
                
        except Exception as e:
            logger.error(f"结束查询监控失败: {e}")
    
    def _update_stats(self, metrics: QueryMetrics):
        """更新统计信息"""
        self.stats.total_queries += 1
        
        if metrics.error:
            self.stats.failed_queries += 1
        else:
            self.stats.successful_queries += 1
        
        if metrics.concurrent_mode:
            self.stats.concurrent_queries += 1
        else:
            self.stats.sequential_queries += 1
        
        # 更新平均查询时间
        if self.stats.total_queries == 1:
            self.stats.avg_query_time = metrics.total_duration
            self.stats.avg_doc_search_time = metrics.doc_search_duration
            self.stats.avg_graph_search_time = metrics.graph_search_duration
            self.stats.avg_parallel_efficiency = metrics.parallel_efficiency
        else:
            self.stats.avg_query_time = (
                (self.stats.avg_query_time * (self.stats.total_queries - 1) + metrics.total_duration) /
                self.stats.total_queries
            )
            
            if metrics.doc_search_duration > 0:
                doc_queries = sum(1 for q in self.query_history if q.doc_search_duration > 0)
                if doc_queries > 0:
                    self.stats.avg_doc_search_time = (
                        (self.stats.avg_doc_search_time * (doc_queries - 1) + metrics.doc_search_duration) /
                        doc_queries
                    )
            
            if metrics.graph_search_duration > 0:
                graph_queries = sum(1 for q in self.query_history if q.graph_search_duration > 0)
                if graph_queries > 0:
                    self.stats.avg_graph_search_time = (
                        (self.stats.avg_graph_search_time * (graph_queries - 1) + metrics.graph_search_duration) /
                        graph_queries
                    )
            
            if metrics.concurrent_mode:
                concurrent_count = sum(1 for q in self.query_history if q.concurrent_mode)
                if concurrent_count > 0:
                    self.stats.avg_parallel_efficiency = (
                        (self.stats.avg_parallel_efficiency * (concurrent_count - 1) + metrics.parallel_efficiency) /
                        concurrent_count
                    )
    
    def update_connection_pool_stats(self, pool_name: str, pool_stats: Dict[str, Any]):
        """更新连接池统计信息"""
        try:
            with self.lock:
                if pool_name not in self.stats.connection_pools:
                    self.stats.connection_pools[pool_name] = ConnectionPoolMetrics(pool_name=pool_name)
                
                pool_metrics = self.stats.connection_pools[pool_name]
                pool_metrics.total_connections = pool_stats.get('total_connections', 0)
                pool_metrics.active_connections = pool_stats.get('active_connections', 0)
                pool_metrics.idle_connections = pool_stats.get('idle_connections', 0)
                pool_metrics.created_connections = pool_stats.get('created_connections', 0)
                pool_metrics.closed_connections = pool_stats.get('closed_connections', 0)
                pool_metrics.connection_errors = pool_stats.get('connection_errors', 0)
                pool_metrics.avg_connection_time = pool_stats.get('avg_connection_time', 0.0)
                pool_metrics.max_connections_reached = pool_stats.get('max_connections_reached', 0)
                
        except Exception as e:
            logger.error(f"更新连接池统计失败: {e}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        with self.lock:
            # 计算百分位数
            if self.query_history:
                query_times = [q.total_duration for q in self.query_history if not q.error]
                if query_times:
                    self.stats.p95_query_time = statistics.quantiles(query_times, n=20)[18]  # 95th percentile
                    self.stats.p99_query_time = statistics.quantiles(query_times, n=100)[98]  # 99th percentile
            
            return {
                'timestamp': datetime.now().isoformat(),
                'query_stats': {
                    'total_queries': self.stats.total_queries,
                    'successful_queries': self.stats.successful_queries,
                    'failed_queries': self.stats.failed_queries,
                    'success_rate': self.stats.success_rate,
                    'concurrent_queries': self.stats.concurrent_queries,
                    'sequential_queries': self.stats.sequential_queries,
                    'concurrent_usage_rate': self.stats.concurrent_usage_rate,
                    'avg_query_time': self.stats.avg_query_time,
                    'avg_doc_search_time': self.stats.avg_doc_search_time,
                    'avg_graph_search_time': self.stats.avg_graph_search_time,
                    'avg_parallel_efficiency': self.stats.avg_parallel_efficiency,
                    'p95_query_time': self.stats.p95_query_time,
                    'p99_query_time': self.stats.p99_query_time
                },
                'cache_stats': {
                    'total_requests': self.stats.cache_metrics.total_requests,
                    'cache_hits': self.stats.cache_metrics.cache_hits,
                    'cache_misses': self.stats.cache_metrics.cache_misses,
                    'cache_errors': self.stats.cache_metrics.cache_errors,
                    'hit_rate': self.stats.cache_metrics.hit_rate,
                    'miss_rate': self.stats.cache_metrics.miss_rate,
                    'avg_hit_time': self.stats.cache_metrics.avg_hit_time,
                    'avg_miss_time': self.stats.cache_metrics.avg_miss_time
                },
                'connection_pool_stats': {
                    pool_name: {
                        'total_connections': metrics.total_connections,
                        'active_connections': metrics.active_connections,
                        'idle_connections': metrics.idle_connections,
                        'utilization_rate': metrics.utilization_rate,
                        'created_connections': metrics.created_connections,
                        'closed_connections': metrics.closed_connections,
                        'connection_errors': metrics.connection_errors,
                        'avg_connection_time': metrics.avg_connection_time,
                        'max_connections_reached': metrics.max_connections_reached
                    }
                    for pool_name, metrics in self.stats.connection_pools.items()
                }
            }
    
    def get_recent_queries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取最近的查询记录"""
        with self.lock:
            recent = list(self.recent_queries)[-limit:]
            return [
                {
                    'query_id': q.query_id,
                    'query_text': q.query_text[:200] + '...' if len(q.query_text) > 200 else q.query_text,
                    'start_time': datetime.fromtimestamp(q.start_time).isoformat(),
                    'total_duration': q.total_duration,
                    'doc_search_duration': q.doc_search_duration,
                    'graph_search_duration': q.graph_search_duration,
                    'cache_hit': q.cache_hit,
                    'concurrent_mode': q.concurrent_mode,
                    'result_count': q.result_count,
                    'parallel_efficiency': q.parallel_efficiency,
                    'error': q.error,
                    'user_id': q.user_id
                }
                for q in recent
            ]
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """获取性能洞察"""
        with self.lock:
            insights = {
                'recommendations': [],
                'alerts': [],
                'trends': {}
            }
            
            # 缓存命中率分析
            if self.stats.cache_metrics.hit_rate < 0.5:
                insights['recommendations'].append({
                    'type': 'cache_optimization',
                    'message': f'缓存命中率较低 ({self.stats.cache_metrics.hit_rate:.1%})，建议优化缓存策略或增加缓存容量',
                    'priority': 'high'
                })
            
            # 并发效率分析
            if self.stats.avg_parallel_efficiency < 0.7 and self.stats.concurrent_queries > 0:
                insights['recommendations'].append({
                    'type': 'concurrency_optimization',
                    'message': f'并发效率较低 ({self.stats.avg_parallel_efficiency:.1%})，可能存在资源竞争或负载不均衡',
                    'priority': 'medium'
                })
            
            # 查询成功率分析
            if self.stats.success_rate < 0.95:
                insights['alerts'].append({
                    'type': 'reliability_issue',
                    'message': f'查询成功率较低 ({self.stats.success_rate:.1%})，需要检查系统稳定性',
                    'priority': 'high'
                })
            
            # 连接池利用率分析
            for pool_name, pool_metrics in self.stats.connection_pools.items():
                if pool_metrics.utilization_rate > 0.8:
                    insights['alerts'].append({
                        'type': 'connection_pool_pressure',
                        'message': f'{pool_name} 连接池利用率过高 ({pool_metrics.utilization_rate:.1%})，建议增加连接数',
                        'priority': 'medium'
                    })
            
            # 性能趋势分析
            if len(self.query_history) >= 100:
                recent_100 = list(self.query_history)[-100:]
                earlier_100 = list(self.query_history)[-200:-100] if len(self.query_history) >= 200 else []
                
                if earlier_100:
                    recent_avg = statistics.mean([q.total_duration for q in recent_100 if not q.error])
                    earlier_avg = statistics.mean([q.total_duration for q in earlier_100 if not q.error])
                    
                    if recent_avg > earlier_avg * 1.2:
                        insights['trends']['performance_degradation'] = {
                            'message': '查询性能有下降趋势',
                            'recent_avg': recent_avg,
                            'earlier_avg': earlier_avg,
                            'degradation_percent': ((recent_avg - earlier_avg) / earlier_avg) * 100
                        }
                    elif recent_avg < earlier_avg * 0.8:
                        insights['trends']['performance_improvement'] = {
                            'message': '查询性能有改善趋势',
                            'recent_avg': recent_avg,
                            'earlier_avg': earlier_avg,
                            'improvement_percent': ((earlier_avg - recent_avg) / earlier_avg) * 100
                        }
            
            return insights
    
    def _cleanup_old_data(self):
        """清理过期数据"""
        while True:
            try:
                time.sleep(300)  # 每5分钟清理一次
                
                cutoff_time = time.time() - (self.stats_window_minutes * 60)
                
                with self.lock:
                    # 清理过期的最近查询记录
                    while self.recent_queries and self.recent_queries[0].start_time < cutoff_time:
                        self.recent_queries.popleft()
                
                logger.debug(f"清理过期数据完成 - 保留最近查询: {len(self.recent_queries)}个")
                
            except Exception as e:
                logger.error(f"清理过期数据失败: {e}")
    
    def export_metrics(self, filepath: str):
        """导出性能指标到文件"""
        try:
            stats = self.get_current_stats()
            recent_queries = self.get_recent_queries(1000)
            insights = self.get_performance_insights()
            
            export_data = {
                'export_time': datetime.now().isoformat(),
                'stats': stats,
                'recent_queries': recent_queries,
                'insights': insights
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"性能指标已导出到: {filepath}")
            
        except Exception as e:
            logger.error(f"导出性能指标失败: {e}")

# 全局性能监控器实例
performance_monitor = PerformanceMonitor()