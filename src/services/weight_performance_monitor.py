#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
权重分配性能监控和效果评估指标
用于监控和评估不同权重策略的性能和效果
"""

import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """指标类型"""
    LATENCY = "latency"  # 延迟
    THROUGHPUT = "throughput"  # 吞吐量
    ACCURACY = "accuracy"  # 准确性
    RELEVANCE = "relevance"  # 相关性
    DIVERSITY = "diversity"  # 多样性
    COVERAGE = "coverage"  # 覆盖率
    PRECISION = "precision"  # 精确率
    RECALL = "recall"  # 召回率
    F1_SCORE = "f1_score"  # F1分数
    NDCG = "ndcg"  # 归一化折损累积增益
    MRR = "mrr"  # 平均倒数排名

@dataclass
class PerformanceMetric:
    """性能指标"""
    name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight_strategy: Optional[str] = None
    query_id: Optional[str] = None

@dataclass
class WeightStrategyStats:
    """权重策略统计"""
    strategy_name: str
    total_queries: int = 0
    avg_latency: float = 0.0
    avg_accuracy: float = 0.0
    avg_relevance: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    metrics_history: List[PerformanceMetric] = field(default_factory=list)

@dataclass
class SystemPerformanceSnapshot:
    """系统性能快照"""
    timestamp: datetime
    total_queries: int
    avg_response_time: float
    memory_usage: float
    cpu_usage: float
    active_strategies: List[str]
    strategy_stats: Dict[str, WeightStrategyStats]
    error_rate: float
    throughput: float

class WeightPerformanceMonitor:
    """权重性能监控器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化性能监控器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 配置参数
        self.max_history_size = self.config.get('max_history_size', 10000)
        self.snapshot_interval = self.config.get('snapshot_interval', 60)  # 秒
        self.metric_retention_days = self.config.get('metric_retention_days', 7)
        self.enable_real_time_monitoring = self.config.get('enable_real_time_monitoring', True)
        
        # 数据存储
        self.metrics_history: deque = deque(maxlen=self.max_history_size)
        self.strategy_stats: Dict[str, WeightStrategyStats] = {}
        self.performance_snapshots: deque = deque(maxlen=1000)
        
        # 实时统计
        self.current_queries = 0
        self.total_queries = 0
        self.start_time = datetime.now()
        self.last_snapshot_time = datetime.now()
        
        # 线程安全
        self.lock = threading.RLock()
        
        # 监控线程
        self.monitoring_thread = None
        self.is_monitoring = False
        
        if self.enable_real_time_monitoring:
            self.start_monitoring()
    
    def start_monitoring(self):
        """启动监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("权重性能监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("权重性能监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                time.sleep(self.snapshot_interval)
                if self.is_monitoring:
                    self._take_performance_snapshot()
                    self._cleanup_old_metrics()
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
    
    def record_query_performance(self, 
                                query_id: str,
                                weight_strategy: str,
                                latency: float,
                                results: List[Dict[str, Any]],
                                expected_results: Optional[List[Dict[str, Any]]] = None,
                                metadata: Optional[Dict[str, Any]] = None):
        """记录查询性能"""
        try:
            with self.lock:
                timestamp = datetime.now()
                metadata = metadata or {}
                
                # 记录基础指标
                self._record_metric(MetricType.LATENCY.value, latency, timestamp, 
                                  weight_strategy, query_id, metadata)
                
                # 计算准确性指标
                if expected_results:
                    accuracy = self._calculate_accuracy(results, expected_results)
                    self._record_metric(MetricType.ACCURACY.value, accuracy, timestamp,
                                      weight_strategy, query_id, metadata)
                    
                    precision, recall, f1 = self._calculate_precision_recall_f1(results, expected_results)
                    self._record_metric(MetricType.PRECISION.value, precision, timestamp,
                                      weight_strategy, query_id, metadata)
                    self._record_metric(MetricType.RECALL.value, recall, timestamp,
                                      weight_strategy, query_id, metadata)
                    self._record_metric(MetricType.F1_SCORE.value, f1, timestamp,
                                      weight_strategy, query_id, metadata)
                    
                    ndcg = self._calculate_ndcg(results, expected_results)
                    self._record_metric(MetricType.NDCG.value, ndcg, timestamp,
                                      weight_strategy, query_id, metadata)
                
                # 计算相关性和多样性
                relevance = self._calculate_relevance(results)
                diversity = self._calculate_diversity(results)
                
                self._record_metric(MetricType.RELEVANCE.value, relevance, timestamp,
                                  weight_strategy, query_id, metadata)
                self._record_metric(MetricType.DIVERSITY.value, diversity, timestamp,
                                  weight_strategy, query_id, metadata)
                
                # 更新策略统计
                self._update_strategy_stats(weight_strategy, latency, 
                                           accuracy if expected_results else None,
                                           relevance)
                
                # 更新总计数
                self.current_queries += 1
                self.total_queries += 1
                
        except Exception as e:
            logger.error(f"记录查询性能失败: {e}")
    
    def _record_metric(self, metric_name: str, value: float, timestamp: datetime,
                      weight_strategy: str, query_id: str, metadata: Dict[str, Any]):
        """记录指标"""
        metric = PerformanceMetric(
            name=metric_name,
            value=value,
            timestamp=timestamp,
            metadata=metadata,
            weight_strategy=weight_strategy,
            query_id=query_id
        )
        self.metrics_history.append(metric)
    
    def _update_strategy_stats(self, strategy_name: str, latency: float,
                              accuracy: Optional[float], relevance: float):
        """更新策略统计"""
        if strategy_name not in self.strategy_stats:
            self.strategy_stats[strategy_name] = WeightStrategyStats(strategy_name=strategy_name)
        
        stats = self.strategy_stats[strategy_name]
        stats.total_queries += 1
        
        # 更新平均延迟
        stats.avg_latency = ((stats.avg_latency * (stats.total_queries - 1) + latency) / 
                            stats.total_queries)
        
        # 更新平均相关性
        stats.avg_relevance = ((stats.avg_relevance * (stats.total_queries - 1) + relevance) / 
                              stats.total_queries)
        
        # 更新平均准确性
        if accuracy is not None:
            if stats.avg_accuracy == 0:
                stats.avg_accuracy = accuracy
            else:
                stats.avg_accuracy = ((stats.avg_accuracy * (stats.total_queries - 1) + accuracy) / 
                                     stats.total_queries)
        
        stats.last_updated = datetime.now()
    
    def _calculate_accuracy(self, results: List[Dict[str, Any]], 
                           expected_results: List[Dict[str, Any]]) -> float:
        """计算准确性"""
        try:
            if not results or not expected_results:
                return 0.0
            
            # 简单的ID匹配准确性
            result_ids = {r.get('id', str(i)) for i, r in enumerate(results)}
            expected_ids = {r.get('id', str(i)) for i, r in enumerate(expected_results)}
            
            intersection = len(result_ids.intersection(expected_ids))
            union = len(result_ids.union(expected_ids))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"计算准确性失败: {e}")
            return 0.0
    
    def _calculate_precision_recall_f1(self, results: List[Dict[str, Any]], 
                                      expected_results: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        """计算精确率、召回率和F1分数"""
        try:
            if not results or not expected_results:
                return 0.0, 0.0, 0.0
            
            result_ids = {r.get('id', str(i)) for i, r in enumerate(results)}
            expected_ids = {r.get('id', str(i)) for i, r in enumerate(expected_results)}
            
            true_positives = len(result_ids.intersection(expected_ids))
            false_positives = len(result_ids - expected_ids)
            false_negatives = len(expected_ids - result_ids)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return precision, recall, f1
            
        except Exception as e:
            logger.error(f"计算精确率/召回率/F1失败: {e}")
            return 0.0, 0.0, 0.0
    
    def _calculate_ndcg(self, results: List[Dict[str, Any]], 
                       expected_results: List[Dict[str, Any]], k: int = 10) -> float:
        """计算NDCG@k"""
        try:
            if not results or not expected_results:
                return 0.0
            
            # 构建相关性分数映射
            relevance_map = {}
            for i, result in enumerate(expected_results):
                result_id = result.get('id', str(i))
                relevance_map[result_id] = result.get('relevance', 1.0)
            
            # 计算DCG
            dcg = 0.0
            for i, result in enumerate(results[:k]):
                result_id = result.get('id', str(i))
                relevance = relevance_map.get(result_id, 0.0)
                dcg += relevance / np.log2(i + 2)
            
            # 计算IDCG
            sorted_relevances = sorted(relevance_map.values(), reverse=True)
            idcg = 0.0
            for i, relevance in enumerate(sorted_relevances[:k]):
                idcg += relevance / np.log2(i + 2)
            
            return dcg / idcg if idcg > 0 else 0.0
            
        except Exception as e:
            logger.error(f"计算NDCG失败: {e}")
            return 0.0
    
    def _calculate_relevance(self, results: List[Dict[str, Any]]) -> float:
        """计算平均相关性"""
        try:
            if not results:
                return 0.0
            
            total_relevance = 0.0
            for result in results:
                # 从结果中获取相关性分数
                relevance = result.get('relevance', result.get('score', 0.5))
                if isinstance(relevance, (int, float)):
                    total_relevance += relevance
            
            return total_relevance / len(results)
            
        except Exception as e:
            logger.error(f"计算相关性失败: {e}")
            return 0.0
    
    def _calculate_diversity(self, results: List[Dict[str, Any]]) -> float:
        """计算结果多样性"""
        try:
            if len(results) <= 1:
                return 0.0
            
            # 基于结果类型的多样性
            types = [result.get('type', 'unknown') for result in results]
            unique_types = len(set(types))
            
            return unique_types / len(results)
            
        except Exception as e:
            logger.error(f"计算多样性失败: {e}")
            return 0.0
    
    def _take_performance_snapshot(self):
        """拍摄性能快照"""
        try:
            with self.lock:
                timestamp = datetime.now()
                
                # 计算吞吐量
                time_diff = (timestamp - self.last_snapshot_time).total_seconds()
                throughput = self.current_queries / time_diff if time_diff > 0 else 0.0
                
                # 计算平均响应时间
                recent_latencies = [m.value for m in self.metrics_history 
                                  if m.name == MetricType.LATENCY.value and 
                                  (timestamp - m.timestamp).total_seconds() < self.snapshot_interval]
                avg_response_time = np.mean(recent_latencies) if recent_latencies else 0.0
                
                # 计算错误率
                total_errors = sum(stats.error_count for stats in self.strategy_stats.values())
                error_rate = total_errors / self.total_queries if self.total_queries > 0 else 0.0
                
                # 创建快照
                snapshot = SystemPerformanceSnapshot(
                    timestamp=timestamp,
                    total_queries=self.total_queries,
                    avg_response_time=avg_response_time,
                    memory_usage=self._get_memory_usage(),
                    cpu_usage=self._get_cpu_usage(),
                    active_strategies=list(self.strategy_stats.keys()),
                    strategy_stats=dict(self.strategy_stats),
                    error_rate=error_rate,
                    throughput=throughput
                )
                
                self.performance_snapshots.append(snapshot)
                
                # 重置当前查询计数
                self.current_queries = 0
                self.last_snapshot_time = timestamp
                
        except Exception as e:
            logger.error(f"拍摄性能快照失败: {e}")
    
    def _get_memory_usage(self) -> float:
        """获取内存使用率"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0
        except Exception as e:
            logger.error(f"获取内存使用率失败: {e}")
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0
        except Exception as e:
            logger.error(f"获取CPU使用率失败: {e}")
            return 0.0
    
    def _cleanup_old_metrics(self):
        """清理过期指标"""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.metric_retention_days)
            
            # 清理指标历史
            self.metrics_history = deque(
                [m for m in self.metrics_history if m.timestamp > cutoff_time],
                maxlen=self.max_history_size
            )
            
            # 清理策略统计中的历史指标
            for stats in self.strategy_stats.values():
                stats.metrics_history = [
                    m for m in stats.metrics_history if m.timestamp > cutoff_time
                ]
            
        except Exception as e:
            logger.error(f"清理过期指标失败: {e}")
    
    def get_strategy_performance(self, strategy_name: str) -> Optional[WeightStrategyStats]:
        """获取策略性能统计"""
        return self.strategy_stats.get(strategy_name)
    
    def get_all_strategy_performance(self) -> Dict[str, WeightStrategyStats]:
        """获取所有策略性能统计"""
        return dict(self.strategy_stats)
    
    def get_recent_metrics(self, metric_type: str, 
                          strategy_name: Optional[str] = None,
                          hours: int = 1) -> List[PerformanceMetric]:
        """获取最近的指标"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        metrics = []
        for metric in self.metrics_history:
            if (metric.timestamp > cutoff_time and 
                metric.name == metric_type and
                (strategy_name is None or metric.weight_strategy == strategy_name)):
                metrics.append(metric)
        
        return sorted(metrics, key=lambda x: x.timestamp)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        try:
            with self.lock:
                now = datetime.now()
                uptime = (now - self.start_time).total_seconds()
                
                # 计算总体统计
                total_latencies = [m.value for m in self.metrics_history 
                                 if m.name == MetricType.LATENCY.value]
                avg_latency = np.mean(total_latencies) if total_latencies else 0.0
                
                total_accuracies = [m.value for m in self.metrics_history 
                                  if m.name == MetricType.ACCURACY.value]
                avg_accuracy = np.mean(total_accuracies) if total_accuracies else 0.0
                
                # 策略比较
                strategy_comparison = {}
                for name, stats in self.strategy_stats.items():
                    strategy_comparison[name] = {
                        'total_queries': stats.total_queries,
                        'avg_latency': stats.avg_latency,
                        'avg_accuracy': stats.avg_accuracy,
                        'avg_relevance': stats.avg_relevance,
                        'success_rate': stats.success_rate
                    }
                
                return {
                    'uptime_seconds': uptime,
                    'total_queries': self.total_queries,
                    'avg_latency': avg_latency,
                    'avg_accuracy': avg_accuracy,
                    'active_strategies': len(self.strategy_stats),
                    'strategy_comparison': strategy_comparison,
                    'recent_snapshots': len(self.performance_snapshots),
                    'metrics_count': len(self.metrics_history)
                }
                
        except Exception as e:
            logger.error(f"获取性能摘要失败: {e}")
            return {}
    
    def export_metrics(self, output_path: str, format: str = 'json'):
        """导出指标数据"""
        try:
            data = {
                'export_timestamp': datetime.now().isoformat(),
                'metrics_history': [
                    {
                        'name': m.name,
                        'value': m.value,
                        'timestamp': m.timestamp.isoformat(),
                        'weight_strategy': m.weight_strategy,
                        'query_id': m.query_id,
                        'metadata': m.metadata
                    }
                    for m in self.metrics_history
                ],
                'strategy_stats': {
                    name: {
                        'strategy_name': stats.strategy_name,
                        'total_queries': stats.total_queries,
                        'avg_latency': stats.avg_latency,
                        'avg_accuracy': stats.avg_accuracy,
                        'avg_relevance': stats.avg_relevance,
                        'success_rate': stats.success_rate,
                        'error_count': stats.error_count,
                        'last_updated': stats.last_updated.isoformat()
                    }
                    for name, stats in self.strategy_stats.items()
                },
                'performance_summary': self.get_performance_summary()
            }
            
            if format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            logger.info(f"指标数据已导出到: {output_path}")
            
        except Exception as e:
            logger.error(f"导出指标数据失败: {e}")
            raise
    
    def reset_metrics(self):
        """重置所有指标"""
        with self.lock:
            self.metrics_history.clear()
            self.strategy_stats.clear()
            self.performance_snapshots.clear()
            self.current_queries = 0
            self.total_queries = 0
            self.start_time = datetime.now()
            self.last_snapshot_time = datetime.now()
        
        logger.info("所有指标已重置")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取监控器统计信息"""
        return {
            'is_monitoring': self.is_monitoring,
            'total_queries': self.total_queries,
            'metrics_count': len(self.metrics_history),
            'strategy_count': len(self.strategy_stats),
            'snapshots_count': len(self.performance_snapshots),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'config': self.config
        }


def main():
    """主函数 - 用于测试"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建性能监控器
    monitor = WeightPerformanceMonitor({
        'max_history_size': 1000,
        'snapshot_interval': 10,
        'enable_real_time_monitoring': True
    })
    
    # 模拟一些查询性能数据
    import random
    strategies = ['static', 'intent_driven', 'quality_driven', 'gnn_driven', 'hybrid']
    
    for i in range(100):
        strategy = random.choice(strategies)
        latency = random.uniform(0.1, 2.0)
        
        # 模拟结果
        results = [
            {'id': f'result_{j}', 'score': random.uniform(0, 1), 'type': f'type_{j%3}'}
            for j in range(random.randint(1, 10))
        ]
        
        expected_results = [
            {'id': f'result_{j}', 'relevance': random.uniform(0, 1)}
            for j in range(random.randint(1, 5))
        ]
        
        monitor.record_query_performance(
            query_id=f'query_{i}',
            weight_strategy=strategy,
            latency=latency,
            results=results,
            expected_results=expected_results
        )
        
        time.sleep(0.1)
    
    # 获取性能摘要
    summary = monitor.get_performance_summary()
    print("性能摘要:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    
    # 导出指标
    monitor.export_metrics('performance_metrics.json')
    
    # 停止监控
    monitor.stop_monitoring()


if __name__ == "__main__":
    main()