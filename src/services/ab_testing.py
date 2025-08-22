#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B测试框架
用于评估不同权重策略的效果，支持实验设计、数据收集和统计分析
"""

import json
import logging
import random
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import statistics
import uuid
from typing import Dict, List, Any, Optional, Tuple

try:
    from scipy import stats
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None
    np = None

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """实验状态枚举"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MetricType(Enum):
    """指标类型枚举"""
    RELEVANCE_SCORE = "relevance_score"
    RESPONSE_TIME = "response_time"
    USER_SATISFACTION = "user_satisfaction"
    CLICK_THROUGH_RATE = "click_through_rate"
    CONVERSION_RATE = "conversion_rate"
    CUSTOM = "custom"


@dataclass
class ExperimentGroup:
    """实验组配置"""
    name: str
    description: str
    weight_strategy: str
    weight_config: Dict[str, Any]
    traffic_allocation: float  # 流量分配比例 (0-1)
    is_control: bool = False


@dataclass
class Metric:
    """评估指标"""
    name: str
    type: MetricType
    description: str
    higher_is_better: bool = True
    target_value: Optional[float] = None
    min_sample_size: int = 30


@dataclass
class ExperimentResult:
    """实验结果数据点"""
    experiment_id: str
    group_name: str
    user_id: str
    query: str
    timestamp: datetime
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalResult:
    """统计分析结果"""
    metric_name: str
    control_mean: float
    treatment_mean: float
    effect_size: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    sample_size_control: int
    sample_size_treatment: int


class ABTestingFramework:
    """A/B测试框架主类"""
    
    def __init__(self, 
                 storage_path: Optional[str] = None,
                 significance_level: float = 0.05,
                 min_effect_size: float = 0.1):
        """
        初始化A/B测试框架
        
        Args:
            storage_path: 数据存储路径
            significance_level: 显著性水平
            min_effect_size: 最小效应大小
        """
        self.storage_path = storage_path or "data/ab_testing"
        self.significance_level = significance_level
        self.min_effect_size = min_effect_size
        
        # 实验管理
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.active_experiments: Dict[str, str] = {}  # user_id -> experiment_id
        self.results: List[ExperimentResult] = []
        
        # 统计缓存
        self._stats_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 300  # 5分钟缓存
        
        logger.info("A/B测试框架初始化完成")
    
    def create_experiment(self,
                         name: str,
                         description: str,
                         groups: List[ExperimentGroup],
                         metrics: List[Metric],
                         duration_days: int = 7,
                         min_sample_size: int = 100) -> str:
        """
        创建新的A/B测试实验
        
        Args:
            name: 实验名称
            description: 实验描述
            groups: 实验组列表
            metrics: 评估指标列表
            duration_days: 实验持续天数
            min_sample_size: 最小样本量
            
        Returns:
            实验ID
        """
        experiment_id = str(uuid.uuid4())
        
        # 验证实验组配置
        total_allocation = sum(group.traffic_allocation for group in groups)
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError(f"流量分配总和必须为1.0，当前为{total_allocation}")
        
        control_groups = [g for g in groups if g.is_control]
        if len(control_groups) != 1:
            raise ValueError("必须有且仅有一个对照组")
        
        experiment = {
            'id': experiment_id,
            'name': name,
            'description': description,
            'groups': {group.name: group for group in groups},
            'metrics': {metric.name: metric for metric in metrics},
            'status': ExperimentStatus.DRAFT,
            'created_at': datetime.now(),
            'start_time': None,
            'end_time': None,
            'duration_days': duration_days,
            'min_sample_size': min_sample_size,
            'results_count': 0
        }
        
        self.experiments[experiment_id] = experiment
        logger.info(f"创建实验: {name} (ID: {experiment_id})")
        
        return experiment_id
    
    def start_experiment(self, experiment_id: str) -> bool:
        """
        启动实验
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            是否启动成功
        """
        if experiment_id not in self.experiments:
            logger.error(f"实验不存在: {experiment_id}")
            return False
        
        experiment = self.experiments[experiment_id]
        
        if experiment['status'] != ExperimentStatus.DRAFT:
            logger.error(f"实验状态不允许启动: {experiment['status']}")
            return False
        
        experiment['status'] = ExperimentStatus.RUNNING
        experiment['start_time'] = datetime.now()
        experiment['end_time'] = experiment['start_time'] + timedelta(days=experiment['duration_days'])
        
        logger.info(f"启动实验: {experiment['name']} (ID: {experiment_id})")
        return True
    
    def assign_user_to_group(self, experiment_id: str, user_id: str) -> Optional[str]:
        """
        为用户分配实验组
        
        Args:
            experiment_id: 实验ID
            user_id: 用户ID
            
        Returns:
            分配的组名，如果实验不活跃则返回None
        """
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        if experiment['status'] != ExperimentStatus.RUNNING:
            return None
        
        # 检查实验是否过期
        if experiment['end_time'] and datetime.now() > experiment['end_time']:
            self._complete_experiment(experiment_id)
            return None
        
        # 如果用户已经分配过组，返回原组
        if user_id in self.active_experiments:
            existing_exp_id = self.active_experiments[user_id]
            if existing_exp_id == experiment_id:
                # 查找用户的组分配
                for result in reversed(self.results):
                    if result.experiment_id == experiment_id and result.user_id == user_id:
                        return result.group_name
        
        # 基于用户ID的哈希进行确定性分配
        random.seed(hash(f"{experiment_id}_{user_id}"))
        rand_value = random.random()
        
        cumulative_allocation = 0.0
        for group_name, group in experiment['groups'].items():
            cumulative_allocation += group.traffic_allocation
            if rand_value <= cumulative_allocation:
                self.active_experiments[user_id] = experiment_id
                logger.debug(f"用户 {user_id} 分配到组 {group_name}")
                return group_name
        
        # 默认分配到最后一个组
        group_names = list(experiment['groups'].keys())
        if group_names:
            self.active_experiments[user_id] = experiment_id
            return group_names[-1]
        
        return None
    
    def record_result(self,
                     experiment_id: str,
                     user_id: str,
                     group_name: str,
                     query: str,
                     metrics: Dict[str, float],
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        记录实验结果
        
        Args:
            experiment_id: 实验ID
            user_id: 用户ID
            group_name: 组名
            query: 查询内容
            metrics: 指标值
            metadata: 元数据
            
        Returns:
            是否记录成功
        """
        if experiment_id not in self.experiments:
            logger.error(f"实验不存在: {experiment_id}")
            return False
        
        experiment = self.experiments[experiment_id]
        
        if experiment['status'] != ExperimentStatus.RUNNING:
            logger.warning(f"实验未运行，跳过结果记录: {experiment_id}")
            return False
        
        # 验证指标
        for metric_name in metrics.keys():
            if metric_name not in experiment['metrics']:
                logger.warning(f"未知指标: {metric_name}")
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            group_name=group_name,
            user_id=user_id,
            query=query,
            timestamp=datetime.now(),
            metrics=metrics,
            metadata=metadata or {}
        )
        
        self.results.append(result)
        experiment['results_count'] += 1
        
        # 清除统计缓存
        if experiment_id in self._stats_cache:
            del self._stats_cache[experiment_id]
        
        logger.debug(f"记录实验结果: {experiment_id}, 用户: {user_id}, 组: {group_name}")
        return True
    
    def get_experiment_stats(self, experiment_id: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        获取实验统计信息
        
        Args:
            experiment_id: 实验ID
            force_refresh: 是否强制刷新缓存
            
        Returns:
            统计信息字典
        """
        if experiment_id not in self.experiments:
            return {}
        
        # 检查缓存
        cache_key = experiment_id
        if not force_refresh and cache_key in self._stats_cache:
            cache_data = self._stats_cache[cache_key]
            if time.time() - cache_data['timestamp'] < self._cache_ttl:
                return cache_data['stats']
        
        experiment = self.experiments[experiment_id]
        experiment_results = [r for r in self.results if r.experiment_id == experiment_id]
        
        # 按组统计
        group_stats = defaultdict(lambda: {
            'sample_size': 0,
            'metrics': defaultdict(list)
        })
        
        for result in experiment_results:
            group_stats[result.group_name]['sample_size'] += 1
            for metric_name, value in result.metrics.items():
                group_stats[result.group_name]['metrics'][metric_name].append(value)
        
        # 计算统计量
        stats_summary = {
            'experiment_id': experiment_id,
            'name': experiment['name'],
            'status': experiment['status'].value,
            'total_results': len(experiment_results),
            'groups': {},
            'statistical_tests': {}
        }
        
        # 计算各组统计量
        for group_name, group_data in group_stats.items():
            group_summary = {
                'sample_size': group_data['sample_size'],
                'metrics': {}
            }
            
            for metric_name, values in group_data['metrics'].items():
                if values:
                    group_summary['metrics'][metric_name] = {
                        'mean': statistics.mean(values),
                        'median': statistics.median(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
            
            stats_summary['groups'][group_name] = group_summary
        
        # 进行统计检验
        if SCIPY_AVAILABLE:
            stats_summary['statistical_tests'] = self._perform_statistical_tests(
                experiment_id, group_stats
            )
        
        # 缓存结果
        self._stats_cache[cache_key] = {
            'timestamp': time.time(),
            'stats': stats_summary
        }
        
        return stats_summary
    
    def _perform_statistical_tests(self, 
                                  experiment_id: str, 
                                  group_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行统计检验
        
        Args:
            experiment_id: 实验ID
            group_stats: 组统计数据
            
        Returns:
            统计检验结果
        """
        if not SCIPY_AVAILABLE:
            return {'error': 'scipy不可用，无法进行统计检验'}
        
        experiment = self.experiments[experiment_id]
        control_group = None
        
        # 找到对照组
        for group_name, group in experiment['groups'].items():
            if group.is_control:
                control_group = group_name
                break
        
        if not control_group or control_group not in group_stats:
            return {'error': '未找到对照组数据'}
        
        test_results = {}
        
        for metric_name, metric in experiment['metrics'].items():
            if metric_name not in group_stats[control_group]['metrics']:
                continue
            
            control_values = group_stats[control_group]['metrics'][metric_name]
            
            for group_name, group_data in group_stats.items():
                if group_name == control_group or metric_name not in group_data['metrics']:
                    continue
                
                treatment_values = group_data['metrics'][metric_name]
                
                if len(control_values) < 2 or len(treatment_values) < 2:
                    continue
                
                # 执行t检验
                try:
                    t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
                    
                    control_mean = np.mean(control_values)
                    treatment_mean = np.mean(treatment_values)
                    
                    # 计算效应大小 (Cohen's d)
                    pooled_std = np.sqrt(((len(control_values) - 1) * np.var(control_values, ddof=1) + 
                                         (len(treatment_values) - 1) * np.var(treatment_values, ddof=1)) / 
                                        (len(control_values) + len(treatment_values) - 2))
                    
                    effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
                    
                    # 计算置信区间
                    se_diff = pooled_std * np.sqrt(1/len(control_values) + 1/len(treatment_values))
                    df = len(control_values) + len(treatment_values) - 2
                    t_critical = stats.t.ppf(1 - self.significance_level/2, df)
                    
                    diff = treatment_mean - control_mean
                    ci_lower = diff - t_critical * se_diff
                    ci_upper = diff + t_critical * se_diff
                    
                    test_key = f"{metric_name}_{control_group}_vs_{group_name}"
                    test_results[test_key] = StatisticalResult(
                        metric_name=metric_name,
                        control_mean=control_mean,
                        treatment_mean=treatment_mean,
                        effect_size=effect_size,
                        p_value=p_value,
                        confidence_interval=(ci_lower, ci_upper),
                        is_significant=p_value < self.significance_level,
                        sample_size_control=len(control_values),
                        sample_size_treatment=len(treatment_values)
                    )
                    
                except Exception as e:
                    logger.error(f"统计检验失败: {e}")
                    continue
        
        return {k: {
            'metric_name': v.metric_name,
            'control_mean': v.control_mean,
            'treatment_mean': v.treatment_mean,
            'effect_size': v.effect_size,
            'p_value': v.p_value,
            'confidence_interval': v.confidence_interval,
            'is_significant': v.is_significant,
            'sample_size_control': v.sample_size_control,
            'sample_size_treatment': v.sample_size_treatment
        } for k, v in test_results.items()}
    
    def _complete_experiment(self, experiment_id: str) -> bool:
        """
        完成实验
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            是否完成成功
        """
        if experiment_id not in self.experiments:
            return False
        
        experiment = self.experiments[experiment_id]
        experiment['status'] = ExperimentStatus.COMPLETED
        experiment['completed_at'] = datetime.now()
        
        # 移除活跃实验分配
        users_to_remove = [user_id for user_id, exp_id in self.active_experiments.items() 
                          if exp_id == experiment_id]
        for user_id in users_to_remove:
            del self.active_experiments[user_id]
        
        logger.info(f"实验完成: {experiment['name']} (ID: {experiment_id})")
        return True
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """
        停止实验
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            是否停止成功
        """
        return self._complete_experiment(experiment_id)
    
    def get_active_experiments(self) -> List[Dict[str, Any]]:
        """
        获取活跃实验列表
        
        Returns:
            活跃实验列表
        """
        active_experiments = []
        
        for experiment_id, experiment in self.experiments.items():
            if experiment['status'] == ExperimentStatus.RUNNING:
                # 检查是否过期
                if experiment['end_time'] and datetime.now() > experiment['end_time']:
                    self._complete_experiment(experiment_id)
                    continue
                
                active_experiments.append({
                    'id': experiment_id,
                    'name': experiment['name'],
                    'description': experiment['description'],
                    'start_time': experiment['start_time'],
                    'end_time': experiment['end_time'],
                    'results_count': experiment['results_count']
                })
        
        return active_experiments
    
    def export_results(self, experiment_id: str, format: str = 'json') -> Optional[str]:
        """
        导出实验结果
        
        Args:
            experiment_id: 实验ID
            format: 导出格式 ('json', 'csv')
            
        Returns:
            导出的数据字符串
        """
        if experiment_id not in self.experiments:
            return None
        
        experiment_results = [r for r in self.results if r.experiment_id == experiment_id]
        
        if format.lower() == 'json':
            export_data = {
                'experiment': self.experiments[experiment_id],
                'results': [{
                    'experiment_id': r.experiment_id,
                    'group_name': r.group_name,
                    'user_id': r.user_id,
                    'query': r.query,
                    'timestamp': r.timestamp.isoformat(),
                    'metrics': r.metrics,
                    'metadata': r.metadata
                } for r in experiment_results]
            }
            return json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
        
        elif format.lower() == 'csv':
            if not experiment_results:
                return ""
            
            # 获取所有指标名称
            all_metrics = set()
            for result in experiment_results:
                all_metrics.update(result.metrics.keys())
            
            # 构建CSV头部
            headers = ['experiment_id', 'group_name', 'user_id', 'query', 'timestamp']
            headers.extend(sorted(all_metrics))
            
            # 构建CSV数据
            csv_lines = [','.join(headers)]
            
            for result in experiment_results:
                row = [
                    result.experiment_id,
                    result.group_name,
                    result.user_id,
                    f'"{result.query}"',  # 引号包围查询内容
                    result.timestamp.isoformat()
                ]
                
                for metric_name in sorted(all_metrics):
                    value = result.metrics.get(metric_name, '')
                    row.append(str(value))
                
                csv_lines.append(','.join(row))
            
            return '\n'.join(csv_lines)
        
        return None
    
    def cleanup_expired_experiments(self) -> int:
        """
        清理过期实验
        
        Returns:
            清理的实验数量
        """
        cleaned_count = 0
        current_time = datetime.now()
        
        for experiment_id, experiment in list(self.experiments.items()):
            if (experiment['status'] == ExperimentStatus.RUNNING and 
                experiment['end_time'] and 
                current_time > experiment['end_time']):
                
                self._complete_experiment(experiment_id)
                cleaned_count += 1
        
        return cleaned_count