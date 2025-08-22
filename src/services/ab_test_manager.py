#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B测试管理器
用于管理权重策略的A/B测试，集成到混合检索服务中
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

try:
    from .ab_testing import (
        ABTestingFramework, ExperimentGroup, Metric, MetricType,
        ExperimentStatus, ExperimentResult
    )
    from .weight_manager import WeightStrategy
except ImportError:
    # 占位符类，用于处理导入失败
    class ABTestingFramework:
        pass
    
    class ExperimentGroup:
        pass
    
    class Metric:
        pass
    
    class MetricType:
        RELEVANCE_SCORE = "relevance_score"
        RESPONSE_TIME = "response_time"
        USER_SATISFACTION = "user_satisfaction"
    
    class WeightStrategy:
        STATIC = "static"
        INTENT_DRIVEN = "intent_driven"
        QUALITY_DRIVEN = "quality_driven"

logger = logging.getLogger(__name__)

@dataclass
class WeightExperimentConfig:
    """权重实验配置"""
    name: str
    description: str
    control_strategy: str
    treatment_strategies: List[str]
    traffic_split: Dict[str, float]  # strategy -> allocation
    duration_days: int = 7
    min_sample_size: int = 100
    target_metrics: List[str] = None

class ABTestManager:
    """A/B测试管理器"""
    
    def __init__(self, 
                 storage_path: Optional[str] = None,
                 significance_level: float = 0.05,
                 auto_cleanup: bool = True):
        """
        初始化A/B测试管理器
        
        Args:
            storage_path: 数据存储路径
            significance_level: 显著性水平
            auto_cleanup: 是否自动清理过期实验
        """
        self.ab_framework = ABTestingFramework(
            storage_path=storage_path,
            significance_level=significance_level
        )
        
        self.auto_cleanup = auto_cleanup
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(hours=1)  # 每小时清理一次
        
        # 预定义的权重策略实验模板
        self.strategy_templates = {
            'basic_comparison': {
                'control': WeightStrategy.STATIC,
                'treatments': [WeightStrategy.INTENT_DRIVEN, WeightStrategy.QUALITY_DRIVEN],
                'metrics': ['relevance_score', 'response_time']
            },
            'advanced_comparison': {
                'control': WeightStrategy.INTENT_DRIVEN,
                'treatments': [WeightStrategy.ADAPTIVE, WeightStrategy.GNN_DRIVEN],
                'metrics': ['relevance_score', 'response_time', 'user_satisfaction']
            },
            'hybrid_comparison': {
                'control': WeightStrategy.QUALITY_DRIVEN,
                'treatments': [WeightStrategy.HYBRID, WeightStrategy.ENSEMBLE],
                'metrics': ['relevance_score', 'response_time', 'user_satisfaction']
            }
        }
        
        logger.info("A/B测试管理器初始化完成")
    
    def create_weight_experiment(self, config: WeightExperimentConfig) -> str:
        """
        创建权重策略A/B测试实验
        
        Args:
            config: 实验配置
            
        Returns:
            实验ID
        """
        # 创建实验组
        groups = []
        
        # 对照组
        control_allocation = config.traffic_split.get(config.control_strategy, 0.5)
        control_group = ExperimentGroup(
            name=f"control_{config.control_strategy}",
            description=f"对照组 - {config.control_strategy}",
            weight_strategy=config.control_strategy,
            weight_config=self._get_default_weight_config(config.control_strategy),
            traffic_allocation=control_allocation,
            is_control=True
        )
        groups.append(control_group)
        
        # 实验组
        remaining_allocation = 1.0 - control_allocation
        treatment_count = len(config.treatment_strategies)
        
        for i, strategy in enumerate(config.treatment_strategies):
            allocation = config.traffic_split.get(strategy, remaining_allocation / treatment_count)
            
            treatment_group = ExperimentGroup(
                name=f"treatment_{strategy}",
                description=f"实验组 - {strategy}",
                weight_strategy=strategy,
                weight_config=self._get_default_weight_config(strategy),
                traffic_allocation=allocation,
                is_control=False
            )
            groups.append(treatment_group)
        
        # 创建评估指标
        metrics = []
        target_metrics = config.target_metrics or ['relevance_score', 'response_time']
        
        for metric_name in target_metrics:
            if metric_name == 'relevance_score':
                metric = Metric(
                    name='relevance_score',
                    type=MetricType.RELEVANCE_SCORE,
                    description='检索结果相关性得分',
                    higher_is_better=True,
                    target_value=0.8,
                    min_sample_size=30
                )
            elif metric_name == 'response_time':
                metric = Metric(
                    name='response_time',
                    type=MetricType.RESPONSE_TIME,
                    description='响应时间（毫秒）',
                    higher_is_better=False,
                    target_value=500.0,
                    min_sample_size=30
                )
            elif metric_name == 'user_satisfaction':
                metric = Metric(
                    name='user_satisfaction',
                    type=MetricType.USER_SATISFACTION,
                    description='用户满意度评分',
                    higher_is_better=True,
                    target_value=4.0,
                    min_sample_size=50
                )
            else:
                # 自定义指标
                metric = Metric(
                    name=metric_name,
                    type=MetricType.CUSTOM,
                    description=f'自定义指标: {metric_name}',
                    higher_is_better=True,
                    min_sample_size=30
                )
            
            metrics.append(metric)
        
        # 创建实验
        experiment_id = self.ab_framework.create_experiment(
            name=config.name,
            description=config.description,
            groups=groups,
            metrics=metrics,
            duration_days=config.duration_days,
            min_sample_size=config.min_sample_size
        )
        
        logger.info(f"创建权重策略实验: {config.name} (ID: {experiment_id})")
        return experiment_id
    
    def create_experiment_from_template(self, 
                                      template_name: str,
                                      experiment_name: str,
                                      duration_days: int = 7) -> Optional[str]:
        """
        从模板创建实验
        
        Args:
            template_name: 模板名称
            experiment_name: 实验名称
            duration_days: 实验持续天数
            
        Returns:
            实验ID，如果模板不存在则返回None
        """
        if template_name not in self.strategy_templates:
            logger.error(f"未知的实验模板: {template_name}")
            return None
        
        template = self.strategy_templates[template_name]
        
        # 构建配置
        config = WeightExperimentConfig(
            name=experiment_name,
            description=f"基于模板 {template_name} 的权重策略对比实验",
            control_strategy=template['control'],
            treatment_strategies=template['treatments'],
            traffic_split={},  # 使用默认分配
            duration_days=duration_days,
            target_metrics=template['metrics']
        )
        
        return self.create_weight_experiment(config)
    
    def start_experiment(self, experiment_id: str) -> bool:
        """
        启动实验
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            是否启动成功
        """
        return self.ab_framework.start_experiment(experiment_id)
    
    def get_user_weight_strategy(self, user_id: str, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        为用户获取权重策略（基于A/B测试分组）
        
        Args:
            user_id: 用户ID
            query: 查询内容
            
        Returns:
            (权重策略名称, 权重配置)
        """
        # 自动清理过期实验
        if self.auto_cleanup:
            self._auto_cleanup()
        
        # 获取活跃实验
        active_experiments = self.ab_framework.get_active_experiments()
        
        if not active_experiments:
            # 没有活跃实验，返回默认策略
            return WeightStrategy.STATIC, self._get_default_weight_config(WeightStrategy.STATIC)
        
        # 选择第一个活跃实验（可以扩展为更复杂的选择逻辑）
        experiment_id = active_experiments[0]['id']
        
        # 为用户分配组
        group_name = self.ab_framework.assign_user_to_group(experiment_id, user_id)
        
        if not group_name:
            # 分配失败，返回默认策略
            return WeightStrategy.STATIC, self._get_default_weight_config(WeightStrategy.STATIC)
        
        # 获取组配置
        experiment = self.ab_framework.experiments[experiment_id]
        group = experiment['groups'][group_name]
        
        return group.weight_strategy, group.weight_config
    
    def record_experiment_result(self,
                               user_id: str,
                               query: str,
                               weight_strategy: str,
                               relevance_score: float,
                               response_time: float,
                               additional_metrics: Optional[Dict[str, float]] = None) -> bool:
        """
        记录实验结果
        
        Args:
            user_id: 用户ID
            query: 查询内容
            weight_strategy: 使用的权重策略
            relevance_score: 相关性得分
            response_time: 响应时间
            additional_metrics: 额外指标
            
        Returns:
            是否记录成功
        """
        # 查找用户当前参与的实验
        if user_id not in self.ab_framework.active_experiments:
            return False
        
        experiment_id = self.ab_framework.active_experiments[user_id]
        
        # 查找用户的组分配
        group_name = None
        for result in reversed(self.ab_framework.results):
            if result.experiment_id == experiment_id and result.user_id == user_id:
                group_name = result.group_name
                break
        
        if not group_name:
            # 尝试重新分配
            group_name = self.ab_framework.assign_user_to_group(experiment_id, user_id)
            if not group_name:
                return False
        
        # 构建指标数据
        metrics = {
            'relevance_score': relevance_score,
            'response_time': response_time
        }
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # 记录结果
        return self.ab_framework.record_result(
            experiment_id=experiment_id,
            user_id=user_id,
            group_name=group_name,
            query=query,
            metrics=metrics,
            metadata={
                'weight_strategy': weight_strategy,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        获取实验结果
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            实验结果统计
        """
        return self.ab_framework.get_experiment_stats(experiment_id)
    
    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """
        获取所有实验信息
        
        Returns:
            实验列表
        """
        experiments = []
        
        for experiment_id, experiment in self.ab_framework.experiments.items():
            exp_info = {
                'id': experiment_id,
                'name': experiment['name'],
                'description': experiment['description'],
                'status': experiment['status'].value,
                'created_at': experiment['created_at'],
                'start_time': experiment.get('start_time'),
                'end_time': experiment.get('end_time'),
                'results_count': experiment['results_count'],
                'groups': list(experiment['groups'].keys())
            }
            experiments.append(exp_info)
        
        return experiments
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """
        停止实验
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            是否停止成功
        """
        return self.ab_framework.stop_experiment(experiment_id)
    
    def export_experiment_data(self, experiment_id: str, format: str = 'json') -> Optional[str]:
        """
        导出实验数据
        
        Args:
            experiment_id: 实验ID
            format: 导出格式
            
        Returns:
            导出的数据字符串
        """
        return self.ab_framework.export_results(experiment_id, format)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要
        
        Returns:
            性能摘要信息
        """
        active_experiments = self.ab_framework.get_active_experiments()
        total_experiments = len(self.ab_framework.experiments)
        total_results = len(self.ab_framework.results)
        
        # 计算各策略的使用情况
        strategy_usage = {}
        for result in self.ab_framework.results:
            strategy = result.metadata.get('weight_strategy', 'unknown')
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
        
        return {
            'total_experiments': total_experiments,
            'active_experiments': len(active_experiments),
            'total_results': total_results,
            'strategy_usage': strategy_usage,
            'last_cleanup': self.last_cleanup.isoformat(),
            'available_templates': list(self.strategy_templates.keys())
        }
    
    def _get_default_weight_config(self, strategy: str) -> Dict[str, Any]:
        """
        获取策略的默认权重配置
        
        Args:
            strategy: 权重策略名称
            
        Returns:
            默认配置字典
        """
        configs = {
            WeightStrategy.STATIC: {
                'doc_weight': 0.7,
                'graph_weight': 0.3
            },
            WeightStrategy.INTENT_DRIVEN: {
                'doc_weight': 0.6,
                'graph_weight': 0.4,
                'intent_threshold': 0.5,
                'adjustment_factor': 0.2
            },
            WeightStrategy.QUALITY_DRIVEN: {
                'doc_weight': 0.5,
                'graph_weight': 0.5,
                'quality_threshold': 0.7,
                'quality_weight': 0.3
            },
            WeightStrategy.ADAPTIVE: {
                'doc_weight': 0.6,
                'graph_weight': 0.4,
                'learning_rate': 0.1,
                'adaptation_window': 10
            },
            WeightStrategy.GNN_DRIVEN: {
                'doc_weight': 0.4,
                'graph_weight': 0.6,
                'gnn_model_path': None,
                'confidence_threshold': 0.8
            },
            WeightStrategy.HYBRID: {
                'strategies': [WeightStrategy.INTENT_DRIVEN, WeightStrategy.QUALITY_DRIVEN],
                'strategy_weights': [0.6, 0.4]
            },
            WeightStrategy.ENSEMBLE: {
                'calculators': [WeightStrategy.INTENT_DRIVEN, WeightStrategy.QUALITY_DRIVEN, WeightStrategy.ADAPTIVE],
                'ensemble_method': 'weighted_average',
                'calculator_weights': [0.4, 0.3, 0.3]
            }
        }
        
        return configs.get(strategy, configs[WeightStrategy.STATIC])
    
    def _auto_cleanup(self):
        """
        自动清理过期实验
        """
        current_time = datetime.now()
        
        if current_time - self.last_cleanup > self.cleanup_interval:
            cleaned_count = self.ab_framework.cleanup_expired_experiments()
            self.last_cleanup = current_time
            
            if cleaned_count > 0:
                logger.info(f"自动清理了 {cleaned_count} 个过期实验")
    
    async def async_record_result(self, *args, **kwargs) -> bool:
        """
        异步记录实验结果
        
        Args:
            与record_experiment_result相同
            
        Returns:
            是否记录成功
        """
        # 在异步环境中执行记录操作
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.record_experiment_result, *args, **kwargs)
    
    def __del__(self):
        """析构函数"""
        try:
            # 清理资源
            if hasattr(self, 'ab_framework'):
                # 可以在这里添加持久化逻辑
                pass
        except Exception as e:
            logger.error(f"A/B测试管理器析构时出错: {e}")