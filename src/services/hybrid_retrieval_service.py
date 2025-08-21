"""
混合检索服务 - 结合文档检索和图谱检索
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# 尝试导入动态权重管理器
try:
    from .weight_manager import DynamicWeightManager, WeightStrategy
except ImportError:
    # 如果导入失败，创建占位符类
    class DynamicWeightManager:
        def __init__(self, *args, **kwargs):
            pass
        
        def calculate_weights(self, *args, **kwargs):
            return {'doc_weight': 0.6, 'graph_weight': 0.4}
        
        async def calculate_weights_async(self, *args, **kwargs):
            return {'doc_weight': 0.6, 'graph_weight': 0.4}
    
    class WeightStrategy:
        STATIC = "static"
        INTENT_DRIVEN = "intent_driven"
        QUALITY_DRIVEN = "quality_driven"
        ADAPTIVE = "adaptive"
        GNN_DRIVEN = "gnn_driven"
        HYBRID = "hybrid"
        ENSEMBLE = "ensemble"

# A/B测试管理器导入
try:
    from .ab_test_manager import ABTestManager
except ImportError:
    # 如果导入失败，创建占位符类
    class ABTestManager:
        def __init__(self, *args, **kwargs):
            pass
        
        def assign_strategy(self, *args, **kwargs):
            return None
        
        def record_result(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)

@dataclass
class HybridResult:
    """混合检索结果"""
    documents: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    combined_score: float
    metadata: Dict[str, Any]

class HybridRetrievalService:
    """混合检索服务 - 结合Weaviate文档检索和Neo4j图谱检索"""
    
    def __init__(self, 
                 document_retrieval_service,  # RetrievalService (Weaviate)
                 graph_retrieval_service,     # GraphRetrievalService (Neo4j)
                 doc_weight: float = 0.6,
                 graph_weight: float = 0.4,
                 weight_strategy: str = WeightStrategy.STATIC,
                 enable_dynamic_weights: bool = False,
                 weight_manager_config: Optional[Dict[str, Any]] = None,
                 enable_ab_testing: bool = False,
                 ab_test_config: Optional[Dict[str, Any]] = None):
        """
        初始化混合检索服务
        
        Args:
            document_retrieval_service: 文档检索服务 (基于Weaviate)
            graph_retrieval_service: 图谱检索服务 (基于Neo4j)
            doc_weight: 文档检索权重 (静态权重时使用)
            graph_weight: 图谱检索权重 (静态权重时使用)
            weight_strategy: 权重策略
            enable_dynamic_weights: 是否启用动态权重
            weight_manager_config: 权重管理器配置
            enable_ab_testing: 是否启用A/B测试
            ab_test_config: A/B测试配置
        """
        self.doc_service = document_retrieval_service
        self.graph_service = graph_retrieval_service
        self.default_doc_weight = doc_weight
        self.default_graph_weight = graph_weight
        self.current_doc_weight = doc_weight
        self.current_graph_weight = graph_weight
        
        # 动态权重管理
        self.enable_dynamic_weights = enable_dynamic_weights
        self.weight_strategy = weight_strategy
        self.weight_manager = None
        
        if enable_dynamic_weights:
            try:
                config = weight_manager_config or {}
                self.weight_manager = DynamicWeightManager(
                    strategy=weight_strategy,
                    **config
                )
                logger.info(f"动态权重管理器初始化完成 - 策略: {weight_strategy}")
            except Exception as e:
                logger.warning(f"动态权重管理器初始化失败，使用静态权重: {e}")
                self.enable_dynamic_weights = False
        
        # 初始化A/B测试管理器
        self.enable_ab_testing = enable_ab_testing
        self.ab_test_manager = None
        if enable_ab_testing:
            try:
                self.ab_test_manager = ABTestManager(
                    config=ab_test_config or {}
                )
                logger.info("A/B测试管理器已初始化")
            except Exception as e:
                logger.error(f"A/B测试管理器初始化失败: {e}")
                self.enable_ab_testing = False
        
        logger.info(f"混合检索服务初始化完成 - 文档权重: {doc_weight}, 图谱权重: {graph_weight}, 动态权重: {enable_dynamic_weights}")
    
    def search_hybrid(self, 
                     query: str, 
                     top_k: int = 10,
                     doc_top_k: Optional[int] = None,
                     graph_top_k: Optional[int] = None,
                     include_intent: bool = True,
                     use_async_weights: bool = False,
                     user_id: Optional[str] = None) -> HybridResult:
        """
        执行混合检索
        
        Args:
            query: 查询文本
            top_k: 总返回结果数
            doc_top_k: 文档检索返回数 (默认根据权重动态计算)
            graph_top_k: 图谱检索返回数 (默认根据权重动态计算)
            include_intent: 是否包含意图识别
            use_async_weights: 是否使用异步权重计算
            user_id: 用户ID (用于A/B测试)
            
        Returns:
            HybridResult: 混合检索结果
        """
        try:
            import time
            start_time = time.time()
            
            # A/B测试策略分配
            assigned_strategy = None
            experiment_id = None
            original_strategy = self.weight_strategy
            if self.enable_ab_testing and self.ab_test_manager and user_id:
                try:
                    assignment = self.ab_test_manager.assign_strategy(user_id, query)
                    if assignment:
                        assigned_strategy = assignment.get('strategy')
                        experiment_id = assignment.get('experiment_id')
                        # 临时更新权重策略
                        if assigned_strategy:
                            self.weight_strategy = assigned_strategy
                            logger.info(f"A/B测试分配策略: {assigned_strategy} (实验ID: {experiment_id})")
                except Exception as e:
                    logger.warning(f"A/B测试策略分配失败: {e}")
            
            # 1. 动态权重计算
            if self.enable_dynamic_weights and self.weight_manager:
                if use_async_weights:
                    # 异步权重计算
                    weights = asyncio.run(self._calculate_dynamic_weights_async(query))
                else:
                    # 同步权重计算
                    weights = self._calculate_dynamic_weights(query)
                
                self.current_doc_weight = weights.get('doc_weight', self.default_doc_weight)
                self.current_graph_weight = weights.get('graph_weight', self.default_graph_weight)
                
                logger.info(f"动态权重计算完成: 文档权重={self.current_doc_weight:.3f}, 图谱权重={self.current_graph_weight:.3f}")
            else:
                # 使用静态权重
                self.current_doc_weight = self.default_doc_weight
                self.current_graph_weight = self.default_graph_weight
            
            # 2. 根据权重设置检索数量
            if doc_top_k is None:
                doc_top_k = max(1, int(top_k * self.current_doc_weight))
            if graph_top_k is None:
                graph_top_k = max(1, int(top_k * self.current_graph_weight))
            
            logger.info(f"开始混合检索: query='{query}', doc_top_k={doc_top_k}, graph_top_k={graph_top_k}")
            
            # 3. 文档检索 (Weaviate)
            doc_results = self._search_documents(query, doc_top_k)
            
            # 4. 图谱检索 (Neo4j)
            graph_results = self._search_graph(query, graph_top_k, include_intent)
            
            # 5. 计算综合得分
            combined_score = self._calculate_combined_score(doc_results, graph_results)
            
            # 6. 构建结果
            result = HybridResult(
                documents=doc_results.get('documents', []),
                entities=graph_results.get('entities', []),
                relationships=graph_results.get('relationships', []),
                combined_score=combined_score,
                metadata={
                    'query': query,
                    'doc_count': len(doc_results.get('documents', [])),
                    'entity_count': len(graph_results.get('entities', [])),
                    'relationship_count': len(graph_results.get('relationships', [])),
                    'doc_weight': self.current_doc_weight,
                    'graph_weight': self.current_graph_weight,
                    'weight_strategy': self.weight_strategy,
                    'dynamic_weights_enabled': self.enable_dynamic_weights,
                    'intent': graph_results.get('intent'),
                    'search_time': time.time() - start_time,
                    'ab_test_enabled': self.enable_ab_testing,
                    'assigned_strategy': assigned_strategy,
                    'experiment_id': experiment_id
                }
            )
            
            # 记录A/B测试结果
            if self.enable_ab_testing and self.ab_test_manager and experiment_id and user_id:
                try:
                    self.ab_test_manager.record_result(
                        experiment_id=experiment_id,
                        user_id=user_id,
                        query=query,
                        result_score=combined_score,
                        result_count=len(result.documents) + len(result.entities),
                        response_time=time.time() - start_time,
                        metadata={
                            'doc_weight': self.current_doc_weight,
                            'graph_weight': self.current_graph_weight,
                            'strategy': assigned_strategy or self.weight_strategy
                        }
                    )
                except Exception as e:
                    logger.warning(f"A/B测试结果记录失败: {e}")
            
            # 恢复原始策略（如果被A/B测试临时修改）
            if assigned_strategy:
                self.weight_strategy = original_strategy
            
            logger.info(f"混合检索完成: 文档{len(result.documents)}个, 实体{len(result.entities)}个, 关系{len(result.relationships)}个")
            return result
            
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            return HybridResult([], [], [], 0.0, {'error': str(e)})
    
    def _search_documents(self, query: str, top_k: int) -> Dict[str, Any]:
        """搜索文档 (Weaviate)"""
        try:
            # 使用混合搜索获取最佳结果
            hybrid_result = self.doc_service.search_hybrid(query, top_k)
            
            # 格式化文档结果 - 使用混合结果
            documents = []
            for result in hybrid_result.hybrid_results:
                documents.append({
                    'content': result.content,
                    'metadata': result.metadata,
                    'score': result.score,
                    'source': 'weaviate'
                })
            
            return {
                'documents': documents,
                'total_score': sum(doc['score'] for doc in documents)
            }
            
        except Exception as e:
            logger.error(f"文档检索失败: {e}")
            return {'documents': [], 'total_score': 0.0}
    
    def _search_graph(self, query: str, top_k: int, include_intent: bool = True) -> Dict[str, Any]:
        """搜索图谱 (Neo4j)"""
        try:
            results = {
                'entities': [],
                'relationships': [],
                'intent': None
            }
            
            # 意图识别
            if include_intent:
                try:
                    intent_result = self.graph_service.recognize_intent(query)
                    results['intent'] = {
                        'intent': intent_result.intent_type,
                        'confidence': intent_result.confidence,
                        'entities': intent_result.entities,
                        'relations': intent_result.relations
                    }
                    
                    # 基于意图的搜索
                    if intent_result.entities:
                        for entity in intent_result.entities[:top_k//2]:
                            entity_results = self.graph_service.search_entities_by_name(entity, top_k//4)
                            results['entities'].extend(entity_results)
                    
                    if intent_result.relations:
                        for relation in intent_result.relations[:top_k//2]:
                            rel_results = self.graph_service.search_relationships_by_type(relation, top_k//4)
                            results['relationships'].extend(rel_results)
                            
                except Exception as e:
                    logger.warning(f"意图识别失败，使用通用搜索: {e}")
            
            # 如果意图搜索结果不足，补充通用搜索
            if len(results['entities']) < top_k // 2:
                # 搜索实体
                entity_results = self.graph_service.search_entities_by_name(query, top_k // 2)
                results['entities'].extend(entity_results)
            
            if len(results['relationships']) < top_k // 2:
                # 搜索关系 - 使用新的查询方法搜索相关关系
                relationship_results = self.graph_service.search_relationships_by_query(query, top_k // 2)
                logger.info(f"关系搜索结果: {relationship_results}")  # 添加调试日志
                results['relationships'].extend(relationship_results)
            
            # 去重并限制数量
            results['entities'] = results['entities'][:top_k//2]
            results['relationships'] = results['relationships'][:top_k//2]
            
            return results
            
        except Exception as e:
            logger.error(f"图谱检索失败: {e}")
            return {'entities': [], 'relationships': [], 'intent': None}
    
    def _calculate_dynamic_weights(self, query: str) -> Dict[str, float]:
        """计算动态权重 (同步)"""
        try:
            if not self.weight_manager:
                return {'doc_weight': self.default_doc_weight, 'graph_weight': self.default_graph_weight}
            
            # 准备权重计算的上下文信息
            context = {
                'query': query,
                'doc_service_available': self.doc_service is not None,
                'graph_service_available': self.graph_service is not None,
                'strategy': self.weight_strategy
            }
            
            weights = self.weight_manager.calculate_weights(
                query=query,
                context=context
            )
            
            return weights
            
        except Exception as e:
            logger.error(f"动态权重计算失败: {e}")
            return {'doc_weight': self.default_doc_weight, 'graph_weight': self.default_graph_weight}
    
    async def _calculate_dynamic_weights_async(self, query: str) -> Dict[str, float]:
        """计算动态权重 (异步)"""
        try:
            if not self.weight_manager:
                return {'doc_weight': self.default_doc_weight, 'graph_weight': self.default_graph_weight}
            
            # 准备权重计算的上下文信息
            context = {
                'query': query,
                'doc_service_available': self.doc_service is not None,
                'graph_service_available': self.graph_service is not None,
                'strategy': self.weight_strategy
            }
            
            weights = await self.weight_manager.calculate_weights_async(
                query=query,
                context=context
            )
            
            return weights
            
        except Exception as e:
            logger.error(f"异步动态权重计算失败: {e}")
            return {'doc_weight': self.default_doc_weight, 'graph_weight': self.default_graph_weight}
    
    def _calculate_combined_score(self, doc_results: Dict[str, Any], graph_results: Dict[str, Any]) -> float:
        """计算综合得分"""
        try:
            doc_score = doc_results.get('total_score', 0.0)
            
            # 计算图谱得分 (基于结果数量和相关性)
            entity_count = len(graph_results.get('entities', []))
            relationship_count = len(graph_results.get('relationships', []))
            graph_score = (entity_count + relationship_count) * 0.1  # 简单的得分计算
            
            # 使用当前权重计算综合得分
            combined_score = (doc_score * self.current_doc_weight + graph_score * self.current_graph_weight)
            
            return combined_score
            
        except Exception as e:
            logger.error(f"计算综合得分失败: {e}")
            return 0.0
    
    def search_with_context(self, 
                           query: str, 
                           context_entities: List[str] = None,
                           context_relations: List[str] = None,
                           top_k: int = 10) -> HybridResult:
        """
        基于上下文的混合检索
        
        Args:
            query: 查询文本
            context_entities: 上下文实体列表
            context_relations: 上下文关系列表
            top_k: 返回结果数
            
        Returns:
            HybridResult: 混合检索结果
        """
        try:
            # 扩展查询以包含上下文
            expanded_query = query
            if context_entities:
                expanded_query += " " + " ".join(context_entities)
            if context_relations:
                expanded_query += " " + " ".join(context_relations)
            
            # 执行混合检索
            result = self.search_hybrid(expanded_query, top_k)
            
            # 在元数据中记录上下文信息
            result.metadata.update({
                'context_entities': context_entities or [],
                'context_relations': context_relations or [],
                'original_query': query,
                'expanded_query': expanded_query
            })
            
            return result
            
        except Exception as e:
            logger.error(f"上下文混合检索失败: {e}")
            return HybridResult([], [], [], 0.0, {'error': str(e)})
    
    def update_weight_strategy(self, strategy: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """更新权重策略"""
        try:
            if not self.enable_dynamic_weights:
                logger.warning("动态权重未启用，无法更新权重策略")
                return False
            
            self.weight_strategy = strategy
            
            if self.weight_manager and hasattr(self.weight_manager, 'update_strategy'):
                self.weight_manager.update_strategy(strategy, config or {})
                logger.info(f"权重策略已更新为: {strategy}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"更新权重策略失败: {e}")
            return False
    
    def get_weight_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取权重历史记录"""
        try:
            if self.weight_manager and hasattr(self.weight_manager, 'get_weight_history'):
                return self.weight_manager.get_weight_history(limit)
            return []
            
        except Exception as e:
            logger.error(f"获取权重历史失败: {e}")
            return []
    
    def clear_weight_cache(self) -> bool:
        """清空权重缓存"""
        try:
            if self.weight_manager and hasattr(self.weight_manager, 'clear_cache'):
                self.weight_manager.clear_cache()
                logger.info("权重缓存已清空")
                return True
            return False
            
        except Exception as e:
            logger.error(f"清空权重缓存失败: {e}")
            return False
    
    def get_weight_cache_stats(self) -> Dict[str, Any]:
        """获取权重缓存统计信息"""
        try:
            if self.weight_manager and hasattr(self.weight_manager, 'get_cache_stats'):
                return self.weight_manager.get_cache_stats()
            return {}
            
        except Exception as e:
            logger.error(f"获取权重缓存统计失败: {e}")
            return {}
    
    def create_ab_experiment(self, 
                           name: str, 
                           strategies: List[str], 
                           traffic_split: Optional[Dict[str, float]] = None,
                           description: str = "") -> Optional[str]:
        """创建A/B测试实验"""
        try:
            if not self.enable_ab_testing or not self.ab_test_manager:
                logger.warning("A/B测试未启用，无法创建实验")
                return None
            
            experiment_id = self.ab_test_manager.create_experiment(
                name=name,
                strategies=strategies,
                traffic_split=traffic_split,
                description=description
            )
            
            if experiment_id:
                logger.info(f"A/B测试实验已创建: {name} (ID: {experiment_id})")
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"创建A/B测试实验失败: {e}")
            return None
    
    def start_ab_experiment(self, experiment_id: str) -> bool:
        """启动A/B测试实验"""
        try:
            if not self.enable_ab_testing or not self.ab_test_manager:
                logger.warning("A/B测试未启用，无法启动实验")
                return False
            
            success = self.ab_test_manager.start_experiment(experiment_id)
            
            if success:
                logger.info(f"A/B测试实验已启动: {experiment_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"启动A/B测试实验失败: {e}")
            return False
    
    def stop_ab_experiment(self, experiment_id: str) -> bool:
        """停止A/B测试实验"""
        try:
            if not self.enable_ab_testing or not self.ab_test_manager:
                logger.warning("A/B测试未启用，无法停止实验")
                return False
            
            success = self.ab_test_manager.stop_experiment(experiment_id)
            
            if success:
                logger.info(f"A/B测试实验已停止: {experiment_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"停止A/B测试实验失败: {e}")
            return False
    
    def get_ab_experiment_stats(self, experiment_id: str) -> Dict[str, Any]:
        """获取A/B测试实验统计信息"""
        try:
            if not self.enable_ab_testing or not self.ab_test_manager:
                return {}
            
            return self.ab_test_manager.get_experiment_stats(experiment_id)
            
        except Exception as e:
            logger.error(f"获取A/B测试实验统计失败: {e}")
            return {}
    
    def get_active_ab_experiments(self) -> List[Dict[str, Any]]:
        """获取活跃的A/B测试实验列表"""
        try:
            if not self.enable_ab_testing or not self.ab_test_manager:
                return []
            
            return self.ab_test_manager.get_active_experiments()
            
        except Exception as e:
            logger.error(f"获取活跃A/B测试实验失败: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取混合检索服务统计信息"""
        try:
            doc_stats = self.doc_service.get_stats() if hasattr(self.doc_service, 'get_stats') else {}
            graph_stats = self.graph_service.get_stats() if hasattr(self.graph_service, 'get_stats') else {}
            
            stats = {
                'document_service': doc_stats,
                'graph_service': graph_stats,
                'weights': {
                    'current_doc_weight': self.current_doc_weight,
                    'current_graph_weight': self.current_graph_weight,
                    'default_doc_weight': self.default_doc_weight,
                    'default_graph_weight': self.default_graph_weight,
                    'strategy': self.weight_strategy,
                    'dynamic_enabled': self.enable_dynamic_weights
                },
                'ab_testing': {
                    'enabled': self.enable_ab_testing,
                    'active_experiments': len(self.get_active_ab_experiments()) if self.enable_ab_testing else 0
                },
                'service_type': 'hybrid_retrieval'
            }
            
            # 添加权重管理器统计信息
            if self.enable_dynamic_weights and self.weight_manager:
                stats['weight_manager'] = {
                    'cache_stats': self.get_weight_cache_stats(),
                    'history_count': len(self.get_weight_history(1))
                }
            
            # 添加A/B测试详细统计信息
            if self.enable_ab_testing and self.ab_test_manager:
                active_experiments = self.get_active_ab_experiments()
                stats['ab_testing'].update({
                    'experiments': [
                        {
                            'id': exp.get('id'),
                            'name': exp.get('name'),
                            'status': exp.get('status'),
                            'strategies': exp.get('strategies', []),
                            'participant_count': exp.get('participant_count', 0)
                        }
                        for exp in active_experiments
                    ]
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {'error': str(e)}