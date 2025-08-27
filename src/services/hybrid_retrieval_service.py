"""
混合检索服务 - 结合文档检索和图谱检索
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from .cache_manager import IntelligentCacheManager
from .performance_monitor import performance_monitor

logger = logging.getLogger(__name__)

# 导入向量化图谱检索服务

# 尝试导入动态权重管理器
try:
    from .weight_manager import DynamicWeightManager, WeightStrategy
except ImportError:
    # 如果导入失败，创建占位符类
    class DynamicWeightManager:
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
    from .ab_testing import ABTestingFramework
except ImportError:
    # 如果导入失败，创建占位符类
    class ABTestingFramework:
        def assign_user(self, *args, **kwargs):
            return None
        
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
    """混合检索服务 - 结合Weaviate文档检索和向量化图谱检索"""
    
    def __init__(self, 
                 document_retrieval_service,  # RetrievalService (Weaviate)
                 graph_retrieval_service,     # VectorizedGraphRetrievalService (向量化图谱检索)
                 doc_weight: float = 0.6,
                 graph_weight: float = 0.4,
                 weight_strategy: str = WeightStrategy.STATIC,
                 enable_dynamic_weights: bool = False,
                 weight_manager_config: Optional[Dict[str, Any]] = None,
                 enable_ab_testing: bool = False,
                 ab_test_config: Optional[Dict[str, Any]] = None,
                 enable_concurrent_queries: bool = True,
                 max_workers: int = 4,
                 cache_config: Optional[Dict[str, Any]] = None,
                 connection_pool_config: Optional[Dict[str, Any]] = None,
                 enable_performance_monitoring: bool = False):
        """
        初始化混合检索服务
        
        Args:
            document_retrieval_service: 文档检索服务 (基于Weaviate)
            graph_retrieval_service: 向量化图谱检索服务 (基于向量相似度)
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
        self.enable_concurrent_queries = enable_concurrent_queries
        self.max_workers = max_workers
        
        # 性能监控配置
        self.enable_performance_monitoring = enable_performance_monitoring
        self.performance_monitor = performance_monitor if enable_performance_monitoring else None
        if self.enable_performance_monitoring:
            logger.info("性能监控已启用")
        else:
            logger.info("性能监控已禁用")
        
        # 初始化线程池执行器
        if enable_concurrent_queries:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            logger.info(f"并发查询已启用 - 最大工作线程: {max_workers}")
        else:
            self.executor = None
            logger.info("使用串行查询模式")
        
        # 初始化智能缓存管理器
        if cache_config:
            self.cache_manager = IntelligentCacheManager(cache_config)
        else:
            self.cache_manager = None
        
        # 初始化连接池
        self.connection_pool_manager = None  # 暂时设置为None，避免未定义错误
        self._setup_connection_pools(connection_pool_config or {})
        
        # 性能统计
        self.query_stats = {
            'total_queries': 0,
            'concurrent_queries': 0,
            'avg_response_time': 0.0,
            'doc_query_time': 0.0,
            'graph_query_time': 0.0
        }
        
        # 动态权重管理
        self.enable_dynamic_weights = enable_dynamic_weights
        self.weight_strategy = weight_strategy
        self.weight_manager = None
        
        if enable_dynamic_weights:
            try:
                config = weight_manager_config or {}
                self.weight_manager = DynamicWeightManager(
                    neo4j_service=graph_retrieval_service,
                    doc_service=document_retrieval_service,
                    config=config
                )
                # 设置权重策略
                if hasattr(self.weight_manager, 'set_strategy'):
                    from .weight_manager import WeightStrategy
                    # 将字符串转换为WeightStrategy枚举
                    if isinstance(weight_strategy, str):
                        try:
                            strategy_enum = WeightStrategy(weight_strategy)
                            self.weight_manager.set_strategy(strategy_enum)
                        except ValueError:
                            logger.warning(f"未知的权重策略: {weight_strategy}，使用默认策略")
                            self.weight_manager.set_strategy(WeightStrategy.ADAPTIVE)
                    else:
                        self.weight_manager.set_strategy(weight_strategy)
                logger.info(f"动态权重管理器初始化完成 - 策略: {weight_strategy}")
            except Exception as e:
                logger.warning(f"动态权重管理器初始化失败，使用静态权重: {e}")
                self.enable_dynamic_weights = False
        
        # 初始化A/B测试管理器
        self.enable_ab_testing = enable_ab_testing
        self.ab_test_manager = None
        if enable_ab_testing:
            try:
                config = ab_test_config or {}
                self.ab_test_manager = ABTestingFramework(
                    storage_path=config.get('storage_path', './ab_test_data'),
                    significance_level=config.get('significance_level', 0.05),
                    min_effect_size=config.get('min_effect_size', 0.1)
                )
                logger.info("A/B测试管理器已初始化")
            except Exception as e:
                logger.error(f"A/B测试管理器初始化失败: {e}")
                self.enable_ab_testing = False
        
        logger.info(f"混合检索服务初始化完成 - 文档权重: {doc_weight}, 图谱权重: {graph_weight}, 动态权重: {enable_dynamic_weights}, 缓存: {bool(cache_config)}, 连接池: {bool(connection_pool_config)}")
    
    def _setup_connection_pools(self, config: Dict[str, Any]):
        """设置连接池"""
        try:
            # 检查连接池管理器是否可用
            if self.connection_pool_manager is None:
                logger.info("连接池管理器未初始化，跳过连接池设置")
                return
            
            # 设置Neo4j连接池
            if 'neo4j' in config and self.graph_service:
                neo4j_config = config['neo4j']
                self.connection_pool_manager.create_neo4j_pool(
                    name='hybrid_neo4j',
                    uri=neo4j_config.get('uri', 'bolt://localhost:7687'),
                    username=neo4j_config.get('username', 'neo4j'),
                    password=neo4j_config.get('password', 'password'),
                    database=neo4j_config.get('database', 'neo4j'),
                    min_connections=neo4j_config.get('min_connections', 2),
                    max_connections=neo4j_config.get('max_connections', 10),
                    max_connection_age=neo4j_config.get('max_connection_age', 3600),
                    max_idle_time=neo4j_config.get('max_idle_time', 300)
                )
                logger.info("Neo4j连接池设置完成")
            
            # 设置Weaviate连接池
            if 'weaviate' in config and self.doc_service:
                weaviate_config = config['weaviate']
                self.connection_pool_manager.create_weaviate_pool(
                    name='hybrid_weaviate',
                    url=weaviate_config.get('url', 'http://localhost:8080'),
                    api_key=weaviate_config.get('api_key'),
                    additional_headers=weaviate_config.get('additional_headers', {}),
                    min_connections=weaviate_config.get('min_connections', 2),
                    max_connections=weaviate_config.get('max_connections', 8),
                    max_connection_age=weaviate_config.get('max_connection_age', 3600),
                    max_idle_time=weaviate_config.get('max_idle_time', 300)
                )
                logger.info("Weaviate连接池设置完成")
                
        except Exception as e:
            logger.warning(f"连接池设置失败: {e}，将使用默认连接方式")
    
    async def search_hybrid(self, 
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
        # 开始性能监控
        query_id = None
        if self.performance_monitor:
            query_id = self.performance_monitor.start_query(query, user_id)
        
        try:
            import time
            start_time = time.time()
            
            # A/B测试策略分配
            assigned_strategy = None
            experiment_id = None
            original_strategy = self.weight_strategy
            if self.enable_ab_testing and self.ab_test_manager and user_id:
                try:
                    assignment = self.ab_test_manager.assign_user(experiment_id='weight_strategy_test', user_id=user_id)
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
            
            # 更新查询统计
            self.query_stats['total_queries'] += 1
            
            # 3. 执行并发或串行查询
            if self.enable_concurrent_queries and self.executor:
                doc_results, graph_results = await self._search_concurrent_async(query, doc_top_k, graph_top_k, include_intent, query_id)
                self.query_stats['concurrent_queries'] += 1
            else:
                # 3. 文档检索 (Weaviate)
                doc_results = await self._search_documents_async(query, doc_top_k, query_id)
                
                # 4. 图谱检索 (Neo4j)
                graph_results = await self._search_graph_async(query, graph_top_k, include_intent, query_id)
            
            # 5. 计算综合得分
            combined_score = self._calculate_combined_score(doc_results, graph_results)
            
            # 更新性能统计
            total_time = time.time() - start_time
            self.query_stats['avg_response_time'] = (
                (self.query_stats['avg_response_time'] * (self.query_stats['total_queries'] - 1) + total_time) /
                self.query_stats['total_queries']
            )
            
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
                        metric_name='combined_score',
                        value=combined_score,
                        metadata={
                            'query': query,
                            'result_count': len(result.documents) + len(result.entities),
                            'response_time': time.time() - start_time,
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
            
            # 结束性能监控
            if self.performance_monitor:
                self.performance_monitor.end_query(query_id, len(result.documents) + len(result.entities), None, self.enable_concurrent_queries)
            
            logger.info(f"混合检索完成: 文档{len(result.documents)}个, 实体{len(result.entities)}个, 关系{len(result.relationships)}个")
            return result
            
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            # 记录错误
            if self.performance_monitor and query_id:
                self.performance_monitor.end_query(query_id, 0, str(e), self.enable_concurrent_queries)
            return HybridResult([], [], [], 0.0, {'error': str(e)})
    
    async def _search_concurrent_async(self, query: str, doc_top_k: int, graph_top_k: int, include_intent: bool = True, query_id: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """异步并发执行文档和图谱检索"""
        try:
            # 创建异步任务
            doc_task = asyncio.create_task(self._search_documents_async(query, doc_top_k, query_id))
            graph_task = asyncio.create_task(self._search_graph_async(query, graph_top_k, include_intent, query_id))
            
            # 等待所有任务完成
            doc_results, graph_results = await asyncio.gather(doc_task, graph_task, return_exceptions=True)
            
            # 处理异常
            if isinstance(doc_results, Exception):
                logger.error(f"文档检索失败: {doc_results}")
                doc_results = {'documents': [], 'total_score': 0.0}
            
            if isinstance(graph_results, Exception):
                logger.error(f"图谱检索失败: {graph_results}")
                graph_results = {'entities': [], 'relationships': [], 'intent': None}
            
            logger.debug(f"异步并发查询完成 - 文档: {len(doc_results.get('documents', []))}个, 图谱: {len(graph_results.get('entities', []))}个实体")
            return doc_results, graph_results
            
        except Exception as e:
            logger.error(f"异步并发查询失败: {e}")
            # 降级到串行查询
            doc_results = await self._search_documents_async(query, doc_top_k, query_id)
            graph_results = await self._search_graph_async(query, graph_top_k, include_intent, query_id)
            return doc_results, graph_results
    
    def _search_concurrent(self, query: str, doc_top_k: int, graph_top_k: int, include_intent: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """并发执行文档和图谱检索（同步版本）"""
        try:
            # 提交并发任务
            future_doc = self.executor.submit(self._search_documents, query, doc_top_k)
            future_graph = self.executor.submit(self._search_graph, query, graph_top_k, include_intent)
            
            # 等待结果
            doc_results = future_doc.result()
            graph_results = future_graph.result()
            
            logger.debug(f"并发查询完成 - 文档: {len(doc_results.get('documents', []))}个, 图谱: {len(graph_results.get('entities', []))}个实体")
            return doc_results, graph_results
            
        except Exception as e:
            logger.error(f"并发查询失败: {e}")
            # 降级到串行查询
            doc_results = self._search_documents(query, doc_top_k)
            graph_results = self._search_graph(query, graph_top_k, include_intent)
            return doc_results, graph_results
    
    async def _search_documents_async(self, query: str, top_k: int, query_id: Optional[str] = None) -> Dict[str, Any]:
        """异步搜索文档 (Weaviate)"""
        try:
            # 记录文档查询开始
            if self.performance_monitor and query_id:
                self.performance_monitor.record_component_start(query_id, 'document_search')
            # 使用混合搜索获取最佳结果
            if hasattr(self.doc_service, 'search_hybrid_async'):
                hybrid_result = await self.doc_service.search_hybrid_async(query, top_k)
            else:
                # 如果没有异步方法，在线程池中运行同步方法
                loop = asyncio.get_event_loop()
                hybrid_result = await loop.run_in_executor(None, self.doc_service.search_hybrid, query, top_k)
            
            # 格式化文档结果 - 使用混合结果
            documents = []
            for result in hybrid_result.hybrid_results:
                documents.append({
                    'content': result.content,
                    'metadata': result.metadata,
                    'score': result.score,
                    'source': 'weaviate'
                })
            
            result = {
                'documents': documents,
                'total_score': sum(doc['score'] for doc in documents)
            }
            
            # 记录文档查询结束
            if self.performance_monitor and query_id:
                self.performance_monitor.record_component_end(query_id, 'document_search', len(documents))
            
            return result
            
        except Exception as e:
            logger.error(f"异步文档检索失败: {e}")
            # 记录文档查询错误
            if self.performance_monitor and query_id:
                self.performance_monitor.record_component_end(query_id, 'document_search', 0, str(e))
            return {'documents': [], 'total_score': 0.0}
    
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
    
    async def _search_graph_async(self, query: str, top_k: int, include_intent: bool = True, query_id: Optional[str] = None) -> Dict[str, Any]:
        """异步搜索向量化图谱 (Neo4j)"""
        try:
            # 记录图谱查询开始
            if self.performance_monitor and query_id:
                self.performance_monitor.record_component_start(query_id, 'graph_search')
            
            # 使用向量化图谱检索服务进行搜索
            if hasattr(self.graph_service, 'search_async'):
                search_results = await self.graph_service.search_async(
                    query=query,
                    limit=top_k,
                    min_similarity=0.7,  # 设置最小相似度阈值
                    max_chain_length=3   # 设置最大知识链长度
                )
            else:
                loop = asyncio.get_event_loop()
                search_results = await loop.run_in_executor(None, self.graph_service.search, query, top_k)
            
            # 初始化结果结构
            results = {
                'entities': [],
                'relationships': [],
                'intent': None
            }
            
            # 转换结果格式以保持兼容性
            for result in search_results:
                if result.get('type') == 'knowledge_chain':
                    # 知识链结果 - 提取其中的实体和关系
                    nodes = result.get('nodes', [])
                    relations = result.get('relations', [])
                    
                    for node in nodes:
                        entity_data = {
                            'id': node.get('neo4j_id', ''),
                            'name': node.get('name', ''),
                            'description': node.get('description', ''),
                            'score': result.get('score', 0.0),
                            'entity_type': node.get('node_type', ''),
                            'source': 'knowledge_chain'
                        }
                        results['entities'].append(entity_data)
                    
                    for relation in relations:
                        relation_data = {
                            'id': relation.get('neo4j_id', ''),
                            'description': relation.get('description', ''),
                            'score': result.get('score', 0.0),
                            'relation_type': relation.get('relation_type', ''),
                            'source': relation.get('source', ''),
                            'target': relation.get('target', ''),
                            'source_chain': 'knowledge_chain'
                        }
                        results['relationships'].append(relation_data)
                        
                elif result.get('type') == 'node':
                    # 节点结果
                    entity_data = {
                        'id': result.get('neo4j_id', ''),
                        'name': result.get('name', ''),
                        'description': result.get('description', ''),
                        'score': result.get('similarity', 0.0),
                        'entity_type': result.get('node_type', ''),
                        'source': 'vector_search'
                    }
                    results['entities'].append(entity_data)
                    
                elif result.get('type') == 'relation':
                    # 关系结果
                    relation_data = {
                        'id': result.get('neo4j_id', ''),
                        'description': result.get('description', ''),
                        'score': result.get('similarity', 0.0),
                        'relation_type': result.get('relation_type', ''),
                        'source': result.get('source', ''),
                        'target': result.get('target', ''),
                        'source_chain': 'vector_search'
                    }
                    results['relationships'].append(relation_data)
            
            # 意图识别（如果启用且有相关方法）
            if include_intent and hasattr(self.graph_service, 'recognize_intent'):
                try:
                    if hasattr(self.graph_service, 'recognize_intent_async'):
                        intent_result = await self.graph_service.recognize_intent_async(query)
                    else:
                        loop = asyncio.get_event_loop()
                        intent_result = await loop.run_in_executor(None, self.graph_service.recognize_intent, query)
                    
                    if intent_result:
                        results['intent'] = {
                            'intent': getattr(intent_result, 'intent_type', None),
                            'confidence': getattr(intent_result, 'confidence', 0.0),
                            'entities': getattr(intent_result, 'entities', []),
                            'relations': getattr(intent_result, 'relations', [])
                        }
                except Exception as e:
                    logger.warning(f"异步意图识别失败: {e}")
            
            # 去重并限制数量
            seen_entities = set()
            unique_entities = []
            for entity in results['entities']:
                entity_id = entity.get('id') or entity.get('name', '')
                if entity_id not in seen_entities:
                    seen_entities.add(entity_id)
                    unique_entities.append(entity)
            
            seen_relationships = set()
            unique_relationships = []
            for rel in results['relationships']:
                rel_id = rel.get('id') or f"{rel.get('source', '')}-{rel.get('target', '')}"
                if rel_id not in seen_relationships:
                    seen_relationships.add(rel_id)
                    unique_relationships.append(rel)
            
            results['entities'] = unique_entities[:top_k//2]
            results['relationships'] = unique_relationships[:top_k//2]
            
            logger.info(f"异步向量化图谱搜索结果: 实体{len(results['entities'])}个, 关系{len(results['relationships'])}个")
            
            # 记录图谱查询结束
            if self.performance_monitor and query_id:
                entity_count = len(results['entities'])
                relationship_count = len(results['relationships'])
                self.performance_monitor.record_component_end(query_id, 'graph_search', entity_count + relationship_count)
            
            return results
            
        except Exception as e:
            logger.error(f"异步向量化图谱检索失败: {e}")
            # 记录图谱查询错误
            if self.performance_monitor and query_id:
                self.performance_monitor.record_component_end(query_id, 'graph_search', 0, str(e))
            return {'entities': [], 'relationships': [], 'intent': None}
    
    def _search_graph(self, query: str, top_k: int, include_intent: bool = True) -> Dict[str, Any]:
        """搜索向量化图谱"""
        try:
            # 使用向量化图谱检索服务进行搜索
            if hasattr(self.graph_service, 'search'):
                search_results = self.graph_service.search(
                    query=query,
                    limit=top_k,
                    min_similarity=0.7,  # 设置最小相似度阈值
                    max_chain_length=3   # 设置最大知识链长度
                )
            else:
                search_results = []
            
            # 初始化结果结构
            results = {
                'entities': [],
                'relationships': [],
                'intent': None
            }
            
            # 转换结果格式以保持兼容性
            for result in search_results:
                if result.get('type') == 'knowledge_chain':
                    # 知识链结果 - 提取其中的实体和关系
                    nodes = result.get('nodes', [])
                    relations = result.get('relations', [])
                    
                    for node in nodes:
                        entity_data = {
                            'id': node.get('neo4j_id', ''),
                            'name': node.get('name', ''),
                            'description': node.get('description', ''),
                            'score': result.get('score', 0.0),
                            'entity_type': node.get('node_type', ''),
                            'source': 'knowledge_chain'
                        }
                        results['entities'].append(entity_data)
                    
                    for relation in relations:
                        relation_data = {
                            'id': relation.get('neo4j_id', ''),
                            'description': relation.get('description', ''),
                            'score': result.get('score', 0.0),
                            'relation_type': relation.get('relation_type', ''),
                            'source': relation.get('source', ''),
                            'target': relation.get('target', ''),
                            'source_chain': 'knowledge_chain'
                        }
                        results['relationships'].append(relation_data)
                        
                elif result.get('type') == 'node':
                    # 节点结果
                    entity_data = {
                        'id': result.get('neo4j_id', ''),
                        'name': result.get('name', ''),
                        'description': result.get('description', ''),
                        'score': result.get('similarity', 0.0),
                        'entity_type': result.get('node_type', ''),
                        'source': 'vector_search'
                    }
                    results['entities'].append(entity_data)
                    
                elif result.get('type') == 'relation':
                    # 关系结果
                    relation_data = {
                        'id': result.get('neo4j_id', ''),
                        'description': result.get('description', ''),
                        'score': result.get('similarity', 0.0),
                        'relation_type': result.get('relation_type', ''),
                        'source': result.get('source', ''),
                        'target': result.get('target', ''),
                        'source_chain': 'vector_search'
                    }
                    results['relationships'].append(relation_data)
            
            # 意图识别（如果启用且有相关方法）
            if include_intent and hasattr(self.graph_service, 'recognize_intent'):
                try:
                    intent_result = self.graph_service.recognize_intent(query)
                    if intent_result:
                        results['intent'] = {
                            'intent': getattr(intent_result, 'intent_type', None),
                            'confidence': getattr(intent_result, 'confidence', 0.0),
                            'entities': getattr(intent_result, 'entities', []),
                            'relations': getattr(intent_result, 'relations', [])
                        }
                except Exception as e:
                    logger.warning(f"意图识别失败: {e}")
            
            # 去重并限制数量
            seen_entities = set()
            unique_entities = []
            for entity in results['entities']:
                entity_id = entity.get('id') or entity.get('name', '')
                if entity_id not in seen_entities:
                    seen_entities.add(entity_id)
                    unique_entities.append(entity)
            
            seen_relationships = set()
            unique_relationships = []
            for rel in results['relationships']:
                rel_id = rel.get('id') or f"{rel.get('source', '')}-{rel.get('target', '')}"
                if rel_id not in seen_relationships:
                    seen_relationships.add(rel_id)
                    unique_relationships.append(rel)
            
            results['entities'] = unique_entities[:top_k//2]
            results['relationships'] = unique_relationships[:top_k//2]
            
            logger.info(f"向量化图谱搜索结果: 实体{len(results['entities'])}个, 关系{len(results['relationships'])}个")
            
            return results
            
        except Exception as e:
            logger.error(f"向量化图谱检索失败: {e}")
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
    
    async def search_with_context(self, 
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
            result = await self.search_hybrid(expanded_query, top_k)
            
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
                description=description,
                groups=[{'name': strategy, 'traffic_percentage': traffic_split.get(strategy, 1.0/len(strategies)) if traffic_split else 1.0/len(strategies)} for strategy in strategies]
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
            
            return self.ab_test_manager.get_statistics(experiment_id)
            
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