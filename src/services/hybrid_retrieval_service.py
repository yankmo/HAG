"""
混合检索服务 - 结合文档检索和图谱检索
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

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
                 graph_weight: float = 0.4):
        """
        初始化混合检索服务
        
        Args:
            document_retrieval_service: 文档检索服务 (基于Weaviate)
            graph_retrieval_service: 图谱检索服务 (基于Neo4j)
            doc_weight: 文档检索权重
            graph_weight: 图谱检索权重
        """
        self.doc_service = document_retrieval_service
        self.graph_service = graph_retrieval_service
        self.doc_weight = doc_weight
        self.graph_weight = graph_weight
        
        logger.info(f"混合检索服务初始化完成 - 文档权重: {doc_weight}, 图谱权重: {graph_weight}")
    
    def search_hybrid(self, 
                     query: str, 
                     top_k: int = 10,
                     doc_top_k: Optional[int] = None,
                     graph_top_k: Optional[int] = None,
                     include_intent: bool = True) -> HybridResult:
        """
        执行混合检索
        
        Args:
            query: 查询文本
            top_k: 总返回结果数
            doc_top_k: 文档检索返回数 (默认为top_k的60%)
            graph_top_k: 图谱检索返回数 (默认为top_k的40%)
            include_intent: 是否包含意图识别
            
        Returns:
            HybridResult: 混合检索结果
        """
        try:
            # 设置默认的检索数量
            if doc_top_k is None:
                doc_top_k = max(1, int(top_k * 0.6))
            if graph_top_k is None:
                graph_top_k = max(1, int(top_k * 0.4))
            
            logger.info(f"开始混合检索: query='{query}', doc_top_k={doc_top_k}, graph_top_k={graph_top_k}")
            
            # 1. 文档检索 (Weaviate)
            doc_results = self._search_documents(query, doc_top_k)
            
            # 2. 图谱检索 (Neo4j)
            graph_results = self._search_graph(query, graph_top_k, include_intent)
            
            # 3. 计算综合得分
            combined_score = self._calculate_combined_score(doc_results, graph_results)
            
            # 4. 构建结果
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
                    'doc_weight': self.doc_weight,
                    'graph_weight': self.graph_weight,
                    'intent': graph_results.get('intent')
                }
            )
            
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
    
    def _calculate_combined_score(self, doc_results: Dict[str, Any], graph_results: Dict[str, Any]) -> float:
        """计算综合得分"""
        try:
            doc_score = doc_results.get('total_score', 0.0)
            
            # 计算图谱得分 (基于结果数量和相关性)
            entity_count = len(graph_results.get('entities', []))
            relationship_count = len(graph_results.get('relationships', []))
            graph_score = (entity_count + relationship_count) * 0.1  # 简单的得分计算
            
            # 加权综合得分
            combined_score = (doc_score * self.doc_weight + graph_score * self.graph_weight)
            
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
    
    def get_stats(self) -> Dict[str, Any]:
        """获取混合检索服务统计信息"""
        try:
            doc_stats = self.doc_service.get_stats()
            graph_stats = self.graph_service.get_stats()
            
            return {
                'document_service': doc_stats,
                'graph_service': graph_stats,
                'weights': {
                    'document_weight': self.doc_weight,
                    'graph_weight': self.graph_weight
                },
                'service_type': 'hybrid_retrieval'
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {'error': str(e)}