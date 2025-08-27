#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j图谱检索服务模块
专门负责知识图谱的节点和关系检索，不涉及向量计算
"""

import sys
import os
import logging
from typing import List, Dict, Any
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入Neo4jIntentRecognizer
from src.knowledge.neo4j_vector_storage import Neo4jIntentRecognizer


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphRetrievalService:
    """基于Neo4j的图谱检索服务 - 专门负责节点和关系查询"""
    
    def __init__(self, graph_db_config):
        """
        初始化图谱检索服务
        
        Args:
            graph_db_config: Neo4j数据库配置对象
        """
        try:
            # 初始化Neo4j连接
            from py2neo import Graph
            self.graph = Graph(
                host=getattr(graph_db_config, 'host', 'localhost'),
                port=getattr(graph_db_config, 'port', 7687),
                user=getattr(graph_db_config, 'user', 'neo4j'),
                password=getattr(graph_db_config, 'password', 'password')
            )
            
            # 初始化意图识别器
            self.intent_recognizer = Neo4jIntentRecognizer(graph_db_config)
            
            logger.info("图谱检索服务初始化成功")
            
        except Exception as e:
            logger.error(f"图谱检索服务初始化失败: {e}")
            raise
    
    def get_stats(self):
        """获取图谱统计信息"""
        try:
            # 获取节点统计
            node_count = self.graph.run("MATCH (n) RETURN count(n) as count").data()[0]['count']
            
            # 获取关系统计
            relationship_count = self.graph.run("MATCH ()-[r]-() RETURN count(r) as count").data()[0]['count']
            
            # 获取实体类型分布
            entity_types = self.graph.run("""
                MATCH (n:Entity)
                RETURN n.type as type, count(n) as count
                ORDER BY count DESC
            """).data()
            
            return {
                'total_nodes': node_count,
                'total_relationships': relationship_count,
                'entity_types': entity_types,
                'status': 'active'
            }
            
        except Exception as e:
            logger.error(f"获取图谱统计失败: {e}")
            return {
                'total_nodes': 0,
                'total_relationships': 0,
                'entity_types': [],
                'status': 'error',
                'error': str(e)
            }
    
    def search_entities_by_type(self, entity_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """根据类型搜索实体"""
        try:
            entities = self.graph.run("""
                MATCH (e:Entity {type: $entity_type})
                RETURN e.name as name, e.type as type, e.description as description
                LIMIT $limit
            """, {
                'entity_type': entity_type,
                'limit': limit
            }).data()
            
            return entities
            
        except Exception as e:
            logger.error(f"按类型搜索实体失败: {e}")
            return []
    
    def search_relationships_by_type(self, relation_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """根据类型搜索关系"""
        try:
            # 数据库中的关系类型是RELATION，需要查找关系的type属性
            relationships = self.graph.run("""
                MATCH (e1:Entity)-[r:RELATION]->(e2:Entity)
                WHERE r.type = $relation_type OR toLower(r.type) CONTAINS toLower($relation_type)
                RETURN e1.name as source, r.type as type, 
                       COALESCE(r.description, '') as description, 
                       e2.name as target,
                       COALESCE(r.source_text, '') as source_text
                LIMIT $limit
            """, {
                'relation_type': relation_type,
                'limit': limit
            }).data()
            
            # 为关系添加更多信息
            for rel in relationships:
                if not rel.get('description') and rel.get('source_text'):
                    source_text = rel['source_text']
                    rel['description'] = source_text[:100] + '...' if len(source_text) > 100 else source_text
                
                # 确保描述不为空
                if not rel.get('description'):
                    rel['description'] = f"{rel['source']} 与 {rel['target']} 之间的 {rel['type']} 关系"
            
            return relationships
            
        except Exception as e:
            logger.error(f"按类型搜索关系失败: {e}")
            return []
    
    def search_relationships_by_query(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """根据查询搜索相关关系 - 新的图谱匹配逻辑"""
        try:
            logger.info(f"开始图谱关系搜索: {query}")
            
            # 第一步：找到最相关的核心节点
            core_entities = self._find_most_relevant_entities(query, limit=5)
            logger.info(f"找到 {len(core_entities)} 个核心实体: {[e['name'] for e in core_entities]}")
            
            if not core_entities:
                logger.info("未找到相关核心实体")
                return []
            
            # 第二步：基于核心节点找到相关的节点和关系
            relationships = []
            
            for core_entity in core_entities:
                entity_name = core_entity['name']
                logger.info(f"查找实体 '{entity_name}' 的关系网络")
                
                # 查找与核心实体直接相关的所有关系
                entity_relationships = self.graph.run("""
                    MATCH (core:Entity {name: $entity_name})-[r:RELATION]-(related:Entity)
                    RETURN core.name as source, 
                           r.type as type, 
                           COALESCE(r.description, '') as description, 
                           related.name as target,
                           COALESCE(r.source_text, '') as source_text,
                           related.type as target_type,
                           COALESCE(related.description, '') as target_description
                    ORDER BY 
                        CASE 
                            WHEN toLower(related.name) CONTAINS toLower($query_keyword) THEN 1
                            WHEN toLower(r.type) CONTAINS toLower($query_keyword) THEN 2
                            WHEN toLower(r.description) CONTAINS toLower($query_keyword) THEN 3
                            ELSE 4
                        END
                    LIMIT $limit
                """, {
                    'entity_name': entity_name,
                    'query_keyword': self._extract_main_keyword(query),
                    'limit': limit
                }).data()
                
                # 处理关系数据
                for rel in entity_relationships:
                    # 确保关系描述不为空
                    if not rel.get('description') or rel['description'].strip() == '':
                        if rel.get('source_text'):
                            source_text = rel['source_text']
                            rel['description'] = source_text[:200] + '...' if len(source_text) > 200 else source_text
                        else:
                            rel['description'] = f"{rel['source']} 与 {rel['target']} 之间存在 {rel['type']} 关系"
                    
                    # 添加关系相关性评分
                    rel['relevance_score'] = self._calculate_relationship_relevance(rel, query)
                    
                    relationships.append(rel)
            
            # 第三步：按相关性排序并去重
            relationships = self._deduplicate_and_rank_relationships(relationships, limit)
            
            logger.info(f"图谱关系搜索完成: {len(relationships)} 个结果")
            return relationships
            
        except Exception as e:
            logger.error(f"图谱关系搜索失败: {e}")
            return []
    
    def _find_most_relevant_entities(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """找到与查询最相关的核心实体"""
        try:
            keywords = self._extract_keywords(query)
            main_keyword = self._extract_main_keyword(query)
            
            # 构建实体搜索查询，按相关性排序
            entities = self.graph.run("""
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($main_keyword)
                   OR toLower(e.description) CONTAINS toLower($main_keyword)
                   OR ANY(keyword IN $keywords WHERE toLower(e.name) CONTAINS toLower(keyword))
                RETURN e.name as name, 
                       e.type as type, 
                       COALESCE(e.description, '') as description,
                       CASE 
                           WHEN toLower(e.name) = toLower($main_keyword) THEN 1
                           WHEN toLower(e.name) CONTAINS toLower($main_keyword) THEN 2
                           WHEN toLower(e.description) CONTAINS toLower($main_keyword) THEN 3
                           ELSE 4
                       END as relevance_rank
                ORDER BY relevance_rank, e.name
                LIMIT $limit
            """, {
                'main_keyword': main_keyword,
                'keywords': keywords,
                'limit': limit
            }).data()
            
            return entities
            
        except Exception as e:
            logger.error(f"查找核心实体失败: {e}")
            return []
    
    def _extract_main_keyword(self, query: str) -> str:
        """提取查询的主要关键词"""
        import jieba
        import re
        
        # 医学关键词优先级
        medical_priority_keywords = ['帕金森', 'Parkinson', '帕金森氏症']
        
        for keyword in medical_priority_keywords:
            if keyword in query:
                return keyword
        
        # 检查是否包含空格（区分中英文）
        if ' ' in query:
            # 英文查询：按空格分割
            words = query.split()
            # 过滤停用词
            stop_words = {'is', 'a', 'an', 'the', 'this', 'that', 'and', 'or', 'but'}
            filtered_words = [word.strip() for word in words if word.strip().lower() not in stop_words and len(word.strip()) > 1]
            if filtered_words:
                return max(filtered_words, key=len)
        else:
            # 中文查询：使用jieba分词
            cleaned_query = re.sub(r'[？?！!。，,、；;：:]', '', query)
            words = list(jieba.cut(cleaned_query, cut_all=False))
            
            # 过滤停用词和短词
            stop_words = {'的', '是', '在', '有', '和', '与', '或', '但', '而', '了', '吗', '呢', '啊', '什么', '怎么', '如何', '这', '那', '一个', '一种'}
            filtered_words = [word.strip() for word in words if len(word.strip()) >= 2 and word.strip() not in stop_words]
            
            # 如果分词后只有一个词或没有有效词，返回原查询
            if len(filtered_words) <= 1:
                return query
            else:
                # 返回最长的有意义词汇
                return max(filtered_words, key=len)
        
        return query
    
    def _calculate_relationship_relevance(self, relationship: Dict[str, Any], query: str) -> float:
        """计算关系与查询的相关性评分"""
        score = 0.0
        query_lower = query.lower()
        target_lower = relationship['target'].lower()
        
        # 检查目标实体名称匹配 - 修复字符串包含逻辑
        # 检查目标实体是否包含在查询中，或查询是否包含目标实体
        if target_lower in query_lower or query_lower in target_lower:
            # 如果是完全匹配，给更高分数
            if target_lower == query_lower:
                score += 5.0
            # 如果目标实体包含在查询中（如'左旋多巴'包含在'左旋多巴治疗'中）
            elif target_lower in query_lower:
                score += 4.0
            # 如果查询包含在目标实体中
            else:
                score += 3.0
        
        # 检查关系类型匹配
        if any(keyword in relationship['type'].lower() for keyword in ['treat', 'therapy', 'cure', '治疗']):
            if any(keyword in query_lower for keyword in ['治疗', 'treat', 'cure', '可以']):
                score += 2.0
        
        # 检查描述匹配
        if query_lower in relationship['description'].lower():
            score += 1.0
        
        return score
    
    def _deduplicate_and_rank_relationships(self, relationships: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """去重并按相关性排序关系"""
        # 去重
        seen = set()
        unique_relationships = []
        
        for rel in relationships:
            rel_key = (rel['source'], rel['type'], rel['target'])
            if rel_key not in seen:
                seen.add(rel_key)
                unique_relationships.append(rel)
        
        # 按相关性评分排序
        unique_relationships.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return unique_relationships[:limit]
    
    def get_entity_relationships(self, entity_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """查询实体的所有关系"""
        try:
            query = """
            MATCH (a {name: $entity_name})-[r]-(b)
            RETURN a.name as source_name, b.name as target_name, 
                   type(r) as relation_type, r.description as description,
                   id(r) as id, r.source_text as source_text
            LIMIT $limit
            """
            
            with self.vector_store.driver.session() as session:
                result = session.run(query, entity_name=entity_name, limit=limit)
                relationships = []
                for record in result:
                    relationships.append({
                        "id": record["id"],
                        "source_name": record["source_name"],
                        "target_name": record["target_name"],
                        "relation_type": record["relation_type"],
                        "description": record["description"],
                        "source_text": record["source_text"]
                    })
                
                logger.info(f"获取实体关系完成: {len(relationships)} 个结果")
                return relationships
                
        except Exception as e:
            logger.error(f"获取实体关系失败: {e}")
            return []
    
    def search_entities_by_name(self, name_pattern: str, limit: int = 10) -> List[Dict[str, Any]]:
        """根据名称模式搜索实体"""
        try:
            # 提取关键词进行搜索
            keywords = self._extract_keywords(name_pattern)
            logger.info(f"从查询 '{name_pattern}' 提取关键词: {keywords}")
            
            # 使用单一查询避免重复
            keyword_conditions = []
            params = {'limit': limit}
            
            for i, keyword in enumerate(keywords):
                param_name = f'keyword_{i}'
                keyword_conditions.append(f"""
                    toLower(e.name) CONTAINS toLower(${param_name})
                """)
                params[param_name] = keyword
            
            if not keyword_conditions:
                return []
            
            query = f"""
                MATCH (e:Entity)
                WHERE {' OR '.join(keyword_conditions)}
                RETURN DISTINCT e.name as name, e.type as type, 
                       COALESCE(e.description, '') as description,
                       e.source_text as source_text
                LIMIT $limit
            """
            
            entities = self.graph.run(query, params).data()
            
            # 为每个实体添加更多信息
            for entity in entities:
                if not entity.get('description'):
                    # 如果没有描述，使用source_text的前100个字符作为描述
                    source_text = entity.get('source_text', '')
                    if source_text:
                        entity['description'] = source_text[:100] + '...' if len(source_text) > 100 else source_text
                    else:
                        entity['description'] = f"{entity['type']}类型的实体"
            
            logger.info(f"搜索实体完成: {len(entities)} 个结果")
            return entities
            
        except Exception as e:
            logger.error(f"按名称搜索实体失败: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词"""
        import re
        import jieba
        
        # 定义医学相关关键词
        medical_keywords = [
            '帕金森', 'Parkinson', '治疗', '症状', '疾病', '药物', '手术', 
            '康复', '运动', '疗法', '诊断', '预防', '病因', '发病', '机制'
        ]
        
        # 移除标点符号
        cleaned_query = re.sub(r'[？?！!。，,、；;：:]', '', query)
        
        # 使用jieba进行中文分词
        words = list(jieba.cut(cleaned_query, cut_all=False))
        
        # 定义停用词
        stop_words = {'的', '是', '在', '有', '和', '与', '或', '但', '而', '了', '吗', '呢', '啊', '什么', '怎么', '如何', '这', '那', '一个', '一种'}
        
        # 提取关键词
        keywords = []
        for word in words:
            word = word.strip()
            if len(word) >= 2 and word not in stop_words:  # 至少2个字符且不是停用词
                keywords.append(word)
        
        # 添加医学关键词匹配
        for keyword in medical_keywords:
            if keyword in query:
                keywords.append(keyword)
        
        # 去重并返回
        return list(set(keywords)) if keywords else [query]
    
    def recognize_intent(self, query: str) -> 'IntentResult':
        """识别查询意图"""
        try:
            return self.intent_recognizer.recognize_intent(query)
        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            from src.knowledge.neo4j_vector_storage import IntentResult
            return IntentResult("unknown", 0.0, [], [], {"error": str(e)})
    
    def search_with_intent(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """基于意图的图谱搜索"""
        try:
            logger.info(f"开始意图感知图谱搜索: {query}")
            
            # 识别意图
            intent_result = self.intent_recognizer.recognize_intent(query)
            logger.info(f"识别意图: {intent_result.intent_type}, 置信度: {intent_result.confidence:.2f}")
            
            # 基于意图搜索相关实体和关系
            entities = []
            relationships = []
            
            # 根据意图类型搜索不同的实体
            if intent_result.intent_type in ["treatment", "therapy"]:
                entities = self.search_entities_by_type("Treatment", limit)
                entities.extend(self.search_entities_by_type("Drug", limit))
            elif intent_result.intent_type in ["symptom", "symptoms"]:
                entities = self.search_entities_by_type("Symptom", limit)
            elif intent_result.intent_type in ["disease", "condition"]:
                entities = self.search_entities_by_type("Disease", limit)
            elif intent_result.intent_type in ["cause", "etiology"]:
                relationships = self.search_relationships_by_type("CAUSES", limit)
            else:
                # 通用搜索
                for entity in intent_result.entities:
                    entity_rels = self.get_entity_relationships(entity, limit)
                    relationships.extend(entity_rels.get('relationships', []))
            
            return {
                'intent': intent_result,
                'entities': entities,
                'relationships': relationships,
                'query': query
            }
            
        except Exception as e:
            logger.error(f"意图感知图谱搜索失败: {e}")
            return {
                'intent': None,
                'entities': [],
                'relationships': [],
                'query': query,
                'error': str(e)
            }
    
    def get_entity_relationships(self, entity_name: str, limit: int = 10) -> Dict[str, Any]:
        """获取实体的关系网络"""
        try:
            # 直接使用Neo4j图谱查询实体关系
            relationships = self.graph.run("""
                MATCH (e:Entity {name: $entity_name})-[r:RELATION]-(related:Entity)
                RETURN e.name as entity,
                       type(r) as relation_type,
                       r.description as relation_description,
                       related.name as related_entity,
                       related.type as related_type,
                       related.description as related_description
                LIMIT $limit
            """, {
                'entity_name': entity_name,
                'limit': limit
            }).data()
            
            return {
                'entity': entity_name,
                'relationships': relationships,
                'count': len(relationships)
            }
            
        except Exception as e:
            logger.error(f"获取实体关系失败: {e}")
            return {'entity': entity_name, 'relationships': [], 'count': 0}
    
    def get_knowledge_graph_summary(self) -> Dict[str, Any]:
        """获取知识图谱摘要"""
        try:
            # 获取实体类型分布
            entity_types = self.graph.run("""
                MATCH (n:Entity)
                RETURN n.type as type, count(n) as count
                ORDER BY count DESC
            """).data()
            
            # 获取关系类型分布
            relation_types = self.graph.run("""
                MATCH ()-[r:RELATION]-()
                RETURN r.type as type, count(r) as count
                ORDER BY count DESC
            """).data()
            
            # 获取连接度最高的实体
            top_connected = self.graph.run("""
                MATCH (n:Entity)-[r:RELATION]-()
                RETURN n.name as entity, n.type as type, count(r) as connections
                ORDER BY connections DESC
                LIMIT 10
            """).data()
            
            return {
                'entity_types': entity_types,
                'relation_types': relation_types,
                'top_connected_entities': top_connected,
                'summary': {
                    'total_entity_types': len(entity_types),
                    'total_relation_types': len(relation_types),
                    'most_common_entity_type': entity_types[0]['type'] if entity_types else None,
                    'most_common_relation_type': relation_types[0]['type'] if relation_types else None
                }
            }
            
        except Exception as e:
            logger.error(f"获取知识图谱摘要失败: {e}")
            return {
                'entity_types': [],
                'relation_types': [],
                'top_connected_entities': [],
                'summary': {}
            }