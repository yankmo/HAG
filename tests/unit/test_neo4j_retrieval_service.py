#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4jRetrievalService单元测试
测试图谱查询、意图识别、连接池管理等功能
"""

import pytest
from unittest.mock import Mock, patch

# 导入被测试的类
from src.services.neo4j_retrieval_service import GraphRetrievalService
from src.knowledge.neo4j_vector_storage import IntentResult


class TestGraphRetrievalService:
    """GraphRetrievalService单元测试类"""
    
    @pytest.fixture
    def mock_graph_config(self):
        """模拟图数据库配置"""
        config = Mock()
        config.host = 'localhost'
        config.port = 7687
        config.user = 'neo4j'
        config.password = 'password'
        return config
    
    @pytest.fixture
    def mock_graph(self):
        """模拟Neo4j图数据库连接"""
        graph = Mock()
        graph.run = Mock()
        return graph
    
    @pytest.fixture
    def mock_intent_recognizer(self):
        """模拟意图识别器"""
        recognizer = Mock()
        recognizer.recognize_intent = Mock()
        return recognizer
    
    @pytest.fixture
    def service(self, mock_graph_config, mock_graph, mock_intent_recognizer):
        """创建GraphRetrievalService实例"""
        with patch('py2neo.Graph', return_value=mock_graph), \
             patch('src.services.neo4j_retrieval_service.Neo4jIntentRecognizer', return_value=mock_intent_recognizer):
            service = GraphRetrievalService(mock_graph_config)
            service.graph = mock_graph
            service.intent_recognizer = mock_intent_recognizer
            return service
    
    def test_init_success(self, mock_graph_config):
        """测试服务初始化成功"""
        with patch('py2neo.Graph') as mock_graph_class, \
             patch('src.services.neo4j_retrieval_service.Neo4jIntentRecognizer') as mock_recognizer_class:
            
            mock_graph = Mock()
            mock_recognizer = Mock()
            mock_graph_class.return_value = mock_graph
            mock_recognizer_class.return_value = mock_recognizer
            
            service = GraphRetrievalService(mock_graph_config)
            
            # 验证初始化调用
            mock_graph_class.assert_called_once_with(
                host='localhost',
                port=7687,
                user='neo4j',
                password='password'
            )
            mock_recognizer_class.assert_called_once_with(mock_graph_config)
            
            assert service.graph == mock_graph
            assert service.intent_recognizer == mock_recognizer
    
    def test_init_failure(self, mock_graph_config):
        """测试服务初始化失败"""
        with patch('py2neo.Graph', side_effect=Exception("连接失败")):
            with pytest.raises(Exception, match="连接失败"):
                GraphRetrievalService(mock_graph_config)
    
    def test_get_stats_success(self, service, mock_graph):
        """测试获取图谱统计信息成功"""
        # 模拟查询结果
        mock_graph.run.side_effect = [
            Mock(data=lambda: [{'count': 100}]),  # 节点数量
            Mock(data=lambda: [{'count': 200}]),  # 关系数量
            Mock(data=lambda: [{'type': 'Disease', 'count': 50}, {'type': 'Drug', 'count': 30}])  # 实体类型
        ]
        
        stats = service.get_stats()
        
        assert stats['total_nodes'] == 100
        assert stats['total_relationships'] == 200
        assert len(stats['entity_types']) == 2
        assert stats['status'] == 'active'
        assert stats['entity_types'][0]['type'] == 'Disease'
    
    def test_get_stats_failure(self, service, mock_graph):
        """测试获取图谱统计信息失败"""
        mock_graph.run.side_effect = Exception("查询失败")
        
        stats = service.get_stats()
        
        assert stats['total_nodes'] == 0
        assert stats['total_relationships'] == 0
        assert stats['entity_types'] == []
        assert stats['status'] == 'error'
        assert 'error' in stats
    
    def test_search_entities_by_type_success(self, service, mock_graph):
        """测试按类型搜索实体成功"""
        mock_entities = [
            {'name': '帕金森病', 'type': 'Disease', 'description': '神经退行性疾病'},
            {'name': '阿尔茨海默病', 'type': 'Disease', 'description': '痴呆症'}
        ]
        mock_graph.run.return_value.data.return_value = mock_entities
        
        entities = service.search_entities_by_type('Disease', 10)
        
        assert len(entities) == 2
        assert entities[0]['name'] == '帕金森病'
        assert entities[1]['type'] == 'Disease'
        
        # 验证查询参数
        mock_graph.run.assert_called_once()
        call_args = mock_graph.run.call_args
        assert 'Disease' in str(call_args)
    
    def test_search_entities_by_type_failure(self, service, mock_graph):
        """测试按类型搜索实体失败"""
        mock_graph.run.side_effect = Exception("查询失败")
        
        entities = service.search_entities_by_type('Disease', 10)
        
        assert entities == []
    
    def test_search_relationships_by_type_success(self, service, mock_graph):
        """测试按类型搜索关系成功"""
        mock_relationships = [
            {
                'source': '帕金森病',
                'type': 'TREATED_BY',
                'target': '左旋多巴',
                'description': '药物治疗',
                'source_text': '左旋多巴是治疗帕金森病的主要药物'
            }
        ]
        mock_graph.run.return_value.data.return_value = mock_relationships
        
        relationships = service.search_relationships_by_type('TREATED_BY', 10)
        
        assert len(relationships) == 1
        assert relationships[0]['source'] == '帕金森病'
        assert relationships[0]['type'] == 'TREATED_BY'
        assert relationships[0]['target'] == '左旋多巴'
    
    def test_search_relationships_by_type_with_empty_description(self, service, mock_graph):
        """测试搜索关系时处理空描述"""
        mock_relationships = [
            {
                'source': '帕金森病',
                'type': 'TREATED_BY',
                'target': '左旋多巴',
                'description': '',
                'source_text': '左旋多巴是治疗帕金森病的主要药物'
            }
        ]
        mock_graph.run.return_value.data.return_value = mock_relationships
        
        relationships = service.search_relationships_by_type('TREATED_BY', 10)
        
        # 验证描述被正确填充
        assert relationships[0]['description'] == '左旋多巴是治疗帕金森病的主要药物'
    
    def test_search_relationships_by_query_success(self, service, mock_graph):
        """测试按查询搜索关系成功"""
        # 模拟核心实体查询结果
        mock_entities = [{'name': '帕金森病', 'type': 'Disease', 'description': '神经退行性疾病'}]
        
        # 模拟关系查询结果
        mock_relationships = [
            {
                'source': '帕金森病',
                'type': 'TREATED_BY',
                'target': '左旋多巴',
                'description': '药物治疗',
                'source_text': '治疗文本',
                'target_type': 'Drug',
                'target_description': '抗帕金森药物'
            }
        ]
        
        # 设置mock返回值
        mock_graph.run.side_effect = [
            Mock(data=lambda: mock_entities),  # _find_most_relevant_entities
            Mock(data=lambda: mock_relationships)  # 关系查询
        ]
        
        relationships = service.search_relationships_by_query('帕金森病的治疗', 10)
        
        assert len(relationships) == 1
        assert relationships[0]['source'] == '帕金森病'
        assert 'relevance_score' in relationships[0]
    
    def test_search_relationships_by_query_no_entities(self, service, mock_graph):
        """测试查询时未找到相关实体"""
        mock_graph.run.return_value.data.return_value = []  # 无实体
        
        relationships = service.search_relationships_by_query('不存在的疾病', 10)
        
        assert relationships == []
    
    def test_find_most_relevant_entities(self, service, mock_graph):
        """测试查找最相关实体"""
        mock_entities = [
            {'name': '帕金森病', 'type': 'Disease', 'description': '神经退行性疾病', 'relevance_rank': 1},
            {'name': '帕金森综合征', 'type': 'Disease', 'description': '相关疾病', 'relevance_rank': 2}
        ]
        mock_graph.run.return_value.data.return_value = mock_entities
        
        entities = service._find_most_relevant_entities('帕金森病', 5)
        
        assert len(entities) == 2
        assert entities[0]['name'] == '帕金森病'
        assert entities[0]['relevance_rank'] == 1
    
    def test_extract_main_keyword(self, service):
        """测试提取主要关键词"""
        # 测试医学关键词优先级
        assert service._extract_main_keyword('帕金森病的症状') == '帕金森'
        assert service._extract_main_keyword('Parkinson disease') == 'Parkinson'
        
        # 测试最长词提取（中文无空格时返回整个查询）
        assert service._extract_main_keyword('这是一个测试查询') == '这是一个测试查询'
        
        # 测试有空格的英文查询
        assert service._extract_main_keyword('this is a test query') == 'query'
        
        # 测试单词查询
        assert service._extract_main_keyword('测试') == '测试'
    
    def test_calculate_relationship_relevance(self, service):
        """测试计算关系相关性"""
        relationship = {
            'target': '左旋多巴',
            'type': 'TREATED_BY',
            'description': '药物治疗帕金森病'
        }
        
        # 测试目标实体匹配（查询包含目标实体名称）
        score1 = service._calculate_relationship_relevance(relationship, '左旋多巴')
        assert score1 >= 3.0
        
        # 测试治疗关系匹配
        score2 = service._calculate_relationship_relevance(relationship, '如何治疗帕金森')
        assert score2 >= 2.0
        
        # 测试描述匹配
        score3 = service._calculate_relationship_relevance(relationship, '药物治疗')
        assert score3 >= 1.0
    
    def test_deduplicate_and_rank_relationships(self, service):
        """测试关系去重和排序"""
        relationships = [
            {'source': 'A', 'type': 'REL', 'target': 'B', 'relevance_score': 2.0},
            {'source': 'A', 'type': 'REL', 'target': 'B', 'relevance_score': 1.0},  # 重复
            {'source': 'C', 'type': 'REL', 'target': 'D', 'relevance_score': 3.0}
        ]
        
        result = service._deduplicate_and_rank_relationships(relationships, 10)
        
        assert len(result) == 2  # 去重后
        assert result[0]['relevance_score'] == 3.0  # 按分数排序
        assert result[1]['relevance_score'] == 2.0
    
    def test_search_entities_by_name_success(self, service, mock_graph):
        """测试按名称搜索实体成功"""
        mock_entities = [
            {
                'name': '帕金森病',
                'type': 'Disease',
                'description': '神经退行性疾病',
                'source_text': '详细的疾病描述文本'
            }
        ]
        mock_graph.run.return_value.data.return_value = mock_entities
        
        entities = service.search_entities_by_name('帕金森', 10)
        
        assert len(entities) == 1
        assert entities[0]['name'] == '帕金森病'
        assert entities[0]['description'] == '神经退行性疾病'
    
    def test_search_entities_by_name_with_empty_description(self, service, mock_graph):
        """测试搜索实体时处理空描述"""
        # 创建一个超过100字符的长文本
        long_text = '这是一个很长的源文本，用于测试描述生成功能，应该被截断到100个字符。' * 3  # 重复3次确保超过100字符
        mock_entities = [
            {
                'name': '帕金森病',
                'type': 'Disease',
                'description': '',
                'source_text': long_text
            }
        ]
        mock_graph.run.return_value.data.return_value = mock_entities
        
        entities = service.search_entities_by_name('帕金森', 10)
        
        # 验证描述被正确生成
        assert len(entities[0]['description']) <= 103  # 100字符 + '...'
        assert entities[0]['description'].endswith('...')
    
    def test_extract_keywords(self, service):
        """测试关键词提取"""
        # 测试医学关键词提取
        keywords = service._extract_keywords('帕金森病的治疗方法')
        assert '帕金森' in keywords
        assert '治疗' in keywords
        
        # 测试标点符号处理
        keywords = service._extract_keywords('什么是帕金森病？')
        assert '帕金森' in keywords
        assert '什么是帕金森病' in keywords
        
        # 测试短词过滤
        keywords = service._extract_keywords('a 帕金森 b')
        assert 'a' not in keywords
        assert 'b' not in keywords
        assert '帕金森' in keywords
    
    def test_recognize_intent_success(self, service, mock_intent_recognizer):
        """测试意图识别成功"""
        mock_result = IntentResult("treatment", 0.9, ['帕金森病'], ['左旋多巴'], {})
        mock_intent_recognizer.recognize_intent.return_value = mock_result
        
        result = service.recognize_intent('如何治疗帕金森病')
        
        assert result.intent_type == "treatment"
        assert result.confidence == 0.9
        mock_intent_recognizer.recognize_intent.assert_called_once_with('如何治疗帕金森病')
    
    def test_recognize_intent_failure(self, service, mock_intent_recognizer):
        """测试意图识别失败"""
        mock_intent_recognizer.recognize_intent.side_effect = Exception("识别失败")
        
        with patch('src.knowledge.neo4j_vector_storage.IntentResult') as mock_intent_result:
            mock_intent_result.return_value = Mock()
            
            result = service.recognize_intent('测试查询')
            
            mock_intent_result.assert_called_once_with("unknown", 0.0, [], [], {"error": "识别失败"})
    
    def test_search_with_intent_treatment(self, service, mock_intent_recognizer, mock_graph):
        """测试基于治疗意图的搜索"""
        # 模拟意图识别结果
        mock_intent = IntentResult("treatment", 0.9, ['帕金森病'], [], {})
        mock_intent_recognizer.recognize_intent.return_value = mock_intent
        
        # 模拟实体搜索结果
        mock_entities = [{'name': '左旋多巴', 'type': 'Treatment', 'description': '抗帕金森药物'}]
        mock_graph.run.return_value.data.return_value = mock_entities
        
        result = service.search_with_intent('如何治疗帕金森病', 10)
        
        assert result['intent'] == mock_intent
        assert len(result['entities']) > 0
        assert result['query'] == '如何治疗帕金森病'
    
    def test_search_with_intent_failure(self, service, mock_intent_recognizer):
        """测试意图搜索失败"""
        mock_intent_recognizer.recognize_intent.side_effect = Exception("搜索失败")
        
        result = service.search_with_intent('测试查询', 10)
        
        assert result['intent'] is None
        assert result['entities'] == []
        assert result['relationships'] == []
        assert 'error' in result
    
    def test_get_entity_relationships_success(self, service, mock_graph):
        """测试获取实体关系成功"""
        mock_relationships = [
            {
                'entity': '帕金森病',
                'relation_type': 'TREATED_BY',
                'relation_description': '药物治疗',
                'related_entity': '左旋多巴',
                'related_type': 'Drug',
                'related_description': '抗帕金森药物'
            }
        ]
        mock_graph.run.return_value.data.return_value = mock_relationships
        
        result = service.get_entity_relationships('帕金森病', 10)
        
        assert result['entity'] == '帕金森病'
        assert result['count'] == 1
        assert len(result['relationships']) == 1
        assert result['relationships'][0]['relation_type'] == 'TREATED_BY'
    
    def test_get_entity_relationships_failure(self, service, mock_graph):
        """测试获取实体关系失败"""
        mock_graph.run.side_effect = Exception("查询失败")
        
        result = service.get_entity_relationships('帕金森病', 10)
        
        assert result['entity'] == '帕金森病'
        assert result['relationships'] == []
        assert result['count'] == 0
    
    def test_get_knowledge_graph_summary_success(self, service, mock_graph):
        """测试获取知识图谱摘要成功"""
        # 模拟查询结果
        mock_graph.run.side_effect = [
            Mock(data=lambda: [{'type': 'Disease', 'count': 50}, {'type': 'Drug', 'count': 30}]),  # 实体类型
            Mock(data=lambda: [{'type': 'TREATED_BY', 'count': 100}]),  # 关系类型
            Mock(data=lambda: [{'entity': '帕金森病', 'type': 'Disease', 'connections': 25}])  # 高连接度实体
        ]
        
        summary = service.get_knowledge_graph_summary()
        
        assert len(summary['entity_types']) == 2
        assert len(summary['relation_types']) == 1
        assert len(summary['top_connected_entities']) == 1
        assert summary['summary']['total_entity_types'] == 2
        assert summary['summary']['most_common_entity_type'] == 'Disease'
    
    def test_get_knowledge_graph_summary_failure(self, service, mock_graph):
        """测试获取知识图谱摘要失败"""
        mock_graph.run.side_effect = Exception("查询失败")
        
        summary = service.get_knowledge_graph_summary()
        
        assert summary['entity_types'] == []
        assert summary['relation_types'] == []
        assert summary['top_connected_entities'] == []
        assert summary['summary'] == {}


if __name__ == '__main__':
    pytest.main([__file__])