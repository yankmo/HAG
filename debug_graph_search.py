#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')
from config.settings import get_config
from src.services.neo4j_retrieval_service import GraphRetrievalService

def debug_graph_search():
    """调试图谱搜索过程"""
    config = get_config()
    
    # 创建图谱检索服务
    try:
        graph_service = GraphRetrievalService(config.neo4j)
        
        query = "帕金森怎么治疗"
        print(f"=== 调试查询: {query} ===\n")
        
        # 1. 测试关键词提取
        keywords = graph_service._extract_keywords(query)
        main_keyword = graph_service._extract_main_keyword(query)
        print(f"提取的关键词: {keywords}")
        print(f"主要关键词: {main_keyword}\n")
        
        # 2. 测试核心实体查找
        core_entities = graph_service._find_most_relevant_entities(query, limit=5)
        print(f"找到的核心实体: {len(core_entities)}")
        for entity in core_entities:
            print(f"  - {entity['name']} ({entity['type']}) - 相关性: {entity.get('relevance_rank', 'N/A')}")
        print()
        
        # 3. 测试关系搜索
        relationships = graph_service.search_relationships_by_query(query, limit=10)
        print(f"找到的关系: {len(relationships)}")
        for rel in relationships:
            score = graph_service._calculate_relationship_relevance(rel, query)
            print(f"  - {rel['source']} -> {rel['type']} -> {rel['target']}")
            print(f"    相关性评分: {score}")
            print(f"    描述: {rel['description'][:100]}...")
            print()
        
        # 4. 测试相关性评分逻辑
        print("=== 测试相关性评分逻辑 ===")
        test_relationship = {
            'source': '帕金森氏症',
            'type': 'TREATS',
            'target': '基因疗法',
            'description': 'Gene therapy is used to treat Parkinson\'s disease.'
        }
        
        score = graph_service._calculate_relationship_relevance(test_relationship, query)
        print(f"测试关系的评分: {score}")
        
        # 分析评分逻辑
        query_lower = query.lower()
        print(f"查询小写: {query_lower}")
        print(f"目标实体匹配: {'基因疗法' in query_lower}")
        print(f"关系类型检查: {'treat' in 'TREATS'.lower()}")
        print(f"查询治疗关键词检查: {any(keyword in query_lower for keyword in ['治疗', 'treat', 'cure', '可以'])}")
        
    except Exception as e:
        print(f"调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_graph_search()