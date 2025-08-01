#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修改后的关系检索逻辑
验证：先找节点 -> 再找关系的正确流程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api import HAGIntegratedAPI

def test_relationship_retrieval_logic():
    """测试关系检索的新逻辑"""
    print("🔍 测试关系检索逻辑...")
    print("=" * 60)
    
    # 初始化API
    api = HAGIntegratedAPI()
    
    # 测试问题
    test_questions = [
        "帕金森病的症状有哪些？",
        "糖尿病和高血压的关系",
        "心脏病的治疗方法"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 测试问题 {i}: {question}")
        print("-" * 40)
        
        # 直接调用关系检索方法
        relationships = api._retrieve_relationships(question)
        
        print("🔗 检索到的关系:")
        if relationships and relationships != "未找到相关关系":
            print(relationships)
        else:
            print("  无相关关系")
        
        print()

def test_step_by_step_logic():
    """逐步测试关系检索的每个步骤"""
    print("\n🔬 逐步测试关系检索逻辑...")
    print("=" * 60)
    
    api = HAGIntegratedAPI()
    question = "帕金森病的症状"
    
    print(f"📝 测试问题: {question}")
    print()
    
    # 第一步：查找相关实体
    print("🎯 第一步：查找相关实体")
    try:
        entities = api.graph_service.search_entities_by_name(question, limit=3)
        print(f"  找到 {len(entities)} 个相关实体:")
        for entity in entities:
            print(f"    - {entity.get('name', 'Unknown')} ({entity.get('type', 'Unknown')})")
    except Exception as e:
        print(f"  实体查找失败: {e}")
        return
    
    print()
    
    # 第二步：基于实体查找关系
    print("🔗 第二步：基于实体查找关系")
    all_relationships = []
    
    for entity in entities:
        entity_name = entity.get('name', '')
        if entity_name:
            print(f"  查找实体 '{entity_name}' 的关系:")
            try:
                entity_rels = api.graph_service.get_entity_relationships(entity_name, limit=5)
                relationships = entity_rels.get('relationships', [])
                print(f"    找到 {len(relationships)} 个关系")
                
                for rel in relationships:
                    source = rel.get('entity', entity_name)
                    target = rel.get('related_entity', '')
                    rel_type = rel.get('relation_type', '')
                    print(f"      {source} --[{rel_type}]--> {target}")
                    all_relationships.append(rel)
                    
            except Exception as e:
                print(f"    关系查找失败: {e}")
    
    print()
    print(f"🎯 总共找到 {len(all_relationships)} 个关系")
    
    # 第三步：测试完整的关系检索方法
    print("\n🔄 第三步：测试完整的关系检索方法")
    final_result = api._retrieve_relationships(question)
    print("最终结果:")
    print(final_result)

if __name__ == "__main__":
    print("🚀 开始测试关系检索逻辑...")
    
    # 基本测试
    test_relationship_retrieval_logic()
    
    # 详细步骤测试
    test_step_by_step_logic()
    
    print("\n✅ 测试完成！")