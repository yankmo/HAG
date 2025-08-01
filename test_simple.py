#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAG整合API简洁测试 - 专门测试帕金森问题
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from api import HAGIntegratedAPI

def test_parkinson_query():
    """测试帕金森相关查询"""
    print("=" * 60)
    print("HAG API - 帕金森病查询测试")
    print("=" * 60)
    
    try:
        # 初始化API
        print("初始化API...")
        api = HAGIntegratedAPI()
        print("✓ API初始化成功")
        
        # 测试帕金森问题
        question = "什么是帕金森病？"
        print(f"\n问题: {question}")
        print("-" * 60)
        
        result = api.query(question)
        
        print(f"\n📝 LLM回答:")
        print(result.answer)
        
        print(f"\n📚 检索到的文档 (应该是Top-5):")
        print(f"实际数量: {len(result.sources['documents'])}个")
        for i, doc in enumerate(result.sources['documents'], 1):
            score = doc.get('score', 'N/A')
            content = doc['content'][:150] + "..." if len(doc['content']) > 150 else doc['content']
            print(f"  {i}. [评分: {score}] {content}")
        
        print(f"\n🏷️ 检索到的实体:")
        print(f"实际数量: {len(result.sources['entities'])}个")
        for i, entity in enumerate(result.sources['entities'], 1):
            print(f"  {i}. {entity.get('name', 'N/A')} ({entity.get('type', 'N/A')})")
        
        print(f"\n🔗 检索到的关系:")
        print(f"实际数量: {len(result.sources['relationships'])}个")
        for i, rel in enumerate(result.sources['relationships'], 1):
            print(f"  {i}. {rel.get('source', 'N/A')} --[{rel.get('type', 'N/A')}]--> {rel.get('target', 'N/A')}")
        
        print(f"\n📊 检索统计:")
        metadata = result.metadata.get('retrieval_metadata', {})
        print(f"  检索元数据: {metadata}")
        
        print("\n" + "=" * 60)
        print("测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parkinson_query()