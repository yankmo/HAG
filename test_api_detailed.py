#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAG整合API详细测试脚本 - 显示检索到的具体内容
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from api import HAGIntegratedAPI, query_knowledge

def print_detailed_results(result, question):
    """打印详细的检索结果"""
    print(f"\n{'='*80}")
    print(f"问题: {question}")
    print(f"{'='*80}")
    
    print(f"\n📝 LLM回答:")
    print(f"{result.answer}")
    
    print(f"\n📚 检索到的文档 (Top-5):")
    if result.sources['documents']:
        for i, doc in enumerate(result.sources['documents'], 1):
            print(f"  {i}. [评分: {doc.get('score', 'N/A'):.4f}]")
            print(f"     内容: {doc['content']}")
            print(f"     元数据: {doc.get('metadata', {})}")
            print()
    else:
        print("  ❌ 未找到相关文档")
    
    print(f"\n🏷️ 检索到的实体:")
    if result.sources['entities']:
        for i, entity in enumerate(result.sources['entities'], 1):
            print(f"  {i}. 名称: {entity.get('name', 'N/A')}")
            print(f"     类型: {entity.get('type', 'N/A')}")
            print(f"     属性: {entity.get('properties', {})}")
            print()
    else:
        print("  ❌ 未找到相关实体")
    
    print(f"\n🔗 检索到的关系:")
    if result.sources['relationships']:
        for i, rel in enumerate(result.sources['relationships'], 1):
            print(f"  {i}. {rel.get('source', 'N/A')} --[{rel.get('type', 'N/A')}]--> {rel.get('target', 'N/A')}")
            print(f"     描述: {rel.get('description', 'N/A')}")
            print()
    else:
        print("  ❌ 未找到相关关系")
    
    print(f"\n📊 检索统计:")
    metadata = result.metadata.get('retrieval_metadata', {})
    print(f"  - 文档数量: {len(result.sources['documents'])}")
    print(f"  - 实体数量: {len(result.sources['entities'])}")
    print(f"  - 关系数量: {len(result.sources['relationships'])}")
    print(f"  - 检索元数据: {metadata}")

def test_detailed_api():
    """详细测试API功能"""
    print("=" * 80)
    print("HAG整合API详细测试 - 帕金森病相关问题")
    print("=" * 80)
    
    try:
        # 初始化API
        print("🚀 1. 初始化API...")
        api = HAGIntegratedAPI()
        print("✅ API初始化成功")
        
        # 检查系统状态
        print("\n🔍 2. 检查系统状态...")
        status = api.get_system_status()
        print(f"✅ 系统状态: {status['status']}")
        print(f"   服务状态: {status.get('services', {})}")
        print(f"   检索统计: {status.get('retrieval_stats', {})}")
        print(f"   图谱统计: {status.get('graph_stats', {})}")
        
        # 测试帕金森相关问题
        print("\n🧠 3. 测试帕金森病相关知识查询...")
        parkinson_questions = [
            "什么是帕金森病？",
            "帕金森病的症状有哪些？",
            "帕金森病的病因是什么？",
            "如何治疗帕金森病？",
            "帕金森病和阿尔茨海默病有什么区别？"
        ]
        
        for i, question in enumerate(parkinson_questions, 1):
            print(f"\n🔬 测试问题 {i}: {question}")
            try:
                result = api.query(question)
                print_detailed_results(result, question)
                
                # 添加分隔线
                print("\n" + "─" * 80)
                
            except Exception as e:
                print(f"❌ 查询失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 测试简化接口
        print(f"\n🎯 4. 测试简化接口...")
        try:
            simple_question = "帕金森病的主要症状"
            result = query_knowledge(simple_question)
            print(f"✅ 简化接口测试成功")
            print(f"   问题: {simple_question}")
            print(f"   回答: {result.answer[:200]}...")
            print(f"   来源: 文档{len(result.sources['documents'])}个, 实体{len(result.sources['entities'])}个, 关系{len(result.sources['relationships'])}个")
        except Exception as e:
            print(f"❌ 简化接口测试失败: {e}")
        
        print("\n" + "=" * 80)
        print("🎉 详细测试完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_detailed_api()