#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAG整合API测试脚本
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from api import HAGIntegratedAPI, query_knowledge

def test_api():
    """测试API功能"""
    print("=" * 60)
    print("HAG整合API测试")
    print("=" * 60)
    
    try:
        # 初始化API
        print("1. 初始化API...")
        api = HAGIntegratedAPI()
        print("✓ API初始化成功")
        
        # 检查系统状态
        print("\n2. 检查系统状态...")
        status = api.get_system_status()
        print(f"✓ 系统状态: {status['status']}")
        print(f"  - 服务状态: {status.get('services', {})}")
        
        # 测试查询
        print("\n3. 测试知识查询...")
        test_questions = [
            "什么是人工智能？",
            "机器学习的基本原理是什么？",
            "深度学习和传统机器学习有什么区别？"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n测试问题 {i}: {question}")
            try:
                result = api.query(question)
                print(f"✓ 回答: {result.answer[:100]}...")
                print(f"  - 文档来源: {len(result.sources['documents'])}个")
                print(f"  - 实体来源: {len(result.sources['entities'])}个") 
                print(f"  - 关系来源: {len(result.sources['relationships'])}个")
            except Exception as e:
                print(f"✗ 查询失败: {e}")
        
        # 测试简化接口
        print("\n4. 测试简化接口...")
        try:
            result = query_knowledge("什么是深度学习？")
            print(f"✓ 简化接口测试成功: {result.answer[:50]}...")
        except Exception as e:
            print(f"✗ 简化接口测试失败: {e}")
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api()