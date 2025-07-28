#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的向量检索验证脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knowledge.vector_storage import WeaviateVectorStore
from src.services.embedding_service import OllamaEmbeddingService as OllamaEmbeddingClient

def main():
    print("🚀 开始向量检索验证")
    print("=" * 50)
    
    # 测试Weaviate连接
    print("\n=== 测试Weaviate连接 ===")
    try:
        vector_store = WeaviateVectorStore()
        stats = vector_store.get_stats()
        print(f"✓ Weaviate连接成功")
        print(f"  实体数量: {stats['entities']}")
        print(f"  关系数量: {stats['relations']}")
        print(f"  总计: {stats['total']}")
        has_data = stats['total'] > 0
    except Exception as e:
        print(f"✗ Weaviate连接失败: {e}")
        return
    
    # 测试Ollama向量化
    print("\n=== 测试Ollama向量化 ===")
    try:
        embedding_client = OllamaEmbeddingClient()
        test_text = "帕金森病的症状"
        vector = embedding_client.embed_text(test_text)
        
        if vector and len(vector) > 0:
            print(f"✓ 向量化成功")
            print(f"  文本: {test_text}")
            print(f"  向量维度: {len(vector)}")
            print(f"  向量前5个值: {vector[:5]}")
        else:
            print(f"✗ 向量化失败，返回空向量")
            return
    except Exception as e:
        print(f"✗ Ollama向量化失败: {e}")
        return
    
    # 测试向量检索
    if has_data:
        print("\n=== 测试向量检索 ===")
        try:
            query = "帕金森病的治疗方法"
            print(f"查询: {query}")
            
            query_vector = embedding_client.embed_text(query)
            results = vector_store.search_entities(query_vector, limit=3)
            
            print(f"找到 {len(results)} 个相关结果:")
            for i, result in enumerate(results, 1):
                distance = result.get('distance', 'N/A')
                name = result.get('name', 'N/A')
                description = result.get('description', 'N/A')
                print(f"  {i}. {name}")
                print(f"     距离: {distance}")
                print(f"     内容: {description[:80]}...")
                
        except Exception as e:
            print(f"✗ 向量检索失败: {e}")
    else:
        print("\n⚠️ 没有数据，跳过检索测试")
    
    print("\n" + "=" * 50)
    print("🎉 向量检索验证完成")

if __name__ == "__main__":
    main()