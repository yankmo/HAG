#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量存储测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knowledge.vector_storage import (
    WeaviateVectorStore,
    VectorKnowledgeProcessor,
    VectorEntity,
    VectorRelation
)
from src.services.embedding_service import OllamaEmbeddingService as OllamaEmbeddingClient
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_embedding_client():
    """测试向量化客户端"""
    print("🧪 测试向量化客户端...")
    
    client = OllamaEmbeddingClient()
    
    # 测试单个文本向量化
    test_text = "帕金森病是一种神经退行性疾病"
    embedding = client.embed_text(test_text)
    
    if embedding:
        print(f"✅ 向量化成功: 文本长度 {len(test_text)}, 向量维度 {len(embedding)}")
        print(f"📊 向量前5个值: {embedding[:5]}")
        assert len(embedding) > 0, "向量维度应该大于0"
    else:
        print("❌ 向量化失败")
        assert False, "单个文本向量化失败"
    
    # 测试批量向量化
    test_texts = [
        "帕金森病的症状包括震颤",
        "多巴胺是重要的神经递质",
        "深部脑刺激是一种治疗方法"
    ]
    
    embeddings = client.embed_batch(test_texts)
    
    if len(embeddings) == len(test_texts):
        print(f"✅ 批量向量化成功: {len(embeddings)} 个向量")
        assert all(len(v) > 0 for v in embeddings if v), "所有向量维度都应该大于0"
    else:
        print(f"❌ 批量向量化失败: 期望 {len(test_texts)} 个向量，实际 {len(embeddings)} 个")
        assert False, "批量文本向量化失败"

def test_weaviate_store():
    """测试Weaviate存储"""
    print("\n=== 测试Weaviate存储 ===")
    
    try:
        # 初始化Weaviate存储
        vector_store = WeaviateVectorStore()
        
        # 设置集合
        if vector_store.setup_collections():
            print("✓ 集合设置成功")
        else:
            print("✗ 集合设置失败")
            return
        
        # 获取统计信息
        stats = vector_store.get_stats()
        print(f"✓ 存储统计: {stats}")
        
        # 测试搜索（使用随机向量）
        import random
        test_vector = [random.random() for _ in range(1024)]  # bge-m3模型的向量维度
        
        entities = vector_store.search_entities(test_vector, limit=5)
        relations = vector_store.search_relations(test_vector, limit=5)
        
        print(f"✓ 搜索测试完成 - 实体: {len(entities)}, 关系: {len(relations)}")
        
        # 关闭连接
        vector_store.client.close()
        
    except Exception as e:
        print(f"✗ Weaviate存储测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_vector_processor():
    """测试向量知识处理器"""
    print("\n=== 测试向量知识处理器 ===")
    
    try:
        # 初始化组件
        embedding_client = OllamaEmbeddingClient()
        vector_store = WeaviateVectorStore()
        processor = VectorKnowledgeProcessor(embedding_client, vector_store)
        
        # 设置Weaviate集合
        if not vector_store.setup_collections():
            print("✗ Weaviate集合设置失败")
            return
        
        # 测试实体处理
        test_entities = [
            {"name": "帕金森病", "type": "Disease", "properties": {"description": "神经退行性疾病"}},
            {"name": "震颤", "type": "Symptom", "properties": {"description": "不自主肌肉收缩"}}
        ]
        
        # 测试关系处理
        test_relations = [
            {"source": "帕金森病", "target": "震颤", "relation_type": "HAS_SYMPTOM", "properties": {"description": "主要症状"}}
        ]
        
        # 处理并存储
        entity_success = processor.process_and_store_entities(test_entities, "测试文本")
        relation_success = processor.process_and_store_relations(test_relations, "测试文本")
        
        if entity_success and relation_success:
            print("✓ 实体和关系处理成功")
        else:
            print(f"✗ 处理失败 - 实体: {entity_success}, 关系: {relation_success}")
        
        # 测试知识搜索
        results = processor.search_knowledge("帕金森病的症状", limit=5)
        print(f"✓ 知识搜索完成，找到 {len(results)} 个结果")
        
        if results:
            print(f"   最相关结果: {results[0].get('name', results[0].get('source', 'Unknown'))}")
        
        # 关闭连接
        vector_store.client.close()
        
    except Exception as e:
        print(f"✗ 向量处理器测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("🚀 开始向量存储功能测试\n")
    
    tests = [
        ("向量化客户端", test_embedding_client),
        ("Weaviate存储", test_weaviate_store),
        ("向量处理器", test_vector_processor)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试出错: {e}")
            results.append((test_name, False))
    
    # 输出测试结果
    print("\n📋 测试结果汇总:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！向量存储功能正常")
    else:
        print("⚠️  部分测试失败，请检查配置和服务状态")

if __name__ == "__main__":
    main()