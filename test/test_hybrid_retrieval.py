#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试混合向量检索功能
验证余弦相似度和欧氏距离的混合检索
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knowledge.vector_storage import (
    WeaviateVectorStore, 
    VectorKnowledgeProcessor
)
from src.services.embedding_service import OllamaEmbeddingService as OllamaEmbeddingClient
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hybrid_retrieval():
    """测试混合检索功能"""
    print("🚀 开始测试混合向量检索功能")
    print("=" * 60)
    
    try:
        # 初始化组件
        print("\n=== 初始化组件 ===")
        embedding_client = OllamaEmbeddingClient()
        vector_store = WeaviateVectorStore()
        processor = VectorKnowledgeProcessor(embedding_client, vector_store)
        
        # 检查数据状态
        stats = vector_store.get_stats()
        print(f"✓ 数据库状态: {stats['entities']} 个实体, {stats['relations']} 个关系")
        
        if stats['total'] == 0:
            print("⚠️ 数据库为空，无法进行检索测试")
            return
        
        # 测试查询
        test_queries = [
            "帕金森病的治疗方法有哪些",
            "帕金森病的症状表现",
            "帕金森病患者的用药注意事项",
            "帕金森病的病因机制",
            "帕金森病的预防措施"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"🔍 测试查询 {i}: {query}")
            print("="*60)
            
            # 执行混合检索
            hybrid_results = processor.search_knowledge_hybrid(query, limit=5)
            
            # 显示检索统计
            stats = hybrid_results.get("retrieval_stats", {})
            print(f"\n📊 检索统计:")
            print(f"  总找到: {stats.get('total_found', 0)} 个片段")
            print(f"  余弦相似度: {stats.get('cosine_count', 0)} 个")
            print(f"  欧氏距离: {stats.get('euclidean_count', 0)} 个")
            print(f"  混合结果: {stats.get('hybrid_count', 0)} 个")
            
            # 显示Top5知识
            top5_knowledge = hybrid_results.get("top5_knowledge", [])
            print(f"\n🏆 Top5 知识片段:")
            for knowledge in top5_knowledge:
                rank = knowledge.get("rank", "N/A")
                source = knowledge.get("source", "未知来源")
                content = knowledge.get("content", "")[:100] + "..." if len(knowledge.get("content", "")) > 100 else knowledge.get("content", "")
                cosine_sim = knowledge.get("cosine_similarity", "N/A")
                euclidean_dist = knowledge.get("euclidean_distance", "N/A")
                
                print(f"  {rank}. 【{source}】")
                print(f"     余弦相似度: {cosine_sim}")
                print(f"     欧氏距离: {euclidean_dist}")
                print(f"     内容: {content}")
                print()
            
            # 测试提示词格式化
            print(f"📝 提示词格式化结果:")
            prompt_knowledge = processor.get_knowledge_for_prompt(query, limit=3)
            print(prompt_knowledge[:300] + "..." if len(prompt_knowledge) > 300 else prompt_knowledge)
            
            if i < len(test_queries):
                input("\n按回车键继续下一个测试...")
        
        print("\n" + "="*60)
        print("🎉 混合向量检索测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_distance_comparison():
    """测试距离度量比较"""
    print("\n" + "="*60)
    print("🔬 测试距离度量比较")
    print("="*60)
    
    try:
        embedding_client = OllamaEmbeddingClient()
        vector_store = WeaviateVectorStore()
        
        query = "帕金森病的治疗"
        print(f"查询: {query}")
        
        # 向量化查询
        query_vector = embedding_client.embed_text(query)
        
        # 分别使用两种距离度量
        print(f"\n📏 余弦相似度检索结果:")
        cosine_results = vector_store.search_entities(query_vector, limit=3, distance_metric="cosine")
        for i, result in enumerate(cosine_results, 1):
            name = result.get("name", "N/A")
            certainty = result.get("certainty", "N/A")
            distance = result.get("distance", "N/A")
            print(f"  {i}. {name}")
            print(f"     余弦相似度(certainty): {certainty}")
            print(f"     距离: {distance}")
        
        print(f"\n📐 欧氏距离检索结果:")
        euclidean_results = vector_store.search_entities(query_vector, limit=3, distance_metric="euclidean")
        for i, result in enumerate(euclidean_results, 1):
            name = result.get("name", "N/A")
            distance = result.get("distance", "N/A")
            print(f"  {i}. {name}")
            print(f"     欧氏距离: {distance}")
        
        print(f"\n🔄 混合检索结果:")
        hybrid_results = vector_store.search_entities_hybrid(query_vector, limit=3)
        for i, result in enumerate(hybrid_results.get("hybrid_results", []), 1):
            name = result.get("name", "N/A")
            cosine_rank = result.get("rank_cosine", "N/A")
            euclidean_rank = result.get("rank_euclidean", "N/A")
            cosine_sim = result.get("cosine_similarity", "N/A")
            euclidean_dist = result.get("distance", "N/A")
            print(f"  {i}. {name}")
            print(f"     余弦排名: {cosine_rank}, 欧氏排名: {euclidean_rank}")
            print(f"     余弦相似度: {cosine_sim}, 欧氏距离: {euclidean_dist}")
        
    except Exception as e:
        print(f"❌ 距离比较测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    test_hybrid_retrieval()
    test_distance_comparison()

if __name__ == "__main__":
    main()