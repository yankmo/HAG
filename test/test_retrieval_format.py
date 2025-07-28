#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试检索结果格式
验证description字段和相似度分数是否正确显示
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.text_processing_service import TextProcessingService
from src.services.retrieval_service import RetrievalService
from src.knowledge.vector_storage import WeaviateVectorStore
from src.services.embedding_service import OllamaEmbeddingService
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_retrieval_format():
    """测试检索结果格式"""
    print("\n=== 测试检索结果格式 ===")
    
    try:
        # 初始化服务
        embedding_service = OllamaEmbeddingService()
        vector_store = WeaviateVectorStore()
        retrieval_service = RetrievalService(embedding_service, vector_store)
        
        # 测试查询
        query = "帕金森病的症状"
        print(f"\n🔍 查询: {query}")
        
        # 余弦相似度搜索
        print("\n📊 余弦相似度搜索结果:")
        cosine_results = retrieval_service.search_by_cosine(query, limit=3)
        
        for i, result in enumerate(cosine_results, 1):
            print(f"\n实体 {i}: {result.id}")
            print(f"  相似度: {result.score:.4f}")
            print(f"  距离: {result.distance:.4f}")
            print(f"  描述: {result.metadata.get('description', 'N/A')}")
            print(f"  内容预览: {result.content[:100]}...")
            print(f"  元数据: {result.metadata}")
        
        # 欧氏距离搜索
        print("\n📊 欧氏距离搜索结果:")
        euclidean_results = retrieval_service.search_by_euclidean(query, limit=3)
        
        for i, result in enumerate(euclidean_results, 1):
            print(f"\n实体 {i}: {result.id}")
            print(f"  相似度: {result.score:.4f}")
            print(f"  距离: {result.distance:.4f}")
            print(f"  描述: {result.metadata.get('description', 'N/A')}")
            print(f"  内容预览: {result.content[:100]}...")
        
        # 混合搜索
        print("\n📊 混合搜索结果:")
        hybrid_result = retrieval_service.search_hybrid(query, limit=3)
        
        print(f"\n统计信息:")
        print(f"  余弦结果数: {len(hybrid_result.cosine_results)}")
        print(f"  欧氏结果数: {len(hybrid_result.euclidean_results)}")
        print(f"  混合结果数: {len(hybrid_result.hybrid_results)}")
        print(f"  统计数据: {hybrid_result.statistics}")
        
        print(f"\n混合结果详情:")
        for i, result in enumerate(hybrid_result.hybrid_results, 1):
            print(f"\n实体 {i}: {result.id}")
            print(f"  相似度: {result.score:.4f}")
            print(f"  距离: {result.distance:.4f}")
            print(f"  描述: {result.metadata.get('description', 'N/A')}")
            print(f"  内容预览: {result.content[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"检索格式测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始检索结果格式测试")
    
    if test_retrieval_format():
        print("\n✅ 检索结果格式测试通过")
    else:
        print("\n❌ 检索结果格式测试失败")

if __name__ == "__main__":
    main()