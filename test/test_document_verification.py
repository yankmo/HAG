#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档存储验证和提示词格式化测试
验证Weaviate中存储的完整文档内容，并测试提示词模板功能
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

def test_document_storage_verification():
    """验证文档存储情况"""
    print("=" * 60)
    print("📚 验证Weaviate文档存储情况")
    print("=" * 60)
    
    try:
        # 初始化组件
        embedding_client = OllamaEmbeddingClient()
        vector_store = WeaviateVectorStore()
        processor = VectorKnowledgeProcessor(embedding_client, vector_store)
        
        # 获取存储统计
        stats = vector_store.get_stats()
        print(f"📊 存储统计:")
        print(f"   实体数量: {stats.get('entities', 0)}")
        print(f"   关系数量: {stats.get('relations', 0)}")
        print(f"   总计: {stats.get('entities', 0) + stats.get('relations', 0)}")
        
        if stats.get('entities', 0) == 0:
            print("⚠️  警告: 未发现存储的实体数据")
            return False
        
        print(f"✅ 发现 {stats.get('entities', 0)} 个存储的文档片段")
        return True
        
    except Exception as e:
        print(f"❌ 文档存储验证失败: {e}")
        return False

def test_knowledge_retrieval_for_prompt():
    """测试知识检索和提示词格式化"""
    print("\n" + "=" * 60)
    print("🔍 测试知识检索和提示词格式化")
    print("=" * 60)
    
    try:
        # 初始化组件
        embedding_client = OllamaEmbeddingClient()
        vector_store = WeaviateVectorStore()
        processor = VectorKnowledgeProcessor(embedding_client, vector_store)
        
        # 测试查询列表
        test_queries = [
            "帕金森病的治疗方法",
            "帕金森病患者的护理",
            "帕金森病的症状表现"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 测试查询 {i}: {query}")
            print("-" * 40)
            
            # 执行混合检索
            hybrid_results = processor.search_knowledge_hybrid(query, limit=5)
            
            # 显示检索统计
            stats = hybrid_results.get("retrieval_stats", {})
            print(f"📊 检索统计:")
            print(f"   总找到: {stats.get('total_found', 0)} 个片段")
            print(f"   余弦相似度: {stats.get('cosine_count', 0)} 个")
            print(f"   欧氏距离: {stats.get('euclidean_count', 0)} 个")
            print(f"   混合结果: {stats.get('hybrid_count', 0)} 个")
            
            # 显示Top5知识片段
            top5_knowledge = hybrid_results.get("top5_knowledge", [])
            print(f"\n📋 Top5 知识片段:")
            for j, knowledge in enumerate(top5_knowledge[:3], 1):  # 只显示前3个
                content = knowledge.get("content", "")[:200]  # 截取前200字符
                source = knowledge.get("source", "")
                cosine_sim = knowledge.get("cosine_similarity", "N/A")
                euclidean_dist = knowledge.get("euclidean_distance", "N/A")
                
                print(f"   {j}. 【{source}】")
                print(f"      内容: {content}...")
                print(f"      余弦相似度: {cosine_sim}")
                print(f"      欧氏距离: {euclidean_dist}")
            
            # 测试提示词格式化
            print(f"\n📝 提示词格式化结果:")
            prompt_knowledge = processor.get_knowledge_for_prompt(query, limit=3)
            print(prompt_knowledge[:500] + "..." if len(prompt_knowledge) > 500 else prompt_knowledge)
            
            if i < len(test_queries):
                print("\n" + "─" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ 知识检索测试失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return False

def test_document_content_verification():
    """验证文档内容的完整性"""
    print("\n" + "=" * 60)
    print("📄 验证文档内容完整性")
    print("=" * 60)
    
    try:
        # 初始化组件
        embedding_client = OllamaEmbeddingClient()
        vector_store = WeaviateVectorStore()
        processor = VectorKnowledgeProcessor(embedding_client, vector_store)
        
        # 执行一个通用查询来获取文档样本
        query = "医疗"
        hybrid_results = processor.search_knowledge_hybrid(query, limit=10)
        
        top_knowledge = hybrid_results.get("top5_knowledge", [])
        
        print(f"📊 文档内容分析:")
        print(f"   检索到的文档片段数: {len(top_knowledge)}")
        
        if not top_knowledge:
            print("⚠️  警告: 未检索到任何文档内容")
            return False
        
        # 分析文档内容
        total_chars = 0
        valid_docs = 0
        
        print(f"\n📋 文档内容样本:")
        for i, knowledge in enumerate(top_knowledge[:5], 1):
            content = knowledge.get("content", "")
            source = knowledge.get("source", "")
            
            if content and len(content.strip()) > 10:  # 有效内容
                valid_docs += 1
                total_chars += len(content)
                
                print(f"   {i}. 【{source}】")
                print(f"      长度: {len(content)} 字符")
                print(f"      内容预览: {content[:100]}...")
                print()
        
        print(f"📈 内容统计:")
        print(f"   有效文档数: {valid_docs}")
        print(f"   总字符数: {total_chars}")
        print(f"   平均长度: {total_chars // valid_docs if valid_docs > 0 else 0} 字符")
        
        if valid_docs > 0:
            print("✅ 文档内容验证通过 - 发现有效的纯文本文档内容")
            return True
        else:
            print("❌ 文档内容验证失败 - 未发现有效内容")
            return False
        
    except Exception as e:
        print(f"❌ 文档内容验证失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始文档存储验证和提示词格式化测试")
    
    # 测试1: 验证文档存储情况
    storage_ok = test_document_storage_verification()
    
    if not storage_ok:
        print("\n❌ 文档存储验证失败，请检查数据是否已正确存储")
        return
    
    # 测试2: 验证文档内容完整性
    content_ok = test_document_content_verification()
    
    if not content_ok:
        print("\n❌ 文档内容验证失败")
        return
    
    # 测试3: 测试知识检索和提示词格式化
    retrieval_ok = test_knowledge_retrieval_for_prompt()
    
    if retrieval_ok:
        print("\n" + "=" * 60)
        print("🎉 所有测试通过!")
        print("✅ Weaviate文档存储正常")
        print("✅ 文档内容完整有效")
        print("✅ 混合向量检索功能正常")
        print("✅ 提示词格式化功能正常")
        print("=" * 60)
    else:
        print("\n❌ 部分测试失败，请检查相关功能")

if __name__ == "__main__":
    main()