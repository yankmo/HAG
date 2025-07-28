#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档向量化和检索测试脚本
测试新的文本处理和检索服务模块
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.text_processing_service import TextProcessingService
from src.services.retrieval_service import RetrievalService
from src.knowledge.vector_storage import WeaviateVectorStore
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_text_processing():
    """测试文本处理服务"""
    print("\n" + "="*60)
    print("🔧 测试文本处理服务")
    print("="*60)
    
    try:
        # 初始化文本处理服务
        text_processor = TextProcessingService()
        
        # 测试文本清理
        test_text = "这是一个测试文本。   包含多余的空格和特殊字符@#$%。\n\n还有换行符。"
        cleaned_text = text_processor.clean_text(test_text)
        print(f"原始文本: {test_text}")
        print(f"清理后文本: {cleaned_text}")
        
        # 测试句子分割
        sentences = text_processor.split_text_by_sentences(cleaned_text)
        print(f"分割句子: {sentences}")
        
        # 测试文本分块
        long_text = "帕金森病是一种慢性神经退行性疾病。" * 20
        chunks = text_processor.chunk_text(long_text, "test_doc")
        print(f"文本分块: {len(chunks)} 个块")
        for i, chunk in enumerate(chunks[:3]):  # 只显示前3个
            print(f"  块{i+1}: {chunk.content[:50]}...")
        
        print("✅ 文本处理服务测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 文本处理服务测试失败: {e}")
        return False

def test_retrieval_service():
    """测试检索服务"""
    print("\n" + "="*60)
    print("🔍 测试检索服务")
    print("="*60)
    
    try:
        # 初始化检索服务
        retrieval_service = RetrievalService()
        
        # 测试相似度计算
        from src.services.retrieval_service import SimilarityCalculator
        calc = SimilarityCalculator()
        
        # 测试向量
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [2.0, 4.0, 6.0]  # vec1的2倍
        vec3 = [1.0, 0.0, 0.0]  # 不同方向
        
        # 计算相似度
        cosine_sim = calc.cosine_similarity(vec1, vec2)
        euclidean_dist = calc.euclidean_distance(vec1, vec2)
        manhattan_dist = calc.manhattan_distance(vec1, vec2)
        dot_product = calc.dot_product_similarity(vec1, vec2)
        
        print(f"向量1: {vec1}")
        print(f"向量2: {vec2}")
        print(f"余弦相似度: {cosine_sim:.4f}")
        print(f"欧氏距离: {euclidean_dist:.4f}")
        print(f"曼哈顿距离: {manhattan_dist:.4f}")
        print(f"点积: {dot_product:.4f}")
        
        # 测试不同向量
        cosine_sim2 = calc.cosine_similarity(vec1, vec3)
        euclidean_dist2 = calc.euclidean_distance(vec1, vec3)
        
        print(f"\n向量1: {vec1}")
        print(f"向量3: {vec3}")
        print(f"余弦相似度: {cosine_sim2:.4f}")
        print(f"欧氏距离: {euclidean_dist2:.4f}")
        
        print("✅ 检索服务测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 检索服务测试失败: {e}")
        return False

def test_weaviate_connection():
    """测试Weaviate连接"""
    print("\n" + "="*60)
    print("🔗 测试Weaviate连接")
    print("="*60)
    
    try:
        vector_store = WeaviateVectorStore()
        
        # 测试连接
        stats = vector_store.get_stats()
        print(f"Weaviate统计: {stats}")
        
        print("✅ Weaviate连接测试通过")
        return vector_store
        
    except Exception as e:
        print(f"❌ Weaviate连接测试失败: {e}")
        return None

def test_document_vectorization(file_path: str):
    """测试文档向量化"""
    print("\n" + "="*60)
    print(f"📄 测试文档向量化: {file_path}")
    print("="*60)
    
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return False
        
        # 初始化服务
        text_processor = TextProcessingService()
        vector_store = WeaviateVectorStore()
        
        # 设置Weaviate集合
        print("🏗️ 设置Weaviate集合...")
        setup_success = vector_store.setup_collections()
        if not setup_success:
            print("❌ Weaviate集合设置失败")
            return False
        
        # 处理文档
        print(f"📖 开始处理文档: {file_path}")
        vector_entities = text_processor.process_document(file_path)
        
        if not vector_entities:
            print("❌ 文档处理失败，没有生成向量实体")
            return False
        
        print(f"✅ 文档处理完成，生成 {len(vector_entities)} 个向量实体")
        
        # 显示前几个实体的信息
        print("\n📋 前3个向量实体:")
        for i, entity in enumerate(vector_entities[:3]):
            print(f"  实体{i+1}:")
            print(f"    名称: {entity.name}")
            print(f"    类型: {entity.type}")
            print(f"    内容长度: {len(entity.source_text)} 字符")
            print(f"    向量维度: {len(entity.vector) if entity.vector else 0}")
            print(f"    内容预览: {entity.source_text[:100]}...")
        
        # 存储到Weaviate
        print("\n💾 存储向量到Weaviate...")
        store_success = text_processor.store_to_weaviate(vector_entities, vector_store)
        
        if not store_success:
            print("❌ 向量存储失败")
            return False
        
        print("✅ 向量存储成功")
        
        # 获取存储统计
        stats = vector_store.get_stats()
        print(f"📊 存储统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ 文档向量化测试失败: {e}")
        return False

def test_document_retrieval():
    """测试文档检索"""
    print("\n" + "="*60)
    print("🔍 测试文档检索")
    print("="*60)
    
    try:
        # 初始化检索服务
        retrieval_service = RetrievalService()
        
        # 测试查询
        test_queries = [
            "帕金森病的症状",
            "帕金森病的治疗方法",
            "神经退行性疾病",
            "震颤和运动障碍"
        ]
        
        for query in test_queries:
            print(f"\n🔎 查询: {query}")
            
            # 余弦相似度搜索
            print("  📐 余弦相似度搜索:")
            cosine_results = retrieval_service.search_by_cosine(query, limit=3)
            for i, result in enumerate(cosine_results):
                print(f"    {i+1}. 分数: {result.score:.4f}, 内容: {result.content[:80]}...")
            
            # 欧氏距离搜索
            print("  📏 欧氏距离搜索:")
            euclidean_results = retrieval_service.search_by_euclidean(query, limit=3)
            for i, result in enumerate(euclidean_results):
                print(f"    {i+1}. 分数: {result.score:.4f}, 内容: {result.content[:80]}...")
            
            # 混合搜索
            print("  🔀 混合搜索:")
            hybrid_result = retrieval_service.search_hybrid(query, limit=3)
            for i, result in enumerate(hybrid_result.hybrid_results):
                print(f"    {i+1}. 分数: {result.score:.4f}, 内容: {result.content[:80]}...")
            
            print(f"  📊 统计: {hybrid_result.statistics}")
        
        print("\n✅ 文档检索测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 文档检索测试失败: {e}")
        return False

def test_distance_comparison():
    """测试距离度量比较"""
    print("\n" + "="*60)
    print("📊 测试距离度量比较")
    print("="*60)
    
    try:
        retrieval_service = RetrievalService()
        
        query = "帕金森病的主要症状"
        print(f"🔎 查询: {query}")
        
        # 比较不同距离度量
        comparison = retrieval_service.compare_distance_metrics(query, limit=5)
        
        print(f"\n📈 比较结果:")
        print(f"  重叠率: {comparison['analysis']['overlap_rate']:.2%}")
        print(f"  仅余弦相似度: {len(comparison['analysis']['cosine_only'])} 个")
        print(f"  仅欧氏距离: {len(comparison['analysis']['euclidean_only'])} 个")
        print(f"  共同结果: {len(comparison['analysis']['common_results'])} 个")
        print(f"  混合独有: {len(comparison['analysis']['hybrid_unique'])} 个")
        
        print("\n✅ 距离度量比较测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 距离度量比较测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始文档向量化和检索测试")
    print("=" * 80)
    
    # 测试1: 文本处理服务
    if not test_text_processing():
        return
    
    # 测试2: 检索服务
    if not test_retrieval_service():
        return
    
    # 测试3: Weaviate连接
    vector_store = test_weaviate_connection()
    if not vector_store:
        return
    
    # 测试4: 文档向量化
    document_path = "e:/Program/Project/HAG/data/pajinsen.txt"
    if not test_document_vectorization(document_path):
        return
    
    # 测试5: 文档检索
    if not test_document_retrieval():
        return
    
    # 测试6: 距离度量比较
    if not test_distance_comparison():
        return
    
    print("\n" + "="*80)
    print("🎉 所有测试通过！文档向量化和检索系统工作正常")
    print("="*80)

if __name__ == "__main__":
    main()