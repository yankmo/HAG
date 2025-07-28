#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量检索验证脚本
测试余弦相似度和欧氏距离检索功能
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
import json
import numpy as np
from typing import List, Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_weaviate_connection():
    """测试Weaviate连接"""
    print("\n=== 测试Weaviate连接 ===")
    try:
        vector_store = WeaviateVectorStore()
        stats = vector_store.get_stats()
        print(f"✓ Weaviate连接成功")
        print(f"  实体数量: {stats['entities']}")
        print(f"  关系数量: {stats['relations']}")
        print(f"  总计: {stats['total']}")
        return vector_store, stats['total'] > 0
    except Exception as e:
        print(f"✗ Weaviate连接失败: {e}")
        return None, False

def test_ollama_embedding():
    """测试Ollama向量化"""
    print("\n=== 测试Ollama向量化 ===")
    try:
        embedding_client = OllamaEmbeddingClient()
        
        # 测试单个文本向量化
        test_text = "帕金森病的症状"
        vector = embedding_client.embed_text(test_text)
        
        if vector and len(vector) > 0:
            print(f"✓ 向量化成功")
            print(f"  文本: {test_text}")
            print(f"  向量维度: {len(vector)}")
            print(f"  向量前5个值: {vector[:5]}")
            return embedding_client, True
        else:
            print(f"✗ 向量化失败，返回空向量")
            return None, False
            
    except Exception as e:
        print(f"✗ Ollama向量化失败: {e}")
        return None, False

def load_document_data():
    """加载文档数据"""
    print("\n=== 加载文档数据 ===")
    try:
        # 读取data目录下的文档
        data_dir = "data"
        documents = []
        
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith('.txt'):
                    filepath = os.path.join(data_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            documents.append({
                                'filename': filename,
                                'content': content,
                                'length': len(content)
                            })
        
        print(f"✓ 找到 {len(documents)} 个文档")
        for doc in documents:
            print(f"  - {doc['filename']}: {doc['length']} 字符")
        
        return documents
        
    except Exception as e:
        print(f"✗ 加载文档数据失败: {e}")
        return []

def store_document_vectors(embedding_client, vector_store, documents):
    """存储文档向量"""
    print("\n=== 存储文档向量 ===")
    try:
        # 重新设置集合
        if not vector_store.setup_collections():
            print("✗ 设置集合失败")
            return False
        
        # 处理文档，创建实体
        entities = []
        for i, doc in enumerate(documents):
            # 将文档分段
            content = doc['content']
            # 简单分段：按句号分割
            sentences = [s.strip() for s in content.split('。') if s.strip()]
            
            for j, sentence in enumerate(sentences[:10]):  # 限制前10个句子
                if len(sentence) > 10:  # 过滤太短的句子
                    entity = VectorEntity(
                        name=f"文档片段_{i+1}_{j+1}",
                        type="文档内容",
                        properties={"description": sentence, "source_file": doc['filename']},
                        source_text=sentence
                    )
                    entities.append(entity)
        
        print(f"  创建了 {len(entities)} 个文档片段实体")
        
        # 向量化并存储
        processor = VectorKnowledgeProcessor(embedding_client, vector_store)
        
        # 批量向量化
        entity_texts = []
        for entity in entities:
            text = f"实体: {entity.name}, 类型: {entity.type}, 描述: {entity.properties.get('description', '')}"
            entity_texts.append(text)
        
        print("  正在向量化文档片段...")
        embeddings = embedding_client.embed_batch(entity_texts)
        
        # 设置向量
        vector_entities = []
        for i, entity in enumerate(entities):
            if i < len(embeddings) and embeddings[i]:
                entity.vector = embeddings[i]
                vector_entities.append(entity)
        
        # 存储到Weaviate
        print("  正在存储到Weaviate...")
        success = vector_store.store_entities(vector_entities)
        
        if success:
            print(f"✓ 成功存储 {len(vector_entities)} 个文档向量")
            return True
        else:
            print("✗ 存储失败")
            return False
            
    except Exception as e:
        print(f"✗ 存储文档向量失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_retrieval(embedding_client, vector_store):
    """测试向量检索功能"""
    print("\n=== 测试向量检索功能 ===")
    
    test_queries = [
        "帕金森病的症状有哪些",
        "如何治疗帕金森病",
        "帕金森病的病因",
        "帕金森病的预防方法",
        "帕金森病患者的护理"
    ]
    
    for query in test_queries:
        print(f"\n🔍 查询: {query}")
        try:
            # 向量化查询
            query_vector = embedding_client.embed_text(query)
            if not query_vector:
                print("  ✗ 查询向量化失败")
                continue
            
            print(f"  查询向量维度: {len(query_vector)}")
            
            # 执行向量检索
            results = vector_store.search_entities(query_vector, limit=5)
            
            print(f"  找到 {len(results)} 个相关结果:")
            for i, result in enumerate(results, 1):
                distance = result.get('distance', 'N/A')
                name = result.get('name', 'N/A')
                description = result.get('description', 'N/A')
                print(f"    {i}. {name}")
                print(f"       距离: {distance}")
                print(f"       内容: {description[:100]}...")
                
        except Exception as e:
            print(f"  ✗ 检索失败: {e}")

def test_cosine_and_euclidean_similarity():
    """测试余弦相似度和欧氏距离计算"""
    print("\n=== 测试相似度计算 ===")
    
    try:
        embedding_client = OllamaEmbeddingClient()
        
        # 测试文本
        text1 = "帕金森病是一种神经退行性疾病"
        text2 = "帕金森病的主要症状包括震颤"
        text3 = "糖尿病是一种代谢性疾病"
        
        # 向量化
        vec1 = embedding_client.embed_text(text1)
        vec2 = embedding_client.embed_text(text2)
        vec3 = embedding_client.embed_text(text3)
        
        if not all([vec1, vec2, vec3]):
            print("✗ 向量化失败")
            return
        
        # 转换为numpy数组
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        v3 = np.array(vec3)
        
        # 计算余弦相似度
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # 计算欧氏距离
        def euclidean_distance(a, b):
            return np.linalg.norm(a - b)
        
        print("相似度计算结果:")
        print(f"  文本1: {text1}")
        print(f"  文本2: {text2}")
        print(f"  文本3: {text3}")
        print()
        
        cos_12 = cosine_similarity(v1, v2)
        cos_13 = cosine_similarity(v1, v3)
        cos_23 = cosine_similarity(v2, v3)
        
        euc_12 = euclidean_distance(v1, v2)
        euc_13 = euclidean_distance(v1, v3)
        euc_23 = euclidean_distance(v2, v3)
        
        print("余弦相似度 (越接近1越相似):")
        print(f"  文本1 vs 文本2: {cos_12:.4f}")
        print(f"  文本1 vs 文本3: {cos_13:.4f}")
        print(f"  文本2 vs 文本3: {cos_23:.4f}")
        print()
        
        print("欧氏距离 (越小越相似):")
        print(f"  文本1 vs 文本2: {euc_12:.4f}")
        print(f"  文本1 vs 文本3: {euc_13:.4f}")
        print(f"  文本2 vs 文本3: {euc_23:.4f}")
        
        # 验证相关性
        print("\n分析:")
        if cos_12 > cos_13 and cos_12 > cos_23:
            print("✓ 余弦相似度正确：帕金森相关文本更相似")
        if euc_12 < euc_13 and euc_12 < euc_23:
            print("✓ 欧氏距离正确：帕金森相关文本距离更近")
            
    except Exception as e:
        print(f"✗ 相似度计算失败: {e}")

def main():
    """主函数"""
    print("🚀 开始向量检索验证")
    print("=" * 50)
    
    # 1. 测试Weaviate连接
    vector_store, has_data = test_weaviate_connection()
    if not vector_store:
        print("❌ Weaviate连接失败，无法继续测试")
        return
    
    # 2. 测试Ollama向量化
    embedding_client, embedding_ok = test_ollama_embedding()
    if not embedding_client:
        print("❌ Ollama向量化失败，无法继续测试")
        return
    
    # 3. 测试相似度计算
    test_cosine_and_euclidean_similarity()
    
    # 4. 加载文档数据
    documents = load_document_data()
    if not documents:
        print("⚠️ 没有找到文档数据，跳过文档向量存储测试")
    else:
        # 5. 存储文档向量
        if store_document_vectors(embedding_client, vector_store, documents):
            has_data = True
    
    # 6. 测试向量检索
    if has_data:
        test_vector_retrieval(embedding_client, vector_store)
    else:
        print("⚠️ 没有数据，跳过检索测试")
    
    print("\n" + "=" * 50)
    print("🎉 向量检索验证完成")

if __name__ == "__main__":
    main()