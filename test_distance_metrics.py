#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试距离度量显示功能
验证余弦相似度和欧氏距离的计算和显示
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services import RetrievalService
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_distance_metrics():
    """测试距离度量功能"""
    try:
        print("🔍 测试距离度量功能...")
        
        # 创建检索服务
        retrieval_service = RetrievalService()
        print("✅ 检索服务创建成功")
        
        # 测试查询
        test_query = "帕金森病的治疗方法"
        print(f"📝 测试查询: {test_query}")
        
        # 执行混合检索
        print("🔄 执行混合检索...")
        hybrid_result = retrieval_service.search_hybrid(test_query, limit=3)
        
        print(f"📊 检索结果统计:")
        print(f"  - 余弦相似度结果: {len(hybrid_result.cosine_results)}")
        print(f"  - 欧氏距离结果: {len(hybrid_result.euclidean_results)}")
        print(f"  - 混合结果: {len(hybrid_result.hybrid_results)}")
        
        # 分析混合结果的距离信息
        print("\n📈 混合结果详细分析:")
        for i, result in enumerate(hybrid_result.hybrid_results[:3], 1):
            print(f"\n结果 {i}:")
            print(f"  - ID: {result.id}")
            print(f"  - 综合评分: {result.score:.3f}")
            print(f"  - 距离: {result.distance:.3f}")
            print(f"  - 距离度量: {result.distance_metric}")
            
            # 检查metadata中的详细信息
            metadata = result.metadata
            if metadata:
                print(f"  - 实体名称: {metadata.get('name', 'N/A')}")
                print(f"  - 实体类型: {metadata.get('type', 'N/A')}")
                
                # 距离度量详情
                cosine_sim = metadata.get('cosine_similarity', 'N/A')
                euclidean_dist = metadata.get('euclidean_distance', 'N/A')
                cosine_rank = metadata.get('cosine_rank', 'N/A')
                euclidean_rank = metadata.get('euclidean_rank', 'N/A')
                
                print(f"  - 余弦相似度: {cosine_sim}")
                print(f"  - 欧氏距离: {euclidean_dist}")
                print(f"  - 余弦排名: {cosine_rank}")
                print(f"  - 欧氏排名: {euclidean_rank}")
        
        # 测试距离度量比较功能
        print("\n🔬 测试距离度量比较功能...")
        comparison_result = retrieval_service.compare_distance_metrics(test_query, limit=3)
        
        if comparison_result:
            print("✅ 距离度量比较成功")
            analysis = comparison_result.get('analysis', {})
            print(f"  - 重叠率: {analysis.get('overlap_rate', 0):.2%}")
            print(f"  - 余弦相似度平均分: {analysis.get('cosine_avg_score', 0):.3f}")
            print(f"  - 欧氏距离平均分: {analysis.get('euclidean_avg_score', 0):.3f}")
        
        print("\n🎉 距离度量功能测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_distance_metrics()
    if success:
        print("\n✅ 所有测试通过！前端应该能正确显示余弦相似度和欧氏距离。")
    else:
        print("\n❌ 测试失败，请检查配置。")