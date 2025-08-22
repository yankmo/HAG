#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用类型定义模块
提供服务间共享的数据类和枚举类型
"""

import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any

# 配置日志
logger = logging.getLogger(__name__)

class DistanceMetric(Enum):
    """距离度量类型"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    DOT_PRODUCT = "dot_product"

@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    content: str
    score: float
    distance: float
    metadata: Dict[str, Any] = None
    distance_metric: str = ""
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class HybridSearchResult:
    """混合搜索结果"""
    cosine_results: List[SearchResult]
    euclidean_results: List[SearchResult]
    hybrid_results: List[SearchResult]
    statistics: Dict[str, Any]

@dataclass
class IntentAwareSearchResult:
    """意图感知搜索结果"""
    intent: Any  # IntentResult类型，避免循环导入
    search_results: HybridSearchResult
    related_knowledge: Dict[str, Any]
    recommendations: List[str]

class SimilarityCalculator:
    """相似度计算器"""
    
    @staticmethod
    def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
        """计算余弦相似度
        
        Args:
            vector1: 向量1
            vector2: 向量2
            
        Returns:
            余弦相似度值 (0-1)
        """
        try:
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            
            # 计算点积
            dot_product = np.dot(v1, v2)
            
            # 计算向量的模长
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            # 避免除零错误
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            
            # 计算余弦相似度
            similarity = dot_product / (norm_v1 * norm_v2)
            
            # 确保结果在[0, 1]范围内
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"计算余弦相似度失败: {e}")
            return 0.0
    
    @staticmethod
    def euclidean_distance(vector1: List[float], vector2: List[float]) -> float:
        """计算欧氏距离
        
        Args:
            vector1: 向量1
            vector2: 向量2
            
        Returns:
            欧氏距离值
        """
        try:
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            
            # 计算欧氏距离
            distance = np.linalg.norm(v1 - v2)
            
            return float(distance)
            
        except Exception as e:
            logger.error(f"计算欧氏距离失败: {e}")
            return float('inf')
    
    @staticmethod
    def manhattan_distance(vector1: List[float], vector2: List[float]) -> float:
        """计算曼哈顿距离
        
        Args:
            vector1: 向量1
            vector2: 向量2
            
        Returns:
            曼哈顿距离值
        """
        try:
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            
            # 计算曼哈顿距离
            distance = np.sum(np.abs(v1 - v2))
            
            return float(distance)
            
        except Exception as e:
            logger.error(f"计算曼哈顿距离失败: {e}")
            return float('inf')
    
    @staticmethod
    def dot_product_similarity(vector1: List[float], vector2: List[float]) -> float:
        """计算点积相似度
        
        Args:
            vector1: 向量1
            vector2: 向量2
            
        Returns:
            点积值
        """
        try:
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            
            # 计算点积
            dot_product = np.dot(v1, v2)
            
            return float(dot_product)
            
        except Exception as e:
            logger.error(f"计算点积相似度失败: {e}")
            return 0.0