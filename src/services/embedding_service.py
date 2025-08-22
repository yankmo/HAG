#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量化服务模块 - 提供统一的文本向量化接口
"""

import sys
import os
import requests
import logging
from typing import List
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import get_config

logger = logging.getLogger(__name__)

class OllamaEmbeddingService:
    """基于 Ollama 的向量化服务"""
    
    def __init__(self, model: str = None, base_url: str = None, timeout: int = None):
        """
        初始化 Ollama 向量化服务
        
        Args:
            model: 向量化模型名称，默认从配置获取
            base_url: Ollama 服务地址，默认从配置获取
            timeout: 请求超时时间，默认从配置获取
        """
        config = get_config()
        
        self.model = model or config.ollama.embedding_model
        self.base_url = base_url or config.ollama.base_url
        self.timeout = timeout or config.ollama.timeout
        
        logger.info(f"Ollama向量化服务初始化: 模型={self.model}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        将文本转换为向量
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 向量表示
        """
        if not text or not text.strip():
            logger.warning("文本为空，无法向量化")
            return []
        
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text.strip()
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            if "embedding" in result:
                return result["embedding"]
            else:
                logger.error("向量化失败，响应中没有embedding字段")
                return []
                
        except Exception as e:
            logger.error(f"向量化请求失败: {e}")
            return []
    
    def embed_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        批量向量化文本
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            
        Returns:
            List[List[float]]: 向量列表
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"正在处理批次 {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            for text in batch:
                embedding = self.embed_text(text)
                embeddings.append(embedding)
                
        return embeddings
    
    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的向量表示（embed_text的别名）
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 向量表示
        """
        return self.embed_text(text)
    
    def get_embedding_dimension(self) -> int:
        """
        获取向量维度
        
        Returns:
            int: 向量维度
        """
        # 使用测试文本获取维度
        test_embedding = self.embed_text("测试文本")
        if test_embedding:
            return len(test_embedding)
        else:
            logger.error("无法获取向量维度")
            return 0

# 为了向后兼容，保留原有的类名
OllamaEmbeddingClient = OllamaEmbeddingService