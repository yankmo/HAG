#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 服务模块 - 提供统一的大语言模型接口
"""

import sys
import os
import requests
import logging
from typing import Dict, Any, Optional, List

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import get_config

logger = logging.getLogger(__name__)

class OllamaLLMService:
    """基于 Ollama 的大语言模型服务"""
    
    def __init__(self, model: str = None, base_url: str = None, timeout: int = None):
        """
        初始化 Ollama LLM 服务
        
        Args:
            model: 模型名称，默认从配置获取
            base_url: Ollama 服务地址，默认从配置获取
            timeout: 请求超时时间，默认从配置获取
        """
        config = get_config()
        
        self.model = model or config.ollama.default_model
        self.base_url = base_url or config.ollama.base_url
        self.timeout = timeout or config.ollama.timeout
        
        logger.info(f"Ollama LLM服务初始化: 模型={self.model}")
    
    def generate_response(self, prompt: str, temperature: float = 0.7, 
                         max_tokens: int = 1000, stream: bool = False, **kwargs) -> str:
        """
        生成文本回答
        
        Args:
            prompt: 输入提示
            temperature: 温度参数
            max_tokens: 最大token数
            stream: 是否流式输出
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
        """
        if not prompt or not prompt.strip():
            logger.warning("提示为空，无法生成回答")
            return ""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt.strip(),
                    "stream": stream,
                    "options": {
                        "temperature": temperature,
                        "top_p": kwargs.get("top_p", 0.9),
                        "num_predict": max_tokens,
                    }
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            if "response" in result:
                return result["response"]
            else:
                logger.error("生成回答失败，响应中没有response字段")
                return "抱歉，无法生成回答。"
                
        except Exception as e:
            logger.error(f"生成回答请求失败: {e}")
            return "抱歉，无法生成回答。"

# 为了向后兼容，保留原有的类名
SimpleOllamaLLM = OllamaLLMService
OllamaLLM = OllamaLLMService