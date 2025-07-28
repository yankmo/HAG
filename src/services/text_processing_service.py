#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本处理和向量化服务
负责文本分词、清理、向量化和存储到Weaviate
"""

import re
import sys
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import get_config
from src.services.embedding_service import OllamaEmbeddingService
from src.knowledge.vector_storage import WeaviateVectorStore, VectorEntity

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """文本块"""
    content: str
    start_pos: int
    end_pos: int
    chunk_id: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class TextProcessingService:
    """文本处理服务"""
    
    def __init__(self, embedding_service: OllamaEmbeddingService = None):
        """初始化文本处理服务
        
        Args:
            embedding_service: 向量化服务实例
        """
        self.embedding_service = embedding_service or OllamaEmbeddingService()
        self.config = get_config()
        
        # 文本分块参数
        self.chunk_size = 500  # 每个文本块的字符数
        self.chunk_overlap = 50  # 文本块之间的重叠字符数
        
    def clean_text(self, text: str) -> str:
        """清理文本
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
            
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符（保留中文、英文、数字、基本标点）
        text = re.sub(r'[^\u4e00-\u9fff\w\s.,;:!?()（）、，。；：！？]', '', text)
        
        # 去除首尾空白
        text = text.strip()
        
        return text
    
    def split_text_by_sentences(self, text: str) -> List[str]:
        """按句子分割文本
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        # 中英文句子分割模式
        sentence_pattern = r'[。！？.!?]+|[\n\r]+'
        sentences = re.split(sentence_pattern, text)
        
        # 过滤空句子并清理
        sentences = [self.clean_text(s) for s in sentences if s.strip()]
        
        return sentences
    
    def chunk_text(self, text: str, source_name: str = "") -> List[TextChunk]:
        """将文本分块
        
        Args:
            text: 输入文本
            source_name: 文本来源名称
            
        Returns:
            文本块列表
        """
        if not text:
            return []
            
        # 清理文本
        cleaned_text = self.clean_text(text)
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(cleaned_text):
            # 计算结束位置
            end = min(start + self.chunk_size, len(cleaned_text))
            
            # 如果不是最后一块，尝试在句子边界处分割
            if end < len(cleaned_text):
                # 向后查找句子结束符
                for i in range(end, min(end + 100, len(cleaned_text))):
                    if cleaned_text[i] in '。！？.!?':
                        end = i + 1
                        break
            
            # 提取文本块
            chunk_content = cleaned_text[start:end].strip()
            
            if chunk_content:
                chunk = TextChunk(
                    content=chunk_content,
                    start_pos=start,
                    end_pos=end,
                    chunk_id=f"{source_name}_chunk_{chunk_id}",
                    metadata={
                        "source": source_name,
                        "chunk_index": chunk_id,
                        "length": len(chunk_content)
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # 移动到下一个位置（考虑重叠）
            start = max(start + self.chunk_size - self.chunk_overlap, end)
        
        logger.info(f"文本分块完成: {len(chunks)} 个块")
        return chunks
    
    def vectorize_chunks(self, chunks: List[TextChunk]) -> List[VectorEntity]:
        """向量化文本块
        
        Args:
            chunks: 文本块列表
            
        Returns:
            向量化实体列表
        """
        if not chunks:
            return []
            
        logger.info(f"开始向量化 {len(chunks)} 个文本块")
        
        # 批量向量化
        texts = [chunk.content for chunk in chunks]
        vectors = self.embedding_service.embed_batch(texts)
        
        vector_entities = []
        for i, chunk in enumerate(chunks):
            if i < len(vectors) and vectors[i]:
                # 生成描述信息
                content_preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
                description = f"文本块 {chunk.metadata.get('chunk_index', i)}，来源: {chunk.metadata.get('source', '未知')}，内容: {content_preview}"
                
                entity = VectorEntity(
                    name=chunk.chunk_id,
                    type="text_chunk",
                    properties={
                        "content": chunk.content,
                        "description": description,  # 添加描述字段
                        "source": chunk.metadata.get("source", ""),
                        "chunk_index": chunk.metadata.get("chunk_index", i),
                        "start_pos": chunk.start_pos,
                        "end_pos": chunk.end_pos,
                        "length": len(chunk.content)
                    },
                    vector=vectors[i],
                    source_text=chunk.content
                )
                vector_entities.append(entity)
            else:
                logger.warning(f"文本块 {chunk.chunk_id} 向量化失败")
        
        logger.info(f"向量化完成: {len(vector_entities)} 个向量实体")
        return vector_entities
    
    def process_document(self, file_path: str) -> List[VectorEntity]:
        """处理文档文件
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            向量化实体列表
        """
        try:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 获取文件名作为来源
            source_name = os.path.basename(file_path)
            
            logger.info(f"开始处理文档: {file_path}")
            logger.info(f"文档长度: {len(content)} 字符")
            
            # 分块
            chunks = self.chunk_text(content, source_name)
            
            # 向量化
            vector_entities = self.vectorize_chunks(chunks)
            
            return vector_entities
            
        except Exception as e:
            logger.error(f"处理文档失败: {e}")
            return []
    
    def store_to_weaviate(self, vector_entities: List[VectorEntity], 
                         vector_store: WeaviateVectorStore = None) -> bool:
        """存储向量实体到Weaviate
        
        Args:
            vector_entities: 向量实体列表
            vector_store: Weaviate存储实例
            
        Returns:
            是否成功
        """
        if not vector_entities:
            logger.warning("没有向量实体需要存储")
            return False
            
        # 使用提供的存储实例或创建新的
        store = vector_store or WeaviateVectorStore()
        
        try:
            # 确保集合已设置
            store.setup_collections()
            
            # 存储实体
            success = store.store_entities(vector_entities)
            
            if success:
                logger.info(f"成功存储 {len(vector_entities)} 个向量实体到Weaviate")
            else:
                logger.error("存储向量实体失败")
                
            return success
            
        except Exception as e:
            logger.error(f"存储到Weaviate失败: {e}")
            return False
    
    def process_and_store_document(self, file_path: str, 
                                  vector_store: WeaviateVectorStore = None) -> bool:
        """处理文档并存储到Weaviate（一站式服务）
        
        Args:
            file_path: 文档文件路径
            vector_store: Weaviate存储实例
            
        Returns:
            是否成功
        """
        logger.info(f"开始处理并存储文档: {file_path}")
        
        # 处理文档
        vector_entities = self.process_document(file_path)
        
        if not vector_entities:
            logger.error("文档处理失败，没有生成向量实体")
            return False
        
        # 存储到Weaviate
        success = self.store_to_weaviate(vector_entities, vector_store)
        
        if success:
            logger.info(f"文档处理和存储完成: {file_path}")
        else:
            logger.error(f"文档存储失败: {file_path}")
            
        return success
    
    # 问题分析
    # 1. VectorEntity的properties中没有description字段
    # 2. 检索结果中description显示为空
    # 3. 相似度分数显示为N/A，可能是因为字段映射问题