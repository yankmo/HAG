#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
帕金森病文本数据处理和Neo4j存储脚本
将帕金森病相关文本数据提取实体和关系，并存储到Neo4j向量数据库
"""

import sys
import os
import logging
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.knowledge.neo4j_vector_storage import Neo4jVectorStore, Neo4jVectorEntity, Neo4jVectorRelation
from src.knowledge.intent_recognition_neo4j import IntentRecognizer, OllamaClient
from src.services.embedding_service import OllamaEmbeddingService
from config import get_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pajinsen_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PajinsenDataProcessor:
    """帕金森病数据处理器"""
    
    def __init__(self):
        """初始化处理器"""
        try:
            # 初始化各个组件
            self.vector_storage = Neo4jVectorStore()
            self.ollama_client = OllamaClient()
            self.intent_recognizer = IntentRecognizer(self.ollama_client)
            self.embedding_service = OllamaEmbeddingService()
            
            logger.info("帕金森病数据处理器初始化完成")
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise
    
    def clear_database(self):
        """清空数据库"""
        try:
            logger.info("正在清空Neo4j数据库...")
            self.vector_storage.clear_database()
            logger.info("数据库清空完成")
        except Exception as e:
            logger.error(f"清空数据库失败: {e}")
            raise
    
    def split_text(self, text: str, chunk_size: int = 800) -> list:
        """智能分割文本"""
        # 清理文本
        text = text.strip()
        
        # 按段落和章节分割
        sections = []
        current_section = ""
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检查是否是章节标题（以==开头和结尾）
            if line.startswith('==') and line.endswith('=='):
                if current_section:
                    sections.append(current_section.strip())
                current_section = line + "\n"
            else:
                current_section += line + "\n"
        
        if current_section:
            sections.append(current_section.strip())
        
        # 进一步分割过长的章节
        chunks = []
        for section in sections:
            if len(section) <= chunk_size:
                chunks.append(section)
            else:
                # 按段落分割长章节
                paragraphs = section.split('\n\n')
                current_chunk = ""
                
                for paragraph in paragraphs:
                    if len(current_chunk) + len(paragraph) <= chunk_size:
                        if current_chunk:
                            current_chunk += "\n\n" + paragraph
                        else:
                            current_chunk = paragraph
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = paragraph
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        # 过滤空块
        chunks = [chunk for chunk in chunks if chunk.strip()]
        return chunks
    
    def process_text_file(self, file_path: str):
        """处理文本文件"""
        try:
            logger.info(f"开始处理文件: {file_path}")
            
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"文件大小: {len(content)} 字符")
            
            # 清空数据库
            self.clear_database()
            
            # 分割文本
            chunks = self.split_text(content)
            logger.info(f"文本分为 {len(chunks)} 个块")
            
            # 统计变量
            total_entities = 0
            total_relations = 0
            all_vector_entities = []
            all_vector_relations = []
            
            # 处理每个文本块
            for i, chunk in enumerate(chunks):
                logger.info(f"处理第 {i+1}/{len(chunks)} 块 (长度: {len(chunk)} 字符)")
                
                try:
                    # 提取实体和关系
                    entities, relations = self.intent_recognizer.extract_entities_and_relations(chunk)
                    
                    if entities:
                        logger.info(f"块 {i+1}: 提取到 {len(entities)} 个实体")
                        
                        # 转换为向量实体
                        for entity in entities:
                            # 生成实体描述文本
                            entity_text = f"实体: {entity.name}, 类型: {entity.type}"
                            if entity.properties and entity.properties.get("description"):
                                entity_text += f", 描述: {entity.properties['description']}"
                            
                            # 向量化
                            vector = self.embedding_service.embed_text(entity_text)
                            if vector:
                                vector_entity = Neo4jVectorEntity(
                                    name=entity.name,
                                    type=entity.type,
                                    properties=entity.properties or {},
                                    vector=vector,
                                    source_text=chunk[:200] + "..." if len(chunk) > 200 else chunk
                                )
                                all_vector_entities.append(vector_entity)
                                total_entities += 1
                    
                    if relations:
                        logger.info(f"块 {i+1}: 提取到 {len(relations)} 个关系")
                        
                        # 转换为向量关系
                        for relation in relations:
                            # 生成关系描述文本
                            relation_text = f"关系: {relation.source} {relation.relation_type} {relation.target}"
                            if relation.properties and relation.properties.get("description"):
                                relation_text += f", 描述: {relation.properties['description']}"
                            
                            # 向量化
                            vector = self.embedding_service.embed_text(relation_text)
                            if vector:
                                vector_relation = Neo4jVectorRelation(
                                    source=relation.source,
                                    target=relation.target,
                                    relation_type=relation.relation_type,
                                    description=relation.properties.get('description', '') if relation.properties else '',
                                    vector=vector,
                                    source_text=chunk[:200] + "..." if len(chunk) > 200 else chunk
                                )
                                all_vector_relations.append(vector_relation)
                                total_relations += 1
                    
                    if not entities and not relations:
                        logger.warning(f"块 {i+1}: 未提取到任何实体或关系")
                        
                except Exception as e:
                    logger.error(f"处理块 {i+1} 时出错: {e}")
                    import traceback
                    logger.error(f"错误详情: {traceback.format_exc()}")
            
            # 批量存储到Neo4j
            logger.info(f"开始存储数据到Neo4j...")
            logger.info(f"准备存储 {len(all_vector_entities)} 个实体向量")
            logger.info(f"准备存储 {len(all_vector_relations)} 个关系向量")
            
            # 存储实体
            if all_vector_entities:
                success = self.vector_storage.store_entities(all_vector_entities)
                if success:
                    logger.info(f"成功存储 {len(all_vector_entities)} 个实体向量")
                else:
                    logger.error("实体向量存储失败")
            
            # 存储关系
            if all_vector_relations:
                success = self.vector_storage.store_relations(all_vector_relations)
                if success:
                    logger.info(f"成功存储 {len(all_vector_relations)} 个关系向量")
                else:
                    logger.error("关系向量存储失败")
            
            # 获取最终统计
            stats = self.vector_storage.get_statistics()
            
            logger.info("=" * 60)
            logger.info("帕金森病数据处理完成！")
            logger.info(f"处理的文本块数: {len(chunks)}")
            logger.info(f"提取的实体总数: {total_entities}")
            logger.info(f"提取的关系总数: {total_relations}")
            logger.info("=" * 60)
            logger.info("Neo4j存储统计:")
            logger.info(f"  Neo4j节点数: {stats['neo4j_nodes']}")
            logger.info(f"  Neo4j关系数: {stats['neo4j_relationships']}")
            logger.info(f"  向量实体数: {stats['vector_entities']}")
            logger.info(f"  向量关系数: {stats['vector_relations']}")
            logger.info("=" * 60)
            
            return {
                "chunks_processed": len(chunks),
                "entities_extracted": total_entities,
                "relations_extracted": total_relations,
                "neo4j_stats": stats
            }
            
        except Exception as e:
            logger.error(f"处理文件失败: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            raise
    
    def test_retrieval(self, query: str = "帕金森病的症状"):
        """测试检索功能"""
        try:
            logger.info(f"测试检索功能，查询: {query}")
            
            # 实体检索
            entity_results = self.vector_storage.search_entities(query, limit=5)
            logger.info(f"实体检索结果: {len(entity_results)} 个")
            for i, result in enumerate(entity_results):
                logger.info(f"  {i+1}. {result['name']} ({result['type']}) - 相似度: {result['similarity']:.3f}")
            
            # 关系检索
            relation_results = self.vector_storage.search_relations(query, limit=5)
            logger.info(f"关系检索结果: {len(relation_results)} 个")
            for i, result in enumerate(relation_results):
                logger.info(f"  {i+1}. {result['source']} -> {result['target']} ({result['relation_type']}) - 相似度: {result['similarity']:.3f}")
            
            # 混合检索
            hybrid_results = self.vector_storage.search_entities_hybrid(query, limit=3)
            logger.info(f"混合检索结果: {len(hybrid_results)} 个")
            for i, result in enumerate(hybrid_results):
                logger.info(f"  {i+1}. {result['name']} ({result['type']}) - 混合分数: {result['score']:.3f}")
            
        except Exception as e:
            logger.error(f"测试检索失败: {e}")

def main():
    """主函数"""
    try:
        # 创建处理器
        processor = PajinsenDataProcessor()
        
        # 处理帕金森病文本文件
        file_path = "e:/Program/Project/HAG/data/pajinsen.txt"
        
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return
        
        # 处理文件
        results = processor.process_text_file(file_path)
        
        # 测试检索功能
        logger.info("\n" + "=" * 60)
        logger.info("测试检索功能")
        logger.info("=" * 60)
        
        test_queries = [
            "帕金森病的症状",
            "帕金森病的治疗方法",
            "帕金森病的病因",
            "多巴胺",
            "震颤"
        ]
        
        for query in test_queries:
            logger.info(f"\n查询: {query}")
            processor.test_retrieval(query)
        
        print("\n✅ 帕金森病数据处理和存储完成！")
        print("📊 详细日志请查看 pajinsen_processing.log 文件")
        print("🔗 可以在Neo4j Browser中查看存储的知识图谱: http://localhost:7474")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        print(f"❌ 执行失败: {e}")

if __name__ == "__main__":
    main()