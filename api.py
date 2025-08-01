#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAG 整合API - 使用LangChain Runnable整合所有功能
包含：Weaviate向量检索、Neo4j图谱检索、LangChain管道
"""

import sys
import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# LangChain imports
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 项目模块导入
from config import get_config
from src.services import (
    RetrievalService, GraphRetrievalService, HybridRetrievalService,
    RAGPipeline, OllamaLLMService, OllamaEmbeddingService
)
from src.knowledge.vector_storage import WeaviateVectorStore
from src.knowledge.neo4j_vector_storage import Neo4jVectorStore

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RetrievalStep:
    """检索步骤"""
    step_name: str
    step_description: str
    start_time: float
    end_time: float
    duration: float
    status: str  # "success", "error", "warning"
    result_count: int
    details: Dict[str, Any]

@dataclass
class IntegratedResponse:
    """整合响应结果"""
    answer: str
    sources: Dict[str, Any]
    metadata: Dict[str, Any]
    retrieval_process: List[RetrievalStep]

class HAGIntegratedAPI:
    """HAG整合API - 使用LangChain Runnable模式"""
    
    def __init__(self):
        """初始化整合API"""
        try:
            logger.info("开始初始化HAG整合API...")
            
            # 加载配置
            self.config = get_config()
            
            # 初始化基础服务
            self._init_services()
            
            # 构建LangChain Runnable管道
            self._build_runnable_chain()
            
            logger.info("HAG整合API初始化完成")
            
        except Exception as e:
            logger.error(f"HAG整合API初始化失败: {e}")
            raise
    
    def _init_services(self):
        """初始化所有服务组件"""
        try:
            # 1. 初始化向量化服务
            self.embedding_service = OllamaEmbeddingService()
            
            # 2. 初始化向量存储服务
            self.vector_store = WeaviateVectorStore()
            
            # 3. 初始化Weaviate检索服务
            self.retrieval_service = RetrievalService(
                embedding_service=self.embedding_service,
                vector_store=self.vector_store
            )
            
            # 4. 初始化Neo4j图谱检索服务
            self.graph_service = GraphRetrievalService(self.config.neo4j)
            
            # 5. 初始化混合检索服务
            self.hybrid_service = HybridRetrievalService(
                document_retrieval_service=self.retrieval_service,
                graph_retrieval_service=self.graph_service,
                doc_weight=0.6,
                graph_weight=0.4
            )
            
            # 6. 初始化LLM服务
            self.llm_service = OllamaLLMService()
            
            # 7. 初始化RAG管道
            self.rag_pipeline = RAGPipeline(
                hybrid_retrieval_service=self.hybrid_service,
                llm_service=self.llm_service
            )
            
            logger.info("所有服务组件初始化完成")
            
        except Exception as e:
            logger.error(f"服务初始化失败: {e}")
            raise
    
    def _build_runnable_chain(self):
        """构建LangChain Runnable管道"""
        try:
            # 定义提示模板
            prompt_template = ChatPromptTemplate.from_template("""
你是一个智能助手，请基于提供的知识库信息回答用户问题。

相关文档:
{documents}

相关实体:
{entities}

相关关系:
{relationships}

用户问题: {question}

请基于以上信息提供准确、详细的回答:
""")
            
            # 构建Runnable链
            self.runnable_chain = (
                {
                    "question": RunnablePassthrough(),
                    "documents": RunnableLambda(self._retrieve_documents),
                    "entities": RunnableLambda(self._retrieve_entities),
                    "relationships": RunnableLambda(self._retrieve_relationships)
                }
                | prompt_template
                | RunnableLambda(self._generate_answer)
                | StrOutputParser()
            )
            
            logger.info("LangChain Runnable管道构建完成")
            
        except Exception as e:
            logger.error(f"Runnable管道构建失败: {e}")
            raise
    
    def _retrieve_documents(self, question: str) -> str:
        """检索相关文档 - Weaviate向量检索（余弦相似度+欧式距离）"""
        try:
            # 使用混合检索获取Top5文档
            hybrid_result = self.retrieval_service.search_hybrid(question, limit=5)
            
            documents = []
            for result in hybrid_result.hybrid_results[:5]:  # 确保只取Top5
                documents.append(f"- {result.content[:200]}...")
            
            return "\n".join(documents) if documents else "未找到相关文档"
            
        except Exception as e:
            logger.error(f"文档检索失败: {e}")
            return "文档检索出错"
    
    def _retrieve_entities(self, question: str) -> str:
        """检索相关实体 - Neo4j图谱检索"""
        try:
            # 将问题转换为向量，然后检索相关实体
            query_vector = self.embedding_service.embed_text(question)
            
            # 使用图谱服务检索实体
            entities = self.graph_service.search_entities_by_name(question, limit=2)
            
            entity_list = []
            for entity in entities[:2]:  # 确保只取2个节点
                name = entity.get('name', '')
                entity_type = entity.get('type', '')
                description = entity.get('description', '')
                entity_list.append(f"- {name} ({entity_type}): {description}")
            
            return "\n".join(entity_list) if entity_list else "未找到相关实体"
            
        except Exception as e:
            logger.error(f"实体检索失败: {e}")
            return "实体检索出错"
    
    def _retrieve_relationships(self, question: str) -> str:
        """检索相关关系 - Neo4j图谱检索（正确逻辑：先找节点，再找关系）"""
        try:
            # 第一步：根据问题找到相关的实体节点
            relevant_entities = self.graph_service.search_entities_by_name(question, limit=3)
            
            if not relevant_entities:
                return "未找到相关实体，无法检索关系"
            
            # 第二步：基于找到的实体节点，查找它们的关系
            all_relationships = []
            
            for entity in relevant_entities:
                entity_name = entity.get('name', '')
                if entity_name:
                    # 获取该实体的关系网络
                    entity_rels = self.graph_service.get_entity_relationships(entity_name, limit=5)
                    relationships = entity_rels.get('relationships', [])
                    
                    # 处理关系数据
                    for rel in relationships:
                        source = rel.get('entity', entity_name)
                        target = rel.get('related_entity', '')
                        rel_type = rel.get('relation_type', '')
                        description = rel.get('relation_description', '') or rel.get('related_description', '')
                        
                        # 确保关系描述不为空
                        if not description:
                            description = f"{source} 与 {target} 之间的 {rel_type} 关系"
                        
                        all_relationships.append({
                            'source': source,
                            'target': target,
                            'type': rel_type,
                            'description': description
                        })
            
            # 第三步：去重并格式化输出
            seen_relations = set()
            unique_relationships = []
            
            for rel in all_relationships:
                rel_key = (rel['source'], rel['type'], rel['target'])
                if rel_key not in seen_relations:
                    seen_relations.add(rel_key)
                    unique_relationships.append(rel)
            
            # 限制输出数量并格式化
            rel_list = []
            for rel in unique_relationships[:10]:  # 最多10个关系
                source = rel['source']
                target = rel['target']
                rel_type = rel['type']
                description = rel['description'][:100] + "..." if len(rel['description']) > 100 else rel['description']
                rel_list.append(f"- {source} --[{rel_type}]--> {target}: {description}")
            
            return "\n".join(rel_list) if rel_list else "未找到相关关系"
            
        except Exception as e:
            logger.error(f"关系检索失败: {e}")
            return "关系检索出错"
    
    def _generate_answer(self, prompt_data: Dict[str, Any]) -> str:
        """使用LLM生成答案"""
        try:
            # 构建完整的提示词
            prompt_text = prompt_data.content if hasattr(prompt_data, 'content') else str(prompt_data)
            
            # 使用LLM生成回答
            answer = self.llm_service.generate_response(
                prompt=prompt_text,
                temperature=0.7,
                max_tokens=1000
            )
            
            return answer
            
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return "抱歉，无法生成答案"
    
    def query(self, question: str) -> IntegratedResponse:
        """
        主要查询接口 - 整合所有功能，记录详细检索过程
        
        Args:
            question: 用户问题
            
        Returns:
            IntegratedResponse: 整合响应结果
        """
        import time
        
        retrieval_steps = []
        
        try:
            logger.info(f"开始处理查询: {question}")
            
            # 步骤1: 问题分析和向量化
            step1_start = time.time()
            try:
                query_vector = self.embedding_service.embed_text(question)
                step1_end = time.time()
                retrieval_steps.append(RetrievalStep(
                    step_name="问题向量化",
                    step_description="将用户问题转换为向量表示",
                    start_time=step1_start,
                    end_time=step1_end,
                    duration=step1_end - step1_start,
                    status="success",
                    result_count=1,
                    details={"vector_dimension": len(query_vector) if query_vector else 0}
                ))
            except Exception as e:
                step1_end = time.time()
                retrieval_steps.append(RetrievalStep(
                    step_name="问题向量化",
                    step_description="将用户问题转换为向量表示",
                    start_time=step1_start,
                    end_time=step1_end,
                    duration=step1_end - step1_start,
                    status="error",
                    result_count=0,
                    details={"error": str(e)}
                ))
            
            # 步骤2: 文档检索 (Weaviate)
            step2_start = time.time()
            try:
                hybrid_result = self.retrieval_service.search_hybrid(question, limit=5)
                documents = hybrid_result.hybrid_results[:5]
                step2_end = time.time()
                retrieval_steps.append(RetrievalStep(
                    step_name="文档检索",
                    step_description="从Weaviate向量数据库检索相关文档",
                    start_time=step2_start,
                    end_time=step2_end,
                    duration=step2_end - step2_start,
                    status="success",
                    result_count=len(documents),
                    details={
                        "search_method": "hybrid_cosine_euclidean",
                        "top_scores": [doc.score for doc in documents[:3]] if documents else []
                    }
                ))
            except Exception as e:
                step2_end = time.time()
                documents = []
                retrieval_steps.append(RetrievalStep(
                    step_name="文档检索",
                    step_description="从Weaviate向量数据库检索相关文档",
                    start_time=step2_start,
                    end_time=step2_end,
                    duration=step2_end - step2_start,
                    status="error",
                    result_count=0,
                    details={"error": str(e)}
                ))
            
            # 步骤3: 实体检索 (Neo4j)
            step3_start = time.time()
            try:
                entities = self.graph_service.search_entities_by_name(question, limit=2)
                step3_end = time.time()
                retrieval_steps.append(RetrievalStep(
                    step_name="实体检索",
                    step_description="从Neo4j图数据库检索相关实体",
                    start_time=step3_start,
                    end_time=step3_end,
                    duration=step3_end - step3_start,
                    status="success",
                    result_count=len(entities),
                    details={
                        "entity_types": [entity.get('type', '') for entity in entities] if entities else []
                    }
                ))
            except Exception as e:
                step3_end = time.time()
                entities = []
                retrieval_steps.append(RetrievalStep(
                    step_name="实体检索",
                    step_description="从Neo4j图数据库检索相关实体",
                    start_time=step3_start,
                    end_time=step3_end,
                    duration=step3_end - step3_start,
                    status="error",
                    result_count=0,
                    details={"error": str(e)}
                ))
            
            # 步骤4: 关系检索 (Neo4j)
            step4_start = time.time()
            try:
                all_relationships = []
                for entity in entities:
                    entity_name = entity.get('name', '')
                    if entity_name:
                        entity_rels = self.graph_service.get_entity_relationships(entity_name, limit=5)
                        relationships = entity_rels.get('relationships', [])
                        all_relationships.extend(relationships)
                
                # 去重
                seen_relations = set()
                unique_relationships = []
                for rel in all_relationships:
                    rel_key = (rel.get('entity', ''), rel.get('relation_type', ''), rel.get('related_entity', ''))
                    if rel_key not in seen_relations:
                        seen_relations.add(rel_key)
                        unique_relationships.append(rel)
                
                step4_end = time.time()
                retrieval_steps.append(RetrievalStep(
                    step_name="关系检索",
                    step_description="从Neo4j图数据库检索实体间关系",
                    start_time=step4_start,
                    end_time=step4_end,
                    duration=step4_end - step4_start,
                    status="success",
                    result_count=len(unique_relationships),
                    details={
                        "relation_types": list(set([rel.get('relation_type', '') for rel in unique_relationships])) if unique_relationships else []
                    }
                ))
            except Exception as e:
                step4_end = time.time()
                unique_relationships = []
                retrieval_steps.append(RetrievalStep(
                    step_name="关系检索",
                    step_description="从Neo4j图数据库检索实体间关系",
                    start_time=step4_start,
                    end_time=step4_end,
                    duration=step4_end - step4_start,
                    status="error",
                    result_count=0,
                    details={"error": str(e)}
                ))
            
            # 步骤5: 混合检索结果整合
            step5_start = time.time()
            try:
                hybrid_result = self.hybrid_service.search_hybrid(question, doc_top_k=5, graph_top_k=4)
                step5_end = time.time()
                retrieval_steps.append(RetrievalStep(
                    step_name="混合检索整合",
                    step_description="整合文档和图谱检索结果",
                    start_time=step5_start,
                    end_time=step5_end,
                    duration=step5_end - step5_start,
                    status="success",
                    result_count=len(hybrid_result.documents) + len(hybrid_result.entities) + len(hybrid_result.relationships),
                    details={
                        "doc_weight": 0.6,
                        "graph_weight": 0.4,
                        "total_documents": len(hybrid_result.documents),
                        "total_entities": len(hybrid_result.entities),
                        "total_relationships": len(hybrid_result.relationships)
                    }
                ))
            except Exception as e:
                step5_end = time.time()
                hybrid_result = None
                retrieval_steps.append(RetrievalStep(
                    step_name="混合检索整合",
                    step_description="整合文档和图谱检索结果",
                    start_time=step5_start,
                    end_time=step5_end,
                    duration=step5_end - step5_start,
                    status="error",
                    result_count=0,
                    details={"error": str(e)}
                ))
            
            # 步骤6: LLM答案生成
            step6_start = time.time()
            try:
                answer = self.runnable_chain.invoke(question)
                step6_end = time.time()
                retrieval_steps.append(RetrievalStep(
                    step_name="答案生成",
                    step_description="使用LLM基于检索结果生成答案",
                    start_time=step6_start,
                    end_time=step6_end,
                    duration=step6_end - step6_start,
                    status="success",
                    result_count=1,
                    details={
                        "model": "gemma3:4b",
                        "answer_length": len(answer),
                        "temperature": 0.7
                    }
                ))
            except Exception as e:
                step6_end = time.time()
                answer = f"抱歉，生成答案时出现错误: {str(e)}"
                retrieval_steps.append(RetrievalStep(
                    step_name="答案生成",
                    step_description="使用LLM基于检索结果生成答案",
                    start_time=step6_start,
                    end_time=step6_end,
                    duration=step6_end - step6_start,
                    status="error",
                    result_count=0,
                    details={"error": str(e)}
                ))
            
            # 构建响应
            sources = {"documents": [], "entities": [], "relationships": []}
            if hybrid_result:
                sources = {
                    "documents": [
                        {
                            "content": doc["content"][:200] + "...",
                            "score": doc["score"],
                            "metadata": doc.get("metadata", {})
                        }
                        for doc in hybrid_result.documents[:5]
                    ],
                    "entities": [
                        {
                            "name": entity.get("name", ""),
                            "type": entity.get("type", ""),
                            "properties": entity.get("properties", {})
                        }
                        for entity in hybrid_result.entities[:2]
                    ],
                    "relationships": [
                        {
                            "source": rel.get("source", ""),
                            "target": rel.get("target", ""),
                            "type": rel.get("type", ""),
                            "description": rel.get("description", "")[:100] + "..."
                        }
                        for rel in hybrid_result.relationships[:10]
                    ]
                }
            
            response = IntegratedResponse(
                answer=answer,
                sources=sources,
                metadata={
                    "question": question,
                    "retrieval_metadata": hybrid_result.metadata if hybrid_result else {},
                    "processing_method": "langchain_runnable_with_detailed_steps",
                    "total_processing_time": sum([step.duration for step in retrieval_steps])
                },
                retrieval_process=retrieval_steps
            )
            
            logger.info(f"查询处理完成: 答案长度={len(answer)}, 检索步骤={len(retrieval_steps)}")
            return response
            
        except Exception as e:
            logger.error(f"查询处理失败: {e}")
            return IntegratedResponse(
                answer=f"抱歉，处理查询时出现错误: {str(e)}",
                sources={"documents": [], "entities": [], "relationships": []},
                metadata={"error": str(e)},
                retrieval_process=retrieval_steps
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            return {
                "status": "active",
                "services": {
                    "weaviate": "connected",
                    "neo4j": "connected", 
                    "ollama": "connected"
                },
                "retrieval_stats": self.retrieval_service.get_stats(),
                "graph_stats": self.graph_service.get_stats()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

# 创建全局API实例
api = None

def get_api() -> HAGIntegratedAPI:
    """获取API实例（单例模式）"""
    global api
    if api is None:
        api = HAGIntegratedAPI()
    return api

# 简化的接口函数
def query_knowledge(question: str) -> IntegratedResponse:
    """
    简化的知识查询接口
    
    Args:
        question: 用户问题
        
    Returns:
        IntegratedResponse: 整合响应结果
    """
    return get_api().query(question)

if __name__ == "__main__":
    # 测试API
    try:
        print("初始化HAG整合API...")
        hag_api = HAGIntegratedAPI()
        
        print("\n系统状态:")
        status = hag_api.get_system_status()
        print(status)
        
        print("\n测试查询...")
        test_question = "什么是人工智能？"
        result = hag_api.query(test_question)
        
        print(f"\n问题: {test_question}")
        print(f"回答: {result.answer}")
        print(f"来源数量: 文档{len(result.sources['documents'])}个, 实体{len(result.sources['entities'])}个, 关系{len(result.sources['relationships'])}个")
        
    except Exception as e:
        print(f"测试失败: {e}")