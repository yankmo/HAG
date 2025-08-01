"""
RAG管道 - 使用LangChain构建检索增强生成管道
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """RAG响应结果"""
    answer: str
    sources: List[Dict[str, Any]]
    context: Dict[str, Any]
    metadata: Dict[str, Any]

class RAGPipeline:
    """RAG管道 - 结合混合检索和大模型生成"""
    
    def __init__(self, 
                 hybrid_retrieval_service,
                 llm_service,
                 max_context_length: int = 4000,
                 temperature: float = 0.7):
        """
        初始化RAG管道
        
        Args:
            hybrid_retrieval_service: 混合检索服务
            llm_service: 大语言模型服务
            max_context_length: 最大上下文长度
            temperature: 生成温度
        """
        self.hybrid_service = hybrid_retrieval_service
        self.llm_service = llm_service
        self.max_context_length = max_context_length
        self.temperature = temperature
        
        logger.info(f"RAG管道初始化完成 - 最大上下文长度: {max_context_length}, 温度: {temperature}")
    
    def generate_answer(self, 
                       question: str, 
                       top_k: int = 10,
                       include_sources: bool = True,
                       context_entities: List[str] = None,
                       context_relations: List[str] = None) -> RAGResponse:
        """
        生成RAG答案
        
        Args:
            question: 用户问题
            top_k: 检索结果数量
            include_sources: 是否包含来源信息
            context_entities: 上下文实体
            context_relations: 上下文关系
            
        Returns:
            RAGResponse: RAG响应结果
        """
        try:
            logger.info(f"开始RAG生成: question='{question}', top_k={top_k}")
            
            # 1. 混合检索
            if context_entities or context_relations:
                retrieval_result = self.hybrid_service.search_with_context(
                    question, context_entities, context_relations, top_k
                )
            else:
                retrieval_result = self.hybrid_service.search_hybrid(question, top_k)
            
            # 2. 构建上下文
            context = self._build_context(retrieval_result)
            
            # 3. 构建提示词
            prompt = self._build_prompt(question, context)
            
            # 4. 生成答案
            answer = self._generate_with_llm(prompt)
            
            # 5. 构建来源信息
            sources = []
            if include_sources:
                sources = self._build_sources(retrieval_result)
            
            # 6. 构建响应
            response = RAGResponse(
                answer=answer,
                sources=sources,
                context=context,
                metadata={
                    'question': question,
                    'retrieval_metadata': retrieval_result.metadata,
                    'context_length': len(prompt),
                    'temperature': self.temperature
                }
            )
            
            logger.info(f"RAG生成完成: 答案长度={len(answer)}, 来源数={len(sources)}")
            return response
            
        except Exception as e:
            logger.error(f"RAG生成失败: {e}")
            return RAGResponse(
                answer=f"抱歉，生成答案时出现错误: {str(e)}",
                sources=[],
                context={},
                metadata={'error': str(e)}
            )
    
    def _build_context(self, retrieval_result) -> Dict[str, Any]:
        """构建上下文信息"""
        try:
            context = {
                'documents': [],
                'entities': [],
                'relationships': [],
                'intent': retrieval_result.metadata.get('intent')
            }
            
            # 处理文档
            for doc in retrieval_result.documents:
                context['documents'].append({
                    'content': doc['content'],
                    'score': doc['score'],
                    'metadata': doc.get('metadata', {})
                })
            
            # 处理实体
            for entity in retrieval_result.entities:
                context['entities'].append({
                    'name': entity.get('name', ''),
                    'type': entity.get('type', ''),
                    'properties': entity.get('properties', {})
                })
            
            # 处理关系
            for rel in retrieval_result.relationships:
                context['relationships'].append({
                    'type': rel.get('type', ''),
                    'source': rel.get('source', ''),
                    'target': rel.get('target', ''),
                    'properties': rel.get('properties', {})
                })
            
            return context
            
        except Exception as e:
            logger.error(f"构建上下文失败: {e}")
            return {'error': str(e)}
    
    def _build_prompt(self, question: str, context: Dict[str, Any]) -> str:
        """构建提示词"""
        try:
            prompt_parts = []
            
            # 系统提示
            prompt_parts.append("你是一个智能助手，请基于提供的上下文信息回答用户问题。")
            prompt_parts.append("请确保答案准确、相关且有帮助。如果上下文信息不足以回答问题，请诚实说明。")
            prompt_parts.append("")
            
            # 文档上下文
            if context.get('documents'):
                prompt_parts.append("相关文档:")
                for i, doc in enumerate(context['documents'][:5], 1):  # 限制文档数量
                    content = doc['content'][:500]  # 限制每个文档长度
                    prompt_parts.append(f"{i}. {content}")
                prompt_parts.append("")
            
            # 实体上下文
            if context.get('entities'):
                prompt_parts.append("相关实体:")
                for entity in context['entities'][:10]:  # 限制实体数量
                    name = entity.get('name', '')
                    entity_type = entity.get('type', '')
                    prompt_parts.append(f"- {name} (类型: {entity_type})")
                prompt_parts.append("")
            
            # 关系上下文
            if context.get('relationships'):
                prompt_parts.append("相关关系:")
                for rel in context['relationships'][:10]:  # 限制关系数量
                    rel_type = rel.get('type', '')
                    source = rel.get('source', '')
                    target = rel.get('target', '')
                    prompt_parts.append(f"- {source} --[{rel_type}]--> {target}")
                prompt_parts.append("")
            
            # 意图信息
            if context.get('intent'):
                intent_info = context['intent']
                prompt_parts.append(f"识别的用户意图: {intent_info.get('intent', '')} (置信度: {intent_info.get('confidence', 0):.2f})")
                prompt_parts.append("")
            
            # 用户问题
            prompt_parts.append(f"用户问题: {question}")
            prompt_parts.append("")
            prompt_parts.append("请基于以上上下文信息回答用户问题:")
            
            # 组合提示词并检查长度
            full_prompt = "\n".join(prompt_parts)
            
            # 如果提示词过长，进行截断
            if len(full_prompt) > self.max_context_length:
                # 优先保留问题和系统提示，然后按重要性截断上下文
                essential_parts = prompt_parts[:3] + prompt_parts[-3:]  # 系统提示 + 问题
                context_parts = prompt_parts[3:-3]
                
                essential_text = "\n".join(essential_parts)
                remaining_length = self.max_context_length - len(essential_text)
                
                # 截断上下文部分
                context_text = "\n".join(context_parts)
                if len(context_text) > remaining_length:
                    context_text = context_text[:remaining_length-100] + "\n...(上下文已截断)"
                
                full_prompt = essential_text.replace(prompt_parts[-3], context_text + "\n" + prompt_parts[-3])
            
            return full_prompt
            
        except Exception as e:
            logger.error(f"构建提示词失败: {e}")
            return f"用户问题: {question}\n\n请回答这个问题。"
    
    def _generate_with_llm(self, prompt: str) -> str:
        """使用大模型生成答案"""
        try:
            # 这里需要根据实际的LLM服务接口进行调用
            # 假设llm_service有一个generate方法
            if hasattr(self.llm_service, 'generate'):
                response = self.llm_service.generate(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=1000
                )
                return response
            elif hasattr(self.llm_service, 'chat'):
                response = self.llm_service.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature
                )
                return response.get('content', response.get('message', str(response)))
            else:
                # 如果没有合适的方法，返回默认响应
                return "抱歉，当前无法生成回答。请检查LLM服务配置。"
                
        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            return f"抱歉，生成答案时出现错误: {str(e)}"
    
    def _build_sources(self, retrieval_result) -> List[Dict[str, Any]]:
        """构建来源信息"""
        try:
            sources = []
            
            # 文档来源
            for doc in retrieval_result.documents:
                sources.append({
                    'type': 'document',
                    'content': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                    'score': doc['score'],
                    'metadata': doc.get('metadata', {}),
                    'source': 'weaviate'
                })
            
            # 实体来源
            for entity in retrieval_result.entities:
                sources.append({
                    'type': 'entity',
                    'name': entity.get('name', ''),
                    'entity_type': entity.get('type', ''),
                    'properties': entity.get('properties', {}),
                    'source': 'neo4j'
                })
            
            # 关系来源
            for rel in retrieval_result.relationships:
                sources.append({
                    'type': 'relationship',
                    'relation_type': rel.get('type', ''),
                    'source_entity': rel.get('source', ''),
                    'target_entity': rel.get('target', ''),
                    'properties': rel.get('properties', {}),
                    'source': 'neo4j'
                })
            
            return sources
            
        except Exception as e:
            logger.error(f"构建来源信息失败: {e}")
            return []
    
    def chat_with_history(self, 
                         question: str,
                         chat_history: List[Dict[str, str]] = None,
                         top_k: int = 10) -> RAGResponse:
        """
        带历史记录的对话
        
        Args:
            question: 当前问题
            chat_history: 对话历史 [{"role": "user/assistant", "content": "..."}]
            top_k: 检索结果数量
            
        Returns:
            RAGResponse: RAG响应结果
        """
        try:
            # 从历史记录中提取上下文实体和关系
            context_entities = []
            context_relations = []
            
            if chat_history:
                # 简单的实体和关系提取（实际实现可以更复杂）
                for turn in chat_history[-3:]:  # 只考虑最近3轮对话
                    content = turn.get('content', '')
                    # 这里可以添加更复杂的实体识别逻辑
                    # 暂时使用简单的关键词提取
                    pass
            
            # 执行RAG生成
            return self.generate_answer(
                question=question,
                top_k=top_k,
                context_entities=context_entities,
                context_relations=context_relations
            )
            
        except Exception as e:
            logger.error(f"历史对话RAG失败: {e}")
            return RAGResponse(
                answer=f"抱歉，处理对话历史时出现错误: {str(e)}",
                sources=[],
                context={},
                metadata={'error': str(e)}
            )
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """获取管道统计信息"""
        try:
            hybrid_stats = self.hybrid_service.get_stats()
            
            return {
                'pipeline_type': 'rag_pipeline',
                'max_context_length': self.max_context_length,
                'temperature': self.temperature,
                'hybrid_service_stats': hybrid_stats,
                'llm_service_type': type(self.llm_service).__name__
            }
            
        except Exception as e:
            logger.error(f"获取管道统计信息失败: {e}")
            return {'error': str(e)}