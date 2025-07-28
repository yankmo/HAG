#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于LangChain的RAG主程序
整合现有的检索功能，实现完整的问答流程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain.schema import BaseRetriever, Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from typing import List, Dict, Any, Optional
import json
import time
from datetime import datetime

# 导入现有的RAG系统
from src.knowledge.modular_rag_system import ModularRAGSystem
import ollama
from ollama import Message

# 导入配置管理器
from config import get_config

class OllamaLLM(LLM):
    """自定义Ollama LLM包装器"""
    
    def __init__(self, model_name: str = None, base_url: str = None):
        super().__init__()
        config = get_config()
        self.model_name = model_name or config.ollama.default_model
        self.base_url = base_url or config.ollama.base_url
    
    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """调用Ollama模型生成回答"""
        try:
            import requests
            import json
            
            # 检查Ollama服务状态
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return "Ollama服务未运行，请启动Ollama服务后重试。"
            
            # 使用HTTP API直接调用
            url = f"{self.base_url}/api/generate"
            data = {
                "model": self.model_name,
                "prompt": f"你是一个专业的医疗知识助手，请基于提供的知识简洁准确地回答问题。\n\n用户问题: {prompt}\n\n请回答:",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            }
            
            response = requests.post(url, json=data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '抱歉，无法生成回答。')
            else:
                return f"Ollama API调用失败: {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return f"无法连接到Ollama服务: {str(e)}"
        except Exception as e:
            return f"生成回答时出错: {str(e)}"

class RAGRetriever(BaseRetriever):
    """自定义RAG检索器"""
    
    def __init__(self, rag_system: ModularRAGSystem, **kwargs):
        super().__init__(**kwargs)
        # 使用object.__setattr__来绕过Pydantic的限制
        object.__setattr__(self, 'rag_system', rag_system)
        object.__setattr__(self, '_last_retrieval_details', {})
    
    @property
    def last_retrieval_details(self):
        return getattr(self, '_last_retrieval_details', {})
    
    def set_retrieval_details(self, details):
        object.__setattr__(self, '_last_retrieval_details', details)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """执行检索并返回相关文档"""
        try:
            # 1. 向量搜索
            print(f"\n🔍 执行向量搜索...")
            vector_results = self.rag_system.retrieval_manager.vector_search(
                query, entity_limit=5, relation_limit=5
            )
            
            # 2. 图谱搜索
            print(f"🕸️ 执行图谱搜索...")
            graph_results = self.rag_system.retrieval_manager.graph_search_topk_nodes(
                query, top_k=5, include_relations=True
            )
            
            # 3. 混合搜索
            print(f"🔄 执行混合搜索...")
            hybrid_results = self.rag_system.search_manager.hybrid_search(
                query, vector_entity_limit=5, graph_top_k=5
            )
            
            # 保存检索详情用于展示
            self.set_retrieval_details({
                'vector_results': vector_results,
                'graph_results': graph_results,
                'hybrid_results': hybrid_results,
                'query': query,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # 构建文档列表
            documents = []
            
            # 添加向量搜索的实体
            for entity in vector_results.get('entities', []):
                content = f"实体: {entity.get('name', '')} - {entity.get('description', '')}"
                documents.append(Document(
                    page_content=content,
                    metadata={'type': 'vector_entity', 'source': 'vector_search'}
                ))
            
            # 添加向量搜索的关系
            for relation in vector_results.get('relations', []):
                content = f"关系: {relation.get('description', '')}"
                documents.append(Document(
                    page_content=content,
                    metadata={'type': 'vector_relation', 'source': 'vector_search'}
                ))
            
            # 添加图谱搜索的节点
            for node in graph_results.get('nodes', []):
                content = f"节点: {node.get('name', '')} - {node.get('properties', {})}"
                documents.append(Document(
                    page_content=content,
                    metadata={'type': 'graph_node', 'source': 'graph_search'}
                ))
            
            # 添加图谱搜索的关系
            for rel in graph_results.get('relationships', []):
                content = f"关系: {rel.get('type', '')} - {rel.get('properties', {})}"
                documents.append(Document(
                    page_content=content,
                    metadata={'type': 'graph_relation', 'source': 'graph_search'}
                ))
            
            return documents
            
        except Exception as e:
            print(f"❌ 检索过程出错: {e}")
            return []

def format_retrieval_results(retriever: RAGRetriever) -> str:
    """格式化检索结果用于展示"""
    details = retriever.last_retrieval_details
    if not details:
        return "无检索详情"
    
    result_text = f"\n📊 检索详情 (查询: {details['query']}, 时间: {details['timestamp']})\n"
    result_text += "=" * 80 + "\n"
    
    # 向量搜索结果
    vector_results = details.get('vector_results', {})
    result_text += f"\n🔍 向量搜索结果:\n"
    result_text += f"  - 实体数量: {len(vector_results.get('entities', []))}\n"
    result_text += f"  - 关系数量: {len(vector_results.get('relations', []))}\n"
    
    for i, entity in enumerate(vector_results.get('entities', [])[:3], 1):
        similarity = entity.get('similarity', 'N/A')
        if isinstance(similarity, (int, float)):
            result_text += f"  - 实体{i}: {entity.get('name', 'N/A')} (相似度: {similarity:.3f})\n"
        else:
            result_text += f"  - 实体{i}: {entity.get('name', 'N/A')} (相似度: {similarity})\n"
    
    # 图谱搜索结果
    graph_results = details.get('graph_results', {})
    result_text += f"\n🕸️ 图谱搜索结果:\n"
    result_text += f"  - 节点数量: {graph_results.get('total_nodes', 0)}\n"
    result_text += f"  - 关系数量: {graph_results.get('total_relationships', 0)}\n"
    
    for i, node in enumerate(graph_results.get('nodes', [])[:3], 1):
        result_text += f"  - 节点{i}: {node.get('name', 'N/A')}\n"
    
    # 混合搜索统计
    hybrid_results = details.get('hybrid_results', {})
    search_stats = hybrid_results.get('search_stats', {})
    result_text += f"\n🔄 混合搜索统计:\n"
    
    total_time = search_stats.get('total_time', 'N/A')
    if isinstance(total_time, (int, float)):
        result_text += f"  - 总检索时间: {total_time:.3f}s\n"
    else:
        result_text += f"  - 总检索时间: {total_time}\n"
    
    vector_time = search_stats.get('vector_search_time', 'N/A')
    if isinstance(vector_time, (int, float)):
        result_text += f"  - 向量搜索时间: {vector_time:.3f}s\n"
    else:
        result_text += f"  - 向量搜索时间: {vector_time}\n"
    
    graph_time = search_stats.get('graph_search_time', 'N/A')
    if isinstance(graph_time, (int, float)):
        result_text += f"  - 图谱搜索时间: {graph_time:.3f}s\n"
    else:
        result_text += f"  - 图谱搜索时间: {graph_time}\n"
    
    return result_text

def create_rag_chain(rag_system: ModularRAGSystem):
    """创建RAG链"""
    
    # 创建检索器
    retriever = RAGRetriever(rag_system)
    
    # 创建LLM
    llm = OllamaLLM()
    
    # 创建提示词模板
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""基于以下检索到的知识内容，简洁准确地回答问题。

检索到的知识:
{context}

问题: {question}

请基于上述知识给出简洁的回答:"""
    )
    
    # 构建RAG链
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever, prompt_template

def main():
    """主函数"""
    print("🚀 启动基于LangChain的RAG系统")
    print("=" * 60)
    
    try:
        # 初始化RAG系统
        print("📡 初始化RAG系统连接...")
        with ModularRAGSystem() as rag_system:
            
            # 检查连接状态
            stats = rag_system.get_stats()
            print(f"✅ 连接状态:")
            print(f"  - Neo4j节点数: {stats.get('neo4j_nodes', 0)}")
            print(f"  - Neo4j关系数: {stats.get('neo4j_relationships', 0)}")
            print(f"  - Weaviate实体数: {stats.get('vector_entities', 0)}")
            print(f"  - Weaviate关系数: {stats.get('vector_relations', 0)}")
            
            # 创建RAG链
            print("\n🔗 构建LangChain RAG管道...")
            rag_chain, retriever, prompt_template = create_rag_chain(rag_system)
            
            # 交互式问答
            print("\n💬 开始交互式问答 (输入 'quit' 退出)")
            print("-" * 60)
            
            while True:
                try:
                    # 用户输入
                    question = input("\n❓ 请输入您的问题: ").strip()
                    
                    if question.lower() in ['quit', 'exit', '退出', 'q']:
                        print("👋 再见！")
                        break
                    
                    if not question:
                        print("⚠️ 请输入有效问题")
                        continue
                    
                    print(f"\n🔄 处理问题: {question}")
                    start_time = time.time()
                    
                    # 执行RAG链
                    answer = rag_chain.invoke(question)
                    
                    processing_time = time.time() - start_time
                    
                    # 展示结果
                    print("\n" + "=" * 80)
                    print("📋 完整处理结果")
                    print("=" * 80)
                    
                    # 1. 检索详情
                    print(format_retrieval_results(retriever))
                    
                    # 2. 构建的提示词
                    print(f"\n📝 构建的提示词:")
                    print("-" * 40)
                    context_docs = retriever._get_relevant_documents(question)
                    context = "\n\n".join([doc.page_content for doc in context_docs])
                    final_prompt = prompt_template.format(context=context, question=question)
                    print(final_prompt[:500] + "..." if len(final_prompt) > 500 else final_prompt)
                    
                    # 3. 大模型回答
                    print(f"\n🤖 大模型回答:")
                    print("-" * 40)
                    print(answer)
                    
                    # 4. 处理统计
                    print(f"\n⏱️ 处理统计:")
                    print(f"  - 总处理时间: {processing_time:.3f}s")
                    print(f"  - 检索文档数: {len(context_docs)}")
                    
                    print("\n" + "=" * 80)
                    
                except KeyboardInterrupt:
                    print("\n👋 用户中断，退出程序")
                    break
                except Exception as e:
                    print(f"❌ 处理问题时出错: {e}")
                    continue
    
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()