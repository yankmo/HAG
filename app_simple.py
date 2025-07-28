import streamlit as st
import sys
import os
import logging
import requests
import json
from typing import List, Dict, Any
import time
import random
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置管理器
from config import get_config
from src.services.llm_service import OllamaLLMService as SimpleOllamaLLM
from src.services.embedding_service import OllamaEmbeddingService as OllamaEmbeddingClient
from src.services.retrieval_service import RetrievalService

# 尝试导入真实的RAG系统组件
REAL_RAG_AVAILABLE = True
try:
    # 使用新的RetrievalService
    if 'import_logged' not in st.session_state:
        logger.info("新的检索服务组件导入成功")
        st.session_state.import_logged = True
except ImportError as e:
    REAL_RAG_AVAILABLE = False
    logger.warning(f"检索服务组件导入失败: {e}")

# 尝试导入简化的RAG组件（避免Weaviate依赖）
try:
    # 先尝试导入Neo4j相关组件
    from src.knowledge.intent_recognition_neo4j import KnowledgeGraphBuilder
    from py2neo import Graph
    SIMPLE_RAG_AVAILABLE = True
    if 'simple_import_logged' not in st.session_state:
        logger.info("简化RAG系统组件导入成功")
        st.session_state.simple_import_logged = True
except ImportError as e:
    SIMPLE_RAG_AVAILABLE = False
    logger.warning(f"简化RAG系统组件导入失败: {e}")

if REAL_RAG_AVAILABLE and 'system_type_logged' not in st.session_state:
    logger.info("使用新的检索服务进行演示")
    st.session_state.system_type_logged = True
elif SIMPLE_RAG_AVAILABLE and 'system_type_logged' not in st.session_state:
    logger.info("使用简化RAG系统进行演示")
    st.session_state.system_type_logged = True
elif 'system_type_logged' not in st.session_state:
    logger.info("使用模拟RAG系统进行演示")
    st.session_state.system_type_logged = True

# 页面配置
st.set_page_config(
    page_title="医疗知识RAG系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    .system-info {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .service-status {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
    }
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online {
        background-color: #4caf50;
    }
    .status-offline {
        background-color: #f44336;
    }
</style>
""", unsafe_allow_html=True)

class NewRAGSystem:
    """使用新的RetrievalService的RAG系统"""
    
    def __init__(self):
        """初始化新的RAG系统"""
        self.retrieval_service = None
        self.initialized = False
        
        try:
            if REAL_RAG_AVAILABLE:
                # 初始化RetrievalService
                self.retrieval_service = RetrievalService()
                self.initialized = True
                logger.info("新的RAG系统初始化成功")
            else:
                logger.warning("新的RAG系统不可用")
        except Exception as e:
            logger.error(f"新的RAG系统初始化失败: {e}")
            self.initialized = False
    
    def get_stats(self):
        """获取系统统计信息"""
        if self.initialized and self.retrieval_service:
            try:
                # 获取统计信息
                stats = self.retrieval_service.get_stats()
                return {
                    "neo4j_nodes": stats.get("neo4j_nodes", 0),
                    "neo4j_relationships": stats.get("neo4j_relationships", 0),
                    "weaviate_entities": stats.get("weaviate_entities", 0),
                    "weaviate_relations": stats.get("weaviate_relations", 0),
                    "status": "新版本"
                }
            except Exception as e:
                logger.error(f"获取统计信息失败: {e}")
                return {
                    "neo4j_nodes": 0,
                    "neo4j_relationships": 0,
                    "weaviate_entities": 0,
                    "weaviate_relations": 0,
                    "error": str(e)
                }
        return {
            "neo4j_nodes": 0,
            "neo4j_relationships": 0,
            "weaviate_entities": 0,
            "weaviate_relations": 0,
            "status": "未初始化"
        }
    
    def search_knowledge(self, query, **kwargs):
        """搜索知识"""
        if self.initialized and self.retrieval_service:
            try:
                import time
                start_time = time.time()
                
                # 使用新的检索服务进行搜索
                hybrid_result = self.retrieval_service.search_hybrid(query, limit=5)
                
                retrieval_time = time.time() - start_time
                logger.info(f"新检索服务完成，耗时 {retrieval_time:.2f}s")
                
                # 转换结果格式以兼容现有的显示代码
                entities = []
                relations = []
                
                # 处理混合检索结果
                for result in hybrid_result.hybrid_results:
                    # 计算相似度（从SearchResult对象获取）
                    similarity = result.score if hasattr(result, 'score') else None
                    distance = result.distance if hasattr(result, 'distance') else None
                    
                    # 从metadata获取信息
                    metadata = result.metadata if hasattr(result, 'metadata') else {}
                    result_type = metadata.get('type', '')
                    name = metadata.get('name', 'N/A')
                    description = metadata.get('description', 'N/A')
                    
                    # 记录调试信息
                    logger.debug(f"处理结果: type={result_type}, name={name}, similarity={similarity}")
                    
                    # 检查是否为关系类型
                    if (result_type == 'relation' and 
                        metadata.get('start_entity') and 
                        metadata.get('end_entity')):
                        # 这是真正的关系
                        relation = {
                            'description': f"{metadata.get('start_entity', 'N/A')} → {metadata.get('relation_type', 'N/A')} → {metadata.get('end_entity', 'N/A')}",
                            'type': metadata.get('relation_type', 'N/A'),
                            'start_entity': metadata.get('start_entity', 'N/A'),
                            'end_entity': metadata.get('end_entity', 'N/A'),
                            'similarity': f"{similarity:.3f}" if similarity is not None else "N/A"
                        }
                        relations.append(relation)
                    else:
                        # 默认作为实体处理（包括所有其他类型）
                        entity = {
                            'name': name,
                            'type': result_type if result_type else 'entity',
                            'description': description,
                            'similarity': f"{similarity:.3f}" if similarity is not None else "N/A",
                            'distance': distance,
                            'source_text': result.content if hasattr(result, 'content') else '',
                            'metadata': metadata
                        }
                        entities.append(entity)
                
                return {
                    "query": query,
                    "search_results": {
                        "vector_search": {
                            "entities": entities,
                            "relations": relations  # 只包含真正的关系
                        },
                        "graph_search": {
                            "nodes": [],
                            "relationships": [],
                            "total_nodes": 0,
                            "total_relationships": 0
                        },
                        "hybrid_search": {
                            "search_stats": {
                                "vector_entities": len(entities),
                                "vector_relations": len(relations),
                                "graph_nodes": 0,
                                "graph_relationships": 0
                            }
                        }
                    },
                    "retrieval_time": retrieval_time
                }
                
            except Exception as e:
                logger.error(f"知识搜索失败: {e}")
                return {
                    "error": str(e),
                    "query": query,
                    "search_results": {}
                }
        else:
            logger.error("新RAG系统未初始化")
            return {
                "error": "新RAG系统未初始化",
                "query": query,
                "search_results": {}
            }
    
    def generate_answer(self, query, search_results=None):
        """生成答案"""
        if self.initialized and self.retrieval_service:
            try:
                # 使用检索服务生成答案
                answer = self.retrieval_service.generate_answer(query, search_results)
                return answer
            except Exception as e:
                logger.error(f"生成答案失败: {e}")
                return f"生成答案时出现错误: {e}"
        else:
            return "新RAG系统未初始化，无法生成答案"


class SimpleRAGSystem:
    """简化的RAG系统，仅使用Neo4j知识图谱"""
    
    def __init__(self):
        """初始化简化RAG系统"""
        self.kg_builder = None
        self.neo4j_graph = None
        self.vector_processor = None
        self.initialized = False
        
        try:
            if SIMPLE_RAG_AVAILABLE:
                # 获取配置
                config = get_config()
                
                # 初始化知识图谱构建器
                self.kg_builder = KnowledgeGraphBuilder()
                # 初始化Neo4j连接
                self.neo4j_graph = Graph(config.neo4j.uri, auth=(config.neo4j.username, config.neo4j.password))
                
                # 尝试初始化向量处理器（可选）
                try:
                    from src.knowledge.vector_storage import WeaviateVectorStore, VectorKnowledgeProcessor
                    embedding_client = OllamaEmbeddingClient()
                    vector_store = WeaviateVectorStore()
                    self.vector_processor = VectorKnowledgeProcessor(embedding_client, vector_store)
                    logger.info("向量处理器初始化成功")
                except Exception as ve:
                    logger.warning(f"向量处理器初始化失败，将仅使用图谱检索: {ve}")
                    self.vector_processor = None
                
                self.initialized = True
                logger.info("简化RAG系统初始化成功")
            else:
                logger.warning("简化RAG系统不可用")
        except Exception as e:
            logger.error(f"简化RAG系统初始化失败: {e}")
            self.initialized = False
    
    def get_stats(self):
        """获取系统统计信息"""
        if self.initialized and self.neo4j_graph:
            try:
                # 查询Neo4j统计信息
                node_count = self.neo4j_graph.run("MATCH (n) RETURN count(n) as count").data()[0]['count']
                rel_count = self.neo4j_graph.run("MATCH ()-[r]->() RETURN count(r) as count").data()[0]['count']
                
                return {
                    "neo4j_nodes": node_count,
                    "neo4j_relationships": rel_count,
                    "weaviate_entities": 0,  # 简化版本不使用Weaviate
                    "weaviate_relations": 0,
                    "status": "简化版本"
                }
            except Exception as e:
                logger.error(f"获取统计信息失败: {e}")
                return {
                    "neo4j_nodes": 0,
                    "neo4j_relationships": 0,
                    "weaviate_entities": 0,
                    "weaviate_relations": 0,
                    "error": str(e)
                }
        return {
            "neo4j_nodes": 0,
            "neo4j_relationships": 0,
            "weaviate_entities": 0,
            "weaviate_relations": 0,
            "status": "未初始化"
        }
    
    def search_knowledge(self, query, **kwargs):
        """搜索知识"""
        if self.initialized and self.neo4j_graph:
            try:
                import time
                start_time = time.time()
                
                # 使用Neo4j进行简单的关键词搜索，支持中英文
                search_query = f"""
                MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower('{query}') 
                   OR toLower(toString(n.description)) CONTAINS toLower('{query}')
                   OR toLower(n.name) CONTAINS toLower('parkinson') 
                   OR toLower(toString(n.description)) CONTAINS toLower('parkinson')
                OPTIONAL MATCH (n)-[r]-(m)
                RETURN n, r, m
                LIMIT 20
                """
                
                logger.info(f"执行Neo4j查询: {search_query}")
                results = self.neo4j_graph.run(search_query).data()
                logger.info(f"Neo4j查询返回 {len(results)} 条结果")
                
                # 格式化结果
                nodes = []
                relationships = []
                seen_nodes = set()
                
                for record in results:
                    if record['n']:
                        node = record['n']
                        node_id = id(node)
                        if node_id not in seen_nodes:
                            node_data = dict(node)
                            node_data['labels'] = list(node.labels)
                            nodes.append(node_data)
                            seen_nodes.add(node_id)
                    
                    if record['r'] and record['m']:
                        rel_data = {
                            'type': type(record['r']).__name__,
                            'properties': dict(record['r']),
                            'start_node': dict(record['n']),
                            'end_node': dict(record['m'])
                        }
                        relationships.append(rel_data)
                
                retrieval_time = time.time() - start_time
                logger.info(f"检索完成，找到 {len(nodes)} 个节点，{len(relationships)} 个关系，耗时 {retrieval_time:.2f}s")
                
                # 尝试向量检索（如果可用）
                vector_entities = []
                vector_relations = []
                hybrid_knowledge = ""
                
                try:
                    # 检查是否有向量存储可用
                    if hasattr(self, 'vector_processor') and self.vector_processor:
                        # 使用混合向量检索
                        hybrid_results = self.vector_processor.search_knowledge_hybrid(query, limit=5)
                        
                        # 提取向量检索结果
                        vector_entities = hybrid_results.get('cosine_results', [])
                        vector_relations = hybrid_results.get('euclidean_results', [])
                        
                        # 获取格式化的知识内容用于提示词
                        hybrid_knowledge = self.vector_processor.get_knowledge_for_prompt(query, limit=5)
                        
                        # 获取检索统计
                        retrieval_stats = hybrid_results.get('retrieval_stats', {})
                        logger.info(f"混合向量检索完成: 总计 {retrieval_stats.get('total_found', 0)} 个片段，"
                                  f"余弦相似度 {retrieval_stats.get('cosine_count', 0)} 个，"
                                  f"欧氏距离 {retrieval_stats.get('euclidean_count', 0)} 个")
                    else:
                        logger.info("向量存储不可用，跳过向量检索")
                except Exception as ve:
                    logger.warning(f"向量检索失败: {ve}")
                
                return {
                    "query": query,
                    "search_results": {
                        "graph_search": {
                            "nodes": nodes,
                            "relationships": relationships,
                            "total_nodes": len(nodes),
                            "total_relationships": len(relationships)
                        },
                        "vector_search": {
                            "entities": vector_entities,
                            "relations": vector_relations,
                            "hybrid_knowledge": hybrid_knowledge
                        },
                        "hybrid_search": {
                            "search_stats": {
                                "graph_nodes": len(nodes),
                                "graph_relationships": len(relationships),
                                "vector_entities": len(vector_entities),
                                "vector_relations": len(vector_relations),
                                "has_hybrid_knowledge": bool(hybrid_knowledge)
                            }
                        }
                    },
                    "retrieval_time": retrieval_time
                }
                
            except Exception as e:
                logger.error(f"知识搜索失败: {e}")
                return {
                    "error": str(e),
                    "query": query,
                    "search_results": {}
                }
        else:
            logger.error("RAG系统未初始化或Neo4j连接失败")
            return {
                "error": "RAG系统未初始化",
                "query": query,
                "search_results": {}
            }
    
    def generate_answer(self, query, search_results=None):
        """生成答案"""
        if self.initialized and self.kg_builder:
            try:
                if search_results and search_results.get("search_results"):
                    # 基于搜索结果构建上下文
                    context_parts = []
                    graph_search = search_results["search_results"].get("graph_search", {})
                    vector_search = search_results["search_results"].get("vector_search", {})
                    
                    # 优先使用混合向量检索的知识内容
                    hybrid_knowledge = vector_search.get("hybrid_knowledge", "")
                    if hybrid_knowledge and hybrid_knowledge != "未找到相关知识内容。":
                        context_parts.append("📚 向量检索知识:")
                        context_parts.append(hybrid_knowledge)
                    
                    # 添加图谱搜索结果
                    if graph_search.get("nodes"):
                        context_parts.append("\n🔍 图谱实体:")
                        for node in graph_search["nodes"][:3]:  # 减少显示数量，避免过长
                            name = node.get("name", "未知")
                            labels = ", ".join(node.get("labels", []))
                            context_parts.append(f"- {name} ({labels})")
                            if node.get("description"):
                                desc = node['description'][:100] + "..." if len(node['description']) > 100 else node['description']
                                context_parts.append(f"  描述: {desc}")
                    
                    if graph_search.get("relationships"):
                        context_parts.append("\n🔗 图谱关系:")
                        for rel in graph_search["relationships"][:2]:  # 减少显示数量
                            start_name = rel.get("start_node", {}).get("name", "未知")
                            end_name = rel.get("end_node", {}).get("name", "未知")
                            rel_type = rel.get("type", "相关")
                            context_parts.append(f"- {start_name} → {rel_type} → {end_name}")
                    
                    context = "\n".join(context_parts)
                    
                    # 构建增强的提示词
                    if hybrid_knowledge and hybrid_knowledge != "未找到相关知识内容。":
                        prompt = f"""基于以下医疗知识回答问题：

问题: {query}

{context}

请基于上述知识内容，提供准确、专业的医疗回答。重点关注向量检索的知识内容，结合图谱信息进行补充。回答要求：
1. 准确性：基于提供的知识内容回答
2. 完整性：尽可能全面地回答问题
3. 专业性：使用医疗专业术语，但要通俗易懂
4. 安全性：如涉及诊断治疗，请提醒咨询专业医生"""
                    else:
                        prompt = f"""基于以下知识图谱信息回答问题：

问题: {query}

知识来源:
{context}

请基于上述知识图谱信息，提供准确、详细的回答。如果信息不足，请说明需要更多哪方面的信息。"""
                    
                    # 使用Ollama生成回答
                    response = self.kg_builder.recognizer.ollama.generate(prompt)
                    return response
                else:
                    # 直接回答
                    prompt = f"请回答以下医疗相关问题：{query}\n\n请提供专业、准确的回答，如涉及诊断治疗建议，请提醒咨询专业医生。"
                    response = self.kg_builder.recognizer.ollama.generate(prompt)
                    return response
                    
            except Exception as e:
                logger.error(f"生成答案失败: {e}")
                return f"生成答案时出现错误: {e}"
        else:
            return "RAG系统未初始化，无法生成答案"


class RealRAGSystem:
    """真实的RAG系统包装器"""
    
    def __init__(self):
        self.rag_system = None
        self.rag_chain = None
        self.retriever = None
        self.prompt_template = None
        self.initialized = False
    
    def initialize(self):
        """初始化RAG系统"""
        try:
            if not REAL_RAG_AVAILABLE:
                raise ImportError("真实RAG系统组件不可用")
            
            # 初始化ModularRAGSystem
            self.rag_system = ModularRAGSystem()
            
            # 创建RAG链
            self.rag_chain, self.retriever, self.prompt_template = create_rag_chain(self.rag_system)
            
            self.initialized = True
            return True
            
        except Exception as e:
            st.error(f"RAG系统初始化失败: {str(e)}")
            return False
    
    def get_stats(self):
        """获取系统统计信息"""
        if not self.initialized or not self.rag_system:
            return {}
        
        try:
            return self.rag_system.get_stats()
        except Exception as e:
            st.error(f"获取统计信息失败: {str(e)}")
            return {}
    
    def search_knowledge(self, query: str) -> Dict[str, Any]:
        """执行知识检索"""
        if not self.initialized:
            return {
                'vector_results': {'entities': [], 'relations': []},
                'graph_results': {'nodes': [], 'relationships': [], 'total_nodes': 0, 'total_relationships': 0},
                'hybrid_results': {'search_stats': {}},
                'error': '系统未初始化'
            }
        
        try:
            # 使用ModularRAGSystem的搜索功能
            start_time = time.time()
            search_result = self.rag_system.search(query)
            retrieval_time = time.time() - start_time
            
            # 从搜索结果中提取检索详情
            details = search_result.get('retrieval_details', {
                'vector_results': {'entities': [], 'relations': []},
                'graph_results': {'nodes': [], 'relationships': [], 'total_nodes': 0, 'total_relationships': 0},
                'hybrid_results': {'search_stats': {}}
            })
            
            # 添加检索时间
            if 'hybrid_results' not in details:
                details['hybrid_results'] = {}
            if 'search_stats' not in details['hybrid_results']:
                details['hybrid_results']['search_stats'] = {}
            details['hybrid_results']['search_stats']['total_time'] = retrieval_time
            
            return details
            
        except Exception as e:
            st.error(f"知识检索失败: {str(e)}")
            return {
                'vector_results': {'entities': [], 'relations': []},
                'graph_results': {'nodes': [], 'relationships': [], 'total_nodes': 0, 'total_relationships': 0},
                'hybrid_results': {'search_stats': {}},
                'error': str(e)
            }
    
    def generate_answer(self, query: str) -> str:
        """生成回答"""
        if not self.initialized:
            return "RAG系统未初始化，无法生成回答。"
        
        try:
            # 使用ModularRAGSystem的搜索功能获取完整结果
            search_result = self.rag_system.search(query)
            answer = search_result.get('answer', '无法生成回答')
            return answer
            
        except Exception as e:
            st.error(f"生成回答失败: {str(e)}")
            return f"生成回答时出错: {str(e)}"
    
    def close(self):
        """关闭RAG系统"""
        if self.rag_system:
            try:
                self.rag_system.close()
            except:
                pass

class MockRAGRetriever:
    """模拟RAG检索器，用于展示检索过程"""
    
    def __init__(self):
        # 模拟知识库
        self.knowledge_base = {
            "帕金森病": {
                "symptoms": ["静止性震颤", "运动迟缓", "肌肉僵直", "姿势不稳"],
                "causes": ["多巴胺神经元退化", "遗传因素", "环境因素"],
                "treatments": ["药物治疗", "深部脑刺激", "康复训练"],
                "source": "医学教科书第12版"
            },
            "高血压": {
                "symptoms": ["头痛", "头晕", "心悸", "视力模糊"],
                "prevention": ["低盐饮食", "规律运动", "控制体重", "戒烟限酒"],
                "treatments": ["ACE抑制剂", "利尿剂", "钙通道阻滞剂"],
                "source": "心血管疾病指南2023"
            },
            "糖尿病": {
                "types": ["1型糖尿病", "2型糖尿病", "妊娠糖尿病"],
                "diet": ["控制碳水化合物", "增加纤维摄入", "定时定量", "避免高糖食物"],
                "complications": ["糖尿病肾病", "糖尿病视网膜病变", "糖尿病足"],
                "source": "糖尿病诊疗指南2023"
            }
        }
    
    def search_knowledge(self, query: str) -> Dict[str, Any]:
        """模拟知识检索过程"""
        # 模拟检索延迟
        time.sleep(0.5)
        
        # 简单的关键词匹配
        results = {
            "query": query,
            "vector_results": [],
            "graph_results": [],
            "hybrid_results": [],
            "total_time": round(random.uniform(0.3, 1.2), 2)
        }
        
        # 模拟向量检索
        for disease, info in self.knowledge_base.items():
            if disease in query:
                similarity = round(random.uniform(0.7, 0.95), 3)
                results["vector_results"].append({
                    "entity": disease,
                    "similarity": similarity,
                    "content": str(info),
                    "source": info.get("source", "未知来源")
                })
        
        # 模拟图谱检索
        if results["vector_results"]:
            main_entity = results["vector_results"][0]["entity"]
            results["graph_results"] = [
                {"relation": "症状", "entities": self.knowledge_base[main_entity].get("symptoms", [])},
                {"relation": "治疗", "entities": self.knowledge_base[main_entity].get("treatments", [])},
            ]
        
        # 模拟混合检索
        results["hybrid_results"] = results["vector_results"][:3]  # 取前3个结果
        
        return results
    
    def generate_answer(self, query: str) -> str:
        """模拟生成回答"""
        # 简单的模拟回答
        for disease, info in self.knowledge_base.items():
            if disease in query:
                answer_parts = [f"关于{disease}的信息："]
                
                if "symptoms" in info:
                    answer_parts.append(f"主要症状包括：{', '.join(info['symptoms'])}")
                
                if "treatments" in info:
                    answer_parts.append(f"治疗方法包括：{', '.join(info['treatments'])}")
                
                if "prevention" in info:
                    answer_parts.append(f"预防措施包括：{', '.join(info['prevention'])}")
                
                answer_parts.append("\n注意：本回答仅供参考，如有疑问请咨询专业医生。")
                
                return "\n\n".join(answer_parts)
        
        return "抱歉，我无法找到相关的医疗信息。建议您咨询专业医生获取准确的医疗建议。"

def display_retrieval_process(retrieval_results: Dict[str, Any]):
    """展示检索过程详情 - 默认折叠"""
    # 使用expander实现折叠功能，默认折叠
    with st.expander("🔍 检索过程详情", expanded=False):
        # 检索统计
        col1, col2, col3, col4 = st.columns(4)
        
        # 处理SimpleRAGSystem的数据格式
        search_results = retrieval_results.get('search_results', {})
        vector_search = search_results.get('vector_search', {})
        graph_search = search_results.get('graph_search', {})
        hybrid_search = search_results.get('hybrid_search', {})
        search_stats = hybrid_search.get('search_stats', {})
        
        with col1:
            vector_count = len(vector_search.get('entities', [])) + len(vector_search.get('relations', []))
            st.metric("向量检索", vector_count)
        
        with col2:
            graph_count = graph_search.get('total_nodes', 0)
            st.metric("图谱检索", graph_count)
        
        with col3:
            hybrid_count = search_stats.get('graph_nodes', 0) + search_stats.get('vector_entities', 0)
            st.metric("混合检索", hybrid_count)
        
        with col4:
            retrieval_time = retrieval_results.get('retrieval_time', 0)
            if isinstance(retrieval_time, (int, float)):
                st.metric("检索耗时", f"{retrieval_time:.2f}s")
            else:
                st.metric("检索耗时", "0.00s")

def display_detailed_results(retrieval_results: Dict[str, Any]):
    """展示详细检索结果 - 默认折叠"""
    # 使用expander实现折叠功能，默认折叠
    with st.expander("📊 详细检索结果", expanded=False):
        # 创建三个标签页
        tab1, tab2, tab3 = st.tabs(["🔍 向量检索", "🕸️ 图谱检索", "🔄 混合检索"])
        
        # 处理SimpleRAGSystem的数据格式
        search_results = retrieval_results.get('search_results', {})
        vector_search = search_results.get('vector_search', {})
        graph_search = search_results.get('graph_search', {})
        hybrid_search = search_results.get('hybrid_search', {})
        
        with tab1:
            entities = vector_search.get('entities', [])
            relations = vector_search.get('relations', [])
            
            if entities:
                st.write("**实体结果:**")
                for i, entity in enumerate(entities[:5], 1):
                    similarity = entity.get('similarity', 'N/A')
                    name = entity.get('name', 'N/A')
                    description = entity.get('description', 'N/A')
                    
                    # 使用容器而不是嵌套expander
                    with st.container():
                        st.markdown(f"**实体 {i}: {name}**")
                        st.write(f"相似度: {similarity}")
                        st.write(f"描述: {description}")
                        st.divider()
            
            if relations:
                st.write("**关系结果:**")
                for i, relation in enumerate(relations[:5], 1):
                    description = relation.get('description', 'N/A')
                    with st.container():
                        st.markdown(f"**关系 {i}**")
                        st.write(f"描述: {description}")
                        st.divider()
            
            if not entities and not relations:
                st.info("未找到相关向量结果")
        
        with tab2:
            nodes = graph_search.get('nodes', [])
            relationships = graph_search.get('relationships', [])
            
            if nodes:
                st.write("**节点结果:**")
                for i, node in enumerate(nodes[:10], 1):  # 显示更多节点
                    name = node.get('name', 'N/A')
                    description = node.get('description', 'N/A')
                    labels = node.get('labels', [])
                    
                    with st.container():
                        st.markdown(f"**节点 {i}: {name}**")
                        st.write(f"标签: {', '.join(labels)}")
                        st.write(f"描述: {description}")
                        # 显示其他属性
                        other_props = {k: v for k, v in node.items() 
                                     if k not in ['name', 'description', 'labels']}
                        if other_props:
                            st.write("其他属性:")
                            for key, value in other_props.items():
                                st.write(f"  - {key}: {value}")
                        st.divider()
            
            if relationships:
                st.write("**关系结果:**")
                for i, rel in enumerate(relationships[:10], 1):  # 显示更多关系
                    rel_type = rel.get('type', 'N/A')
                    start_node = rel.get('start_node', {}).get('name', 'N/A')
                    end_node = rel.get('end_node', {}).get('name', 'N/A')
                    
                    with st.container():
                        st.markdown(f"**关系 {i}: {rel_type}**")
                        st.write(f"起始节点: {start_node}")
                        st.write(f"结束节点: {end_node}")
                        if rel.get('properties'):
                            st.write(f"属性: {rel['properties']}")
                        st.divider()
            
            if not nodes and not relationships:
                st.info("未找到相关图谱结果")
        
        with tab3:
            search_stats = hybrid_search.get('search_stats', {})
            
            if search_stats:
                st.write("**检索统计:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    graph_nodes = search_stats.get('graph_nodes', 0)
                    graph_relationships = search_stats.get('graph_relationships', 0)
                    vector_entities = search_stats.get('vector_entities', 0)
                    vector_relations = search_stats.get('vector_relations', 0)
                    
                    st.metric("图谱节点", graph_nodes)
                    st.metric("图谱关系", graph_relationships)
                    
                with col2:
                    st.metric("向量实体", vector_entities)
                    st.metric("向量关系", vector_relations)
                    
                    retrieval_time = retrieval_results.get('retrieval_time', 0)
                    if isinstance(retrieval_time, (int, float)):
                        st.metric("检索耗时", f"{retrieval_time:.3f}s")
                    else:
                        st.metric("检索耗时", "N/A")
            else:
                st.info("暂无混合检索统计信息")

def display_knowledge_sources(retrieval_results: Dict[str, Any]):
    """展示知识来源 - 默认折叠"""
    # 使用expander实现折叠功能，默认折叠
    with st.expander("📚 知识来源", expanded=False):
        # 处理SimpleRAGSystem的数据格式
        search_results = retrieval_results.get('search_results', {})
        vector_search = search_results.get('vector_search', {})
        graph_search = search_results.get('graph_search', {})
        
        sources = []
        seen_sources = set()  # 用于去重
        
        # 收集向量检索来源
        for entity in vector_search.get('entities', []):
            if entity.get('name'):
                source_key = f"vector_entity_{entity.get('name')}"
                if source_key not in seen_sources:
                    sources.append({
                        'type': '向量实体',
                        'name': entity.get('name', 'N/A'),
                        'source': 'Weaviate向量数据库',
                        'description': entity.get('description', 'N/A')
                    })
                    seen_sources.add(source_key)
        
        for relation in vector_search.get('relations', []):
            if relation.get('description'):
                desc = relation.get('description', 'N/A')[:50] + '...'
                source_key = f"vector_relation_{desc}"
                if source_key not in seen_sources:
                    sources.append({
                        'type': '向量关系',
                        'name': desc,
                        'source': 'Weaviate向量数据库'
                    })
                    seen_sources.add(source_key)
        
        # 收集图谱检索来源（去重）
        for node in graph_search.get('nodes', []):
            if node.get('name'):
                node_name = node.get('name', 'N/A')
                source_key = f"graph_node_{node_name}"
                if source_key not in seen_sources:
                    sources.append({
                        'type': '图谱节点',
                        'name': node_name,
                        'source': 'Neo4j知识图谱',
                        'description': node.get('description', 'N/A'),
                        'labels': ', '.join(node.get('labels', []))
                    })
                    seen_sources.add(source_key)
        
        for rel in graph_search.get('relationships', []):
            if rel.get('type'):
                start_name = rel.get('start_node', {}).get('name', 'N/A')
                end_name = rel.get('end_node', {}).get('name', 'N/A')
                rel_name = f"{start_name} → {rel.get('type', 'N/A')} → {end_name}"
                source_key = f"graph_relation_{rel_name}"
                if source_key not in seen_sources:
                    sources.append({
                        'type': '图谱关系',
                        'name': rel_name,
                        'source': 'Neo4j知识图谱'
                    })
                    seen_sources.add(source_key)
        
        if sources:
            for i, source in enumerate(sources[:10], 1):
                # 使用容器而不是嵌套expander
                with st.container():
                    st.markdown(f"**来源 {i}: {source['name']}**")
                    st.write(f"类型: {source['type']}")
                    st.write(f"数据源: {source['source']}")
                    if source.get('description') and source['description'] != 'N/A':
                        st.write(f"描述: {source['description']}")
                    if source.get('labels'):
                        st.write(f"标签: {source['labels']}")
                    st.divider()
        else:
            st.info("暂无知识来源信息")

def check_service_status():
    """检查各服务状态"""
    services = {
        "Ollama": "http://localhost:11434/api/tags",
        "Neo4j": "http://localhost:7474",  # Neo4j Browser端口
        "Weaviate": "http://localhost:8080/v1/meta"
    }
    
    status = {}
    for service, url in services.items():
        try:
            response = requests.get(url, timeout=3)
            status[service] = response.status_code == 200
        except:
            status[service] = False
    
    return status

def display_service_status():
    """显示服务状态"""
    st.subheader("🔧 服务状态")
    
    status = check_service_status()
    
    for service, is_online in status.items():
        status_class = "status-online" if is_online else "status-offline"
        status_text = "在线" if is_online else "离线"
        
        st.markdown(f"""
        <div class="service-status">
            <div class="status-indicator {status_class}"></div>
            <strong>{service}</strong>: {status_text}
        </div>
        """, unsafe_allow_html=True)
    
    return status

@st.cache_resource
def initialize_llm():
    """初始化LLM（使用缓存）"""
    return SimpleOllamaLLM()

@st.cache_resource
def initialize_retriever():
    """初始化检索器（使用缓存）"""
    logger.info(f"开始初始化检索器，REAL_RAG_AVAILABLE={REAL_RAG_AVAILABLE}")
    
    if REAL_RAG_AVAILABLE:
        logger.info("尝试初始化新RAG系统")
        rag_system = NewRAGSystem()
        if rag_system.initialized:
            logger.info("新RAG系统初始化成功")
            return rag_system
        else:
            logger.warning("新RAG系统初始化失败，尝试简化RAG系统")
            if SIMPLE_RAG_AVAILABLE:
                rag_system = SimpleRAGSystem()
                if rag_system.initialized:
                    logger.info("简化RAG系统初始化成功")
                    return rag_system
            logger.warning("所有RAG系统初始化失败，使用模拟系统")
            return MockRAGRetriever()
    elif SIMPLE_RAG_AVAILABLE:
        logger.info("尝试初始化简化RAG系统")
        rag_system = SimpleRAGSystem()
        if rag_system.initialized:
            logger.info("简化RAG系统初始化成功")
            return rag_system
        else:
            logger.warning("简化RAG系统初始化失败，使用模拟系统")
            return MockRAGRetriever()
    else:
        logger.info("使用模拟RAG系统")
        return MockRAGRetriever()

def create_medical_prompt(question: str, retrieval_context: str = "") -> str:
    """创建医疗领域的提示词"""
    base_prompt = f"""你是一个专业的医疗知识助手。请基于医学知识和提供的上下文信息回答以下问题，确保回答准确、专业且易于理解。

用户问题：{question}"""
    
    if retrieval_context:
        base_prompt += f"""

参考信息：
{retrieval_context}"""
    
    base_prompt += """

请提供详细的回答，包括：
1. 直接回答问题
2. 相关的医学解释
3. 注意事项或建议

注意：本回答仅供参考，不能替代专业医疗建议。如有疑问，请咨询专业医生。

回答："""
    
    return base_prompt

def format_retrieval_context(retrieval_results: Dict[str, Any]) -> str:
    """格式化检索结果为上下文"""
    context_parts = []
    
    vector_results = retrieval_results.get("vector_results", [])
    for result in vector_results:
        context_parts.append(f"实体: {result.get('entity', 'N/A')}")
        context_parts.append(f"内容: {result.get('content', 'N/A')}")
        context_parts.append(f"来源: {result.get('source', 'N/A')}")
        context_parts.append("---")
    
    return "\n".join(context_parts)

def main():
    """主函数"""
    # 页面标题
    st.markdown('<h1 class="main-header">🏥 医疗知识RAG系统</h1>', unsafe_allow_html=True)
    
    # 初始化会话状态
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    
    if 'max_tokens' not in st.session_state:
        st.session_state.max_tokens = 1000
    
    # 使用缓存的初始化函数，避免重复初始化
    with st.spinner("🚀 正在初始化RAG系统..."):
        rag_system = initialize_retriever()
        
        # 显示初始化结果（只在首次加载时显示）
        if 'init_message_shown' not in st.session_state:
            if isinstance(rag_system, NewRAGSystem):
                if rag_system.initialized:
                    st.success("✅ 新RAG系统初始化成功！")
                    
                    # 显示系统统计
                    stats = rag_system.get_stats()
                    if stats:
                        st.info(f"📊 系统状态: Neo4j节点 {stats.get('neo4j_nodes', 0)} 个, "
                               f"关系 {stats.get('neo4j_relationships', 0)} 个, "
                               f"向量实体 {stats.get('vector_entities', 0)} 个")
                else:
                    st.error("❌ 新RAG系统初始化失败，使用模拟系统")
            elif isinstance(rag_system, SimpleRAGSystem):
                if rag_system.initialized:
                    st.success("✅ 简化RAG系统初始化成功！")
                    
                    # 显示系统统计
                    stats = rag_system.get_stats()
                    if stats:
                        st.info(f"📊 系统状态: Neo4j节点 {stats.get('neo4j_nodes', 0)} 个, "
                               f"关系 {stats.get('neo4j_relationships', 0)} 个, "
                               f"状态: {stats.get('status', '未知')}")
                else:
                    st.error("❌ 简化RAG系统初始化失败，使用模拟系统")
            elif isinstance(rag_system, RealRAGSystem):
                if rag_system.initialized:
                    st.success("✅ 真实RAG系统初始化成功！")
                    
                    # 显示系统统计
                    stats = rag_system.get_stats()
                    if stats:
                        st.info(f"📊 系统状态: Neo4j节点 {stats.get('neo4j_nodes', 0)} 个, "
                               f"关系 {stats.get('neo4j_relationships', 0)} 个, "
                               f"向量实体 {stats.get('vector_entities', 0)} 个")
                else:
                    st.error("❌ 真实RAG系统初始化失败，使用模拟系统")
            else:
                st.warning("⚠️ 使用模拟RAG系统")
            
            st.session_state.init_message_shown = True
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统设置")
        
        # 服务状态
        service_status = display_service_status()
        
        st.divider()
        
        # 模型设置
        st.header("🤖 模型设置")
        model_name = st.selectbox(
            "选择模型",
            ["gemma3:4b", "llama3:8b", "qwen2:7b"],
            index=0
        )
        
        st.session_state.temperature = st.slider(
            "温度 (创造性)",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1
        )
        
        st.session_state.max_tokens = st.slider(
            "最大输出长度",
            min_value=100,
            max_value=2000,
            value=st.session_state.max_tokens,
            step=100
        )
        
        st.divider()
        
        # 使用说明
        st.header("📖 使用说明")
        st.markdown("""
        1. 在下方输入框中输入医疗相关问题
        2. 系统将检索相关知识并生成专业回答
        3. 可在侧边栏调整模型参数
        4. 查看详细的检索过程和知识来源
        """)
        
        st.divider()
        
        # 系统配置信息（移除RAG系统状态）
        st.header("⚙️ 当前配置")
        st.info(f"""
        **模型设置:**
        - 模型: {model_name}
        - 温度: {st.session_state.temperature}
        - 最大长度: {st.session_state.max_tokens}
        - 服务端口: 11434
        """)
        
        st.divider()
        
        # 功能说明
        st.header("🔍 功能特色")
        st.markdown("""
        - **智能检索**: 基于知识图谱的精准检索
        - **多维度分析**: 向量+图谱+混合检索
        - **检索可视化**: 详细展示检索过程
        - **专业问答**: 基于医疗知识的智能回答
        - **参数调节**: 可调节模型创造性和长度
        """)
        
        # 注意事项
        st.warning("⚠️ 本系统仅供学习研究使用，医疗建议请咨询专业医生")
    
    # 主内容区域
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 智能问答")
        
        # 检查Ollama连接
        if not service_status.get("Ollama", False):
            st.error("❌ Ollama服务未连接，请确保Ollama正在运行")
            st.stop()
        
        # 初始化LLM
        llm = initialize_llm()
        llm.model = model_name
        
        # 显示聊天历史
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    with col2:
        st.header("🔧 功能面板")
        
        # 清空聊天历史
        if st.button("🗑️ 清空聊天记录"):
            st.session_state.messages = []
            st.success("聊天记录已清空！")
        
        st.divider()
        
        # 示例问题
        st.subheader("💡 示例问题")
        example_questions = [
            "帕金森病的主要症状有哪些？",
            "高血压的预防措施是什么？",
            "糖尿病患者的饮食注意事项？",
            "如何预防心脏病？",
            "感冒和流感的区别是什么？"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(question, key=f"example_{i}"):
                # 设置会话状态来处理示例问题
                st.session_state.example_question = question
        
        st.divider()
        
        # 统计信息
        st.subheader("📊 会话统计")
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        assistant_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        
        st.metric("用户消息", user_messages)
        st.metric("AI回复", assistant_messages)
    
    # 处理示例问题
    if hasattr(st.session_state, 'example_question'):
        prompt = st.session_state.example_question
        del st.session_state.example_question
        
        # 处理问题的逻辑
        process_user_question(prompt, model_name, st.session_state.temperature, st.session_state.max_tokens)
    
    # 用户输入（移到columns外面）
    if prompt := st.chat_input("请输入您的医疗相关问题..."):
        process_user_question(prompt, model_name, st.session_state.temperature, st.session_state.max_tokens)

def process_user_question(prompt: str, model_name: str, temperature: float, max_tokens: int):
    """处理用户问题的统一函数 - 优化用户体验"""
    # 立即添加用户消息到历史并显示
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 立即显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 立即显示AI思考状态
    with st.chat_message("assistant"):
        # 创建一个容器用于自动滚动
        question_container = st.container()
        with question_container:
            # 添加一个空的占位符，用于JavaScript滚动定位
            st.markdown('<div id="question-anchor"></div>', unsafe_allow_html=True)
        
        # 创建占位符用于流式更新
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("🤖 **思考中...**")
        
        # 检查Ollama连接（静默检查）
        service_status = check_service_status()
        if not service_status.get("Ollama", False):
            thinking_placeholder.error("❌ Ollama服务未连接，请确保Ollama正在运行")
            return
        
        # 初始化组件
        llm = initialize_llm()
        llm.model = model_name
        retriever = initialize_retriever()  # 使用缓存的初始化函数
        
        # 更新思考状态
        thinking_placeholder.markdown("🔍 **正在检索相关知识...**")
        
        # 执行检索
        try:
            retrieval_results = retriever.search_knowledge(prompt)
            
            # 更新思考状态
            thinking_placeholder.markdown("🧠 **正在分析检索结果...**")
            
            # 创建完整响应的占位符
            response_placeholder = st.empty()
            
            # 构建完整响应内容
            with response_placeholder.container():
                # 显示检索过程
                display_retrieval_process(retrieval_results)
                
                # 显示详细结果
                display_detailed_results(retrieval_results)
                
                # 显示知识来源
                display_knowledge_sources(retrieval_results)
                
                # 更新思考状态为生成回答
                thinking_placeholder.markdown("✍️ **正在生成专业回答...**")
                
                # 生成回答
                if isinstance(retriever, (SimpleRAGSystem, RealRAGSystem)) and hasattr(retriever, 'generate_answer'):
                    # 简化或真实RAG系统
                    response = retriever.generate_answer(prompt, retrieval_results)
                else:
                    # 模拟RAG系统 - 使用简单LLM
                    retrieval_context = format_retrieval_context(retrieval_results)
                    medical_prompt = create_medical_prompt(prompt, retrieval_context)
                    response = llm.generate_response(medical_prompt, temperature, max_tokens)
                
                # 清除思考状态，显示最终回答
                thinking_placeholder.empty()
                
                # 显示AI回答
                st.markdown("### 🤖 AI回答")
                
                # 模拟流式输出效果
                answer_placeholder = st.empty()
                
                # 分段显示回答（模拟流式效果）
                words = response.split()
                displayed_text = ""
                
                for i, word in enumerate(words):
                    displayed_text += word + " "
                    if i % 3 == 0:  # 每3个词更新一次
                        answer_placeholder.markdown(displayed_text)
                        time.sleep(0.05)  # 短暂延迟模拟流式效果
                
                # 显示完整回答
                answer_placeholder.markdown(response)
                
                # 构建完整的历史记录
                search_results = retrieval_results.get('search_results', {})
                graph_search = search_results.get('graph_search', {})
                vector_search = search_results.get('vector_search', {})
                
                full_response = f"""### 🔍 检索结果
- 向量检索: {len(vector_search.get('entities', []))} 个实体
- 图谱检索: {len(graph_search.get('nodes', []))} 个节点
- 检索耗时: {retrieval_results.get('retrieval_time', 'N/A')}s

### 🤖 AI回答
{response}
"""
                
                # 添加助手消息到历史
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # 添加JavaScript自动滚动到问题位置
                st.markdown("""
                <script>
                setTimeout(function() {
                    const anchor = document.getElementById('question-anchor');
                    if (anchor) {
                        anchor.scrollIntoView({
                            behavior: 'smooth',
                            block: 'start'
                        });
                    }
                }, 100);
                </script>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            thinking_placeholder.error(f"❌ 处理过程中出现错误: {str(e)}")
            logger.error(f"处理用户问题时出错: {e}")

if __name__ == "__main__":
    main()