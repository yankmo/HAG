import streamlit as st
import sys
import os
import logging
from typing import List, Dict, Any
import time

# 添加项目路径到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入现有的RAG系统组件
from src.knowledge.modular_rag_system import ModularRAGSystem
from main import OllamaLLM, RAGRetriever

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    .retrieval-info {
        background-color: #f3e5f5;
        border: 1px solid #9c27b0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """初始化RAG系统（使用缓存避免重复初始化）"""
    try:
        with st.spinner("🔄 正在初始化RAG系统..."):
            # 初始化ModularRAGSystem
            rag_system = ModularRAGSystem()
            
            # 初始化LLM
            llm = OllamaLLM(
                model="gemma3:4b",
                base_url="http://localhost:11434"
            )
            
            # 初始化检索器
            retriever = RAGRetriever(rag_system=rag_system)
            
            return rag_system, llm, retriever
    except Exception as e:
        st.error(f"❌ RAG系统初始化失败: {str(e)}")
        return None, None, None

def display_system_status(rag_system):
    """显示系统状态"""
    if rag_system is None:
        st.error("❌ 系统未初始化")
        return
    
    try:
        # 获取存储统计信息
        stats = rag_system.get_storage_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Neo4j 节点",
                value=stats.get('neo4j_nodes', 0),
                delta=None
            )
        
        with col2:
            st.metric(
                label="Neo4j 关系",
                value=stats.get('neo4j_relationships', 0),
                delta=None
            )
        
        with col3:
            st.metric(
                label="向量实体",
                value=stats.get('vector_entities', 0),
                delta=None
            )
        
        with col4:
            st.metric(
                label="向量关系",
                value=stats.get('vector_relations', 0),
                delta=None
            )
            
    except Exception as e:
        st.error(f"❌ 获取系统状态失败: {str(e)}")

def format_retrieval_results(results: Dict[str, Any]) -> str:
    """格式化检索结果"""
    if not results:
        return "未找到相关信息"
    
    formatted = []
    
    # 向量搜索结果
    if 'vector_results' in results:
        vector_results = results['vector_results']
        if vector_results.get('entities') or vector_results.get('relations'):
            formatted.append("🔍 **向量搜索结果:**")
            if vector_results.get('entities'):
                formatted.append(f"  - 实体数量: {len(vector_results['entities'])}")
                for i, entity in enumerate(vector_results['entities'][:3], 1):
                    name = entity.get('name', '未知')
                    score = entity.get('score', 0)
                    formatted.append(f"  - 实体{i}: {name} (相似度: {score:.3f})")
            
            if vector_results.get('relations'):
                formatted.append(f"  - 关系数量: {len(vector_results['relations'])}")
    
    # 图谱搜索结果
    if 'graph_results' in results:
        graph_results = results['graph_results']
        if graph_results.get('entities') or graph_results.get('relations'):
            formatted.append("\n🕸️ **图谱搜索结果:**")
            if graph_results.get('entities'):
                formatted.append(f"  - 实体数量: {len(graph_results['entities'])}")
            if graph_results.get('relations'):
                formatted.append(f"  - 关系数量: {len(graph_results['relations'])}")
    
    # 混合搜索结果
    if 'hybrid_results' in results:
        hybrid_results = results['hybrid_results']
        if hybrid_results.get('entities') or hybrid_results.get('relations'):
            formatted.append("\n🔄 **混合搜索结果:**")
            if hybrid_results.get('entities'):
                formatted.append(f"  - 实体数量: {len(hybrid_results['entities'])}")
            if hybrid_results.get('relations'):
                formatted.append(f"  - 关系数量: {len(hybrid_results['relations'])}")
    
    return "\n".join(formatted) if formatted else "未找到相关信息"

def process_query(query: str, rag_system, llm, retriever):
    """处理用户查询"""
    try:
        # 执行检索
        with st.spinner("🔍 正在检索相关信息..."):
            retrieval_results = retriever.get_relevant_documents(query)
        
        # 显示检索详情
        if retrieval_results:
            with st.expander("📊 检索详情", expanded=False):
                # 假设retriever返回的是Document对象列表
                if hasattr(retrieval_results[0], 'metadata') and 'retrieval_details' in retrieval_results[0].metadata:
                    details = retrieval_results[0].metadata['retrieval_details']
                    st.markdown(f"```\n{format_retrieval_results(details)}\n```")
                else:
                    st.write(f"检索到 {len(retrieval_results)} 个相关文档")
        
        # 构建上下文
        context_parts = []
        for doc in retrieval_results:
            context_parts.append(doc.page_content)
        context = "\n".join(context_parts)
        
        # 生成回答
        with st.spinner("🤖 正在生成回答..."):
            prompt = f"""基于以下医疗知识信息，请回答用户的问题。请确保回答准确、专业且易于理解。

相关知识信息：
{context}

用户问题：{query}

请提供详细的回答："""
            
            response = llm._call(prompt)
        
        return response, retrieval_results
        
    except Exception as e:
        st.error(f"❌ 处理查询时出错: {str(e)}")
        return None, None

def main():
    """主函数"""
    # 页面标题
    st.markdown('<h1 class="main-header">🏥 医疗知识RAG系统</h1>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统设置")
        
        # 初始化系统
        if st.button("🔄 重新初始化系统"):
            st.cache_resource.clear()
            st.rerun()
        
        st.divider()
        
        # 系统信息
        st.header("📊 系统状态")
        rag_system, llm, retriever = initialize_rag_system()
        
        if rag_system is not None:
            st.success("✅ 系统已就绪")
            display_system_status(rag_system)
        else:
            st.error("❌ 系统初始化失败")
            st.stop()
        
        st.divider()
        
        # 使用说明
        st.header("📖 使用说明")
        st.markdown("""
        1. 在下方输入框中输入您的医疗相关问题
        2. 系统将自动检索相关知识
        3. 基于检索结果生成专业回答
        4. 可查看检索详情了解信息来源
        """)
    
    # 主内容区域
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 智能问答")
        
        # 初始化聊天历史
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # 显示聊天历史
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # 显示检索信息（如果有）
                if "retrieval_info" in message:
                    with st.expander("检索详情", expanded=False):
                        st.text(message["retrieval_info"])
        
        # 用户输入
        if prompt := st.chat_input("请输入您的医疗相关问题..."):
            # 添加用户消息到历史
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # 显示用户消息
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 处理查询并生成回答
            with st.chat_message("assistant"):
                response, retrieval_results = process_query(prompt, rag_system, llm, retriever)
                
                if response:
                    st.markdown(response)
                    
                    # 添加助手消息到历史
                    message_data = {"role": "assistant", "content": response}
                    
                    # 添加检索信息
                    if retrieval_results:
                        retrieval_info = f"检索到 {len(retrieval_results)} 个相关文档"
                        message_data["retrieval_info"] = retrieval_info
                    
                    st.session_state.messages.append(message_data)
                else:
                    st.error("抱歉，处理您的问题时出现了错误。")
    
    with col2:
        st.header("🔧 功能面板")
        
        # 清空聊天历史
        if st.button("🗑️ 清空聊天记录"):
            st.session_state.messages = []
            st.rerun()
        
        # 示例问题
        st.subheader("💡 示例问题")
        example_questions = [
            "帕金森病的主要症状有哪些？",
            "帕金森病如何诊断？",
            "帕金森病有哪些治疗方法？",
            "帕金森病的病因是什么？",
            "如何预防帕金森病？"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{question}"):
                # 将示例问题添加到输入框
                st.session_state.example_question = question
                st.rerun()
        
        # 系统信息
        st.subheader("ℹ️ 系统信息")
        st.info("""
        **模型信息:**
        - 生成模型: gemma3:4b
        - 向量模型: bge-m3:latest
        - 图数据库: Neo4j
        - 向量数据库: Weaviate
        """)

if __name__ == "__main__":
    main()