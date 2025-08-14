#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAG FastAPI 后端包装器
为前端提供RESTful API接口
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
import logging
import json
import asyncio
import os
import uuid
from datetime import datetime
from collections import defaultdict

# 导入现有的HAG API
from api import HAGIntegratedAPI, IntegratedResponse

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="HAG API", description="Hybrid Augmented Generation API", version="1.0.0")

# 配置CORS - 从环境变量获取允许的来源
allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    include_history: bool = True

class RetrievalStep(BaseModel):
    step_name: str
    step_description: str
    start_time: float
    end_time: float
    duration: float
    status: str
    result_count: int
    details: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    sources: Dict[str, Any]
    metadata: Dict[str, Any]
    retrieval_process: List[RetrievalStep] = []

# 全局HAG API实例
hag_api: Optional[HAGIntegratedAPI] = None

# 会话管理
class SessionManager:
    def __init__(self, max_history_length: int = 10):
        self.sessions = defaultdict(list)  # session_id -> list of messages
        self.max_history_length = max_history_length
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """获取或创建会话ID"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str, sources: Optional[Dict] = None):
        """添加消息到会话历史"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'sources': sources
        }
        self.sessions[session_id].append(message)
        
        # 保持历史记录长度限制
        if len(self.sessions[session_id]) > self.max_history_length * 2:  # *2 因为包含用户和助手消息
            self.sessions[session_id] = self.sessions[session_id][-self.max_history_length * 2:]
    
    def get_history(self, session_id: str, include_sources: bool = False) -> List[Dict]:
        """获取会话历史"""
        history = self.sessions.get(session_id, [])
        if not include_sources:
            return [{'role': msg['role'], 'content': msg['content']} for msg in history]
        return history
    
    def get_context_for_query(self, session_id: str, current_question: str) -> str:
        """为当前查询构建上下文"""
        history = self.get_history(session_id)
        if not history:
            return current_question
        
        # 构建包含历史对话的上下文
        context_parts = []
        
        # 添加最近的对话历史（最多3轮）
        recent_history = history[-6:] if len(history) > 6 else history
        if recent_history:
            context_parts.append("对话历史：")
            for msg in recent_history:
                role_name = "用户" if msg['role'] == 'user' else "助手"
                context_parts.append(f"{role_name}: {msg['content']}")
            context_parts.append("")
        
        context_parts.append(f"当前问题: {current_question}")
        
        return "\n".join(context_parts)

# 全局会话管理器
session_manager = SessionManager()

@app.on_event("startup")
async def startup_event():
    """启动时初始化HAG API"""
    global hag_api
    try:
        logger.info("初始化HAG API...")
        hag_api = HAGIntegratedAPI()
        logger.info("HAG API初始化完成")
    except Exception as e:
        logger.error(f"HAG API初始化失败: {e}")
        raise

@app.get("/")
async def root():
    """根路径"""
    return {"message": "HAG API is running"}

@app.get("/health")
async def health_check():
    """健康检查"""
    if hag_api is None:
        raise HTTPException(status_code=503, detail="HAG API not initialized")
    
    try:
        status = hag_api.get_system_status()
        return {"status": "healthy", "system_status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_knowledge(request: QueryRequest):
    """知识查询接口"""
    if hag_api is None:
        raise HTTPException(status_code=503, detail="HAG API not initialized")
    
    try:
        logger.info(f"处理查询: {request.question}")
        result = hag_api.query(request.question)
        
        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            metadata=result.metadata,
            retrieval_process=[
                RetrievalStep(
                    step_name=step.step_name,
                    step_description=step.step_description,
                    start_time=step.start_time,
                    end_time=step.end_time,
                    duration=step.duration,
                    status=step.status,
                    result_count=step.result_count,
                    details=step.details
                ) for step in result.retrieval_process
            ]
        )
    except Exception as e:
        logger.error(f"查询处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/query/stream")
async def query_knowledge_stream(request: QueryRequest):
    """流式知识查询接口 - 集成图谱检索功能和上下文记忆"""
    if hag_api is None:
        raise HTTPException(status_code=503, detail="HAG API not initialized")
    
    async def generate_stream():
        try:
            # 获取或创建会话ID
            session_id = session_manager.get_or_create_session(request.session_id)
            
            logger.info(f"处理流式查询 [会话: {session_id}]: {request.question}")
            
            # 发送会话ID
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id}, ensure_ascii=False)}\n\n"
            
            # 发送开始信号
            yield f"data: {json.dumps({'type': 'start', 'message': '开始处理您的问题...'}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.1)
            
            # 发送检索状态
            yield f"data: {json.dumps({'type': 'status', 'message': '正在检索相关信息...'}, ensure_ascii=False)}\n\n"
            
            # 1. 文档检索 - 直接使用用户问题
            yield f"data: {json.dumps({'type': 'status', 'message': '正在检索相关文档...'}, ensure_ascii=False)}\n\n"
            retrieval_result = hag_api.retrieval_service.search_hybrid(request.question, limit=5)
            
            # 2. 图谱实体检索
            yield f"data: {json.dumps({'type': 'status', 'message': '正在检索相关实体...'}, ensure_ascii=False)}\n\n"
            entities = hag_api.graph_service.search_entities_by_name(request.question, limit=3)
            
            # 3. 图谱关系检索
            yield f"data: {json.dumps({'type': 'status', 'message': '正在检索相关关系...'}, ensure_ascii=False)}\n\n"
            relationships = hag_api.graph_service.search_relationships_by_query(request.question, limit=10)
            
            # 构建完整的上下文（包含文档、实体、关系）
            context = ""
            
            # 添加文档信息
            if retrieval_result.hybrid_results:
                context += "相关文档：\n"
                for doc in retrieval_result.hybrid_results[:3]:
                    context += f"- {doc.content[:200]}...\n"
                context += "\n"
            
            # 添加实体信息
            if entities:
                context += "相关实体：\n"
                for entity in entities[:3]:
                    name = entity.get('name', '')
                    entity_type = entity.get('type', '')
                    description = entity.get('description', '')
                    context += f"- {name} ({entity_type}): {description}\n"
                context += "\n"
            
            # 添加关系信息
            if relationships:
                context += "相关关系：\n"
                for rel in relationships[:5]:
                    source = rel.get('source', '')
                    target = rel.get('target', '')
                    rel_type = rel.get('type', '')
                    description = rel.get('description', '')
                    score = rel.get('relevance_score', 0)
                    context += f"- {source} --[{rel_type}]--> {target}: {description} (相关性: {score:.2f})\n"
                context += "\n"
            
            # 发送生成状态
            yield f"data: {json.dumps({'type': 'status', 'message': '正在生成回答...'}, ensure_ascii=False)}\n\n"
            
            # 使用ollama进行真正的流式生成
            import requests
            from config import get_config
            
            config = get_config()
            
            # 构建messages数组
            messages = []
            
            # 添加系统消息（包含检索结果）
            if context.strip():
                messages.append({
                    "role": "system",
                    "content": f"基于以下知识库信息回答用户问题：\n\n{context}\n\n请基于以上文档、实体和关系信息提供准确、简短的回答。如果知识库中没有相关信息，请说明无法找到相关信息。"
                })
            
            # 添加对话历史
            if request.include_history and session_manager.get_history(session_id):
                history = session_manager.get_history(session_id)
                for msg in history:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
            
            # 添加当前用户问题
            messages.append({
                "role": "user",
                "content": request.question
            })
            
            # 调用ollama的chat API
            response = requests.post(
                f"{config.ollama.base_url}/api/chat",
                json={
                    "model": config.ollama.default_model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 2000,
                    }
                },
                stream=True,
                timeout=config.ollama.timeout
            )
            
            if response.status_code == 200:
                current_text = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            # /api/chat接口返回的是message.content而不是response
                            if 'message' in chunk and 'content' in chunk['message']:
                                current_text += chunk['message']['content']
                                yield f"data: {json.dumps({'type': 'content', 'content': current_text}, ensure_ascii=False)}\n\n"
                                
                            if chunk.get('done', False):
                                break
                                
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.error(f"处理流式响应块失败: {e}")
                            continue
                
                # 保存用户问题和AI回答到会话历史
                session_manager.add_message(session_id, 'user', request.question)
                session_manager.add_message(session_id, 'assistant', current_text)
                
            else:
                # 如果流式请求失败，回退到普通模式
                logger.warning("流式请求失败，回退到普通模式")
                result = hag_api.query(request.question)
                current_text = result.answer
                yield f"data: {json.dumps({'type': 'content', 'content': current_text}, ensure_ascii=False)}\n\n"
                retrieval_result = {'sources': result.sources}
                
                # 保存用户问题和AI回答到会话历史
                session_manager.add_message(session_id, 'user', request.question)
                session_manager.add_message(session_id, 'assistant', current_text)
            
            # 检查是否有检索到的内容，只有在有内容时才发送来源信息
            has_documents = hasattr(retrieval_result, 'hybrid_results') and retrieval_result.hybrid_results
            has_entities = entities and len(entities) > 0
            has_relationships = relationships and len(relationships) > 0
            
            # 只有当检索到相关信息时才发送参考资料
            if has_documents or has_entities or has_relationships:
                sources = {
                    'documents': [],
                    'entities': [],
                    'relationships': [],
                    'statistics': {}
                }
                
                # 添加文档来源
                if has_documents:
                    sources['documents'] = [
                        {
                            'content': doc.content[:300] + "..." if len(doc.content) > 300 else doc.content,
                            'score': doc.score,
                            'metadata': doc.metadata
                        } for doc in retrieval_result.hybrid_results[:3]
                    ]
                    sources['statistics'] = retrieval_result.statistics if hasattr(retrieval_result, 'statistics') else {}
                
                # 添加实体来源
                if has_entities:
                    sources['entities'] = [
                        {
                            'name': entity.get('name', ''),
                            'type': entity.get('type', ''),
                            'description': entity.get('description', '')
                        } for entity in entities[:3]
                    ]
                
                # 添加关系来源
                if has_relationships:
                    sources['relationships'] = [
                        {
                            'source': rel.get('source', ''),
                            'target': rel.get('target', ''),
                            'type': rel.get('type', ''),
                            'description': rel.get('description', ''),
                            'relevance_score': rel.get('relevance_score', 0)
                        } for rel in relationships[:5]
                    ]
                
                # 保存来源信息到会话历史
                session_manager.sessions[session_id][-1]['sources'] = sources
                
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"
            
            # 发送完成信号
            yield f"data: {json.dumps({'type': 'done', 'message': '回答完成'}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            logger.error(f"流式查询处理失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': f'处理失败: {str(e)}'}, ensure_ascii=False)}\n\n"
            # 即使出错也要发送完成信号
            yield f"data: {json.dumps({'type': 'done', 'message': '处理完成'}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str, include_sources: bool = False):
    """获取会话历史"""
    try:
        history = session_manager.get_history(session_id, include_sources=include_sources)
        return {
            "session_id": session_id,
            "history": history,
            "total_messages": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session history: {str(e)}")

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """清除会话历史"""
    try:
        if session_id in session_manager.sessions:
            del session_manager.sessions[session_id]
            return {"message": f"Session {session_id} cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")

@app.get("/status")
async def get_system_status():
    """获取系统状态"""
    if hag_api is None:
        raise HTTPException(status_code=503, detail="HAG API not initialized")
    
    try:
        return hag_api.get_system_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "backend_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )