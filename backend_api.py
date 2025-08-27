#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAG FastAPI 后端包装器
为前端提供RESTful API接口
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
import logging
import asyncio
from datetime import datetime
import os
import uuid
from collections import defaultdict
import tempfile
from pathlib import Path
import json

# 导入现有的HAG API
from api import HAGIntegratedAPI

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="HAG API", description="Hybrid Augmented Generation API", version="1.0.0")

# 配置CORS - 从环境变量获取允许的来源
allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
# 添加通配符支持以确保开发环境正常工作
if "*" not in allowed_origins:
    allowed_origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
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

# 存储功能相关模型
class DocumentUploadResponse(BaseModel):
    task_id: str
    filename: str
    file_size: int
    status: str
    message: str

class ProcessingProgress(BaseModel):
    task_id: str
    status: str  # 'uploading', 'parsing', 'extracting', 'storing_neo4j', 'storing_weaviate', 'completed', 'failed'
    progress: int  # 0-100
    current_stage: str
    message: str
    details: Dict[str, Any] = {}

class StorageStats(BaseModel):
    neo4j_stats: Dict[str, int]
    weaviate_stats: Dict[str, Any]  # 改为Any类型以支持浮点数avg_similarity
    total_documents: int
    last_updated: str

class SearchTestRequest(BaseModel):
    query: str
    search_type: str = "both"  # 'neo4j', 'weaviate', 'vectorized_graph', 'both', 'all'

class SearchTestResponse(BaseModel):
    neo4j_results: Dict[str, Any] = {}
    weaviate_results: Dict[str, Any] = {}
    vectorized_graph_results: Dict[str, Any] = {}
    query: str
    search_type: str

class ProcessingTask(BaseModel):
    task_id: str
    filename: str
    status: str
    progress: float
    start_time: str
    end_time: Optional[str] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

class CacheStats(BaseModel):
    cache_type: str
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    redis_connected: bool
    redis_info: Dict[str, Any]
    local_cache_size: int
    remote_cache_size: int
    last_updated: str

class CacheHealthResponse(BaseModel):
    redis_connected: bool
    redis_ping_time_ms: float
    redis_info: Dict[str, Any]
    cache_manager_status: str
    last_check: str

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

# 文档处理任务管理
class DocumentProcessor:
    def __init__(self):
        self.tasks = {}  # task_id -> ProcessingProgress
        self.supported_formats = {'.txt', '.pdf', '.docx', '.doc'}
        # 初始化向量化数据导入器
        try:
            from src.knowledge.vectorized_data_importer import VectorizedDataImporter
            self.vectorized_importer = VectorizedDataImporter()
            logger.info("VectorizedDataImporter初始化成功")
        except Exception as e:
            logger.error(f"VectorizedDataImporter初始化失败: {e}")
            self.vectorized_importer = None
    
    def create_task(self, filename: str, file_size: int) -> str:
        """创建新的处理任务"""
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = ProcessingProgress(
            task_id=task_id,
            status='uploading',
            progress=0,
            current_stage='文件上传中',
            message='正在上传文件...',
            details={'filename': filename, 'file_size': file_size}
        )
        return task_id
    
    def update_progress(self, task_id: str, status: str, progress: int, stage: str, message: str, details: Dict = None):
        """更新任务进度"""
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            self.tasks[task_id].progress = progress
            self.tasks[task_id].current_stage = stage
            self.tasks[task_id].message = message
            if details:
                self.tasks[task_id].details.update(details)
    
    def get_progress(self, task_id: str) -> Optional[ProcessingProgress]:
        """获取任务进度"""
        return self.tasks.get(task_id)
    
    def is_supported_format(self, filename: str) -> bool:
        """检查文件格式是否支持"""
        return Path(filename).suffix.lower() in self.supported_formats
    
    async def process_document(self, task_id: str, file_path: str, filename: str):
        """处理文档的后台任务"""
        import time
        import os
        start_time = time.time()
        
        try:
            # 1. 预检查阶段
            self.update_progress(task_id, 'validating', 5, '验证文档', '正在验证文档格式和大小...')
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise Exception(f"文件不存在: {file_path}")
            
            # 检查文件大小（限制为10MB）
            file_size = os.path.getsize(file_path)
            max_size = 10 * 1024 * 1024  # 10MB
            if file_size > max_size:
                raise Exception(f"文件过大: {file_size / 1024 / 1024:.2f}MB，最大支持10MB")
            
            # 检查文件扩展名
            allowed_extensions = {'.txt', '.pdf', '.docx', '.doc'}
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in allowed_extensions:
                raise Exception(f"不支持的文件格式: {file_ext}，支持的格式: {', '.join(allowed_extensions)}")
            
            # 检查向量化导入器是否可用
            if not self.vectorized_importer:
                raise Exception("VectorizedDataImporter未初始化，请检查系统配置")
            
            # 2. 文档解析阶段
            self.update_progress(task_id, 'parsing', 10, '文档解析中', '正在读取文档内容...')
            
            # 读取文档内容（仅支持文本文件）
            if file_ext == '.txt':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    logger.info(f"成功读取文档 [{filename}]: {len(content)} 字符")
                except UnicodeDecodeError:
                    try:
                        # 尝试其他编码
                        with open(file_path, 'r', encoding='gbk') as f:
                            content = f.read()
                        logger.info(f"使用GBK编码读取文档 [{filename}]: {len(content)} 字符")
                    except UnicodeDecodeError:
                        # 最后尝试latin-1编码
                        with open(file_path, 'r', encoding='latin-1') as f:
                            content = f.read()
                        logger.info(f"使用Latin-1编码读取文档 [{filename}]: {len(content)} 字符")
            else:
                # 对于非文本文件，暂时抛出异常
                raise Exception(f"暂不支持 {file_ext} 格式的文档解析，请转换为TXT格式")
            
            # 验证内容
            if not content or not content.strip():
                raise Exception("文档内容为空或仅包含空白字符")
            
            # 检查内容长度
            if len(content) < 10:
                raise Exception("文档内容过短，至少需要10个字符")
            
            if len(content) > 1000000:  # 1MB文本
                logger.warning(f"文档内容较大: {len(content)} 字符，处理可能需要较长时间")
            
            self.update_progress(task_id, 'parsing', 20, '文档解析完成', f'文档大小: {len(content)} 字符')
            
            # 3. 实体关系提取阶段
            self.update_progress(task_id, 'extracting', 30, '实体关系提取中', '正在使用AI模型提取实体和关系...')
             
            # 使用VectorizedDataImporter处理文档
            # 注意：process_and_vectorize_file是同步方法，需要在线程池中运行
            import asyncio
            import concurrent.futures
             
            try:
                # 在线程池中运行同步的文档处理，设置超时时间
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # 设置超时时间为5分钟
                    stats = await asyncio.wait_for(
                        loop.run_in_executor(
                            executor, 
                            self.vectorized_importer.process_and_vectorize_file, 
                            file_path
                        ),
                        timeout=300  # 5分钟超时
                    )
                 
                # 验证处理结果
                if not stats:
                    raise Exception("文档处理返回空结果")
                 
                logger.info(f"文档处理统计 [{task_id}]: 实体={stats.total_entities}, 关系={stats.total_relations}, 向量化节点={stats.vectorized_nodes}")
                 
            except asyncio.TimeoutError:
                raise Exception("文档处理超时（5分钟），请尝试处理较小的文档")
            except AttributeError as e:
                raise Exception(f"VectorizedDataImporter接口错误: {str(e)}")
            except Exception as e:
                if "Neo4j" in str(e):
                    raise Exception(f"Neo4j数据库连接或操作失败: {str(e)}")
                elif "Weaviate" in str(e):
                    raise Exception(f"Weaviate向量数据库操作失败: {str(e)}")
                elif "Ollama" in str(e):
                    raise Exception(f"AI模型服务不可用: {str(e)}")
                else:
                    raise Exception(f"文档处理过程中发生错误: {str(e)}")
             
            # 4. 更新进度 - Neo4j存储
            self.update_progress(task_id, 'storing_neo4j', 60, 'Neo4j存储完成', 
                               f'已存储 {stats.total_entities} 个实体和 {stats.total_relations} 个关系')
            
            # 5. 更新进度 - Weaviate向量化
            self.update_progress(task_id, 'storing_weaviate', 80, 'Weaviate向量化完成', 
                               f'已向量化 {stats.vectorized_nodes} 个节点和 {stats.vectorized_relations} 个关系')
            
            # 6. 处理完成
            processing_time = time.time() - start_time
            self.update_progress(
                task_id, 'completed', 100, '处理完成', '文档处理完成',
                {
                    'filename': filename,
                    'neo4j_entities': stats.total_entities,
                    'neo4j_relationships': stats.total_relations,
                    'weaviate_nodes': stats.vectorized_nodes,
                    'weaviate_relations': stats.vectorized_relations,
                    'failed_nodes': stats.failed_nodes,
                    'failed_relations': stats.failed_relations,
                    'processing_time': f'{processing_time:.2f}秒',
                    'content_length': len(content)
                }
            )
            
            logger.info(f"文档处理完成 [{task_id}]: {filename}, 耗时 {processing_time:.2f}秒")
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"文档处理失败 [{task_id}]: {error_msg}")
            self.update_progress(
                task_id, 'failed', 0, '处理失败', 
                f'处理失败: {error_msg}',
                {
                    'filename': filename,
                    'error': error_msg,
                    'processing_time': f'{processing_time:.2f}秒'
                }
            )
        
        finally:
            # 清理临时文件
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"已清理临时文件: {file_path}")
            except Exception as e:
                logger.warning(f"清理临时文件失败: {e}")

# 全局文档处理器
doc_processor = DocumentProcessor()

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
                                # 只发送新增的内容片段，而不是累积的全部内容
                                new_content = chunk['message']['content']
                                current_text += new_content
                                # 只发送增量内容，并立即刷新缓冲区
                                yield f"data: {json.dumps({'type': 'content', 'content': new_content}, ensure_ascii=False)}\n\n"
                                
                            if chunk.get('done', False):
                                break
                                
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.error(f"处理流式响应块失败: {e}")
                            continue
                
            else:
                # 如果流式请求失败，回退到普通模式
                logger.warning("流式请求失败，回退到普通模式")
                result = hag_api.query(request.question)
                current_text = result.answer
                yield f"data: {json.dumps({'type': 'content', 'content': current_text}, ensure_ascii=False)}\n\n"
                retrieval_result = {'sources': result.sources}
            
            # 保存用户问题和AI回答到会话历史（统一在这里处理）
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

# ==================== 存储功能API接口 ====================

@app.post("/storage/upload", response_model=DocumentUploadResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """文档上传接口"""
    try:
        # 检查文件格式
        if not doc_processor.is_supported_format(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件格式。支持的格式: {', '.join(doc_processor.supported_formats)}"
            )
        
        # 检查文件大小 (限制为10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        file_size = 0
        
        # 创建临时文件
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_file_path, "wb") as buffer:
            while chunk := await file.read(1024):
                file_size += len(chunk)
                if file_size > max_size:
                    os.remove(temp_file_path)
                    os.rmdir(temp_dir)
                    raise HTTPException(status_code=400, detail="文件大小超过10MB限制")
                buffer.write(chunk)
        
        # 创建处理任务
        task_id = doc_processor.create_task(file.filename, file_size)
        
        # 启动后台处理任务
        background_tasks.add_task(doc_processor.process_document, task_id, temp_file_path, file.filename)
        
        logger.info(f"文档上传成功 [{task_id}]: {file.filename} ({file_size} bytes)")
        
        return DocumentUploadResponse(
            task_id=task_id,
            filename=file.filename,
            file_size=file_size,
            status="uploading",
            message="文件上传成功，开始处理"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

@app.get("/storage/progress/{task_id}", response_model=ProcessingProgress)
async def get_processing_progress(task_id: str):
    """获取文档处理进度"""
    progress = doc_processor.get_progress(task_id)
    if not progress:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return progress

@app.get("/storage/stats", response_model=StorageStats)
async def get_storage_stats():
    """获取存储统计信息"""
    try:
        # 模拟统计数据 - 实际应用中应该从Neo4j和Weaviate获取真实数据
        neo4j_stats = {
            "entities": 1247,
            "relationships": 856,
            "documents": 23,
            "entity_types": 8
        }
        
        weaviate_stats = {
            "vectors": 1892,
            "documents": 23,
            "collections": 3,
            "avg_similarity": 0.85
        }
        
        return StorageStats(
            neo4j_stats=neo4j_stats,
            weaviate_stats=weaviate_stats,
            total_documents=23,
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"获取存储统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

@app.post("/storage/search/test", response_model=SearchTestResponse)
async def test_search(request: SearchTestRequest):
    """检索测试接口 - 支持Neo4j、Weaviate和向量化图谱检索"""
    if hag_api is None:
        raise HTTPException(status_code=503, detail="HAG API not initialized")
    
    try:
        logger.info(f"执行检索测试: {request.query} (类型: {request.search_type})")
        
        neo4j_results = {}
        weaviate_results = {}
        vectorized_graph_results = {}
        
        # Neo4j检索
        if request.search_type in ['neo4j', 'both', 'all']:
            try:
                entities = hag_api.graph_service.search_entities_by_name(request.query, limit=5)
                relationships = hag_api.graph_service.search_relationships_by_query(request.query, limit=8)
                
                neo4j_results = {
                    "entities": [
                        {
                            "name": entity.get('name', ''),
                            "type": entity.get('type', ''),
                            "description": entity.get('description', '')
                        } for entity in entities[:5]
                    ],
                    "relationships": [
                        {
                            "source": rel.get('source', ''),
                            "target": rel.get('target', ''),
                            "type": rel.get('type', ''),
                            "description": rel.get('description', ''),
                            "relevance_score": rel.get('relevance_score', 0)
                        } for rel in relationships[:8]
                    ],
                    "total_entities": len(entities),
                    "total_relationships": len(relationships)
                }
            except Exception as e:
                logger.warning(f"Neo4j检索失败: {e}")
                neo4j_results = {"error": f"Neo4j检索失败: {str(e)}"}
        
        # Weaviate检索
        if request.search_type in ['weaviate', 'both', 'all']:
            try:
                retrieval_result = hag_api.retrieval_service.search_hybrid(request.query, limit=5)
                
                weaviate_results = {
                    "documents": [
                        {
                            "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                            "score": doc.score,
                            "metadata": doc.metadata
                        } for doc in retrieval_result.hybrid_results[:5]
                    ],
                    "total_documents": len(retrieval_result.hybrid_results),
                    "statistics": retrieval_result.statistics if hasattr(retrieval_result, 'statistics') else {}
                }
            except Exception as e:
                logger.warning(f"Weaviate检索失败: {e}")
                weaviate_results = {"error": f"Weaviate检索失败: {str(e)}"}
        
        # 向量化图谱检索
        if request.search_type in ['vectorized_graph', 'both', 'all']:
            try:
                # 使用向量化图谱检索服务
                graph_result = hag_api.graph_service.search_by_query(request.query)
                
                vectorized_graph_results = {
                    "nodes": [
                        {
                            "name": node.get('name'),
                            "type": node.get('type'),
                            "description": node.get('description'),
                            "score": node.get('similarity', 0.0),
                            "properties": node.get('properties', {})
                        } for node in graph_result.nodes[:5]
                    ],
                    "relationships": [
                        {
                            "source": rel.get('source_node'),
                            "target": rel.get('target_node'),
                            "type": rel.get('relation_type'),
                            "description": rel.get('description'),
                            "score": rel.get('similarity', 0.0),
                            "properties": rel.get('properties', {})
                        } for rel in graph_result.relations[:8]
                    ],
                    "total_nodes": len(graph_result.nodes),
                    "total_relationships": len(graph_result.relations),
                    "search_metadata": {
                        "search_time_seconds": graph_result.search_time,
                        "total_results": graph_result.total_results,
                        "query_vector_length": len(graph_result.query_vector) if graph_result.query_vector else 0
                    }
                }
                
                logger.info(f"向量化图谱检索完成: 找到 {len(graph_result.nodes)} 个节点, {len(graph_result.relations)} 个关系")
                
            except Exception as e:
                logger.warning(f"向量化图谱检索失败: {e}")
                vectorized_graph_results = {"error": f"向量化图谱检索失败: {str(e)}"}
        
        # 构建响应，根据search_type决定返回哪些结果
        response_data = {
            "query": request.query,
            "search_type": request.search_type
        }
        
        if neo4j_results:
            response_data["neo4j_results"] = neo4j_results
        if weaviate_results:
            response_data["weaviate_results"] = weaviate_results
        if vectorized_graph_results:
            response_data["vectorized_graph_results"] = vectorized_graph_results
        
        return SearchTestResponse(**response_data)
        
    except Exception as e:
        logger.error(f"检索测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"检索测试失败: {str(e)}")

@app.delete("/storage/tasks/{task_id}")
async def delete_processing_task(task_id: str):
    """删除处理任务记录"""
    if task_id in doc_processor.tasks:
        del doc_processor.tasks[task_id]
        return {"message": f"任务 {task_id} 已删除"}
    else:
        raise HTTPException(status_code=404, detail="任务不存在")

@app.get("/storage/tasks")
async def list_processing_tasks():
    """获取所有处理任务列表"""
    return {
        "tasks": list(doc_processor.tasks.values()),
        "total_tasks": len(doc_processor.tasks)
    }

@app.get("/cache/stats", response_model=CacheStats)
async def get_cache_stats():
    """获取缓存统计信息"""
    if hag_api is None:
        raise HTTPException(status_code=503, detail="HAG API not initialized")
    
    try:
        # 获取缓存管理器统计信息 - 从hybrid_service获取
        cache_manager = hag_api.hybrid_service.cache_manager
        cache_stats = cache_manager.get_stats()
        
        # 处理不同缓存类型的统计数据
        cache_type_info = cache_stats.get('cache_type', {})
        cache_type = cache_type_info.get('cache_type', 'unknown') if isinstance(cache_type_info, dict) else str(cache_type_info)
        
        if cache_type == 'hybrid':
            # 混合模式：合并本地和远程缓存统计
            local_stats = cache_stats.get('local_cache', {})
            remote_stats = cache_stats.get('remote_cache', {})
            
            total_requests = local_stats.get('total_requests', 0) + remote_stats.get('total_requests', 0)
            cache_hits = local_stats.get('cache_hits', 0) + remote_stats.get('cache_hits', 0)
            cache_misses = local_stats.get('cache_misses', 0) + remote_stats.get('cache_misses', 0)
            hit_rate = cache_stats.get('total_hit_rate', 0.0)
            local_cache_size = local_stats.get('total_size_bytes', 0)
            remote_cache_size = remote_stats.get('total_size_bytes', 0)
        else:
            # 单一缓存模式
            total_requests = cache_stats.get('total_requests', 0)
            cache_hits = cache_stats.get('cache_hits', 0)
            cache_misses = cache_stats.get('cache_misses', 0)
            hit_rate = (cache_hits / total_requests) if total_requests > 0 else 0.0
            local_cache_size = cache_stats.get('total_size_bytes', 0)
            remote_cache_size = 0
        
        # 检查Redis连接状态
        redis_connected = False
        redis_info = {}
        
        if cache_manager.cache_type in ['redis', 'hybrid']:
            try:
                if hasattr(cache_manager, 'remote_cache') and cache_manager.remote_cache:
                    redis_client = cache_manager.remote_cache.redis_client
                    redis_connected = redis_client.ping()
                    if redis_connected:
                        redis_info = redis_client.info()
                elif hasattr(cache_manager, 'cache') and hasattr(cache_manager.cache, 'redis_client'):
                    redis_client = cache_manager.cache.redis_client
                    redis_connected = redis_client.ping()
                    if redis_connected:
                        redis_info = redis_client.info()
            except Exception as e:
                logger.warning(f"Redis连接检查失败: {e}")
                redis_connected = False
        
        return CacheStats(
            cache_type=cache_type,
            total_requests=total_requests,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            hit_rate=round(hit_rate, 4),
            redis_connected=redis_connected,
            redis_info={
                'used_memory_human': redis_info.get('used_memory_human', 'N/A'),
                'connected_clients': redis_info.get('connected_clients', 0),
                'total_commands_processed': redis_info.get('total_commands_processed', 0),
                'keyspace_hits': redis_info.get('keyspace_hits', 0),
                'keyspace_misses': redis_info.get('keyspace_misses', 0)
            } if redis_info else {},
            local_cache_size=local_cache_size,
            remote_cache_size=remote_cache_size,
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取缓存统计失败: {str(e)}")

@app.get("/cache/health", response_model=CacheHealthResponse)
async def get_cache_health():
    """检查缓存健康状态"""
    if hag_api is None:
        raise HTTPException(status_code=503, detail="HAG API not initialized")
    
    try:
        cache_manager = hag_api.hybrid_service.cache_manager
        
        # 检查Redis连接和响应时间
        redis_connected = False
        ping_time_ms = 0.0
        redis_info = {}
        cache_status = "unknown"
        
        if cache_manager.cache_type in ['redis', 'hybrid']:
            try:
                import time
                start_time = time.time()
                
                if hasattr(cache_manager, 'remote_cache') and cache_manager.remote_cache:
                    redis_client = cache_manager.remote_cache.redis_client
                    redis_connected = redis_client.ping()
                elif hasattr(cache_manager, 'cache') and hasattr(cache_manager.cache, 'redis_client'):
                    redis_client = cache_manager.cache.redis_client
                    redis_connected = redis_client.ping()
                
                ping_time_ms = (time.time() - start_time) * 1000
                
                if redis_connected:
                    redis_info = redis_client.info()
                    cache_status = "healthy"
                else:
                    cache_status = "redis_disconnected"
                    
            except Exception as e:
                logger.warning(f"Redis健康检查失败: {e}")
                redis_connected = False
                cache_status = "redis_error"
        else:
            # LRU缓存模式
            cache_status = "lru_only"
            redis_connected = False
        
        return CacheHealthResponse(
            redis_connected=redis_connected,
            redis_ping_time_ms=round(ping_time_ms, 2),
            redis_info={
                'version': redis_info.get('redis_version', 'N/A'),
                'uptime_in_seconds': redis_info.get('uptime_in_seconds', 0),
                'used_memory_human': redis_info.get('used_memory_human', 'N/A'),
                'connected_clients': redis_info.get('connected_clients', 0)
            } if redis_info else {},
            cache_manager_status=cache_status,
            last_check=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"缓存健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"缓存健康检查失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "backend_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )