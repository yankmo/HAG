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

# 导入现有的HAG API
from api import HAGIntegratedAPI, IntegratedResponse

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="HAG API", description="Hybrid Augmented Generation API", version="1.0.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React开发服务器
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class QueryRequest(BaseModel):
    question: str

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
    """流式知识查询接口 - 集成图谱检索功能"""
    if hag_api is None:
        raise HTTPException(status_code=503, detail="HAG API not initialized")
    
    async def generate_stream():
        try:
            logger.info(f"处理流式查询: {request.question}")
            
            # 发送开始信号
            yield f"data: {json.dumps({'type': 'start', 'message': '开始处理您的问题...'}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.1)
            
            # 发送检索状态
            yield f"data: {json.dumps({'type': 'status', 'message': '正在检索相关信息...'}, ensure_ascii=False)}\n\n"
            
            # 1. 文档检索
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
            
            # 构建完整的prompt
            prompt = f"""基于以下知识库信息回答用户问题：

{context}

用户问题：{request.question}

请基于以上文档、实体和关系信息提供准确、详细的回答。如果知识库中没有相关信息，请说明无法找到相关信息。"""
            
            # 发送生成状态
            yield f"data: {json.dumps({'type': 'status', 'message': '正在生成回答...'}, ensure_ascii=False)}\n\n"
            
            # 使用ollama进行真正的流式生成
            import requests
            from config import get_config
            
            config = get_config()
            
            # 调用ollama的流式API
            response = requests.post(
                f"{config.ollama.base_url}/api/generate",
                json={
                    "model": config.ollama.default_model,
                    "prompt": prompt,
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
                            if 'response' in chunk:
                                current_text += chunk['response']
                                yield f"data: {json.dumps({'type': 'content', 'content': current_text}, ensure_ascii=False)}\n\n"
                                
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
                yield f"data: {json.dumps({'type': 'content', 'content': result.answer}, ensure_ascii=False)}\n\n"
                retrieval_result = {'sources': result.sources}
            
            # 发送来源信息（包含文档、实体、关系）
            sources = {
                'documents': [],
                'entities': [],
                'relationships': [],
                'statistics': {}
            }
            
            # 添加文档来源
            if hasattr(retrieval_result, 'hybrid_results') and retrieval_result.hybrid_results:
                sources['documents'] = [
                    {
                        'content': doc.content[:300] + "..." if len(doc.content) > 300 else doc.content,
                        'score': doc.score,
                        'metadata': doc.metadata
                    } for doc in retrieval_result.hybrid_results[:3]
                ]
                sources['statistics'] = retrieval_result.statistics if hasattr(retrieval_result, 'statistics') else {}
            
            # 添加实体来源
            if entities:
                sources['entities'] = [
                    {
                        'name': entity.get('name', ''),
                        'type': entity.get('type', ''),
                        'description': entity.get('description', '')
                    } for entity in entities[:3]
                ]
            
            # 添加关系来源
            if relationships:
                sources['relationships'] = [
                    {
                        'source': rel.get('source', ''),
                        'target': rel.get('target', ''),
                        'type': rel.get('type', ''),
                        'description': rel.get('description', ''),
                        'relevance_score': rel.get('relevance_score', 0)
                    } for rel in relationships[:5]
                ]
            
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