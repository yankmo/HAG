#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAG FastAPI 后端包装器
为前端提供RESTful API接口
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
import logging

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