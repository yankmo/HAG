# HAG整合API使用指南

## 概述

HAG整合API是一个基于LangChain Runnable的知识检索和问答系统，整合了以下核心功能：

1. **Weaviate向量检索** - 使用余弦相似度和欧式距离的混合检索，获取Top5相关文档
2. **Neo4j图谱检索** - 检索相关的节点和关系信息
3. **LangChain管道** - 使用Runnable模式整合检索结果，生成智能回答

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置服务

确保以下服务正在运行：
- Weaviate (http://localhost:8080)
- Neo4j (bolt://localhost:7687)
- Ollama (http://localhost:11434)

### 3. 基本使用

```python
from api import HAGIntegratedAPI, query_knowledge

# 方式1: 使用完整API
api = HAGIntegratedAPI()
result = api.query("什么是人工智能？")

print(f"回答: {result.answer}")
print(f"文档来源: {len(result.sources['documents'])}个")
print(f"实体来源: {len(result.sources['entities'])}个")
print(f"关系来源: {len(result.sources['relationships'])}个")

# 方式2: 使用简化接口
result = query_knowledge("机器学习的基本原理是什么？")
print(result.answer)
```

## API接口说明

### HAGIntegratedAPI类

#### 主要方法

- `query(question: str) -> IntegratedResponse`: 主要查询接口
- `get_system_status() -> Dict[str, Any]`: 获取系统状态

#### 返回结果结构

```python
@dataclass
class IntegratedResponse:
    answer: str                    # LLM生成的回答
    sources: Dict[str, Any]        # 检索来源信息
    metadata: Dict[str, Any]       # 元数据信息
```

### 简化接口

- `query_knowledge(question: str) -> IntegratedResponse`: 直接查询接口

## 系统架构

### LangChain Runnable管道

```
用户问题 → 并行检索 → 提示词构建 → LLM生成 → 结构化输出
         ↓
    ┌─ 文档检索 (Weaviate)
    ├─ 实体检索 (Neo4j)  
    └─ 关系检索 (Neo4j)
```

### 检索策略

1. **文档检索**: 使用Weaviate的混合检索（余弦相似度 + 欧式距离），获取Top5文档
2. **实体检索**: 从Neo4j检索2个最相关的节点
3. **关系检索**: 检索节点间的所有关系信息

## 测试

运行测试脚本验证功能：

```bash
python test_api.py
```

## 配置

系统配置位于 `config/config.yaml`，包含：
- Neo4j连接配置
- Ollama模型配置  
- Weaviate连接配置
- 应用参数配置

## 注意事项

1. 确保所有依赖服务正常运行
2. 检查配置文件中的连接参数
3. 首次使用时会初始化所有服务连接
4. 支持单例模式，避免重复初始化

## 错误处理

API包含完整的错误处理机制：
- 服务连接失败时会返回错误信息
- 检索失败时会提供降级响应
- 所有异常都会被捕获并记录日志