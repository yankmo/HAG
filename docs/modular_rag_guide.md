# 模块化RAG系统使用指南

## 📋 概述

模块化RAG系统将原有的混合向量图谱系统重构为独立的功能模块，提供更清晰的架构和更灵活的使用方式。

## 🏗️ 系统架构

### 核心模块

1. **KnowledgeStorageManager** - 知识存储管理器
   - 负责实体和关系的存储
   - 支持Neo4j图谱和Weaviate向量的双重存储
   - 提供批量存储和统计功能

2. **KnowledgeRetrievalManager** - 知识检索管理器
   - 负责向量检索和图谱检索
   - 支持TopK节点检索和图谱扩展
   - 提供灵活的检索参数配置

3. **HybridSearchManager** - 混合搜索管理器
   - 结合向量检索和图谱检索
   - 支持智能答案生成
   - 提供完整的对话流程

4. **ModularRAGSystem** - 模块化RAG系统
   - 整合所有功能模块
   - 提供统一的接口
   - 支持端到端的知识库构建和检索

## 🚀 快速开始

### 1. 基本使用

```python
from src.knowledge.modular_rag_system import ModularRAGSystem

# 创建系统
rag_system = ModularRAGSystem()

# 构建知识库
result = rag_system.build_knowledge_base("path/to/your/file.txt")

# 搜索知识
answer = rag_system.search("你的问题")
print(answer['answer'])
```

### 2. 模块化使用

```python
from src.knowledge.modular_rag_system import (
    KnowledgeStorageManager,
    KnowledgeRetrievalManager,
    HybridSearchManager
)

# 独立使用存储模块
storage = KnowledgeStorageManager()
storage.setup_storage()

# 独立使用检索模块
retrieval = KnowledgeRetrievalManager()
results = retrieval.vector_search("查询内容")

# 独立使用混合搜索
search = HybridSearchManager()
answer = search.chat("问题")
```

## 🔧 功能特性

### 存储功能

- ✅ 双重存储：Neo4j图谱 + Weaviate向量
- ✅ 批量处理：支持大量实体和关系的批量存储
- ✅ 数据一致性：确保图谱和向量数据的同步
- ✅ 统计监控：实时获取存储状态和统计信息

### 检索功能

- ✅ 向量检索：基于语义相似度的实体和关系检索
- ✅ 图谱检索：支持TopK节点检索和关系扩展
- ✅ 混合检索：结合向量和图谱的综合检索
- ✅ 智能扩展：基于图谱结构的知识扩展

### 高级特性

- ✅ 模块化设计：各模块可独立使用
- ✅ 参数可配置：支持自定义检索参数
- ✅ 性能优化：支持批量操作和并行处理
- ✅ 错误处理：完善的异常处理和日志记录

## 📊 图谱检索优化

### TopK节点检索

新的图谱检索逻辑支持检索最相关的K个节点及其关系：

```python
retrieval_manager = KnowledgeRetrievalManager()

# 检索top5个最相关节点及其关系
results = retrieval_manager.graph_search_topk_nodes(
    query="帕金森病治疗",
    top_k=5,
    include_relations=True
)

print(f"找到 {results['total_nodes']} 个节点")
print(f"找到 {results['total_relationships']} 个关系")
```

### 图谱扩展检索

支持从指定节点扩展子图：

```python
# 从特定节点扩展子图
expanded = retrieval_manager.graph_expand_from_nodes(
    node_ids=["1", "2", "3"],
    depth=2,
    max_nodes=50
)

print(f"扩展后: {expanded['total_nodes']} 个节点")
print(f"发现路径: {expanded['total_paths']} 条")
```

## 🎯 使用示例

### 示例1：完整系统使用

```python
# 运行完整演示
python demo_modular_rag.py
```

### 示例2：模块测试

```python
# 运行模块测试
python test_modular_rag.py
```

### 示例3：自定义搜索参数

```python
result = rag_system.search(
    "帕金森病的治疗方法",
    vector_entity_limit=5,      # 向量搜索实体数量
    vector_relation_limit=5,    # 向量搜索关系数量
    graph_top_k=10,            # 图谱搜索节点数量
    expand_depth=3,            # 图谱扩展深度
    max_expand_nodes=30        # 最大扩展节点数
)
```

## 📈 性能监控

### 获取系统统计

```python
stats = rag_system.get_stats()
print(f"Neo4j节点: {stats['neo4j_nodes']}")
print(f"Neo4j关系: {stats['neo4j_relationships']}")
print(f"向量实体: {stats['vector_entities']}")
print(f"向量关系: {stats['vector_relations']}")
```

### 搜索统计

每次搜索都会返回详细的统计信息：

```python
result = rag_system.search("查询内容")
search_stats = result['search_results']['search_stats']

print(f"向量检索: {search_stats['vector_entities']} 实体 + {search_stats['vector_relations']} 关系")
print(f"图谱检索: {search_stats['graph_nodes']} 节点 + {search_stats['graph_relationships']} 关系")
print(f"图谱扩展: {search_stats['expanded_nodes']} 节点 + {search_stats['expanded_relationships']} 关系")
```

## 🔗 系统访问

- **Neo4j Browser**: http://localhost:7474
- **Weaviate**: http://localhost:8080

## 📝 注意事项

1. **依赖服务**: 确保Neo4j和Weaviate服务正在运行
2. **内存使用**: 大规模数据处理时注意内存使用情况
3. **参数调优**: 根据具体需求调整检索参数
4. **错误处理**: 注意查看日志信息以排查问题

## 🆚 与原系统对比

| 特性 | 原系统 | 模块化系统 |
|------|--------|------------|
| 架构设计 | 单一类 | 模块化设计 |
| 代码复用 | 低 | 高 |
| 功能扩展 | 困难 | 容易 |
| 测试维护 | 复杂 | 简单 |
| 图谱检索 | 基础 | TopK + 扩展 |
| 参数配置 | 固定 | 灵活可配 |

## 🔮 未来扩展

- [ ] 支持更多向量模型
- [ ] 增加缓存机制
- [ ] 支持分布式部署
- [ ] 添加可视化界面
- [ ] 支持实时数据更新