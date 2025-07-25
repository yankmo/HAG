# HAG (HybridRAG)

> 🚀 基于双数据库的智能混合检索系统，融合图谱与向量的优势

## 📊 对比分析

| 特性 | 传统RAG | HAG |
|------|---------|-----|
| 检索方式 | 单一向量检索 | 图谱+向量混合检索 |
| 相似度算法 | 单一算法 | 欧氏距离+余弦相似度 |
| 知识获取 | 文档片段 | 实体+关系+节点+文档 |
| 结果融合 | 无 | 智能去重排序 |
| 检索透明度 | 黑盒 | 完整过程展示 |


## ✨ 核心特性

### 🔄 双数据库混合检索
- **Neo4j + Weaviate**: 图谱结构检索 + 向量语义检索
- **双相似度算法**: 欧氏距离 + 余弦相似度
- **智能融合**: 并行检索，结果自动去重排序

### 🎯 全方位知识获取
- **实体 + 关系 + 节点 + 文档**: 四维度统一检索
- **统一存储**: 检索结果可直接存储管理
- **实时统计**: 完整的检索过程监控


## 🛠️ 技术栈

- **后端**: Python + LangChain
- **图数据库**: Neo4j
- **向量数据库**: Weaviate
- **大语言模型**: Ollama
- **前端**: Streamlit

## 📦 快速开始

### 环境要求
- Python 3.8+
- Docker
- Neo4j
- Ollama

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动服务
```bash
# 启动 Weaviate
docker-compose up -d

# 启动 Ollama 并下载模型
ollama serve
ollama pull gemma3:4b
ollama pull bge-m3:latest

# 启动 Web 应用
streamlit run app_simple.py
```
### web界面
<!-- 在这里添加使用界面截图 -->
![Web Interface](./images/finalanwser.png)
### 向量检索
![vector Interface](./images/vector.png)
### 图谱检索
![graph Interface](./images/graph.png)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License