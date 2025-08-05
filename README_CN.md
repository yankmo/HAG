# HAG: 混合增强生成框架

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/yankmo/HAG?style=social)](https://github.com/yankmo/HAG)
[![GitHub forks](https://img.shields.io/github/forks/yankmo/HAG?style=social)](https://github.com/yankmo/HAG)
[![GitHub issues](https://img.shields.io/github/issues/yankmo/HAG)](https://github.com/yankmo/HAG/issues)
[![GitHub license](https://img.shields.io/github/license/yankmo/HAG)](https://github.com/yankmo/HAG/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green.svg)](https://langchain.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.0+-red.svg)](https://neo4j.com/)
[![Weaviate](https://img.shields.io/badge/Weaviate-1.20+-orange.svg)](https://weaviate.io/)

[English](README.md) | [中文](README_CN.md)

**作者**: [YankMo](https://github.com/yankmo)

</div>

---

## 🚀 HAG 是什么？

HAG（混合增强生成）是一个先进的知识增强生成框架，结合了向量数据库和知识图谱的强大功能，提供智能问答能力。基于 LangChain、Neo4j 和 Weaviate 构建，HAG 在领域特定知识检索和推理方面表现卓越。

## ✨ 核心功能

### 🎯 智能意图识别
- **多维度理解**：深度解析用户查询意图，精准匹配知识需求
- **上下文感知**：基于对话历史和语义理解，提供个性化响应

### 🔄 双数据库集成架构
- **向量数据库**：Weaviate 提供高效语义相似性搜索
- **知识图谱**：Neo4j 实现复杂关系推理和实体发现
- **混合检索**：智能融合两种数据源，确保检索准确性和完整性

### 🚀 可调用API服务
- **RESTful接口**：标准化API设计，支持多种编程语言调用
- **模块化架构**：独立的嵌入、检索、生成服务，灵活组合
- **LangChain集成**：可运行管道架构，支持复杂工作流编排

### 🎨 LINEAR风格前端设计
- **现代化界面**：简洁优雅的用户体验，遵循LINEAR设计理念
- **实时反馈**：流式响应显示，即时状态更新
- **智能交互**：直观的对话界面，支持多轮对话和历史记录

## 系统架构

![工作流程](./docs/images/black.svg)
*What is HAG*
## 📸 效果展示

### 1. Web 界面
![HAG Web 界面](./docs/images/Newapp.png)
*LINEAR设计风格前端界面*

### 2. 检索效果
![HAG 检索效果](./docs/images/NewSearch.png)
*混合检索工作流程展示，融合向量数据库和知识图谱*

### 3. 最终回答
![HAG 最终回答](./docs/images/Newanswer.png)
*智能问答结果展示，包含完整的知识来源和推理过程*

## 📦 安装

### 前置要求
- Python 3.8 或更高版本
- Docker 和 Docker Compose
- Git

### 快速开始

1. **克隆仓库**
```bash
git clone https://github.com/yankmo/HAG.git
cd HAG
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **启动必需服务**
```bash
# 启动 Neo4j
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:latest

# 启动 Weaviate
docker run -d --name weaviate \
  -p 8080:8080 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  semitechnologies/weaviate:latest

# 启动 Ollama
docker run -d --name ollama \
  -p 11434:11434 \
  ollama/ollama:latest
```

4. **配置系统**
```bash
# 编辑配置文件
cp config/config.yaml.example config/config.yaml
# 更新数据库凭据和服务 URL
```

5. **运行应用程序**
```bash
# 启动 Web 界面
streamlit run app_simple.py

# 或直接使用 API
python api.py
```

## 🔧 配置

编辑 `config/config.yaml` 来自定义您的设置：

```yaml
# Neo4j 配置
neo4j:
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "your_password"

# Ollama 配置
ollama:
  base_url: "http://localhost:11434"
  default_model: "gemma3:4b"
  embedding_model: "bge-m3:latest"

# Weaviate 配置
weaviate:
  url: "http://localhost:8080"
```

## 🧪 使用示例

### Web 界面
```bash
streamlit run app_simple.py
```
导航到 `http://localhost:8501` 并开始提问！

### API 使用
```python
from api import HAGIntegratedAPI

# 初始化系统
hag = HAGIntegratedAPI()

# 提问
response = hag.runnable_chain.invoke("帕金森病的症状是什么？")
print(response)
```

### 直接服务访问
```python
from src.services import HybridRetrievalService

# 直接使用混合检索
hybrid_service = HybridRetrievalService(...)
results = hybrid_service.search("医疗查询", limit=5)
```

## 🧪 测试

运行测试套件以验证您的安装：

```bash
# 测试基本功能
python -c "from api import HAGIntegratedAPI; api = HAGIntegratedAPI(); print('✅ HAG 初始化成功')"
```

## 🤝 贡献

我们欢迎贡献！请查看我们的[贡献指南](CONTRIBUTING.md)了解详情。

1. Fork 仓库
2. 创建您的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 👨‍💻 作者

**YankMo**
- GitHub: [@yankmo](https://github.com/yankmo)
- CSDN 博客: [YankMo 的技术博客](https://blog.csdn.net/YankMo)

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个 Star！**

</div>
