# HAG: Hybrid Augmented Generation Framework

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

**Author**: [YankMo](https://github.com/yankmo)

</div>

---

## 🚀 What is HAG?

HAG (Hybrid Augmented Generation) is an advanced knowledge-enhanced generation framework that combines the powerful capabilities of vector databases and knowledge graphs to provide intelligent Q&A capabilities. Built on LangChain, Neo4j, and Weaviate, HAG excels in domain-specific knowledge retrieval and reasoning.

## ✨ Core Features

### 🎯 Intelligent Intent Recognition
- **Multi-dimensional Understanding**: Deep analysis of user query intent with precise knowledge need matching
- **Context Awareness**: Personalized responses based on conversation history and semantic understanding

### 🔄 Dual Database Integration Architecture
- **Vector Database**: Weaviate provides efficient semantic similarity search
- **Knowledge Graph**: Neo4j enables complex relationship reasoning and entity discovery
- **Hybrid Retrieval**: Intelligent fusion of two data sources ensuring retrieval accuracy and completeness

### 🚀 Callable API Services
- **RESTful Interface**: Standardized API design supporting multiple programming language calls
- **Modular Architecture**: Independent embedding, retrieval, and generation services with flexible composition
- **LangChain Integration**: Runnable pipeline architecture supporting complex workflow orchestration

### 🎨 LINEAR Style Frontend Design
- **Modern Interface**: Clean and elegant user experience following LINEAR design principles
- **Real-time Feedback**: Streaming response display with instant status updates
- **Intelligent Interaction**: Intuitive chat interface supporting multi-turn conversations and history

## System Architecture

![Workflow](./docs/images/black.svg)
*What is HAG*

## 📸 Demo Gallery

### 1. Web Interface
![HAG Web Interface](./docs/images/Newapp.png)
*LINEAR style frontend interface*

### 2. Retrieval Effects
![HAG Retrieval Effects](./docs/images/NewSearch.png)
*Hybrid retrieval workflow demonstration, integrating vector database and knowledge graph*

### 3. Final Answer
![HAG Final Answer](./docs/images/Newanswer.png)
*Intelligent Q&A result display with complete knowledge sources and reasoning process*

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Docker and Docker Compose
- Git

### Quick Start

1. **Clone Repository**
```bash
git clone https://github.com/yankmo/HAG.git
cd HAG
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Start Required Services**
```bash
# Start Neo4j
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:latest

# Start Weaviate
docker run -d --name weaviate \
  -p 8080:8080 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  semitechnologies/weaviate:latest

# Start Ollama
docker run -d --name ollama \
  -p 11434:11434 \
  ollama/ollama:latest
```

4. **Configure System**
```bash
# Edit configuration file
cp config/config.yaml.example config/config.yaml
# Update database credentials and service URLs
```

5. **Run Application**
```bash
# Start Web Interface
streamlit run app_simple.py

# Or use API directly
python api.py
```

## 🔧 Configuration

Edit `config/config.yaml` to customize your settings:

```yaml
# Neo4j Configuration
neo4j:
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "your_password"

# Ollama Configuration
ollama:
  base_url: "http://localhost:11434"
  default_model: "gemma3:4b"
  embedding_model: "bge-m3:latest"

# Weaviate Configuration
weaviate:
  url: "http://localhost:8080"
```

## 🧪 Usage Examples

### Web Interface
```bash
streamlit run app_simple.py
```
Navigate to `http://localhost:8501` and start asking questions!

### API Usage
```python
from api import HAGIntegratedAPI

# Initialize system
hag = HAGIntegratedAPI()

# Ask questions
response = hag.runnable_chain.invoke("What are the symptoms of Parkinson's disease?")
print(response)
```

### Direct Service Access
```python
from src.services import HybridRetrievalService

# Use hybrid retrieval directly
hybrid_service = HybridRetrievalService(...)
results = hybrid_service.search("medical query", limit=5)
```

## 🧪 Testing

Run the test suite to verify your installation:

```bash
# Test basic functionality
python -c "from api import HAGIntegratedAPI; api = HAGIntegratedAPI(); print('✅ HAG initialized successfully')"
```

## 🤝 Contributing

We welcome contributions! Please check our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**YankMo**
- GitHub: [@yankmo](https://github.com/yankmo)
- CSDN Blog: [YankMo's Tech Blog](https://blog.csdn.net/YankMo)

---

<div align="center">

**⭐ If this project helps you, please give us a Star!**

</div>