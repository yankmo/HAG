#!/usr/bin/env python3
"""
配置初始化脚本
帮助用户快速设置 HAG 项目的配置文件
"""

import os
import sys
import shutil
from pathlib import Path


def create_config_yaml():
    """创建默认的 config.yaml 文件"""
    config_dir = Path("config")
    config_file = config_dir / "config.yaml"
    
    if config_file.exists():
        response = input(f"配置文件 {config_file} 已存在，是否覆盖？(y/N): ")
        if response.lower() != 'y':
            print("跳过创建 config.yaml")
            return False
    
    config_content = """# HAG 项目配置文件
# 请根据实际环境修改以下配置

# Neo4j 图数据库配置
neo4j:
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "your_password_here"  # 请修改为实际密码
  database: "neo4j"

# Ollama 大语言模型服务配置
ollama:
  base_url: "http://localhost:11434"
  default_model: "gemma3:4b"
  embedding_model: "bge-m3:latest"
  timeout: 30

# Weaviate 向量数据库配置
weaviate:
  url: "http://localhost:8080"
  host: "localhost"
  port: 8080

# 应用程序配置
app:
  debug: false
  log_level: "INFO"
  chunk_size: 1000
  chunk_overlap: 200
  max_results: 10
"""
    
    config_dir.mkdir(exist_ok=True)
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ 已创建配置文件: {config_file}")
    return True


def create_env_file():
    """创建 .env 文件"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        response = input(f"环境变量文件 {env_file} 已存在，是否覆盖？(y/N): ")
        if response.lower() != 'y':
            print("跳过创建 .env")
            return False
    
    if env_example.exists():
        shutil.copy(env_example, env_file)
        print(f"✅ 已从 {env_example} 复制创建 {env_file}")
    else:
        env_content = """# HAG 项目环境变量配置
# 请根据实际环境修改以下配置

# Neo4j 数据库配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_DATABASE=neo4j

# Ollama 服务配置
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=gemma3:4b
OLLAMA_EMBEDDING_MODEL=bge-m3:latest
OLLAMA_TIMEOUT=30

# Weaviate 向量数据库配置
WEAVIATE_URL=http://localhost:8080
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080

# 应用程序配置
APP_DEBUG=false
APP_LOG_LEVEL=INFO
APP_CHUNK_SIZE=1000
APP_CHUNK_OVERLAP=200
APP_MAX_RESULTS=10
"""
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        print(f"✅ 已创建环境变量文件: {env_file}")
    
    return True


def create_gitignore():
    """更新 .gitignore 文件"""
    gitignore_file = Path(".gitignore")
    
    # 需要添加的忽略规则
    ignore_rules = [
        "# 配置文件",
        ".env",
        "config/config.json",
        "",
        "# 日志文件",
        "*.log",
        "logs/",
        "",
        "# 临时文件",
        "*.tmp",
        "*.temp",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".Python",
        "",
        "# IDE 文件",
        ".vscode/",
        ".idea/",
        "*.swp",
        "*.swo",
        "",
        "# 数据文件",
        "data/",
        "*.db",
        "*.sqlite",
        "",
    ]
    
    existing_content = ""
    if gitignore_file.exists():
        with open(gitignore_file, 'r', encoding='utf-8') as f:
            existing_content = f.read()
    
    # 检查哪些规则需要添加
    rules_to_add = []
    for rule in ignore_rules:
        if rule and rule not in existing_content:
            rules_to_add.append(rule)
    
    if rules_to_add:
        with open(gitignore_file, 'a', encoding='utf-8') as f:
            if existing_content and not existing_content.endswith('\n'):
                f.write('\n')
            f.write('\n'.join(rules_to_add))
            f.write('\n')
        print(f"✅ 已更新 .gitignore 文件")
    else:
        print("ℹ️  .gitignore 文件已包含必要的忽略规则")


def print_next_steps():
    """打印后续步骤"""
    print("\n" + "=" * 60)
    print("🎉 配置文件初始化完成！")
    print("\n📝 后续步骤:")
    print("1. 编辑配置文件，修改为实际的服务配置:")
    print("   - config/config.yaml (推荐)")
    print("   - .env (环境变量方式)")
    print("\n2. 启动必要的服务:")
    print("   - Neo4j 数据库")
    print("   - Ollama 服务: ollama serve")
    print("   - Weaviate 服务: docker-compose up -d")
    print("\n3. 下载 Ollama 模型:")
    print("   - ollama pull gemma3:4b")
    print("   - ollama pull bge-m3:latest")
    print("\n4. 验证配置:")
    print("   - python validate_config.py")
    print("\n5. 启动应用:")
    print("   - streamlit run app_simple.py")
    print("\n📚 更多信息请参考:")
    print("   - docs/配置迁移指南.md")
    print("   - README.md")


def main():
    """主函数"""
    print("🚀 HAG 项目配置初始化")
    print("=" * 60)
    
    # 检查是否在项目根目录
    if not Path("config").exists() and not Path("src").exists():
        print("❌ 请在 HAG 项目根目录下运行此脚本")
        sys.exit(1)
    
    try:
        # 创建配置文件
        print("\n📁 创建配置文件...")
        create_config_yaml()
        create_env_file()
        
        # 更新 .gitignore
        print("\n🔒 更新 .gitignore...")
        create_gitignore()
        
        # 打印后续步骤
        print_next_steps()
        
    except Exception as e:
        print(f"❌ 初始化过程中出现错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()