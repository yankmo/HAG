#!/usr/bin/env python3
"""
配置验证脚本
用于检查 HAG 项目的配置是否正确，以及各个服务是否可用
"""

import sys
import os
import requests
import time
from typing import Dict, Any, Tuple

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from config import get_config
    from neo4j import GraphDatabase
    import weaviate
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装所有依赖: pip install -r requirements.txt")
    sys.exit(1)


class ConfigValidator:
    """配置验证器"""
    
    def __init__(self):
        self.config = None
        self.results = {}
    
    def load_config(self) -> bool:
        """加载配置"""
        try:
            self.config = get_config()
            print("✅ 配置文件加载成功")
            return True
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            return False
    
    def validate_neo4j(self) -> Tuple[bool, str]:
        """验证 Neo4j 连接"""
        try:
            driver = GraphDatabase.driver(
                self.config.neo4j.uri,
                auth=self.config.neo4j.to_auth_tuple()
            )
            
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record and record["test"] == 1:
                    driver.close()
                    return True, "连接成功"
                else:
                    driver.close()
                    return False, "查询测试失败"
                    
        except Exception as e:
            return False, f"连接失败: {str(e)}"
    
    def validate_ollama(self) -> Tuple[bool, str]:
        """验证 Ollama 服务"""
        try:
            # 检查 Ollama 服务是否运行
            response = requests.get(f"{self.config.ollama.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                
                # 检查默认模型是否存在
                default_model = self.config.ollama.default_model
                embedding_model = self.config.ollama.embedding_model
                
                missing_models = []
                if default_model not in model_names:
                    missing_models.append(default_model)
                if embedding_model not in model_names:
                    missing_models.append(embedding_model)
                
                if missing_models:
                    return False, f"缺少模型: {', '.join(missing_models)}"
                else:
                    return True, f"服务正常，已安装模型: {len(model_names)} 个"
            else:
                return False, f"服务响应异常: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, "无法连接到 Ollama 服务"
        except Exception as e:
            return False, f"检查失败: {str(e)}"
    
    def validate_weaviate(self) -> Tuple[bool, str]:
        """验证 Weaviate 服务"""
        try:
            client = weaviate.Client(self.config.weaviate.url)
            
            # 检查服务是否可用
            if client.is_ready():
                # 获取集群信息
                meta = client.get_meta()
                version = meta.get("version", "未知")
                return True, f"服务正常，版本: {version}"
            else:
                return False, "服务未就绪"
                
        except Exception as e:
            return False, f"连接失败: {str(e)}"
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("\n📋 配置摘要:")
        print(f"  Neo4j URI: {self.config.neo4j.uri}")
        print(f"  Neo4j 用户: {self.config.neo4j.username}")
        print(f"  Neo4j 数据库: {self.config.neo4j.database}")
        print(f"  Ollama URL: {self.config.ollama.base_url}")
        print(f"  Ollama 默认模型: {self.config.ollama.default_model}")
        print(f"  Ollama 嵌入模型: {self.config.ollama.embedding_model}")
        print(f"  Weaviate URL: {self.config.weaviate.url}")
        print(f"  Weaviate 主机: {self.config.weaviate.host}:{self.config.weaviate.port}")
    
    def print_service_urls(self):
        """打印服务访问地址"""
        print("\n🌐 服务访问地址:")
        print(f"  Neo4j Browser: {self.config.get_neo4j_browser_url()}")
        print(f"  Ollama API: {self.config.get_ollama_api_url()}")
        print(f"  Weaviate: {self.config.get_weaviate_url()}")
    
    def run_validation(self):
        """运行完整验证"""
        print("🔍 HAG 项目配置验证")
        print("=" * 50)
        
        # 加载配置
        if not self.load_config():
            return False
        
        # 打印配置摘要
        self.print_config_summary()
        
        print("\n🔧 服务连接测试:")
        
        # 验证 Neo4j
        print("  检查 Neo4j...", end=" ")
        neo4j_ok, neo4j_msg = self.validate_neo4j()
        if neo4j_ok:
            print(f"✅ {neo4j_msg}")
        else:
            print(f"❌ {neo4j_msg}")
        
        # 验证 Ollama
        print("  检查 Ollama...", end=" ")
        ollama_ok, ollama_msg = self.validate_ollama()
        if ollama_ok:
            print(f"✅ {ollama_msg}")
        else:
            print(f"❌ {ollama_msg}")
        
        # 验证 Weaviate
        print("  检查 Weaviate...", end=" ")
        weaviate_ok, weaviate_msg = self.validate_weaviate()
        if weaviate_ok:
            print(f"✅ {weaviate_msg}")
        else:
            print(f"❌ {weaviate_msg}")
        
        # 打印服务地址
        self.print_service_urls()
        
        # 总结
        all_ok = neo4j_ok and ollama_ok and weaviate_ok
        print("\n" + "=" * 50)
        if all_ok:
            print("🎉 所有服务配置正确，可以启动应用！")
            print("\n启动命令:")
            print("  streamlit run app_simple.py")
        else:
            print("⚠️  部分服务配置有问题，请检查上述错误信息")
            print("\n常见解决方案:")
            if not neo4j_ok:
                print("  - 启动 Neo4j 服务")
                print("  - 检查用户名和密码")
            if not ollama_ok:
                print("  - 启动 Ollama 服务: ollama serve")
                print("  - 下载模型: ollama pull gemma3:4b")
                print("  - 下载嵌入模型: ollama pull bge-m3:latest")
            if not weaviate_ok:
                print("  - 启动 Weaviate 服务: docker-compose up -d")
        
        return all_ok


def main():
    """主函数"""
    validator = ConfigValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()