"""
配置管理模块
统一管理Neo4j、Ollama、Weaviate等服务的配置
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
import yaml


@dataclass
class Neo4jConfig:
    """Neo4j数据库配置"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "hrx274700"
    database: str = "neo4j"
    
    def to_auth_tuple(self):
        """返回认证元组格式"""
        return (self.username, self.password)


@dataclass
class OllamaConfig:
    """Ollama服务配置"""
    base_url: str = "http://localhost:11434"
    default_model: str = "gemma3:4b"
    embedding_model: str = "bge-m3:latest"
    timeout: int = 30
    
    @property
    def api_generate_url(self):
        """生成API URL"""
        return f"{self.base_url}/api/generate"
    
    @property
    def api_tags_url(self):
        """标签API URL"""
        return f"{self.base_url}/api/tags"


@dataclass
class WeaviateConfig:
    """Weaviate向量数据库配置"""
    url: str = "http://localhost:8080"
    host: str = "localhost"
    port: int = 8080
    
    @property
    def meta_url(self):
        """元数据API URL"""
        return f"{self.url}/v1/meta"


@dataclass
class AppConfig:
    """应用程序配置"""
    debug: bool = False
    log_level: str = "INFO"
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    max_results: int = 10


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._get_default_config_file()
        self._load_config()
    
    def _get_default_config_file(self) -> str:
        """获取默认配置文件路径"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "config.yaml")
    
    def _load_config(self):
        """加载配置"""
        # 首先设置默认配置
        self.neo4j = Neo4jConfig()
        self.ollama = OllamaConfig()
        self.weaviate = WeaviateConfig()
        self.app = AppConfig()
        
        # 从环境变量加载配置
        self._load_from_env()
        
        # 从配置文件加载配置（如果存在）
        if os.path.exists(self.config_file):
            self._load_from_file()
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        # Neo4j配置
        if os.getenv('NEO4J_URI'):
            self.neo4j.uri = os.getenv('NEO4J_URI')
        if os.getenv('NEO4J_USERNAME'):
            self.neo4j.username = os.getenv('NEO4J_USERNAME')
        if os.getenv('NEO4J_PASSWORD'):
            self.neo4j.password = os.getenv('NEO4J_PASSWORD')
        if os.getenv('NEO4J_DATABASE'):
            self.neo4j.database = os.getenv('NEO4J_DATABASE')
        
        # Ollama配置
        if os.getenv('OLLAMA_BASE_URL'):
            self.ollama.base_url = os.getenv('OLLAMA_BASE_URL')
        if os.getenv('OLLAMA_DEFAULT_MODEL'):
            self.ollama.default_model = os.getenv('OLLAMA_DEFAULT_MODEL')
        if os.getenv('OLLAMA_EMBEDDING_MODEL'):
            self.ollama.embedding_model = os.getenv('OLLAMA_EMBEDDING_MODEL')
        if os.getenv('OLLAMA_TIMEOUT'):
            self.ollama.timeout = int(os.getenv('OLLAMA_TIMEOUT'))
        
        # Weaviate配置
        if os.getenv('WEAVIATE_URL'):
            self.weaviate.url = os.getenv('WEAVIATE_URL')
        if os.getenv('WEAVIATE_HOST'):
            self.weaviate.host = os.getenv('WEAVIATE_HOST')
        if os.getenv('WEAVIATE_PORT'):
            self.weaviate.port = int(os.getenv('WEAVIATE_PORT'))
        
        # 应用配置
        if os.getenv('DEBUG'):
            self.app.debug = os.getenv('DEBUG').lower() == 'true'
        if os.getenv('LOG_LEVEL'):
            self.app.log_level = os.getenv('LOG_LEVEL')
    
    def _load_from_file(self):
        """从配置文件加载配置"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # 更新Neo4j配置
            if 'neo4j' in config_data:
                neo4j_data = config_data['neo4j']
                for key, value in neo4j_data.items():
                    if hasattr(self.neo4j, key):
                        setattr(self.neo4j, key, value)
            
            # 更新Ollama配置
            if 'ollama' in config_data:
                ollama_data = config_data['ollama']
                for key, value in ollama_data.items():
                    if hasattr(self.ollama, key):
                        setattr(self.ollama, key, value)
            
            # 更新Weaviate配置
            if 'weaviate' in config_data:
                weaviate_data = config_data['weaviate']
                for key, value in weaviate_data.items():
                    if hasattr(self.weaviate, key):
                        setattr(self.weaviate, key, value)
            
            # 更新应用配置
            if 'app' in config_data:
                app_data = config_data['app']
                for key, value in app_data.items():
                    if hasattr(self.app, key):
                        setattr(self.app, key, value)
                        
        except Exception as e:
            print(f"警告：加载配置文件失败 {self.config_file}: {e}")
    
    def save_config(self):
        """保存配置到文件"""
        config_data = {
            'neo4j': {
                'uri': self.neo4j.uri,
                'username': self.neo4j.username,
                'password': self.neo4j.password,
                'database': self.neo4j.database
            },
            'ollama': {
                'base_url': self.ollama.base_url,
                'default_model': self.ollama.default_model,
                'embedding_model': self.ollama.embedding_model,
                'timeout': self.ollama.timeout
            },
            'weaviate': {
                'url': self.weaviate.url,
                'host': self.weaviate.host,
                'port': self.weaviate.port
            },
            'app': {
                'debug': self.app.debug,
                'log_level': self.app.log_level,
                'max_chunk_size': self.app.max_chunk_size,
                'chunk_overlap': self.app.chunk_overlap,
                'max_results': self.app.max_results
            }
        }
        
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            else:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def get_service_status_urls(self) -> Dict[str, str]:
        """获取服务状态检查URL"""
        return {
            "Neo4j": self.neo4j.uri,
            "Ollama": self.ollama.api_tags_url,
            "Weaviate": self.weaviate.meta_url
        }


# 全局配置实例
config = ConfigManager()


def get_config() -> ConfigManager:
    """获取配置管理器实例"""
    return config


def reload_config():
    """重新加载配置"""
    global config
    config = ConfigManager()
    return config