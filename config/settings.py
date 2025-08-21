"""
配置管理模块
统一管理Neo4j、Ollama、Weaviate等服务的配置
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json
import yaml


@dataclass
class Neo4jConfig:
    """Neo4j数据库配置"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = ""  # 从环境变量读取，不设置默认值
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
    available_models: List[str] = field(default_factory=lambda: ["gemma3:4b", "llama3:8b", "qwen2:7b"])
    
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
class WeightConfig:
    """权重策略配置"""
    # 默认权重策略
    default_strategy: str = "static"
    
    # 是否启用动态权重
    enable_dynamic_weights: bool = True
    
    # 是否启用异步权重计算
    enable_async_weights: bool = False
    
    # 默认文档和图谱权重
    default_doc_weight: float = 0.6
    default_graph_weight: float = 0.4
    
    # 静态权重配置
    static_weights: Dict[str, float] = field(default_factory=lambda: {
        "document": 0.6,
        "graph": 0.4
    })
    
    # 意图驱动权重配置
    intent_driven_config: Dict[str, Any] = field(default_factory=lambda: {
        "factual_doc_weight": 0.7,
        "factual_graph_weight": 0.3,
        "analytical_doc_weight": 0.4,
        "analytical_graph_weight": 0.6,
        "exploratory_doc_weight": 0.3,
        "exploratory_graph_weight": 0.7,
        "default_doc_weight": 0.5,
        "default_graph_weight": 0.5
    })
    
    # 质量驱动权重配置
    quality_driven_config: Dict[str, Any] = field(default_factory=lambda: {
        "relevance_threshold": 0.7,
        "confidence_threshold": 0.6,
        "freshness_weight": 0.1,
        "authority_weight": 0.2,
        "completeness_weight": 0.3,
        "accuracy_weight": 0.4
    })
    
    # 自适应权重配置
    adaptive_config: Dict[str, Any] = field(default_factory=lambda: {
        "learning_rate": 0.01,
        "momentum": 0.9,
        "decay_factor": 0.95,
        "min_samples": 10,
        "update_frequency": 100
    })
    
    # GNN权重配置
    gnn_config: Dict[str, Any] = field(default_factory=lambda: {
        "model_type": "gat",  # gat 或 gcn
        "hidden_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "batch_size": 32,
        "max_epochs": 100,
        "early_stopping_patience": 10,
        "model_save_path": "models/gnn_weights.pth",
        "enable_training": False,
        "training_data_size": 1000
    })
    
    # 混合策略配置
    hybrid_config: Dict[str, Any] = field(default_factory=lambda: {
        "intent_weight": 0.3,
        "quality_weight": 0.4,
        "gnn_weight": 0.3,
        "enable_normalization": True
    })
    
    # 集成策略配置
    ensemble_config: Dict[str, Any] = field(default_factory=lambda: {
        "method": "weighted_average",  # weighted_average 或 majority_vote
        "calculator_weights": {
            "intent_driven": 0.25,
            "quality_driven": 0.25,
            "gnn_driven": 0.25,
            "hybrid": 0.25
        },
        "confidence_threshold": 0.5
    })
    
    # 缓存配置
    cache_config: Dict[str, Any] = field(default_factory=lambda: {
        "enable_cache": True,
        "cache_size": 1000,
        "cache_ttl": 3600,  # 秒
        "cleanup_interval": 300  # 秒
    })
    
    # 性能监控配置
    monitoring_config: Dict[str, Any] = field(default_factory=lambda: {
        "enable_monitoring": True,
        "history_size": 1000,
        "metrics_interval": 60,  # 秒
        "log_performance": True
    })


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
        self.weight = WeightConfig()
        
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
        
        # 权重配置
        if os.getenv('WEIGHT_DEFAULT_STRATEGY'):
            self.weight.default_strategy = os.getenv('WEIGHT_DEFAULT_STRATEGY')
        if os.getenv('WEIGHT_ENABLE_DYNAMIC'):
            self.weight.enable_dynamic_weights = os.getenv('WEIGHT_ENABLE_DYNAMIC').lower() == 'true'
        if os.getenv('WEIGHT_ENABLE_ASYNC'):
            self.weight.enable_async_weights = os.getenv('WEIGHT_ENABLE_ASYNC').lower() == 'true'
        if os.getenv('WEIGHT_DEFAULT_DOC'):
            self.weight.default_doc_weight = float(os.getenv('WEIGHT_DEFAULT_DOC'))
        if os.getenv('WEIGHT_DEFAULT_GRAPH'):
            self.weight.default_graph_weight = float(os.getenv('WEIGHT_DEFAULT_GRAPH'))
    
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
            
            # 更新权重配置
            if 'weight' in config_data:
                weight_data = config_data['weight']
                for key, value in weight_data.items():
                    if hasattr(self.weight, key):
                        setattr(self.weight, key, value)
                        
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"加载配置文件失败 {self.config_file}: {e}")
    
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
            },
            'weight': {
                'default_strategy': self.weight.default_strategy,
                'enable_dynamic_weights': self.weight.enable_dynamic_weights,
                'enable_async_weights': self.weight.enable_async_weights,
                'default_doc_weight': self.weight.default_doc_weight,
                'default_graph_weight': self.weight.default_graph_weight,
                'static_weights': self.weight.static_weights,
                'intent_driven_config': self.weight.intent_driven_config,
                'quality_driven_config': self.weight.quality_driven_config,
                'adaptive_config': self.weight.adaptive_config,
                'gnn_config': self.weight.gnn_config,
                'hybrid_config': self.weight.hybrid_config,
                'ensemble_config': self.weight.ensemble_config,
                'cache_config': self.weight.cache_config,
                'monitoring_config': self.weight.monitoring_config
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