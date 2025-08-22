#!/usr/bin/env python3
"""
数据库连接池优化模块 - 优化Neo4j和Weaviate连接管理
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from contextlib import contextmanager
from queue import Queue, Empty
import weakref

try:
    from neo4j import GraphDatabase, Driver, Session
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None
    Driver = None
    Session = None

try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    weaviate = None

logger = logging.getLogger(__name__)

@dataclass
class ConnectionStats:
    """连接池统计信息"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    created_connections: int = 0
    closed_connections: int = 0
    connection_errors: int = 0
    avg_connection_time: float = 0.0
    max_connections_reached: int = 0
    
    @property
    def connection_utilization(self) -> float:
        """连接利用率"""
        if self.total_connections == 0:
            return 0.0
        return self.active_connections / self.total_connections

class ConnectionWrapper:
    """连接包装器"""
    
    def __init__(self, connection: Any, pool: 'BaseConnectionPool'):
        self.connection = connection
        self.pool = weakref.ref(pool)
        self.created_at = time.time()
        self.last_used = time.time()
        self.use_count = 0
        self.is_active = False
        self.lock = threading.Lock()
    
    def acquire(self):
        """获取连接"""
        with self.lock:
            self.is_active = True
            self.last_used = time.time()
            self.use_count += 1
            return self.connection
    
    def release(self):
        """释放连接"""
        with self.lock:
            self.is_active = False
            self.last_used = time.time()
    
    def is_expired(self, max_age: float) -> bool:
        """检查连接是否过期"""
        return time.time() - self.created_at > max_age
    
    def is_idle_too_long(self, max_idle: float) -> bool:
        """检查连接是否空闲过久"""
        return not self.is_active and (time.time() - self.last_used > max_idle)
    
    def close(self):
        """关闭连接"""
        try:
            if hasattr(self.connection, 'close'):
                self.connection.close()
        except Exception as e:
            logger.warning(f"关闭连接时出错: {e}")

class BaseConnectionPool:
    """基础连接池"""
    
    def __init__(self,
                 min_connections: int = 2,
                 max_connections: int = 10,
                 max_connection_age: float = 3600,  # 1小时
                 max_idle_time: float = 300,  # 5分钟
                 connection_timeout: float = 30,
                 cleanup_interval: float = 60):  # 1分钟
        
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.max_connection_age = max_connection_age
        self.max_idle_time = max_idle_time
        self.connection_timeout = connection_timeout
        self.cleanup_interval = cleanup_interval
        
        self.connections: Queue[ConnectionWrapper] = Queue(maxsize=max_connections)
        self.all_connections: List[ConnectionWrapper] = []
        self.stats = ConnectionStats()
        self.lock = threading.RLock()
        
        # 启动清理线程
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"连接池初始化 - 最小: {min_connections}, 最大: {max_connections}")
    
    def _create_connection(self) -> Any:
        """创建新连接 - 子类需要实现"""
        raise NotImplementedError
    
    def _validate_connection(self, connection: Any) -> bool:
        """验证连接有效性 - 子类需要实现"""
        raise NotImplementedError
    
    @contextmanager
    def get_connection(self):
        """获取连接的上下文管理器"""
        wrapper = None
        try:
            wrapper = self._acquire_connection()
            yield wrapper.connection
        finally:
            if wrapper:
                self._release_connection(wrapper)
    
    def _acquire_connection(self) -> ConnectionWrapper:
        """获取连接"""
        start_time = time.time()
        
        try:
            # 尝试从池中获取空闲连接
            try:
                wrapper = self.connections.get(timeout=0.1)
                if self._validate_connection(wrapper.connection):
                    wrapper.acquire()
                    self.stats.active_connections += 1
                    return wrapper
                else:
                    # 连接无效，关闭并创建新的
                    self._close_connection(wrapper)
            except Empty:
                pass
            
            # 如果没有空闲连接，创建新连接
            with self.lock:
                if len(self.all_connections) < self.max_connections:
                    wrapper = self._create_new_connection()
                    if wrapper:
                        wrapper.acquire()
                        self.stats.active_connections += 1
                        return wrapper
                else:
                    self.stats.max_connections_reached += 1
            
            # 等待连接可用
            wrapper = self.connections.get(timeout=self.connection_timeout)
            if self._validate_connection(wrapper.connection):
                wrapper.acquire()
                self.stats.active_connections += 1
                return wrapper
            else:
                self._close_connection(wrapper)
                raise Exception("获取到无效连接")
                
        except Exception as e:
            self.stats.connection_errors += 1
            logger.error(f"获取连接失败: {e}")
            raise
        finally:
            # 更新平均连接时间
            connection_time = time.time() - start_time
            self.stats.avg_connection_time = (
                (self.stats.avg_connection_time * self.stats.created_connections + connection_time) /
                max(self.stats.created_connections + 1, 1)
            )
    
    def _create_new_connection(self) -> Optional[ConnectionWrapper]:
        """创建新连接"""
        try:
            connection = self._create_connection()
            wrapper = ConnectionWrapper(connection, self)
            
            self.all_connections.append(wrapper)
            self.stats.total_connections += 1
            self.stats.created_connections += 1
            
            logger.debug(f"创建新连接 - 总数: {len(self.all_connections)}")
            return wrapper
            
        except Exception as e:
            self.stats.connection_errors += 1
            logger.error(f"创建连接失败: {e}")
            return None
    
    def _release_connection(self, wrapper: ConnectionWrapper):
        """释放连接"""
        try:
            wrapper.release()
            self.stats.active_connections = max(0, self.stats.active_connections - 1)
            
            # 检查连接是否仍然有效
            if (not wrapper.is_expired(self.max_connection_age) and 
                self._validate_connection(wrapper.connection)):
                self.connections.put(wrapper, timeout=1.0)
            else:
                self._close_connection(wrapper)
                
        except Exception as e:
            logger.error(f"释放连接失败: {e}")
            self._close_connection(wrapper)
    
    def _close_connection(self, wrapper: ConnectionWrapper):
        """关闭连接"""
        try:
            wrapper.close()
            
            with self.lock:
                if wrapper in self.all_connections:
                    self.all_connections.remove(wrapper)
                    self.stats.total_connections = max(0, self.stats.total_connections - 1)
                    self.stats.closed_connections += 1
                    
        except Exception as e:
            logger.error(f"关闭连接失败: {e}")
    
    def _cleanup_loop(self):
        """清理循环"""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired_connections()
            except Exception as e:
                logger.error(f"连接池清理失败: {e}")
    
    def _cleanup_expired_connections(self):
        """清理过期连接"""
        with self.lock:
            expired_connections = []
            
            for wrapper in self.all_connections[:]:
                if (wrapper.is_expired(self.max_connection_age) or 
                    wrapper.is_idle_too_long(self.max_idle_time)):
                    expired_connections.append(wrapper)
            
            for wrapper in expired_connections:
                self._close_connection(wrapper)
            
            if expired_connections:
                logger.info(f"清理过期连接: {len(expired_connections)}个")
            
            # 更新统计信息
            self.stats.idle_connections = len(self.all_connections) - self.stats.active_connections
    
    def get_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        with self.lock:
            stats_dict = {
                'total_connections': self.stats.total_connections,
                'active_connections': self.stats.active_connections,
                'idle_connections': len(self.all_connections) - self.stats.active_connections,
                'created_connections': self.stats.created_connections,
                'closed_connections': self.stats.closed_connections,
                'connection_errors': self.stats.connection_errors,
                'avg_connection_time': self.stats.avg_connection_time,
                'max_connections_reached': self.stats.max_connections_reached,
                'connection_utilization': self.stats.connection_utilization,
                'pool_config': {
                    'min_connections': self.min_connections,
                    'max_connections': self.max_connections,
                    'max_connection_age': self.max_connection_age,
                    'max_idle_time': self.max_idle_time
                }
            }
            return stats_dict
    
    def close_all(self):
        """关闭所有连接"""
        with self.lock:
            for wrapper in self.all_connections[:]:
                self._close_connection(wrapper)
            
            # 清空队列
            while not self.connections.empty():
                try:
                    self.connections.get_nowait()
                except Empty:
                    break
            
            logger.info("所有连接已关闭")

class Neo4jConnectionPool(BaseConnectionPool):
    """Neo4j连接池"""
    
    def __init__(self, 
                 uri: str,
                 username: str,
                 password: str,
                 database: str = "neo4j",
                 **kwargs):
        
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available. Install with: pip install neo4j")
        
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        
        # 创建驱动器
        self.driver = GraphDatabase.driver(
            uri, 
            auth=(username, password),
            max_connection_lifetime=kwargs.get('max_connection_lifetime', 3600),
            max_connection_pool_size=kwargs.get('max_connections', 10),
            connection_acquisition_timeout=kwargs.get('connection_timeout', 30)
        )
        
        super().__init__(**kwargs)
        
        # 预创建最小连接数
        self._ensure_min_connections()
        
        logger.info(f"Neo4j连接池初始化完成 - URI: {uri}, 数据库: {database}")
    
    def _create_connection(self) -> Session:
        """创建Neo4j会话"""
        return self.driver.session(database=self.database)
    
    def _validate_connection(self, session: Session) -> bool:
        """验证Neo4j会话有效性"""
        try:
            # 执行简单查询测试连接
            result = session.run("RETURN 1 as test")
            result.single()
            return True
        except Exception:
            return False
    
    def _ensure_min_connections(self):
        """确保最小连接数"""
        with self.lock:
            while len(self.all_connections) < self.min_connections:
                wrapper = self._create_new_connection()
                if wrapper:
                    self.connections.put(wrapper)
                else:
                    break
    
    def close_all(self):
        """关闭所有连接和驱动器"""
        super().close_all()
        if self.driver:
            self.driver.close()
            logger.info("Neo4j驱动器已关闭")

class WeaviateConnectionPool(BaseConnectionPool):
    """Weaviate连接池"""
    
    def __init__(self, 
                 url: str,
                 api_key: Optional[str] = None,
                 additional_headers: Optional[Dict[str, str]] = None,
                 **kwargs):
        
        if not WEAVIATE_AVAILABLE:
            raise ImportError("Weaviate client not available. Install with: pip install weaviate-client")
        
        self.url = url
        self.api_key = api_key
        self.additional_headers = additional_headers or {}
        
        super().__init__(**kwargs)
        
        # 预创建最小连接数
        self._ensure_min_connections()
        
        logger.info(f"Weaviate连接池初始化完成 - URL: {url}")
    
    def _create_connection(self):
        """创建Weaviate客户端"""
        try:
            import weaviate.classes.init as wvc
            from urllib.parse import urlparse
            
            # 解析URL获取主机和端口
            parsed_url = urlparse(self.url)
            host = parsed_url.hostname or 'localhost'
            port = parsed_url.port or 8080
            
            # 使用新版本的连接方式
            client = weaviate.connect_to_local(
                host=host,
                port=port,
                grpc_port=50051,
                headers=self.additional_headers,
                skip_init_checks=True,
                additional_config=wvc.AdditionalConfig(
                    timeout=wvc.Timeout(init=30, query=60, insert=120)
                )
            )
            
            return client
        except Exception as e:
            logger.error(f"创建连接失败: {e}")
            return None
    
    def _validate_connection(self, client) -> bool:
        """验证Weaviate客户端有效性"""
        try:
            if client is None:
                return False
            # 检查客户端是否连接
            client.collections.list_all()
            return True
        except Exception:
            return False
    
    def _ensure_min_connections(self):
        """确保最小连接数"""
        with self.lock:
            while len(self.all_connections) < self.min_connections:
                wrapper = self._create_new_connection()
                if wrapper:
                    self.connections.put(wrapper)
                else:
                    break

class ConnectionPoolManager:
    """连接池管理器"""
    
    def __init__(self):
        self.pools: Dict[str, BaseConnectionPool] = {}
        self.lock = threading.Lock()
        logger.info("连接池管理器初始化完成")
    
    def create_neo4j_pool(self, 
                          name: str,
                          uri: str,
                          username: str,
                          password: str,
                          database: str = "neo4j",
                          **kwargs) -> Neo4jConnectionPool:
        """创建Neo4j连接池"""
        with self.lock:
            if name in self.pools:
                raise ValueError(f"连接池 '{name}' 已存在")
            
            pool = Neo4jConnectionPool(
                uri=uri,
                username=username,
                password=password,
                database=database,
                **kwargs
            )
            
            self.pools[name] = pool
            logger.info(f"Neo4j连接池 '{name}' 创建成功")
            return pool
    
    def create_weaviate_pool(self,
                            name: str,
                            url: str,
                            api_key: Optional[str] = None,
                            additional_headers: Optional[Dict[str, str]] = None,
                            **kwargs) -> WeaviateConnectionPool:
        """创建Weaviate连接池"""
        with self.lock:
            if name in self.pools:
                raise ValueError(f"连接池 '{name}' 已存在")
            
            pool = WeaviateConnectionPool(
                url=url,
                api_key=api_key,
                additional_headers=additional_headers,
                **kwargs
            )
            
            self.pools[name] = pool
            logger.info(f"Weaviate连接池 '{name}' 创建成功")
            return pool
    
    def get_pool(self, name: str) -> Optional[BaseConnectionPool]:
        """获取连接池"""
        return self.pools.get(name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有连接池统计信息"""
        stats = {}
        for name, pool in self.pools.items():
            stats[name] = pool.get_stats()
        return stats
    
    def close_all_pools(self):
        """关闭所有连接池"""
        with self.lock:
            for name, pool in self.pools.items():
                try:
                    pool.close_all()
                    logger.info(f"连接池 '{name}' 已关闭")
                except Exception as e:
                    logger.error(f"关闭连接池 '{name}' 失败: {e}")
            
            self.pools.clear()
            logger.info("所有连接池已关闭")

# 全局连接池管理器实例
connection_pool_manager = ConnectionPoolManager()