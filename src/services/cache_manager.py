#!/usr/bin/env python3
"""
智能缓存管理器 - 支持Redis缓存、TTL和LRU策略
"""

import json
import time
import hashlib
import logging
from collections import OrderedDict, defaultdict
import threading
from threading import Lock
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Set

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    timestamp: float
    ttl: Optional[float] = None
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0
    tags: Set[str] = field(default_factory=set)
    data_version: Optional[str] = None
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def update_access(self):
        """更新访问信息"""
        self.access_count += 1
        self.last_access = time.time()

@dataclass
class CacheStats:
    """缓存统计信息"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    expired_entries: int = 0
    total_size_bytes: int = 0
    avg_response_time_ms: float = 0.0
    invalidations: int = 0
    auto_invalidations: int = 0
    
    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """缓存未命中率"""
        return 1.0 - self.hit_rate

@dataclass
class InvalidationRule:
    """缓存失效规则"""
    tag: str
    max_age: Optional[float] = None  # 最大存活时间（秒）
    max_idle_time: Optional[float] = None  # 最大空闲时间（秒）
    min_access_count: Optional[int] = None  # 最小访问次数
    data_version_check: bool = False  # 是否检查数据版本

@dataclass
class CacheConfig:
    """缓存配置"""
    cache_type: str = 'lru'  # 'lru', 'redis', 'hybrid'
    max_size: int = 1000
    default_ttl: Optional[float] = 3600
    redis_config: Optional[Dict[str, Any]] = None
    enable_compression: bool = False
    enable_async: bool = True

class LRUCache:
    """本地LRU缓存实现"""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = Lock()
        self.stats = CacheStats()
    
    def _calculate_size(self, value: Any) -> int:
        """计算值的大小（字节）"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value.encode('utf-8') if isinstance(value, str) else value)
            elif isinstance(value, (dict, list)):
                return len(json.dumps(value, ensure_ascii=False).encode('utf-8'))
            else:
                return len(str(value).encode('utf-8'))
        except Exception:
            return 100  # 默认大小
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        start_time = time.time()
        
        with self.lock:
            self.stats.total_requests += 1
            
            if key not in self.cache:
                self.stats.cache_misses += 1
                return None
            
            entry = self.cache[key]
            
            # 检查是否过期
            if entry.is_expired():
                del self.cache[key]
                self.stats.cache_misses += 1
                self.stats.expired_entries += 1
                return None
            
            # 更新访问信息并移到末尾（最近使用）
            entry.update_access()
            self.cache.move_to_end(key)
            
            self.stats.cache_hits += 1
            
            # 更新平均响应时间
            response_time = (time.time() - start_time) * 1000
            self.stats.avg_response_time_ms = (
                (self.stats.avg_response_time_ms * (self.stats.total_requests - 1) + response_time) /
                self.stats.total_requests
            )
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        with self.lock:
            # 如果键已存在，先删除
            if key in self.cache:
                old_entry = self.cache[key]
                self.stats.total_size_bytes -= old_entry.size_bytes
                del self.cache[key]
            
            # 创建新条目
            size_bytes = self._calculate_size(value)
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            
            # 检查是否需要驱逐
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # 添加新条目
            self.cache[key] = entry
            self.stats.total_size_bytes += size_bytes
            
            return True
    
    def _evict_lru(self):
        """驱逐最少使用的条目"""
        if not self.cache:
            return
        
        # 移除最旧的条目（OrderedDict的第一个）
        key, entry = self.cache.popitem(last=False)
        self.stats.total_size_bytes -= entry.size_bytes
        self.stats.evictions += 1
        logger.debug(f"LRU驱逐缓存条目: {key}")
    
    def delete(self, key: str) -> bool:
        """删除缓存条目"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                self.stats.total_size_bytes -= entry.size_bytes
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.stats = CacheStats()
    
    def cleanup_expired(self) -> int:
        """清理过期条目"""
        with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self.cache[key]
                self.stats.total_size_bytes -= entry.size_bytes
                del self.cache[key]
                self.stats.expired_entries += 1
            
            return len(expired_keys)

class RedisCache:
    """Redis缓存实现"""
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 default_ttl: Optional[float] = 3600,
                 key_prefix: str = 'hag_cache:'):
        
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")
        
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.stats = CacheStats()
        
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # 测试连接
            self.redis_client.ping()
            logger.info(f"Redis缓存连接成功: {host}:{port}/{db}")
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            raise
    
    def _make_key(self, key: str) -> str:
        """生成带前缀的键"""
        return f"{self.key_prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        start_time = time.time()
        
        try:
            self.stats.total_requests += 1
            
            redis_key = self._make_key(key)
            value_str = self.redis_client.get(redis_key)
            
            if value_str is None:
                self.stats.cache_misses += 1
                return None
            
            # 反序列化值
            try:
                value = json.loads(value_str)
            except json.JSONDecodeError:
                value = value_str
            
            self.stats.cache_hits += 1
            
            # 更新平均响应时间
            response_time = (time.time() - start_time) * 1000
            self.stats.avg_response_time_ms = (
                (self.stats.avg_response_time_ms * (self.stats.total_requests - 1) + response_time) /
                self.stats.total_requests
            )
            
            return value
            
        except Exception as e:
            logger.error(f"Redis获取失败: {e}")
            self.stats.cache_misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        try:
            redis_key = self._make_key(key)
            
            # 序列化值
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value, ensure_ascii=False)
            else:
                value_str = str(value)
            
            # 设置TTL
            ttl_seconds = int(ttl or self.default_ttl or 3600)
            
            # 存储到Redis
            result = self.redis_client.setex(redis_key, ttl_seconds, value_str)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis设置失败: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """删除缓存条目"""
        try:
            redis_key = self._make_key(key)
            result = self.redis_client.delete(redis_key)
            return bool(result)
        except Exception as e:
            logger.error(f"Redis删除失败: {e}")
            return False
    
    def clear(self):
        """清空缓存"""
        try:
            pattern = f"{self.key_prefix}*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
            self.stats = CacheStats()
        except Exception as e:
            logger.error(f"Redis清空失败: {e}")

class IntelligentCacheManager:
    """智能缓存管理器"""
    
    def __init__(self,
                 cache_type: str = 'lru',  # 'lru', 'redis', 'hybrid'
                 max_size: int = 1000,
                 default_ttl: Optional[float] = 3600,
                 redis_config: Optional[Dict[str, Any]] = None,
                 enable_compression: bool = False,
                 enable_async: bool = True):
        
        self.cache_type = cache_type
        self.enable_compression = enable_compression
        self.enable_async = enable_async
        self.executor = ThreadPoolExecutor(max_workers=4) if enable_async else None
        
        # 失效策略相关
        self.invalidation_rules: Dict[str, InvalidationRule] = {}
        self.tag_to_keys: Dict[str, Set[str]] = defaultdict(set)
        self.data_versions: Dict[str, str] = {}
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # 初始化缓存后端
        if cache_type == 'redis' and REDIS_AVAILABLE:
            config = redis_config or {}
            self.cache = RedisCache(
                default_ttl=default_ttl,
                **config
            )
        elif cache_type == 'hybrid' and REDIS_AVAILABLE:
            # 混合模式：本地LRU + Redis
            self.local_cache = LRUCache(max_size=max_size//2, default_ttl=300)  # 本地缓存TTL较短
            redis_config = redis_config or {}
            self.remote_cache = RedisCache(default_ttl=default_ttl, **redis_config)
            self.cache = None  # 混合模式不使用单一缓存
        else:
            # 默认使用LRU缓存
            self.cache = LRUCache(max_size=max_size, default_ttl=default_ttl)
            if cache_type == 'redis':
                logger.warning("Redis不可用，回退到LRU缓存")
        
        # 启动后台清理任务
        self._start_cleanup_task()
        
        logger.info(f"智能缓存管理器初始化完成 - 类型: {cache_type}, 最大大小: {max_size}")
    
    def _generate_cache_key(self, 
                           service: str, 
                           method: str, 
                           query: str, 
                           params: Optional[Dict[str, Any]] = None) -> str:
        """生成缓存键"""
        # 创建唯一的缓存键
        key_data = {
            'service': service,
            'method': method,
            'query': query,
            'params': params or {}
        }
        
        # 使用MD5哈希生成短键
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        key_hash = hashlib.md5(key_str.encode('utf-8')).hexdigest()
        
        return f"{service}:{method}:{key_hash}"
    
    def get(self, 
            service: str, 
            method: str, 
            query: str, 
            params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """获取缓存值"""
        cache_key = self._generate_cache_key(service, method, query, params)
        
        # 检查是否需要失效
        if self._should_invalidate(cache_key):
            self._invalidate_key(cache_key)
            return None
        
        if self.cache_type == 'hybrid':
            # 混合模式：先查本地，再查远程
            value = self.local_cache.get(cache_key)
            if value is not None:
                # 更新访问模式
                self._update_access_pattern(cache_key)
                return value
            
            value = self.remote_cache.get(cache_key)
            if value is not None:
                # 将远程缓存的值存到本地缓存
                self.local_cache.set(cache_key, value, ttl=300)
                # 更新访问模式
                self._update_access_pattern(cache_key)
                return value
            
            return None
        else:
            value = self.cache.get(cache_key)
            if value is not None:
                # 更新访问模式
                self._update_access_pattern(cache_key)
            return value
    
    def set(self, 
            service: str, 
            method: str, 
            query: str, 
            value: Any, 
            params: Optional[Dict[str, Any]] = None,
            ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        cache_key = self._generate_cache_key(service, method, query, params)
        
        if self.cache_type == 'hybrid':
            # 混合模式：同时存储到本地和远程
            local_result = self.local_cache.set(cache_key, value, ttl=min(ttl or 300, 300))
            remote_result = self.remote_cache.set(cache_key, value, ttl=ttl)
            return local_result and remote_result
        else:
            return self.cache.set(cache_key, value, ttl=ttl)
    
    async def get_async(self, 
                       service: str, 
                       method: str, 
                       query: str, 
                       params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """异步获取缓存值"""
        if not self.enable_async or not self.executor:
            return self.get(service, method, query, params)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.get, 
            service, method, query, params
        )
    
    async def set_async(self, 
                       service: str, 
                       method: str, 
                       query: str, 
                       value: Any, 
                       params: Optional[Dict[str, Any]] = None,
                       ttl: Optional[float] = None) -> bool:
        """异步设置缓存值"""
        if not self.enable_async or not self.executor:
            return self.set(service, method, query, value, params, ttl)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.set, 
            service, method, query, value, params, ttl
        )
    
    def invalidate_pattern(self, pattern: str) -> int:
        """根据模式失效缓存"""
        # 这里简化实现，实际应该支持模式匹配
        logger.info(f"缓存失效模式: {pattern}")
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if self.cache_type == 'hybrid':
            local_stats = asdict(self.local_cache.stats)
            remote_stats = asdict(self.remote_cache.stats)
            
            # 合并统计信息
            combined_stats = {
                'cache_type': 'hybrid',
                'local_cache': local_stats,
                'remote_cache': remote_stats,
                'total_hit_rate': (
                    (local_stats['cache_hits'] + remote_stats['cache_hits']) /
                    max(local_stats['total_requests'] + remote_stats['total_requests'], 1)
                ),
                'invalidations': local_stats.get('invalidations', 0) + remote_stats.get('invalidations', 0),
                'auto_invalidations': local_stats.get('auto_invalidations', 0) + remote_stats.get('auto_invalidations', 0),
                'active_rules': len(self.invalidation_rules),
                'tracked_tags': len(self.tag_to_keys)
            }
            return combined_stats
        else:
            stats = asdict(self.cache.stats)
            stats['cache_type'] = self.cache_type
            stats['active_rules'] = len(self.invalidation_rules)
            stats['tracked_tags'] = len(self.tag_to_keys)
            return stats
    
    def add_invalidation_rule(self, rule: InvalidationRule):
        """添加缓存失效规则"""
        self.invalidation_rules[rule.tag] = rule
        logger.info(f"添加缓存失效规则: {rule.tag}")
    
    def remove_invalidation_rule(self, tag: str):
        """移除缓存失效规则"""
        if tag in self.invalidation_rules:
            del self.invalidation_rules[tag]
            logger.info(f"移除缓存失效规则: {tag}")
    
    def invalidate_by_tag(self, tag: str):
        """根据标签失效缓存"""
        if tag in self.tag_to_keys:
            keys_to_invalidate = list(self.tag_to_keys[tag])
            for key in keys_to_invalidate:
                self._invalidate_key(key)
                if self.cache_type == 'hybrid':
                    self.local_cache.stats.invalidations += 1
                    self.remote_cache.stats.invalidations += 1
                else:
                    self.cache.stats.invalidations += 1
            logger.info(f"根据标签 {tag} 失效了 {len(keys_to_invalidate)} 个缓存项")
    
    def update_data_version(self, data_source: str, version: str):
        """更新数据版本"""
        old_version = self.data_versions.get(data_source)
        if old_version != version:
            self.data_versions[data_source] = version
            # 失效相关的缓存
            self.invalidate_by_tag(f"data_source:{data_source}")
            logger.info(f"数据源 {data_source} 版本更新: {old_version} -> {version}")
    
    def _should_invalidate(self, key: str) -> bool:
        """检查是否应该失效缓存"""
        try:
            # 获取缓存项的标签
            cache_item = self._get_cache_item_info(key)
            if not cache_item:
                return False
            
            current_time = time.time()
            
            for tag in cache_item.get('tags', set()):
                if tag in self.invalidation_rules:
                    rule = self.invalidation_rules[tag]
                    
                    # 检查最大存活时间
                    if rule.max_age and (current_time - cache_item['created_at']) > rule.max_age:
                        return True
                    
                    # 检查最大空闲时间
                    if rule.max_idle_time and (current_time - cache_item['last_accessed']) > rule.max_idle_time:
                        return True
                    
                    # 检查最小访问次数
                    if rule.min_access_count and cache_item['access_count'] < rule.min_access_count:
                        # 如果缓存项存在时间超过一定阈值但访问次数不足，则失效
                        if (current_time - cache_item['created_at']) > 3600:  # 1小时
                            return True
                    
                    # 检查数据版本
                    if rule.data_version_check and cache_item.get('data_version'):
                        data_source = tag.replace('data_source:', '')
                        if data_source in self.data_versions:
                            if cache_item['data_version'] != self.data_versions[data_source]:
                                return True
            
            return False
            
        except Exception as e:
            logger.error(f"检查缓存失效失败: {e}")
            return False
    
    def _invalidate_key(self, key: str):
        """失效指定的缓存键"""
        try:
            # 从所有标签映射中移除
            for tag_keys in self.tag_to_keys.values():
                tag_keys.discard(key)
            
            # 从访问模式中移除
            if key in self.access_patterns:
                del self.access_patterns[key]
            
            # 从缓存中删除
            if self.cache_type == 'hybrid':
                self.local_cache.delete(key)
                self.remote_cache.delete(key)
                self.local_cache.stats.auto_invalidations += 1
                self.remote_cache.stats.auto_invalidations += 1
            else:
                if hasattr(self.cache, 'delete'):
                    self.cache.delete(key)
                elif hasattr(self.cache, 'pop'):
                    self.cache.pop(key, None)
                self.cache.stats.auto_invalidations += 1
            
        except Exception as e:
            logger.error(f"失效缓存键失败: {e}")
    
    def _update_access_pattern(self, key: str):
        """更新访问模式"""
        current_time = time.time()
        self.access_patterns[key].append(current_time)
        
        # 只保留最近的访问记录（最多100个）
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-100:]
    
    def _get_cache_item_info(self, key: str) -> Optional[Dict[str, Any]]:
        """获取缓存项信息"""
        try:
            # 这里需要根据具体的缓存实现来获取缓存项的元信息
            # 简化实现，返回基本信息
            if hasattr(self.cache, 'get_item_info'):
                return self.cache.get_item_info(key)
            else:
                # 默认实现
                access_times = self.access_patterns.get(key, [])
                if access_times:
                    return {
                        'created_at': access_times[0] if access_times else time.time(),
                        'last_accessed': access_times[-1] if access_times else time.time(),
                        'access_count': len(access_times),
                        'tags': set(),
                        'data_version': None
                    }
                return None
        except Exception as e:
            logger.error(f"获取缓存项信息失败: {e}")
            return None
    
    def _start_cleanup_task(self):
        """启动后台清理任务"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(300)  # 每5分钟执行一次
                    self._perform_cleanup()
                except Exception as e:
                    logger.error(f"后台清理任务失败: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("后台清理任务已启动")
    
    def _perform_cleanup(self):
        """执行清理操作"""
        try:
            current_time = time.time()
            keys_to_check = list(self.access_patterns.keys())
            
            for key in keys_to_check:
                if self._should_invalidate(key):
                    self._invalidate_key(key)
            
            # 清理过期的访问模式记录
            cutoff_time = current_time - 86400  # 24小时前
            for key, access_times in list(self.access_patterns.items()):
                # 移除24小时前的访问记录
                recent_accesses = [t for t in access_times if t > cutoff_time]
                if recent_accesses:
                    self.access_patterns[key] = recent_accesses
                else:
                    del self.access_patterns[key]
            
            logger.debug("缓存清理完成")
            
        except Exception as e:
            logger.error(f"执行清理操作失败: {e}")
    
    def cleanup(self):
        """清理过期缓存"""
        if self.cache_type == 'hybrid':
            expired_local = self.local_cache.cleanup_expired()
            logger.info(f"清理本地过期缓存: {expired_local}个")
        elif hasattr(self.cache, 'cleanup_expired'):
            expired = self.cache.cleanup_expired()
            logger.info(f"清理过期缓存: {expired}个")
    
    def __del__(self):
        """析构函数"""
        if self.executor:
            self.executor.shutdown(wait=False)