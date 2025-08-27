import pytest
from unittest.mock import Mock

from src.services.cache_manager import (
    LRUCache,
    CacheConfig,
    IntelligentCacheManager
)


class TestLRUCache:
    """测试LRUCache基本功能"""
    
    def test_lru_cache_set_and_get(self):
        """测试LRUCache设置和获取"""
        cache = LRUCache(max_size=3, default_ttl=300.0)
        
        # 设置缓存项
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # 获取缓存项
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("nonexistent") is None
    
    def test_lru_cache_eviction(self):
        """测试LRU缓存驱逐策略"""
        cache = LRUCache(max_size=2, default_ttl=300.0)
        
        # 填满缓存
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # 添加新项，应该驱逐最旧的项
        cache.set("key3", "value3")
        
        assert cache.get("key3") == "value3"
        # 验证缓存大小限制
        assert len([k for k in ["key1", "key2", "key3"] if cache.get(k) is not None]) <= 2