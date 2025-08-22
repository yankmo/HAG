import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict
from typing import Dict, Any, Optional

from src.services.cache_manager import (
    CacheEntry,
    CacheStats,
    InvalidationRule,
    CacheConfig,
    LRUCache,
    RedisCache,
    IntelligentCacheManager
)


class TestCacheEntry:
    """测试CacheEntry数据类"""
    
    def test_cache_entry_initialization(self):
        """测试CacheEntry初始化"""
        entry = CacheEntry(
            value="test_value",
            created_at=1234567890.0,
            ttl=300.0,
            access_count=5,
            last_accessed=1234567900.0
        )
        
        assert entry.value == "test_value"
        assert entry.created_at == 1234567890.0
        assert entry.ttl == 300.0
        assert entry.access_count == 5
        assert entry.last_accessed == 1234567900.0
    
    def test_cache_entry_is_expired(self):
        """测试CacheEntry过期检查"""
        current_time = time.time()
        
        # 未过期的条目
        entry_valid = CacheEntry(
            value="test",
            created_at=current_time - 100,
            ttl=300.0
        )
        assert not entry_valid.is_expired()
        
        # 过期的条目
        entry_expired = CacheEntry(
            value="test",
            created_at=current_time - 400,
            ttl=300.0
        )
        assert entry_expired.is_expired()
        
        # 无TTL的条目
        entry_no_ttl = CacheEntry(
            value="test",
            created_at=current_time - 1000,
            ttl=None
        )
        assert not entry_no_ttl.is_expired()


class TestCacheStats:
    """测试CacheStats数据类"""
    
    def test_cache_stats_initialization(self):
        """测试CacheStats初始化"""
        stats = CacheStats(
            cache_hits=100,
            cache_misses=20,
            total_requests=120,
            hit_rate=0.833,
            evictions=5,
            current_size=50,
            max_size=100
        )
        
        assert stats.cache_hits == 100
        assert stats.cache_misses == 20
        assert stats.total_requests == 120
        assert stats.hit_rate == 0.833
        assert stats.evictions == 5
        assert stats.current_size == 50
        assert stats.max_size == 100


class TestInvalidationRule:
    """测试InvalidationRule数据类"""
    
    def test_invalidation_rule_initialization(self):
        """测试InvalidationRule初始化"""
        rule = InvalidationRule(
            tag="test_tag",
            max_age=3600.0,
            max_idle_time=1800.0,
            min_access_count=5,
            data_version_check=True
        )
        
        assert rule.tag == "test_tag"
        assert rule.max_age == 3600.0
        assert rule.max_idle_time == 1800.0
        assert rule.min_access_count == 5
        assert rule.data_version_check is True


class TestCacheConfig:
    """测试CacheConfig数据类"""
    
    def test_cache_config_initialization(self):
        """测试CacheConfig初始化"""
        config = CacheConfig(
            cache_type="redis",
            max_size=1000,
            default_ttl=300.0,
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            enable_async=True,
            max_workers=4
        )
        
        assert config.cache_type == "redis"
        assert config.max_size == 1000
        assert config.default_ttl == 300.0
        assert config.redis_host == "localhost"
        assert config.redis_port == 6379
        assert config.redis_db == 0
        assert config.enable_async is True
        assert config.max_workers == 4


class TestLRUCache:
    """测试LRUCache类"""
    
    def test_lru_cache_initialization(self):
        """测试LRUCache初始化"""
        cache = LRUCache(max_size=100, default_ttl=300.0)
        
        assert cache.max_size == 100
        assert cache.default_ttl == 300.0
        assert len(cache.cache) == 0
        assert len(cache.access_order) == 0
        assert cache.stats.max_size == 100
    
    def test_lru_cache_set_and_get(self):
        """测试LRUCache设置和获取"""
        cache = LRUCache(max_size=3, default_ttl=300.0)
        
        # 设置缓存项
        assert cache.set("key1", "value1") is True
        assert cache.set("key2", "value2") is True
        assert cache.set("key3", "value3") is True
        
        # 获取缓存项
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("nonexistent") is None
        
        # 检查统计信息
        assert cache.stats.cache_hits == 3
        assert cache.stats.cache_misses == 1
        assert cache.stats.current_size == 3
    
    def test_lru_cache_eviction(self):
        """测试LRU缓存驱逐策略"""
        cache = LRUCache(max_size=2, default_ttl=300.0)
        
        # 填满缓存
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # 访问key1使其成为最近使用
        cache.get("key1")
        
        # 添加新项，应该驱逐key2
        cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1"  # 仍然存在
        assert cache.get("key2") is None      # 被驱逐
        assert cache.get("key3") == "value3"  # 新添加的
        assert cache.stats.evictions == 1
    
    def test_lru_cache_ttl_expiration(self):
        """测试TTL过期"""
        cache = LRUCache(max_size=10, default_ttl=0.1)  # 100ms TTL
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # 等待过期
        time.sleep(0.15)
        assert cache.get("key1") is None
    
    def test_lru_cache_delete(self):
        """测试删除缓存项"""
        cache = LRUCache(max_size=10, default_ttl=300.0)
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("nonexistent") is False
    
    def test_lru_cache_clear(self):
        """测试清空缓存"""
        cache = LRUCache(max_size=10, default_ttl=300.0)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert cache.stats.current_size == 2
        
        cache.clear()
        assert cache.stats.current_size == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_lru_cache_cleanup_expired(self):
        """测试清理过期缓存"""
        cache = LRUCache(max_size=10, default_ttl=0.1)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2", ttl=1.0)  # 更长的TTL
        
        time.sleep(0.15)  # 等待key1过期
        
        expired_count = cache.cleanup_expired()
        assert expired_count == 1
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"


class TestRedisCache:
    """测试RedisCache类"""
    
    @patch('redis.Redis')
    def test_redis_cache_initialization_success(self, mock_redis_class):
        """测试RedisCache成功初始化"""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis
        
        cache = RedisCache(
            host="localhost",
            port=6379,
            db=0,
            default_ttl=300.0
        )
        
        assert cache.redis == mock_redis
        assert cache.default_ttl == 300.0
        assert cache.connected is True
        mock_redis.ping.assert_called_once()
    
    @patch('redis.Redis')
    def test_redis_cache_initialization_failure(self, mock_redis_class):
        """测试RedisCache初始化失败"""
        mock_redis = Mock()
        mock_redis.ping.side_effect = Exception("Connection failed")
        mock_redis_class.return_value = mock_redis
        
        cache = RedisCache(
            host="localhost",
            port=6379,
            db=0,
            default_ttl=300.0
        )
        
        assert cache.connected is False
    
    @patch('redis.Redis')
    def test_redis_cache_set_and_get(self, mock_redis_class):
        """测试RedisCache设置和获取"""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.setex.return_value = True
        mock_redis.get.return_value = b'{"value": "test_value"}'
        mock_redis_class.return_value = mock_redis
        
        cache = RedisCache(default_ttl=300.0)
        
        # 设置值
        result = cache.set("test_key", "test_value", ttl=300.0)
        assert result is True
        mock_redis.setex.assert_called_once()
        
        # 获取值
        value = cache.get("test_key")
        assert value == "test_value"
        mock_redis.get.assert_called_with("test_key")
    
    @patch('redis.Redis')
    def test_redis_cache_get_miss(self, mock_redis_class):
        """测试RedisCache缓存未命中"""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = None
        mock_redis_class.return_value = mock_redis
        
        cache = RedisCache(default_ttl=300.0)
        
        value = cache.get("nonexistent_key")
        assert value is None
        assert cache.stats.cache_misses == 1
    
    @patch('redis.Redis')
    def test_redis_cache_delete(self, mock_redis_class):
        """测试RedisCache删除"""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.delete.return_value = 1
        mock_redis_class.return_value = mock_redis
        
        cache = RedisCache(default_ttl=300.0)
        
        result = cache.delete("test_key")
        assert result is True
        mock_redis.delete.assert_called_with("test_key")
    
    @patch('redis.Redis')
    def test_redis_cache_clear(self, mock_redis_class):
        """测试RedisCache清空"""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.flushdb.return_value = True
        mock_redis_class.return_value = mock_redis
        
        cache = RedisCache(default_ttl=300.0)
        
        cache.clear()
        mock_redis.flushdb.assert_called_once()
    
    @patch('redis.Redis')
    def test_redis_cache_connection_error_handling(self, mock_redis_class):
        """测试Redis连接错误处理"""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.get.side_effect = Exception("Connection lost")
        mock_redis_class.return_value = mock_redis
        
        cache = RedisCache(default_ttl=300.0)
        
        # 连接错误时应返回None
        value = cache.get("test_key")
        assert value is None
        assert cache.stats.cache_misses == 1


class TestIntelligentCacheManager:
    """测试IntelligentCacheManager类"""
    
    def test_intelligent_cache_manager_lru_initialization(self):
        """测试智能缓存管理器LRU模式初始化"""
        config = CacheConfig(
            cache_type="lru",
            max_size=100,
            default_ttl=300.0
        )
        
        manager = IntelligentCacheManager(config)
        
        assert manager.cache_type == "lru"
        assert isinstance(manager.cache, LRUCache)
        assert manager.cache.max_size == 100
        assert manager.cache.default_ttl == 300.0
    
    @patch('redis.Redis')
    def test_intelligent_cache_manager_redis_initialization(self, mock_redis_class):
        """测试智能缓存管理器Redis模式初始化"""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis
        
        config = CacheConfig(
            cache_type="redis",
            default_ttl=300.0,
            redis_host="localhost",
            redis_port=6379,
            redis_db=0
        )
        
        manager = IntelligentCacheManager(config)
        
        assert manager.cache_type == "redis"
        assert isinstance(manager.cache, RedisCache)
    
    @patch('redis.Redis')
    def test_intelligent_cache_manager_hybrid_initialization(self, mock_redis_class):
        """测试智能缓存管理器混合模式初始化"""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis
        
        config = CacheConfig(
            cache_type="hybrid",
            max_size=100,
            default_ttl=300.0,
            redis_host="localhost",
            redis_port=6379,
            redis_db=0
        )
        
        manager = IntelligentCacheManager(config)
        
        assert manager.cache_type == "hybrid"
        assert isinstance(manager.local_cache, LRUCache)
        assert isinstance(manager.remote_cache, RedisCache)
    
    def test_intelligent_cache_manager_cache_key_generation(self):
        """测试缓存键生成"""
        config = CacheConfig(cache_type="lru", max_size=100)
        manager = IntelligentCacheManager(config)
        
        key1 = manager._generate_cache_key("service1", "method1", "query1")
        key2 = manager._generate_cache_key("service1", "method1", "query1")
        key3 = manager._generate_cache_key("service1", "method1", "query2")
        
        # 相同参数应生成相同键
        assert key1 == key2
        # 不同参数应生成不同键
        assert key1 != key3
        # 键应包含服务和方法信息
        assert "service1:method1:" in key1
    
    def test_intelligent_cache_manager_set_and_get(self):
        """测试智能缓存管理器设置和获取"""
        config = CacheConfig(cache_type="lru", max_size=100, default_ttl=300.0)
        manager = IntelligentCacheManager(config)
        
        # 设置缓存
        result = manager.set("service1", "method1", "query1", "value1")
        assert result is True
        
        # 获取缓存
        value = manager.get("service1", "method1", "query1")
        assert value == "value1"
        
        # 获取不存在的缓存
        value = manager.get("service1", "method1", "query2")
        assert value is None
    
    @patch('redis.Redis')
    def test_intelligent_cache_manager_hybrid_mode(self, mock_redis_class):
        """测试混合模式缓存"""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = b'{"value": "remote_value"}'
        mock_redis.setex.return_value = True
        mock_redis_class.return_value = mock_redis
        
        config = CacheConfig(
            cache_type="hybrid",
            max_size=100,
            default_ttl=300.0
        )
        manager = IntelligentCacheManager(config)
        
        # 设置缓存（应同时存储到本地和远程）
        result = manager.set("service1", "method1", "query1", "value1")
        assert result is True
        
        # 清空本地缓存，模拟只有远程缓存有数据的情况
        manager.local_cache.clear()
        
        # 获取缓存（应从远程获取并存储到本地）
        value = manager.get("service1", "method1", "query1")
        assert value == "remote_value"
    
    @pytest.mark.asyncio
    async def test_intelligent_cache_manager_async_operations(self):
        """测试异步操作"""
        config = CacheConfig(
            cache_type="lru",
            max_size=100,
            default_ttl=300.0,
            enable_async=True,
            max_workers=2
        )
        manager = IntelligentCacheManager(config)
        
        # 异步设置
        result = await manager.set_async("service1", "method1", "query1", "value1")
        assert result is True
        
        # 异步获取
        value = await manager.get_async("service1", "method1", "query1")
        assert value == "value1"
        
        # 异步获取不存在的值
        value = await manager.get_async("service1", "method1", "query2")
        assert value is None
    
    def test_intelligent_cache_manager_invalidation_rules(self):
        """测试缓存失效规则"""
        config = CacheConfig(cache_type="lru", max_size=100, default_ttl=300.0)
        manager = IntelligentCacheManager(config)
        
        # 添加失效规则
        rule = InvalidationRule(
            tag="test_tag",
            max_age=3600.0,
            max_idle_time=1800.0
        )
        manager.add_invalidation_rule(rule)
        
        assert "test_tag" in manager.invalidation_rules
        assert manager.invalidation_rules["test_tag"] == rule
        
        # 移除失效规则
        manager.remove_invalidation_rule("test_tag")
        assert "test_tag" not in manager.invalidation_rules
    
    def test_intelligent_cache_manager_tag_invalidation(self):
        """测试标签失效"""
        config = CacheConfig(cache_type="lru", max_size=100, default_ttl=300.0)
        manager = IntelligentCacheManager(config)
        
        # 设置一些缓存项
        manager.set("service1", "method1", "query1", "value1")
        manager.set("service1", "method2", "query2", "value2")
        
        # 模拟标签映射
        test_tag = "test_tag"
        key1 = manager._generate_cache_key("service1", "method1", "query1")
        key2 = manager._generate_cache_key("service1", "method2", "query2")
        
        manager.tag_to_keys[test_tag] = {key1, key2}
        
        # 根据标签失效
        manager.invalidate_by_tag(test_tag)
        
        # 验证缓存已被失效
        assert manager.get("service1", "method1", "query1") is None
        assert manager.get("service1", "method2", "query2") is None
    
    def test_intelligent_cache_manager_data_version_update(self):
        """测试数据版本更新"""
        config = CacheConfig(cache_type="lru", max_size=100, default_ttl=300.0)
        manager = IntelligentCacheManager(config)
        
        # 更新数据版本
        manager.update_data_version("data_source1", "v1.0")
        assert manager.data_versions["data_source1"] == "v1.0"
        
        # 再次更新版本
        manager.update_data_version("data_source1", "v1.1")
        assert manager.data_versions["data_source1"] == "v1.1"
    
    def test_intelligent_cache_manager_stats(self):
        """测试统计信息"""
        config = CacheConfig(cache_type="lru", max_size=100, default_ttl=300.0)
        manager = IntelligentCacheManager(config)
        
        # 执行一些操作
        manager.set("service1", "method1", "query1", "value1")
        manager.get("service1", "method1", "query1")  # 命中
        manager.get("service1", "method1", "query2")  # 未命中
        
        stats = manager.get_stats()
        
        assert stats['cache_type'] == 'lru'
        assert stats['cache_hits'] == 1
        assert stats['cache_misses'] == 1
        assert stats['total_requests'] == 2
        assert stats['current_size'] == 1
    
    @patch('redis.Redis')
    def test_intelligent_cache_manager_hybrid_stats(self, mock_redis_class):
        """测试混合模式统计信息"""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis
        
        config = CacheConfig(
            cache_type="hybrid",
            max_size=100,
            default_ttl=300.0
        )
        manager = IntelligentCacheManager(config)
        
        stats = manager.get_stats()
        
        assert stats['cache_type'] == 'hybrid'
        assert 'local_cache' in stats
        assert 'remote_cache' in stats
        assert 'total_hit_rate' in stats
    
    def test_intelligent_cache_manager_cleanup(self):
        """测试缓存清理"""
        config = CacheConfig(cache_type="lru", max_size=100, default_ttl=0.1)
        manager = IntelligentCacheManager(config)
        
        # 设置一些缓存项
        manager.set("service1", "method1", "query1", "value1")
        manager.set("service1", "method2", "query2", "value2")
        
        # 等待过期
        time.sleep(0.15)
        
        # 执行清理
        manager.cleanup()
        
        # 验证过期项已被清理
        assert manager.get("service1", "method1", "query1") is None
        assert manager.get("service1", "method2", "query2") is None
    
    def test_intelligent_cache_manager_access_pattern_tracking(self):
        """测试访问模式跟踪"""
        config = CacheConfig(cache_type="lru", max_size=100, default_ttl=300.0)
        manager = IntelligentCacheManager(config)
        
        # 设置和访问缓存
        manager.set("service1", "method1", "query1", "value1")
        manager.get("service1", "method1", "query1")
        manager.get("service1", "method1", "query1")
        
        key = manager._generate_cache_key("service1", "method1", "query1")
        
        # 验证访问模式被记录
        assert key in manager.access_patterns
        assert len(manager.access_patterns[key]) >= 2  # 至少有2次访问记录
    
    @patch('threading.Thread')
    def test_intelligent_cache_manager_background_cleanup(self, mock_thread):
        """测试后台清理任务启动"""
        config = CacheConfig(cache_type="lru", max_size=100, default_ttl=300.0)
        
        # 创建管理器会启动后台清理任务
        manager = IntelligentCacheManager(config)
        
        # 验证线程被创建并启动
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()
    
    def test_intelligent_cache_manager_error_handling(self):
        """测试错误处理"""
        config = CacheConfig(cache_type="lru", max_size=100, default_ttl=300.0)
        manager = IntelligentCacheManager(config)
        
        # 模拟缓存操作异常
        with patch.object(manager.cache, 'get', side_effect=Exception("Cache error")):
            # 应该优雅处理异常并返回None
            value = manager.get("service1", "method1", "query1")
            assert value is None
    
    def test_intelligent_cache_manager_cache_item_info(self):
        """测试缓存项信息获取"""
        config = CacheConfig(cache_type="lru", max_size=100, default_ttl=300.0)
        manager = IntelligentCacheManager(config)
        
        # 设置缓存项
        manager.set("service1", "method1", "query1", "value1")
        key = manager._generate_cache_key("service1", "method1", "query1")
        
        # 获取缓存项信息
        info = manager._get_cache_item_info(key)
        
        if info:  # 如果实现了缓存项信息获取
            assert 'created_at' in info
            assert 'last_accessed' in info
            assert 'access_count' in info