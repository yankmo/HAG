#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试辅助工具模块
提供通用的测试辅助函数、类和装饰器
"""

import asyncio
import functools
import inspect
import json
import os
import random
import string
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Callable, Generator
from unittest.mock import Mock
import pytest
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MetricsData:
    """测试指标数据类"""
    response_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    cache_hit_rate: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def finish(self, success: bool = True, error_message: Optional[str] = None):
        """完成测试并记录指标"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error_message = error_message


class AsyncTestHelper:
    """异步测试辅助类"""
    
    @staticmethod
    def run_async(coro):
        """运行异步协程"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    @staticmethod
    async def wait_for_condition(
        condition: Callable[[], bool],
        timeout: float = 10.0,
        interval: float = 0.1
    ) -> bool:
        """等待条件满足"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition():
                return True
            await asyncio.sleep(interval)
        return False
    
    @staticmethod
    async def run_with_timeout(coro, timeout: float = 30.0):
        """运行协程并设置超时"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"操作超时 ({timeout}秒)")


class MockServiceFactory:
    """模拟服务工厂"""
    
    @staticmethod
    def create_embedding_service(dimension: int = 384) -> Mock:
        """创建模拟向量化服务"""
        service = Mock()
        service.embed_text.return_value = [random.random() for _ in range(dimension)]
        service.embed_documents.return_value = [
            [random.random() for _ in range(dimension)] for _ in range(3)
        ]
        service.get_embedding_dimension.return_value = dimension
        return service
    
    @staticmethod
    def create_vector_store() -> Mock:
        """创建模拟向量存储"""
        store = Mock()
        store.search.return_value = [
            {
                'content': f'测试文档内容 {i}',
                'score': 0.9 - i * 0.1,
                'metadata': {'source': f'doc_{i}.txt', 'chunk_id': i}
            }
            for i in range(3)
        ]
        store.add_documents.return_value = True
        store.delete_documents.return_value = True
        store.get_stats.return_value = {
            'total_documents': 1000,
            'total_vectors': 1000,
            'index_size': '50MB'
        }
        return store
    
    @staticmethod
    def create_graph_service() -> Mock:
        """创建模拟图谱服务"""
        service = Mock()
        
        # 模拟实体搜索
        service.search_entities_by_name.return_value = [
            {
                'name': f'实体_{i}',
                'type': 'Technology',
                'description': f'测试实体描述 {i}',
                'properties': {'field': 'Computer Science'}
            }
            for i in range(3)
        ]
        
        # 模拟关系搜索
        service.search_relationships_by_query.return_value = [
            {
                'source': f'实体_{i}',
                'target': f'实体_{i+1}',
                'type': 'RELATES_TO',
                'description': f'关系描述 {i}',
                'relevance_score': 0.9 - i * 0.1
            }
            for i in range(2)
        ]
        
        service.get_stats.return_value = {
            'total_nodes': 1000,
            'total_relationships': 2500,
            'query_count': 150
        }
        
        return service
    
    @staticmethod
    def create_llm_service() -> Mock:
        """创建模拟LLM服务"""
        service = Mock()
        service.generate_response.return_value = "这是一个测试回答，包含了相关的信息和解释。"
        service.generate_response_async.return_value = service.generate_response.return_value
        return service
    
    @staticmethod
    def create_cache_manager() -> Mock:
        """创建模拟缓存管理器"""
        cache = Mock()
        cache_data = {}
        
        def mock_get(key):
            return cache_data.get(key)
        
        def mock_set(key, value, ttl=None):
            cache_data[key] = value
            return True
        
        def mock_delete(key):
            return cache_data.pop(key, None) is not None
        
        def mock_clear():
            cache_data.clear()
            return True
        
        cache.get.side_effect = mock_get
        cache.set.side_effect = mock_set
        cache.delete.side_effect = mock_delete
        cache.clear.side_effect = mock_clear
        cache.get_stats.return_value = {
            'hits': 100,
            'misses': 50,
            'hit_rate': 0.67,
            'size': len(cache_data)
        }
        
        return cache


class DataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def random_string(length: int = 10) -> str:
        """生成随机字符串"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    @staticmethod
    def random_text(word_count: int = 50) -> str:
        """生成随机文本"""
        words = [
            '人工智能', '机器学习', '深度学习', '神经网络', '算法', '数据', '模型',
            '训练', '预测', '分类', '回归', '聚类', '特征', '向量', '矩阵',
            '计算机', '科学', '技术', '系统', '应用', '研究', '开发', '创新'
        ]
        return ' '.join(random.choices(words, k=word_count))
    
    @staticmethod
    def generate_document() -> Dict[str, Any]:
        """生成单个测试文档"""
        return {
            'id': f'doc_{random.randint(1, 1000):03d}',
            'content': DataGenerator.random_text(random.randint(20, 100)),
            'metadata': {
                'type': random.choice(['article', 'report', 'manual', 'guide']),
                'category': random.choice(['技术', '业务', '管理', '培训']),
                'created_at': datetime.now().isoformat(),
                'author': f'author_{random.randint(1, 10)}',
                'tags': random.sample(['重要', '紧急', '参考', '草稿', '已审核'], k=random.randint(1, 3))
            }
        }
    
    @staticmethod
    def generate_documents(count: int = 5) -> List[Dict[str, Any]]:
        """生成测试文档"""
        documents = []
        for i in range(count):
            doc = {
                'id': f'doc_{i+1:03d}',
                'content': DataGenerator.random_text(random.randint(20, 100)),
                'metadata': {
                    'type': random.choice(['article', 'report', 'manual', 'guide']),
                    'category': random.choice(['技术', '业务', '管理', '培训']),
                    'created_at': datetime.now().isoformat(),
                    'author': f'author_{random.randint(1, 10)}',
                    'tags': random.sample(['重要', '紧急', '参考', '草稿', '已审核'], k=random.randint(1, 3))
                }
            }
            documents.append(doc)
        return documents
    
    @staticmethod
    def generate_entities(count: int = 5) -> List[Dict[str, Any]]:
        """生成测试实体"""
        entity_types = ['Technology', 'Person', 'Organization', 'Concept']
        return [
            {
                'name': f'实体_{i}',
                'type': random.choice(entity_types),
                'description': DataGenerator.random_text(10),
                'properties': {
                    'field': random.choice(['Computer Science', 'Mathematics', 'Physics']),
                    'importance': random.uniform(0.1, 1.0)
                }
            }
            for i in range(count)
        ]
    
    @staticmethod
    def generate_relationships(count: int = 5) -> List[Dict[str, Any]]:
        """生成测试关系"""
        relation_types = ['RELATES_TO', 'INCLUDES', 'DEPENDS_ON', 'SIMILAR_TO']
        return [
            {
                'source': f'实体_{i}',
                'target': f'实体_{i+1}',
                'type': random.choice(relation_types),
                'description': DataGenerator.random_text(5),
                'properties': {
                    'strength': random.uniform(0.1, 1.0),
                    'confidence': random.uniform(0.5, 1.0)
                }
            }
            for i in range(count)
        ]
    
    @staticmethod
    def generate_queries(count: int = 5) -> List[Dict[str, Any]]:
        """生成测试查询"""
        questions = [
            '什么是人工智能？',
            '机器学习的基本原理是什么？',
            '深度学习和传统机器学习有什么区别？',
            '神经网络是如何工作的？',
            '自然语言处理的应用有哪些？'
        ]
        
        return [
            {
                'question': random.choice(questions),
                'expected_entities': [f'实体_{random.randint(0, 10)}' for _ in range(random.randint(1, 3))],
                'expected_keywords': DataGenerator.random_text(5).split(),
                'difficulty': random.choice(['easy', 'medium', 'hard']),
                'category': random.choice(['技术', '概念', '应用', '原理'])
            }
            for _ in range(count)
        ]


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = []
        self._start_time = None
        self._start_memory = None
    
    def __enter__(self):
        """进入上下文管理器"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        self.stop_monitoring()
        return False
    
    def start_monitoring(self):
        """开始监控"""
        self._start_time = time.time()
        try:
            import psutil
            process = psutil.Process()
            self._start_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            self._start_memory = None
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """停止监控并返回指标"""
        end_time = time.time()
        duration = end_time - self._start_time if self._start_time else 0
        
        memory_usage = None
        if self._start_memory:
            try:
                import psutil
                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage = end_memory - self._start_memory
            except ImportError:
                pass
        
        metrics = {
            'execution_time': duration,
            'memory_usage': memory_usage,
            'timestamp': end_time
        }
        
        self.metrics.append(metrics)
        return metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取最新的性能指标"""
        if self.metrics:
            return self.metrics[-1]
        else:
            # 如果还没有停止监控，返回当前状态
            if self._start_time:
                current_time = time.time()
                current_memory = None
                if self._start_memory:
                    try:
                        import psutil
                        process = psutil.Process()
                        end_memory = process.memory_info().rss / 1024 / 1024  # MB
                        current_memory = end_memory - self._start_memory
                    except ImportError:
                        current_memory = 0
                return {
                    'execution_time': current_time - self._start_time,
                    'memory_usage': current_memory or 0,
                    'timestamp': current_time
                }
            return {'execution_time': 0, 'memory_usage': 0, 'timestamp': time.time()}
    
    @contextmanager
    def monitor(self):
        """监控上下文管理器"""
        self.start_monitoring()
        try:
            yield self
        finally:
            self.stop_monitoring()


class ConcurrencyTestHelper:
    """并发测试辅助类"""
    
    @staticmethod
    def run_concurrent_tasks(
        task_func: Callable,
        task_args_list: List[tuple],
        max_workers: int = 10,
        timeout: float = 30.0
    ) -> List[Any]:
        """运行并发任务"""
        import concurrent.futures
        
        results = []
        errors = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_args = {
                executor.submit(task_func, *args): args
                for args in task_args_list
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_args, timeout=timeout):
                args = future_to_args[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors.append((args, str(e)))
        
        if errors:
            print(f"并发执行中发生 {len(errors)} 个错误:")
            for args, error in errors:
                print(f"  参数 {args}: {error}")
        
        return results
    
    @staticmethod
    async def run_concurrent_async_tasks(
        task_func: Callable,
        task_args_list: List[tuple],
        max_concurrent: int = 10,
        timeout: float = 30.0
    ) -> List[Any]:
        """运行并发异步任务"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_task(args):
            async with semaphore:
                return await task_func(*args)
        
        tasks = [run_task(args) for args in task_args_list]
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            return results
        except asyncio.TimeoutError:
            raise TimeoutError(f"并发异步任务超时 ({timeout}秒)")


class DatabaseTestHelper:
    """数据库测试辅助类"""
    
    @staticmethod
    @contextmanager
    def temporary_database():
        """临时数据库上下文管理器"""
        # 这里可以根据需要实现临时数据库的创建和清理
        # 例如使用 testcontainers 创建临时的 Neo4j 或 PostgreSQL 实例
        db_instance = None
        try:
            # 创建临时数据库实例
            yield db_instance
        finally:
            # 清理临时数据库实例
            if db_instance:
                pass  # 清理逻辑
    
    @staticmethod
    def wait_for_database_ready(
        check_func: Callable[[], bool],
        timeout: float = 30.0,
        interval: float = 1.0
    ) -> bool:
        """等待数据库就绪"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if check_func():
                    return True
            except Exception:
                pass
            time.sleep(interval)
        return False


class FileTestHelper:
    """文件测试辅助类"""
    
    @staticmethod
    @contextmanager
    def temporary_file(content: str = "", suffix: str = ".txt") -> Generator[str, None, None]:
        """临时文件上下文管理器"""
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        try:
            yield temp_path
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @staticmethod
    @contextmanager
    def temporary_directory() -> Generator[str, None, None]:
        """临时目录上下文管理器"""
        temp_dir = tempfile.mkdtemp()
        try:
            yield temp_dir
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class ConfigTestHelper:
    """配置测试辅助类"""
    
    @staticmethod
    @contextmanager
    def mock_environment(**env_vars):
        """模拟环境变量"""
        original_env = {}
        for key, value in env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = str(value)
        
        try:
            yield
        finally:
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
    
    @staticmethod
    def create_test_config(**overrides) -> Dict[str, Any]:
        """创建测试配置"""
        default_config = {
            'neo4j': {
                'uri': 'bolt://localhost:7687',
                'username': 'neo4j',
                'password': 'test_password',
                'database': 'test_db'
            },
            'weaviate': {
                'url': 'http://localhost:8080',
                'api_key': None
            },
            'ollama': {
                'base_url': 'http://localhost:11434',
                'model': 'gemma3:4b',
                'embedding_model': 'nomic-embed-text'
            },
            'cache': {
                'type': 'lru',
                'max_size': 1000,
                'ttl': 3600
            }
        }
        
        # 递归更新配置
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(default_config, overrides)
        return default_config


# 装饰器
def timeout(seconds: float):
    """超时装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if inspect.iscoroutinefunction(func):
                # 异步函数
                async def async_wrapper():
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
                return async_wrapper()
            else:
                # 同步函数 - 使用threading.Timer替代signal（Windows兼容）
                import threading
                
                result = [None]
                exception = [None]
                
                def target():
                    try:
                        result[0] = func(*args, **kwargs)
                    except Exception as e:
                        exception[0] = e
                
                thread = threading.Thread(target=target)
                thread.daemon = True
                thread.start()
                thread.join(timeout=seconds)
                
                if thread.is_alive():
                    raise TimeoutError(f"函数执行超时 ({seconds}秒)")
                
                if exception[0]:
                    raise exception[0]
                
                return result[0]
        
        return wrapper
    return decorator


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """重试装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise e
                    
                    print(f"第 {attempts} 次尝试失败: {e}，{current_delay}秒后重试...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        
        return wrapper
    return decorator


def measure_performance(func):
    """性能测量装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        monitor = PerformanceMonitor()
        with monitor.monitor():
            result = func(*args, **kwargs)
        
        metrics = monitor.metrics[-1]
        print(f"函数 {func.__name__} 性能指标:")
        print(f"  执行时间: {metrics['execution_time']:.4f}秒")
        if metrics['memory_usage']:
            print(f"  内存使用: {metrics['memory_usage']:.2f}MB")
        
        return result
    
    return wrapper


def skip_if_no_service(service_check_func: Callable[[], bool], reason: str = "服务不可用"):
    """服务不可用时跳过测试的装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not service_check_func():
                pytest.skip(reason)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# 断言辅助函数
def assert_response_structure(response: Dict[str, Any], expected_keys: List[str]):
    """断言响应结构"""
    assert isinstance(response, dict), "响应应该是字典类型"
    
    for key in expected_keys:
        assert key in response, f"响应中缺少必需的键: {key}"


def assert_performance_threshold(
    duration: float,
    max_duration: float,
    memory_usage: Optional[float] = None,
    max_memory: Optional[float] = None
):
    """断言性能在阈值内"""
    assert duration <= max_duration, f"执行时间 {duration:.4f}s 超过阈值 {max_duration}s"
    
    if memory_usage is not None and max_memory is not None:
        assert memory_usage <= max_memory, f"内存使用 {memory_usage:.2f}MB 超过阈值 {max_memory}MB"


def assert_cache_hit_rate(hit_rate: float, min_hit_rate: float = 0.5):
    """断言缓存命中率"""
    assert hit_rate >= min_hit_rate, f"缓存命中率 {hit_rate:.2%} 低于最小要求 {min_hit_rate:.2%}"


def assert_concurrent_success_rate(success_count: int, total_count: int, min_success_rate: float = 0.8):
    """断言并发成功率"""
    success_rate = success_count / total_count if total_count > 0 else 0
    assert success_rate >= min_success_rate, (
        f"并发成功率 {success_rate:.2%} 低于最小要求 {min_success_rate:.2%} "
        f"({success_count}/{total_count})"
    )


# 测试数据验证
def validate_document_structure(document: Dict[str, Any]) -> bool:
    """验证文档结构"""
    required_keys = ['content', 'metadata']
    return all(key in document for key in required_keys)


def validate_entity_structure(entity: Dict[str, Any]) -> bool:
    """验证实体结构"""
    required_keys = ['name', 'type']
    return all(key in entity for key in required_keys)


def validate_relationship_structure(relationship: Dict[str, Any]) -> bool:
    """验证关系结构"""
    required_keys = ['source', 'target', 'type']
    return all(key in relationship for key in required_keys)


def validate_search_result_structure(result: Dict[str, Any]) -> bool:
    """验证搜索结果结构"""
    required_keys = ['content', 'score']
    return all(key in result for key in required_keys)


# 测试报告生成
class ReportGenerator:
    """测试报告生成器"""
    
    def __init__(self):
        self.results = []
        self.start_time = datetime.now()
    
    def add_test_result(
        self,
        test_name: str,
        success: bool,
        duration: float,
        error_message: Optional[str] = None,
        **metadata
    ):
        """添加测试结果"""
        self.results.append({
            'test_name': test_name,
            'success': success,
            'duration': duration,
            'error_message': error_message,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata
        })
    
    def generate_summary(self) -> Dict[str, Any]:
        """生成测试摘要"""
        total_tests = len(self.results)
        successful_tests = sum(1 for result in self.results if result['success'])
        failed_tests = total_tests - successful_tests
        
        total_duration = sum(result['duration'] for result in self.results)
        avg_duration = total_duration / total_tests if total_tests > 0 else 0
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'total_duration': total_duration,
            'average_duration': avg_duration,
            'timestamp': datetime.now().isoformat()
        }
    
    def export_to_json(self, file_path: str):
        """导出为JSON文件"""
        report_data = {
            'summary': self.generate_summary(),
            'test_results': self.results
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)


# 全局测试报告生成器实例
test_report_generator = ReportGenerator()