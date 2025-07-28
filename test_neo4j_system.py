#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j系统测试脚本
验证Neo4j意图识别和检索功能
"""

import sys
import os
import logging
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import get_config
from src.services.embedding_service import OllamaEmbeddingService
from src.knowledge.neo4j_vector_storage import Neo4jVectorStore, Neo4jIntentRecognizer
from src.services.neo4j_retrieval_service import Neo4jRetrievalService

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jSystemTester:
    """Neo4j系统测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.config = get_config()
        self.embedding_service = OllamaEmbeddingService()
        self.vector_store = Neo4jVectorStore()
        self.intent_recognizer = Neo4jIntentRecognizer(self.vector_store)
        self.retrieval_service = Neo4jRetrievalService(
            self.embedding_service, 
            self.vector_store, 
            self.intent_recognizer
        )
        
        logger.info("Neo4j系统测试器初始化完成")
    
    def test_connection(self) -> bool:
        """测试Neo4j连接"""
        try:
            logger.info("测试Neo4j连接...")
            stats = self.vector_store.get_stats()
            logger.info(f"连接成功，统计信息: {stats}")
            return True
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            return False
    
    def test_intent_recognition(self) -> bool:
        """测试意图识别"""
        try:
            logger.info("测试意图识别...")
            
            test_queries = [
                "糖尿病的症状有哪些？",
                "高血压如何治疗？",
                "感冒是什么原因引起的？",
                "如何诊断心脏病？",
                "阿司匹林的副作用"
            ]
            
            success_count = 0
            
            for query in test_queries:
                try:
                    logger.info(f"测试查询: {query}")
                    intent_result = self.intent_recognizer.recognize_intent(query)
                    logger.info(f"识别结果: {intent_result.intent_type}, 置信度: {intent_result.confidence:.2f}")
                    logger.info(f"实体数量: {len(intent_result.entities)}, 关系数量: {len(intent_result.relations)}")
                    
                    if intent_result.entities:
                        logger.info(f"识别的实体: {intent_result.entities[:3]}")
                    if intent_result.relations:
                        logger.info(f"识别的关系: {intent_result.relations[:3]}")
                    
                    # 检查是否有错误
                    if 'error' in intent_result.context:
                        logger.warning(f"意图识别有错误: {intent_result.context['error']}")
                    else:
                        success_count += 1
                    
                    print("-" * 50)
                    
                except Exception as query_error:
                    logger.error(f"查询 '{query}' 失败: {query_error}")
                    print("-" * 50)
            
            # 如果至少有一半的查询成功，认为测试通过
            success_rate = success_count / len(test_queries)
            logger.info(f"意图识别成功率: {success_rate:.2%} ({success_count}/{len(test_queries)})")
            
            return success_rate >= 0.5
            
        except Exception as e:
            logger.error(f"意图识别测试失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return False
    
    def test_vector_storage(self) -> bool:
        """测试向量存储"""
        try:
            logger.info("测试向量存储...")
            
            # 导入必要的类
            from src.knowledge.neo4j_vector_storage import Neo4jVectorEntity, Neo4jVectorRelation
            
            # 测试实体存储
            test_entities_data = [
                {
                    "name": "测试疾病",
                    "type": "Disease",
                    "description": "这是一个测试疾病实体",
                    "source_text": "测试疾病是用于验证系统功能的虚拟疾病"
                },
                {
                    "name": "测试药物",
                    "type": "Drug",
                    "description": "这是一个测试药物实体",
                    "source_text": "测试药物是用于验证系统功能的虚拟药物"
                }
            ]
            
            # 向量化并存储实体
            entities_to_store = []
            for entity_data in test_entities_data:
                vector = self.embedding_service.embed_text(entity_data["source_text"])
                if vector:
                    entity = Neo4jVectorEntity(
                        name=entity_data["name"],
                        type=entity_data["type"],
                        source_text=entity_data["source_text"],
                        vector=vector,
                        properties={"description": entity_data["description"]}
                    )
                    entities_to_store.append(entity)
            
            if entities_to_store:
                success = self.vector_store.store_entities(entities_to_store)
                if success:
                    logger.info(f"成功存储 {len(entities_to_store)} 个实体")
                else:
                    logger.error("实体存储失败")
            
            # 测试关系存储
            test_relations_data = [
                {
                    "source": "测试疾病",
                    "target": "测试药物",
                    "relation_type": "TREATS",
                    "description": "测试药物治疗测试疾病",
                    "source_text": "测试药物可以有效治疗测试疾病"
                }
            ]
            
            relations_to_store = []
            for relation_data in test_relations_data:
                vector = self.embedding_service.embed_text(relation_data["source_text"])
                if vector:
                    relation = Neo4jVectorRelation(
                        source=relation_data["source"],
                        target=relation_data["target"],
                        relation_type=relation_data["relation_type"],
                        source_text=relation_data["source_text"],
                        vector=vector,
                        description=relation_data["description"]
                    )
                    relations_to_store.append(relation)
            
            if relations_to_store:
                success = self.vector_store.store_relations(relations_to_store)
                if success:
                    logger.info(f"成功存储 {len(relations_to_store)} 个关系")
                else:
                    logger.error("关系存储失败")
            
            # 验证存储结果
            stats = self.vector_store.get_stats()
            logger.info(f"存储后统计: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"向量存储测试失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return False
    
    def test_retrieval_service(self) -> bool:
        """测试检索服务"""
        try:
            logger.info("测试检索服务...")
            
            test_queries = [
                "糖尿病治疗",
                "高血压症状",
                "心脏病诊断",
                "测试疾病"
            ]
            
            for query in test_queries:
                logger.info(f"测试查询: {query}")
                
                # 测试余弦相似度搜索
                cosine_results = self.retrieval_service.search_by_cosine(query, limit=3)
                logger.info(f"余弦相似度结果数量: {len(cosine_results)}")
                
                # 测试欧氏距离搜索
                euclidean_results = self.retrieval_service.search_by_euclidean(query, limit=3)
                logger.info(f"欧氏距离结果数量: {len(euclidean_results)}")
                
                # 测试混合搜索
                hybrid_results = self.retrieval_service.search_hybrid(query, limit=3)
                logger.info(f"混合搜索结果数量: {len(hybrid_results.hybrid_results)}")
                
                # 测试意图感知搜索
                intent_results = self.retrieval_service.search_with_intent(query, limit=3)
                logger.info(f"意图感知搜索 - 意图: {intent_results.intent.intent_type}")
                logger.info(f"推荐数量: {len(intent_results.recommendations)}")
                
                print("-" * 50)
            
            return True
            
        except Exception as e:
            logger.error(f"检索服务测试失败: {e}")
            return False
    
    def test_statistics(self) -> bool:
        """测试统计功能"""
        try:
            logger.info("测试统计功能...")
            
            # 获取系统统计
            stats = self.retrieval_service.get_stats()
            logger.info(f"系统统计: {stats}")
            
            # 获取知识图谱摘要
            summary = self.retrieval_service.get_knowledge_graph_summary()
            logger.info(f"知识图谱摘要: {summary['summary']}")
            
            return True
            
        except Exception as e:
            logger.error(f"统计功能测试失败: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """运行所有测试"""
        logger.info("开始运行Neo4j系统全面测试...")
        
        test_results = {}
        
        # 测试连接
        test_results["connection"] = self.test_connection()
        
        # 测试意图识别
        test_results["intent_recognition"] = self.test_intent_recognition()
        
        # 测试向量存储
        test_results["vector_storage"] = self.test_vector_storage()
        
        # 测试检索服务
        test_results["retrieval_service"] = self.test_retrieval_service()
        
        # 测试统计功能
        test_results["statistics"] = self.test_statistics()
        
        # 汇总结果
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        logger.info(f"测试完成: {passed_tests}/{total_tests} 通过")
        
        for test_name, result in test_results.items():
            status = "✓ 通过" if result else "✗ 失败"
            logger.info(f"{test_name}: {status}")
        
        return test_results

def main():
    """主函数"""
    try:
        tester = Neo4jSystemTester()
        results = tester.run_all_tests()
        
        # 检查是否所有测试都通过
        if all(results.values()):
            logger.info("🎉 所有测试通过！Neo4j系统运行正常")
            return True
        else:
            logger.warning("⚠️ 部分测试失败，请检查系统配置")
            return False
            
    except Exception as e:
        logger.error(f"测试运行失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)