#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理Neo4j数据库脚本
"""

from py2neo import Graph
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_neo4j_database():
    """清理Neo4j数据库"""
    try:
        # 连接Neo4j
        graph = Graph('bolt://localhost:7687', auth=('neo4j', 'hrx274700'))
        logger.info("连接Neo4j成功")
        
        # 删除所有节点和关系
        graph.delete_all()
        logger.info("✅ Neo4j数据库已清空")
        
        # 验证清理结果
        result = graph.run("MATCH (n) RETURN count(n) as node_count").data()
        node_count = result[0]['node_count'] if result else 0
        
        result = graph.run("MATCH ()-[r]->() RETURN count(r) as rel_count").data()
        rel_count = result[0]['rel_count'] if result else 0
        
        print(f"📊 清理后统计: {node_count} 个节点, {rel_count} 个关系")
        
        return True
        
    except Exception as e:
        logger.error(f"清理Neo4j数据库失败: {e}")
        print(f"❌ 清理失败: {e}")
        return False

def main():
    """主函数"""
    print("🧹 开始清理Neo4j数据库...")
    
    if clear_neo4j_database():
        print("🎉 Neo4j数据库清理完成！")
    else:
        print("❌ Neo4j数据库清理失败！")

if __name__ == "__main__":
    main()