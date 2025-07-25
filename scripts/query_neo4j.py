#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j数据查询验证脚本
"""

from py2neo import Graph
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def query_neo4j_data():
    """查询Neo4j中的数据"""
    try:
        # 连接Neo4j
        graph = Graph('bolt://localhost:7687', auth=('neo4j', 'hrx274700'))
        logger.info("连接Neo4j成功")
        
        # 查询节点统计
        print("📊 数据库统计信息:")
        print("=" * 50)
        
        # 总节点数
        result = graph.run("MATCH (n) RETURN count(n) as node_count").data()
        node_count = result[0]['node_count'] if result else 0
        print(f"总节点数: {node_count}")
        
        # 总关系数
        result = graph.run("MATCH ()-[r]->() RETURN count(r) as rel_count").data()
        rel_count = result[0]['rel_count'] if result else 0
        print(f"总关系数: {rel_count}")
        
        # 按类型统计节点
        print("\n📋 节点类型统计:")
        result = graph.run("MATCH (n) RETURN labels(n) as labels, count(n) as count ORDER BY count DESC").data()
        for record in result:
            labels = record['labels']
            count = record['count']
            if labels:
                print(f"  {labels[0]}: {count}")
        
        # 按类型统计关系
        print("\n🔗 关系类型统计:")
        result = graph.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC").data()
        for record in result:
            rel_type = record['rel_type']
            count = record['count']
            print(f"  {rel_type}: {count}")
        
        # 显示一些示例节点
        print("\n🔍 示例节点 (前10个):")
        result = graph.run("MATCH (n) RETURN n.name as name, labels(n) as labels LIMIT 10").data()
        for record in result:
            name = record['name']
            labels = record['labels']
            if name and labels:
                print(f"  {name} ({labels[0]})")
        
        # 显示一些示例关系
        print("\n🔗 示例关系 (前10个):")
        result = graph.run("""
            MATCH (a)-[r]->(b) 
            RETURN a.name as source, type(r) as relation, b.name as target 
            LIMIT 10
        """).data()
        for record in result:
            source = record['source']
            relation = record['relation']
            target = record['target']
            if source and target:
                print(f"  {source} -{relation}-> {target}")
        
        # 查找帕金森氏症相关的核心节点
        print("\n🎯 帕金森氏症相关核心节点:")
        result = graph.run("""
            MATCH (n) 
            WHERE toLower(n.name) CONTAINS 'parkinson' 
            RETURN n.name as name, labels(n) as labels
        """).data()
        for record in result:
            name = record['name']
            labels = record['labels']
            if name and labels:
                print(f"  {name} ({labels[0]})")
        
        return True
        
    except Exception as e:
        logger.error(f"查询Neo4j数据失败: {e}")
        print(f"❌ 查询失败: {e}")
        return False

def main():
    """主函数"""
    print("🔍 开始查询Neo4j数据...")
    
    if query_neo4j_data():
        print("\n🎉 数据查询完成！")
        print("💡 你可以在Neo4j Browser中进一步探索数据:")
        print("   http://localhost:7474")
        print("\n📝 推荐查询语句:")
        print("   MATCH (n) RETURN n LIMIT 25  // 查看所有节点")
        print("   MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 25  // 查看节点和关系")
        print("   MATCH (n) WHERE toLower(n.name) CONTAINS 'parkinson' RETURN n  // 查找帕金森相关节点")
    else:
        print("❌ 数据查询失败！")

if __name__ == "__main__":
    main()