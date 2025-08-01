#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from neo4j import GraphDatabase

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def check_parkinson_data():
    """检查帕金森相关数据的详细结构"""
    
    # Neo4j连接配置
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "hrx274700"
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    try:
        with driver.session() as session:
            print("=== 检查帕金森相关实体的详细信息 ===")
            
            # 1. 查找所有包含"帕金森"的实体
            query1 = """
            MATCH (e:Entity) 
            WHERE toLower(e.name) CONTAINS '帕金森' OR toLower(e.name) CONTAINS 'parkinson'
            RETURN e.name as name, e.type as type, e.description as description, 
                   keys(e) as all_properties, e as full_entity
            """
            result1 = session.run(query1)
            entities = list(result1)
            
            print(f"找到 {len(entities)} 个帕金森相关实体:")
            for i, record in enumerate(entities, 1):
                print(f"\n实体 {i}:")
                print(f"  名称: {record['name']}")
                print(f"  类型: {record['type']}")
                print(f"  描述: {record['description']}")
                print(f"  所有属性: {record['all_properties']}")
                print(f"  完整实体: {dict(record['full_entity'])}")
            
            print("\n=== 检查帕金森相关关系 ===")
            
            # 2. 查找帕金森实体的所有关系
            query2 = """
            MATCH (e1:Entity)-[r]->(e2:Entity)
            WHERE toLower(e1.name) CONTAINS '帕金森' OR toLower(e1.name) CONTAINS 'parkinson'
               OR toLower(e2.name) CONTAINS '帕金森' OR toLower(e2.name) CONTAINS 'parkinson'
            RETURN e1.name as from_entity, type(r) as relation_type, e2.name as to_entity,
                   r as full_relation
            LIMIT 20
            """
            result2 = session.run(query2)
            relations = list(result2)
            
            print(f"找到 {len(relations)} 个帕金森相关关系:")
            for i, record in enumerate(relations, 1):
                print(f"\n关系 {i}:")
                print(f"  从: {record['from_entity']}")
                print(f"  关系类型: {record['relation_type']}")
                print(f"  到: {record['to_entity']}")
                print(f"  完整关系: {dict(record['full_relation'])}")
            
            print("\n=== 检查治疗相关实体 ===")
            
            # 3. 查找治疗相关实体
            query3 = """
            MATCH (e:Entity) 
            WHERE toLower(e.name) CONTAINS '治疗' OR toLower(e.name) CONTAINS 'treatment'
               OR toLower(e.name) CONTAINS '药物' OR toLower(e.name) CONTAINS 'medicine'
            RETURN e.name as name, e.type as type, e.description as description
            LIMIT 10
            """
            result3 = session.run(query3)
            treatments = list(result3)
            
            print(f"找到 {len(treatments)} 个治疗相关实体:")
            for i, record in enumerate(treatments, 1):
                print(f"  {i}. {record['name']} (类型: {record['type']})")
            
            print("\n=== 检查所有关系类型 ===")
            
            # 4. 查看所有关系类型
            query4 = """
            MATCH ()-[r]->()
            RETURN DISTINCT type(r) as relation_type, count(r) as count
            ORDER BY count DESC
            """
            result4 = session.run(query4)
            relation_types = list(result4)
            
            print("所有关系类型:")
            for record in relation_types:
                print(f"  {record['relation_type']}: {record['count']} 个")
                
    except Exception as e:
        print(f"错误: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    check_parkinson_data()