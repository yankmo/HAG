#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from py2neo import Graph
import sys
sys.path.append('.')
from config.settings import get_config

def check_parkinson_relations():
    config = get_config()
    graph = Graph(config.neo4j.uri, auth=(config.neo4j.username, config.neo4j.password))
    
    print('=== 帕金森相关的关系类型 ===')
    result = graph.run("""
        MATCH (e:Entity)-[r:RELATION]-(related:Entity)
        WHERE toLower(e.name) CONTAINS '帕金森'
        RETURN DISTINCT r.type as relation_type, count(r) as count
        ORDER BY count DESC
    """).data()
    
    for item in result:
        print(f'{item["relation_type"]}: {item["count"]}')
    
    print('\n=== 治疗相关的关系 ===')
    result = graph.run("""
        MATCH (e:Entity)-[r:RELATION]-(related:Entity)
        WHERE toLower(e.name) CONTAINS '帕金森'
        AND (toLower(r.type) CONTAINS '治疗' OR toLower(r.description) CONTAINS '治疗' 
             OR toLower(related.name) CONTAINS '治疗' OR toLower(related.type) CONTAINS 'treatment')
        RETURN e.name as source, r.type as relation_type, r.description as description, 
               related.name as target, related.type as target_type
        LIMIT 10
    """).data()
    
    for item in result:
        print(f'{item["source"]} -> {item["relation_type"]} -> {item["target"]} ({item["target_type"]})')
        if item["description"]:
            print(f'  描述: {item["description"][:100]}...')
        print()
    
    print('\n=== 所有帕金森相关的关系 ===')
    result = graph.run("""
        MATCH (e:Entity)-[r:RELATION]-(related:Entity)
        WHERE toLower(e.name) CONTAINS '帕金森'
        RETURN e.name as source, r.type as relation_type, r.description as description, 
               related.name as target, related.type as target_type
        LIMIT 20
    """).data()
    
    for item in result:
        print(f'{item["source"]} -> {item["relation_type"]} -> {item["target"]} ({item["target_type"]})')
        if item["description"]:
            print(f'  描述: {item["description"][:100]}...')
        print()

if __name__ == "__main__":
    check_parkinson_relations()