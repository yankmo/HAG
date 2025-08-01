from py2neo import Graph
import logging

# 连接Neo4j
graph = Graph('bolt://localhost:7687', auth=('neo4j', 'hrx274700'))

print('=== 检查数据库连接 ===')
try:
    result = graph.run('RETURN 1 as test').data()
    print('Neo4j连接成功')
except Exception as e:
    print(f'Neo4j连接失败: {e}')
    exit(1)

print('\n=== 检查节点标签 ===')
labels = graph.run('CALL db.labels()').data()
for label in labels:
    print(f'标签: {label}')

print('\n=== 检查节点数量 ===')
node_counts = graph.run('MATCH (n) RETURN labels(n) as labels, count(n) as count').data()
for item in node_counts:
    print(f'标签 {item["labels"]}: {item["count"]} 个节点')

print('\n=== 检查实体节点 ===')
entities = graph.run('MATCH (n:Entity) RETURN n.name as name, n.type as type LIMIT 5').data()
if entities:
    print('找到Entity节点:')
    for entity in entities:
        print(f'  - {entity["name"]} (类型: {entity["type"]})')
else:
    print('未找到Entity节点')

print('\n=== 检查所有节点（前5个）===')
all_nodes = graph.run('MATCH (n) RETURN n LIMIT 5').data()
for i, node in enumerate(all_nodes):
    print(f'节点 {i+1}: {node}')

print('\n=== 检查关系 ===')
relations = graph.run('MATCH ()-[r]-() RETURN type(r) as rel_type, count(r) as count').data()
for rel in relations:
    print(f'关系类型 {rel["rel_type"]}: {rel["count"]} 个')

print('\n=== 搜索帕金森相关节点 ===')
parkinson_nodes = graph.run('MATCH (n) WHERE n.name CONTAINS "帕金森" OR n.description CONTAINS "帕金森" RETURN n LIMIT 10').data()
print(f'找到 {len(parkinson_nodes)} 个帕金森相关节点')
for i, node in enumerate(parkinson_nodes):
    print(f'  节点 {i+1}: {node}')