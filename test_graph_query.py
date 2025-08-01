from py2neo import Graph

# 连接Neo4j
graph = Graph('bolt://localhost:7687', auth=('neo4j', 'hrx274700'))

print('=== 测试图谱检索查询 ===')

# 测试当前的查询逻辑
query_text = "帕金森额可以治疗吗"
print(f'查询文本: {query_text}')

print('\n1. 测试当前的CONTAINS查询:')
entities = graph.run("""
    MATCH (e:Entity)
    WHERE e.name CONTAINS $name_pattern
    RETURN e.name as name, e.type as type, e.description as description
    LIMIT 10
""", {'name_pattern': query_text}).data()
print(f'CONTAINS查询结果: {len(entities)} 个')
for entity in entities:
    print(f'  - {entity["name"]} ({entity["type"]})')

print('\n2. 测试包含"帕金森"的查询:')
entities = graph.run("""
    MATCH (e:Entity)
    WHERE e.name CONTAINS "帕金森" OR e.description CONTAINS "帕金森"
    RETURN e.name as name, e.type as type, e.description as description
    LIMIT 10
""", {}).data()
print(f'包含"帕金森"的查询结果: {len(entities)} 个')
for entity in entities:
    print(f'  - {entity["name"]} ({entity["type"]}): {entity["description"][:100]}...')

print('\n3. 测试包含"治疗"的查询:')
entities = graph.run("""
    MATCH (e:Entity)
    WHERE e.name CONTAINS "治疗" OR e.description CONTAINS "治疗"
    RETURN e.name as name, e.type as type, e.description as description
    LIMIT 10
""", {}).data()
print(f'包含"治疗"的查询结果: {len(entities)} 个')
for entity in entities:
    print(f'  - {entity["name"]} ({entity["type"]}): {entity["description"][:100]}...')

print('\n4. 测试Parkinson相关查询:')
entities = graph.run("""
    MATCH (e:Entity)
    WHERE e.name CONTAINS "Parkinson" OR e.description CONTAINS "Parkinson"
    RETURN e.name as name, e.type as type, e.description as description
    LIMIT 10
""", {}).data()
print(f'包含"Parkinson"的查询结果: {len(entities)} 个')
for entity in entities:
    print(f'  - {entity["name"]} ({entity["type"]}): {entity["description"][:100]}...')