from py2neo import Graph, Subgraph
from py2neo import Node, Relationship, Path

# 连接数据库
# graph = Graph('http://localhost:7474', username='neo4j', password='123456') # 旧版本
graph = Graph('bolt://localhost:7687', auth=('neo4j', 'hrx274700'))

# 删除所有已有节点
graph.delete_all()
