import sys
import os
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py2neo import Graph, Subgraph
from py2neo import Node, Relationship, Path

# 导入配置管理器
from config import get_config

# 连接数据库
config = get_config()
graph = Graph(config.neo4j.uri, auth=(config.neo4j.username, config.neo4j.password))

# 删除所有已有节点
graph.delete_all()
