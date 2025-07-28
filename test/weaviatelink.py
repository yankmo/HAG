import weaviate
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import get_config
from langchain_community.document_loaders import DirectoryLoader, WebBaseLoader
import pandas as pd

# 获取配置并创建客户端
config = get_config()
client = weaviate.Client(url=config.weaviate.url)
class_name = 'Stephen_Chow'  # class的名字

# v4版本的集合定义
class_obj = {
    'class': class_name,
    'description': 'A collection for Stephen Chow information',
    'vectorizer': 'none',  # 如果你要上传自己的向量
    'vectorIndexType': 'hnsw',
    'vectorIndexConfig': {
        'distance': 'l2-squared',
        'efConstruction': 200,
        'maxConnections': 64
    },
    'properties': [
        {
            'name': 'content',
            'description': 'The content text',
            'dataType': ['text'],
            'tokenization': 'word'
        }
    ]
}