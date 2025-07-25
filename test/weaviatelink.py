import weaviate
from langchain_community.document_loaders import DirectoryLoader, WebBaseLoader
import pandas as pd

client = weaviate.Client(url='http://localhost:8080')
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