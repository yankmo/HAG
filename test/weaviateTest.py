import weaviate
from weaviate.auth import AuthApiKey
import random
import time

# 连接到本地部署的 Weaviate

def test_connection(client: weaviate.WeaviateClient):
    """
    测试Weaviate连接状态
    :param client: Weaviate 客户端
    :return: 连接状态信息
    """
    try:
        # 检查连接状态
        is_ready = client.is_ready()
        print(f"🔗 Weaviate连接状态: {'✅ 正常' if is_ready else '❌ 异常'}")
        
        # 获取版本信息
        meta = client.get_meta()
        version = meta.get('version', '未知版本')
        print(f"📊 Weaviate版本: {version}")
        
        # 获取现有集合信息
        collections = client.collections.list_all()
        print(f"📁 现有集合数量: {len(collections)}")
        if collections:
            print(f"📋 集合列表: {list(collections.keys())}")
        
        return is_ready
    except Exception as e:
        print(f"❌ 连接测试失败: {e}")
        return False

def check_collection_exists(client: weaviate.WeaviateClient, collection_name: str) -> bool:
    """
    检查集合是否存在
    :param client: Weaviate 客户端
    :param collection_name: 集合名称
    :return: True 或 False
    """
    try:
        collections = client.collections.list_all()
        return collection_name in collections
    except Exception as e:
        print(f"检查集合异常: {e}")
        return False

def create_collection(client: weaviate.WeaviateClient, collection_name: str):
    """
    创建集合
    :param client: Weaviate 客户端
    :param collection_name: 集合名称
    """
    print(f"🏗️ 开始创建集合 '{collection_name}'...")
    
    collection_obj = {
        "class": collection_name,
        "description": "A test collection for RAG functionality",
        "vectorizer": "none",  # 使用自定义向量
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {
            "distance": "cosine",
            "efConstruction": 200,
            "maxConnections": 64,
            "vectorCacheMaxObjects": 1000000
        },
        "properties": [
            {
                "name": "text",
                "description": "The text content",
                "dataType": ["text"],
                "tokenization": "word",
                "indexFilterable": True,
                "indexSearchable": True
            },
            {
                "name": "source",
                "description": "Source of the document",
                "dataType": ["text"],
                "indexFilterable": True
            }
        ]
    }
    try:
        client.collections.create_from_dict(collection_obj)
        print(f"✅ 集合 '{collection_name}' 创建成功！")
        print(f"📋 集合配置:")
        print(f"   - 向量索引: HNSW")
        print(f"   - 距离度量: 余弦相似度")
        print(f"   - 向量维度: 384")
    except weaviate.exceptions.UnexpectedStatusCodeException as e:
        print(f"❌ 创建集合失败: {e}")
    except Exception as e:
        print(f"❌ 创建集合时发生未知错误: {e}")

def save_documents(client: weaviate.WeaviateClient, collection_name: str, documents: list):
    """
    向集合中插入数据
    :param client: Weaviate 客户端
    :param collection_name: 集合名称
    :param documents: 文档列表
    """
    collection = client.collections.get(collection_name)
    print(f"📝 开始插入 {len(documents)} 个文档...")
    
    for i, doc in enumerate(documents):
        content = doc  # 假设文档是简单的字符串
        # 生成随机向量用于测试（实际应用中应该使用真实的embedding）
        vector = [random.random() for _ in range(384)]  # 384维向量，常见的embedding维度
        properties = {
            "text": content
        }
        try:
            uuid = collection.data.insert(properties=properties, vector=vector)
            print(f"✅ 文档 {i+1} 添加成功: {content[:50]}{'...' if len(content) > 50 else ''}")
            print(f"   UUID: {uuid}")
        except Exception as e:
            print(f"❌ 文档 {i+1} 添加失败: {e}")
    
    print(f"📝 文档插入完成！")

def query_vector_collection(client: weaviate.WeaviateClient, collection_name: str, query: str, k: int) -> list:
    """
    基于向量查询集合
    :param client: Weaviate 客户端
    :param collection_name: 集合名称
    :param query: 查询字符串
    :param k: 返回结果数量
    :return: 查询结果列表
    """
    print(f"🔍 开始向量查询: '{query}' (返回前{k}个结果)")
    
    collection = client.collections.get(collection_name)
    
    # 生成查询向量（实际应用中应该使用真实的embedding）
    query_vector = [random.random() for _ in range(384)]
    
    try:
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=k,
            return_metadata=["distance", "score"]
        )
        
        results = []
        print(f"📊 查询结果 (共找到 {len(response.objects)} 个结果):")
        
        for i, obj in enumerate(response.objects):
            text = obj.properties.get("text", "")
            distance = obj.metadata.distance if obj.metadata else "未知"
            score = obj.metadata.score if obj.metadata else "未知"
            
            print(f"  {i+1}. 文本: {text[:100]}{'...' if len(text) > 100 else ''}")
            print(f"     距离: {distance:.4f}, 分数: {score:.4f}")
            print(f"     UUID: {obj.uuid}")
            print()
            
            results.append({
                "text": text,
                "distance": distance,
                "score": score,
                "uuid": str(obj.uuid)
            })
        
        return results
        
    except Exception as e:
        print(f"❌ 向量查询失败: {e}")
        return []

def delete_collection(client: weaviate.WeaviateClient, collection_name: str):
    """
    删除集合
    :param client: Weaviate 客户端
    :param collection_name: 集合名称
    """
    print(f"🗑️ 开始删除集合 '{collection_name}'...")
    
    try:
        # 先检查集合是否存在
        if not check_collection_exists(client, collection_name):
            print(f"⚠️ 集合 '{collection_name}' 不存在，无需删除")
            return
            
        client.collections.delete(collection_name)
        print(f"✅ 集合 '{collection_name}' 删除成功！")
        
    except weaviate.exceptions.UnexpectedStatusCodeException as e:
        print(f"❌ 删除集合失败: {e}")
    except Exception as e:
        print(f"❌ 删除集合时发生未知错误: {e}")


def get_all_documents(client: weaviate.WeaviateClient, collection_name: str) -> list:
    """
    获取集合中的所有文档
    :param client: Weaviate 客户端
    :param collection_name: 集合名称
    :return: 所有文档列表
    """
    print(f"📄 获取集合 '{collection_name}' 中的所有文档...")
    
    try:
        collection = client.collections.get(collection_name)
        response = collection.query.fetch_objects(limit=1000)  # 限制返回数量
        
        documents = []
        print(f"📊 找到 {len(response.objects)} 个文档:")
        
        for i, obj in enumerate(response.objects):
            text = obj.properties.get('text', '')
            documents.append(text)
            print(f"  {i+1}. {text[:80]}{'...' if len(text) > 80 else ''}")
            
        return documents
        
    except Exception as e:
        print(f"❌ 获取文档失败: {e}")
        return []


if __name__ == "__main__":
    print("🚀 开始Weaviate功能测试...")
    print("=" * 60)
    
    # 连接到本地Weaviate实例
    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051
    )
    
    try:
        # 1. 测试连接
        print("\n1️⃣ 测试连接状态")
        print("-" * 30)
        if not test_connection(client):
            print("❌ 连接失败，退出测试")
            exit(1)
        
        # 2. 测试集合操作
        collection_name = "TestCollection"
        print(f"\n2️⃣ 测试集合操作")
        print("-" * 30)
        
        # 检查集合是否存在
        if check_collection_exists(client, collection_name):
            print(f"⚠️ 集合 '{collection_name}' 已存在，先删除...")
            delete_collection(client, collection_name)
            time.sleep(1)  # 等待删除完成
        
        # 创建新集合
        create_collection(client, collection_name)
        time.sleep(1)  # 等待创建完成
        
        # 3. 测试文档操作
        print(f"\n3️⃣ 测试文档操作")
        print("-" * 30)
        
        # 准备测试文档
        test_documents = [
            "这是第一个测试文档，包含关于人工智能的内容。",
            "第二个文档讨论机器学习和深度学习的应用。",
            "第三个文档介绍自然语言处理技术。",
            "第四个文档探讨计算机视觉的发展。",
            "最后一个文档总结了RAG技术的优势。"
        ]
        
        # 插入文档
        save_documents(client, collection_name, test_documents)
        time.sleep(2)  # 等待索引完成
        
        # 4. 测试查询功能
        print(f"\n4️⃣ 测试查询功能")
        print("-" * 30)
        
        # 向量查询测试
        query_results = query_vector_collection(client, collection_name, "人工智能", 3)
        
        # 5. 测试获取所有文档
        print(f"\n5️⃣ 测试获取所有文档")
        print("-" * 30)
        all_docs = get_all_documents(client, collection_name)
        
        # 6. 最终状态检查
        print(f"\n6️⃣ 最终状态检查")
        print("-" * 30)
        test_connection(client)
        
        print(f"\n✅ 所有测试完成！")
        print("=" * 60)
        print(f"📊 测试总结:")
        print(f"   - 插入文档数: {len(test_documents)}")
        print(f"   - 查询结果数: {len(query_results)}")
        print(f"   - 获取文档数: {len(all_docs)}")
        
        # 可选：清理测试数据
        cleanup = input("\n🗑️ 是否删除测试集合？(y/N): ").lower().strip()
        if cleanup == 'y':
            delete_collection(client, collection_name)
            print("🧹 测试数据已清理")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        
    finally:
        # 关闭连接
        client.close()
        print("🔌 连接已关闭")
