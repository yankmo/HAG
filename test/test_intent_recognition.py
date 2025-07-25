#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图识别和Neo4j导入测试脚本
"""

from intent_recognition_neo4j import KnowledgeGraphBuilder, OllamaClient
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ollama_connection():
    """测试Ollama连接"""
    print("🔍 测试Ollama连接...")
    try:
        client = OllamaClient()
        response = client.generate("Hello, please respond with 'Connection successful'")
        if response:
            print(f"✅ Ollama连接成功: {response[:50]}...")
            return True
        else:
            print("❌ Ollama响应为空")
            return False
    except Exception as e:
        print(f"❌ Ollama连接失败: {e}")
        return False

def test_small_text_processing():
    """测试小文本处理"""
    print("\n🧪 测试小文本处理...")
    
    test_text = """
    Parkinson's disease is a chronic neurodegenerative disease that affects the central nervous system. 
    The main symptoms include tremors, limb stiffness, and decreased motor function. 
    L-dopa is commonly used to treat initial symptoms.
    """
    
    try:
        builder = KnowledgeGraphBuilder()
        entities, relations = builder.recognizer.extract_entities_and_relations(test_text)
        
        print(f"✅ 提取到 {len(entities)} 个实体:")
        for entity in entities[:5]:  # 只显示前5个
            print(f"   - {entity.name} ({entity.type})")
        
        print(f"✅ 提取到 {len(relations)} 个关系:")
        for relation in relations[:5]:  # 只显示前5个
            print(f"   - {relation.source} -{relation.relation_type}-> {relation.target}")
        
        return True
        
    except Exception as e:
        print(f"❌ 文本处理失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始意图识别和Neo4j导入测试\n")
    
    # 测试Ollama连接
    if not test_ollama_connection():
        print("\n❌ 请确保Ollama服务正在运行，并且已安装qwen2:0.5b模型")
        print("   启动命令: ollama serve")
        print("   安装模型: ollama pull qwen2:0.5b")
        return
    
    # 测试小文本处理
    if not test_small_text_processing():
        return
    
    # 询问是否继续完整处理
    print("\n" + "="*50)
    choice = input("🤔 是否继续处理完整的帕金森氏症文档？(y/n): ").lower().strip()
    
    if choice == 'y':
        print("\n📚 开始处理完整文档...")
        try:
            builder = KnowledgeGraphBuilder()
            file_path = "e:/Program/Project/rag-first/knowledgeBase/帕金森氏症en.txt"
            builder.process_text_file(file_path, chunk_size=800)  # 使用较小的块大小
            
            print("\n🎉 完整文档处理完成！")
            print("📊 请在Neo4j Browser中查看构建的知识图谱")
            print("🔗 Neo4j Browser: http://localhost:7474")
            
        except Exception as e:
            print(f"\n❌ 完整文档处理失败: {e}")
    else:
        print("\n👋 测试完成，跳过完整文档处理")

if __name__ == "__main__":
    main()