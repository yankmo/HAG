#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试gemma3:4b模型的实际响应
"""

import json
import re
import requests

def test_actual_text_chunk():
    """测试实际的文本块"""
    
    # 读取实际文件
    with open("e:/Program/Project/rag-first/data/pajinsen.txt", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 分块处理（模拟实际程序的分块逻辑）
    chunks = split_text(content, 1000)
    
    print(f"文件总长度: {len(content)} 字符")
    print(f"分为 {len(chunks)} 个块")
    
    # 测试前几个块
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n{'='*50}")
        print(f"测试第 {i+1} 个块 (长度: {len(chunk)} 字符)")
        print(f"{'='*50}")
        print("块内容预览:")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        print(f"{'='*50}")
        
        # 发送到模型
        response = test_ollama_with_chunk(chunk)
        if response:
            print("✅ 成功获得响应")
            # 测试解析
            parsed = parse_response(response)
            if parsed:
                print(f"✅ 解析成功: {len(parsed.get('entities', []))} 个实体, {len(parsed.get('relations', []))} 个关系")
            else:
                print("❌ 解析失败")
        else:
            print("❌ 未获得响应")

def split_text(text: str, chunk_size: int):
    """分割文本（与主程序相同的逻辑）"""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def test_ollama_with_chunk(text_chunk):
    """测试Ollama响应"""
    url = "http://localhost:11434/api/generate"
    
    system_prompt = """你是一个专业的医学知识图谱构建专家。你的任务是从医学文本中提取实体和关系，构建知识图谱。

请严格按照以下JSON格式输出结果：
{
    "entities": [
        {
            "name": "实体名称",
            "type": "实体类型",
            "properties": {"description": "实体描述"}
        }
    ],
    "relations": [
        {
            "source": "源实体名称",
            "target": "目标实体名称", 
            "relation_type": "关系类型",
            "properties": {"description": "关系描述"}
        }
    ]
}

实体类型包括：Disease(疾病), Symptom(症状), Treatment(治疗), Drug(药物), Gene(基因), Protein(蛋白质), BodyPart(身体部位), Cause(病因), Risk(风险因素)

关系类型包括：CAUSES(导致), TREATS(治疗), HAS_SYMPTOM(有症状), AFFECTS(影响), RELATED_TO(相关), LOCATED_IN(位于), INCREASES_RISK(增加风险), DECREASES_RISK(降低风险)

请确保输出的是有效的JSON格式。"""

    user_prompt = f"""请分析以下医学文本，提取其中的实体和关系：

{text_chunk}

请提取文本中的关键医学实体和它们之间的关系，构建知识图谱。"""

    data = {
        "model": "gemma3:4b",
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 2048
        }
    }
    
    try:
        print("🔄 发送请求到Ollama...")
        response = requests.post(url, json=data, timeout=120)
        response.raise_for_status()
        result = response.json()
        raw_response = result.get("response", "")
        
        print("📝 原始响应:")
        print("-" * 30)
        print(raw_response)
        print("-" * 30)
        
        return raw_response
        
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

def parse_response(response: str):
    """解析响应（与主程序相同的逻辑）"""
    try:
        cleaned_response = response.strip()
        
        # 方法1: 直接解析
        if cleaned_response.startswith('{') and cleaned_response.endswith('}'):
            try:
                json_data = json.loads(cleaned_response)
                if "entities" in json_data or "relations" in json_data:
                    print(f"✅ 直接解析成功")
                    return json_data
            except json.JSONDecodeError as e:
                print(f"❌ 直接解析失败: {e}")
        
        # 方法2: 查找```json代码块
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, cleaned_response, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                json_str = match.strip()
                json_data = json.loads(json_str)
                if "entities" in json_data or "relations" in json_data:
                    print(f"✅ 正则解析成功")
                    return json_data
            except json.JSONDecodeError as e:
                print(f"❌ 正则解析失败: {e}")
                continue
        
        # 方法3: 手动提取
        start_idx = cleaned_response.find('```json')
        if start_idx != -1:
            content_start = start_idx + 7  # len('```json')
            end_idx = cleaned_response.find('```', content_start)
            if end_idx != -1:
                json_str = cleaned_response[content_start:end_idx].strip()
                try:
                    json_data = json.loads(json_str)
                    if "entities" in json_data or "relations" in json_data:
                        print(f"✅ 手动提取成功")
                        return json_data
                except json.JSONDecodeError as e:
                    print(f"❌ 手动提取失败: {e}")
        
        # 方法4: 查找任何JSON对象
        start_idx = cleaned_response.find('{')
        if start_idx != -1:
            end_idx = cleaned_response.rfind('}')
            if end_idx != -1 and end_idx > start_idx:
                json_str = cleaned_response[start_idx:end_idx+1]
                try:
                    json_data = json.loads(json_str)
                    if "entities" in json_data or "relations" in json_data:
                        print(f"✅ 通用提取成功")
                        return json_data
                except json.JSONDecodeError as e:
                    print(f"❌ 通用提取失败: {e}")
        
        print("❌ 所有解析方法都失败了")
        return None
        
    except Exception as e:
        print(f"❌ 解析异常: {e}")
        return None

if __name__ == "__main__":
    test_actual_text_chunk()