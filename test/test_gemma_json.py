#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试gemma3:4b模型的JSON输出格式
"""

import json
import re
import requests

def test_ollama_response():
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

    user_prompt = """请分析以下医学文本，提取其中的实体和关系：

Parkinson's disease (PD) is a chronic neurodegenerative disease that affects the central nervous system, mainly affecting the motor nervous system. Symptoms usually appear slowly over time, with the most obvious early symptoms being tremors, limb stiffness, decreased motor function, and gait abnormality.

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
        print("=" * 50)
        print(raw_response)
        print("=" * 50)
        
        # 测试JSON解析
        print("\n🔍 测试JSON解析...")
        parsed_data = parse_response(raw_response)
        
        if parsed_data:
            print(f"✅ 解析成功!")
            print(f"实体数量: {len(parsed_data.get('entities', []))}")
            print(f"关系数量: {len(parsed_data.get('relations', []))}")
            
            print("\n📋 实体列表:")
            for entity in parsed_data.get('entities', []):
                print(f"  - {entity.get('name')} ({entity.get('type')})")
                
            print("\n🔗 关系列表:")
            for relation in parsed_data.get('relations', []):
                print(f"  - {relation.get('source')} -{relation.get('relation_type')}-> {relation.get('target')}")
        else:
            print("❌ 解析失败")
            
    except Exception as e:
        print(f"❌ 请求失败: {e}")

def parse_response(response: str):
    """解析响应"""
    try:
        cleaned_response = response.strip()
        
        # 方法1: 直接解析
        if cleaned_response.startswith('{') and cleaned_response.endswith('}'):
            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError:
                pass
        
        # 方法2: 查找```json代码块
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, cleaned_response, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                json_str = match.strip()
                data = json.loads(json_str)
                if "entities" in data or "relations" in data:
                    return data
            except json.JSONDecodeError:
                continue
        
        # 方法3: 手动提取
        start_idx = cleaned_response.find('```json')
        if start_idx != -1:
            content_start = start_idx + 7  # len('```json')
            end_idx = cleaned_response.find('```', content_start)
            if end_idx != -1:
                json_str = cleaned_response[content_start:end_idx].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        
        # 方法4: 查找任何JSON对象
        start_idx = cleaned_response.find('{')
        if start_idx != -1:
            # 找到最后一个}
            end_idx = cleaned_response.rfind('}')
            if end_idx != -1 and end_idx > start_idx:
                json_str = cleaned_response[start_idx:end_idx+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        
        return None
        
    except Exception as e:
        print(f"解析异常: {e}")
        return None

if __name__ == "__main__":
    test_ollama_response()