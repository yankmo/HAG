#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图识别和Neo4j数据导入模块
使用Ollama的deepseek-r1:7b模型进行意图识别，提取实体和关系，并导入Neo4j
"""

import json
import re
import requests
from typing import Dict, List, Tuple, Any
from py2neo import Graph, Node, Relationship
from dataclasses import dataclass
import logging

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import get_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """实体类"""
    name: str
    type: str
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

@dataclass
class Relation:
    """关系类"""
    source: str
    target: str
    relation_type: str
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

class OllamaClient:
    """Ollama客户端"""
    
    def __init__(self, base_url: str = None, model: str = None):
        config = get_config()
        
        self.base_url = base_url or config.ollama.base_url
        self.model = model or config.ollama.default_model
        
    def generate(self, prompt: str, system_prompt: str = None, timeout: int = 180) -> str:
        """生成文本"""
        url = f"{self.base_url}/api/generate"
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # 降低温度以获得更稳定的结果
                "top_p": 0.9,
                "num_predict": 8192  # 增加最大输出长度以避免响应被截断
            }
        }
        
        if system_prompt:
            data["system"] = system_prompt
            
        try:
            response = requests.post(url, json=data, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            logger.error(f"Ollama请求失败: {e}")
            return ""

class IntentRecognizer:
    """意图识别器"""
    
    def __init__(self, ollama_client: OllamaClient):
        self.ollama = ollama_client
        
    def extract_entities_and_relations(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        """从文本中提取实体和关系"""
        
        # 系统提示词 (英文版本以提高处理效率)
        system_prompt = """You are a professional medical knowledge graph construction expert. Your task is to extract entities and relationships from medical texts to build knowledge graphs.

Please strictly follow the JSON format below for output:
{
    "entities": [
        {
            "name": "entity_name",
            "type": "entity_type",
            "properties": {"description": "entity_description"}
        }
    ],
    "relations": [
        {
            "source": "source_entity_name",
            "target": "target_entity_name", 
            "relation_type": "relationship_type",
            "properties": {"description": "relationship_description"}
        }
    ]
}

Entity types include: Disease, Symptom, Treatment, Drug, Gene, Protein, BodyPart, Cause, Risk

Relationship types include: CAUSES, TREATS, HAS_SYMPTOM, AFFECTS, RELATED_TO, LOCATED_IN, INCREASES_RISK, DECREASES_RISK, PART_OF

Please ensure the output is valid JSON format."""

        # 用户提示词 (英文版本)
        user_prompt = f"""Please analyze the following medical text and extract entities and relationships:

{text}

Extract key medical entities and their relationships to build a knowledge graph."""

        try:
            response = self.ollama.generate(user_prompt, system_prompt)
            logger.info(f"LLM响应长度: {len(response)} 字符")
            
            # 尝试解析JSON
            entities, relations = self._parse_llm_response(response)
            
            if not entities and not relations:
                logger.warning("无法从响应中提取有效的JSON数据")
                return [], []
            
            return entities, relations
            
        except Exception as e:
            logger.error(f"实体关系提取失败: {e}")
            return [], []
    
    def _parse_llm_response(self, response: str) -> Tuple[List[Entity], List[Relation]]:
        """解析LLM响应"""
        entities = []
        relations = []
        
        try:
            if not response:
                logger.warning("响应为空")
                return entities, relations
            
            # 清理响应
            cleaned_response = response.strip()
            
            # 方法1: 直接解析JSON
            try:
                json_data = json.loads(cleaned_response)
                if "entities" in json_data or "relations" in json_data:
                    logger.info("直接JSON解析成功")
                    return self._extract_entities_relations_from_json(json_data)
            except Exception as e:
                pass
            
            # 方法2: 查找json代码块
            json_pattern = r'```json\s*(.*?)\s*```'
            matches = re.findall(json_pattern, cleaned_response, re.DOTALL)
            if matches:
                try:
                    json_str = matches[0].strip()
                    json_data = json.loads(json_str)
                    if "entities" in json_data or "relations" in json_data:
                        logger.info("json代码块解析成功")
                        return self._extract_entities_relations_from_json(json_data)
                except Exception as e:
                    pass
            
            # 方法3: 查找任何代码块
            code_pattern = r'```\s*(.*?)\s*```'
            matches = re.findall(code_pattern, cleaned_response, re.DOTALL)
            if matches:
                for i, match in enumerate(matches):
                    try:
                        json_str = match.strip()
                        json_data = json.loads(json_str)
                        if "entities" in json_data or "relations" in json_data:
                            logger.info(f"通用代码块解析成功")
                            return self._extract_entities_relations_from_json(json_data)
                    except Exception as e:
                        continue
            
            # 方法4: 手动提取JSON对象
            start_idx = cleaned_response.find('{')
            if start_idx != -1:
                brace_count = 0
                for i, char in enumerate(cleaned_response[start_idx:], start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            try:
                                json_str = cleaned_response[start_idx:i+1]
                                json_data = json.loads(json_str)
                                if "entities" in json_data or "relations" in json_data:
                                    logger.info("手动JSON对象提取成功")
                                    return self._extract_entities_relations_from_json(json_data)
                            except Exception as e:
                                break
            
            logger.warning("所有解析方法都失败了")
            return entities, relations
            
        except Exception as e:
            logger.error(f"解析LLM响应时出错: {e}")
            return entities, relations
    
    def _extract_entities_relations_from_json(self, json_data: dict) -> Tuple[List[Entity], List[Relation]]:
        """从JSON数据中提取实体和关系"""
        entities = []
        relations = []
        
        try:
            # 解析实体
            entity_list = json_data.get("entities", [])
            for entity_data in entity_list:
                if isinstance(entity_data, dict):
                    entity = Entity(
                        name=entity_data.get("name", "").strip(),
                        type=entity_data.get("type", "Unknown").strip(),
                        properties=entity_data.get("properties", {})
                    )
                    if entity.name:  # 只添加有名称的实体
                        entities.append(entity)
            
            # 解析关系
            relation_list = json_data.get("relations", [])
            for relation_data in relation_list:
                if isinstance(relation_data, dict):
                    relation = Relation(
                        source=relation_data.get("source", "").strip(),
                        target=relation_data.get("target", "").strip(),
                        relation_type=relation_data.get("relation_type", "RELATED_TO").strip(),
                        properties=relation_data.get("properties", {})
                    )
                    if relation.source and relation.target:  # 只添加有效的关系
                        relations.append(relation)
                        
        except Exception as e:
            logger.error(f"从JSON数据提取实体关系时出错: {e}")
            
        return entities, relations

class Neo4jImporter:
    """Neo4j数据导入器"""
    
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        config = get_config()
        
        uri = uri or config.neo4j.uri
        username = username or config.neo4j.username
        password = password or config.neo4j.password
        
        try:
            self.graph = Graph(uri, auth=(username, password))
            logger.info("Neo4j连接成功")
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            raise
    
    def clear_database(self):
        """清空数据库"""
        try:
            self.graph.delete_all()
            logger.info("数据库已清空")
        except Exception as e:
            logger.error(f"清空数据库失败: {e}")
    
    def import_entities_and_relations(self, entities: List[Entity], relations: List[Relation]):
        """导入实体和关系"""
        try:
            # 创建实体节点
            entity_nodes = {}
            for entity in entities:
                node = Node(entity.type, name=entity.name, **entity.properties)
                self.graph.create(node)
                entity_nodes[entity.name] = node
            
            logger.info(f"创建了 {len(entities)} 个实体节点")
            
            # 创建关系
            created_relations = 0
            for relation in relations:
                if relation.source in entity_nodes and relation.target in entity_nodes:
                    source_node = entity_nodes[relation.source]
                    target_node = entity_nodes[relation.target]
                    rel = Relationship(source_node, relation.relation_type, target_node, **relation.properties)
                    self.graph.create(rel)
                    created_relations += 1
                else:
                    logger.warning(f"关系中的实体不存在: {relation.source} -> {relation.target}")
            
            logger.info(f"创建了 {created_relations} 个关系")
                    
        except Exception as e:
            logger.error(f"数据导入失败: {e}")

class KnowledgeGraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self):
        self.ollama = OllamaClient()
        self.recognizer = IntentRecognizer(self.ollama)
        self.importer = Neo4jImporter()
    
    def process_text_file(self, file_path: str, chunk_size: int = 500):
        """处理文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"开始处理文件: {file_path}")
            logger.info(f"文件大小: {len(content)} 字符")
            
            # 清空数据库
            self.importer.clear_database()
            
            # 分块处理文本
            chunks = self._split_text(content, chunk_size)
            logger.info(f"文本分为 {len(chunks)} 个块")
            
            all_entities = []
            all_relations = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"处理第 {i+1}/{len(chunks)} 块")
                
                try:
                    entities, relations = self.recognizer.extract_entities_and_relations(chunk)
                    all_entities.extend(entities)
                    all_relations.extend(relations)
                    logger.info(f"块 {i+1}: 提取到 {len(entities)} 个实体, {len(relations)} 个关系")
                            
                except Exception as e:
                    logger.error(f"处理块 {i+1} 时出错: {e}")
                    import traceback
                    logger.error(f"错误详情: {traceback.format_exc()}")
            
            # 去重
            unique_entities = self._deduplicate_entities(all_entities)
            unique_relations = self._deduplicate_relations(all_relations)
            
            logger.info(f"去重后: {len(unique_entities)} 个实体, {len(unique_relations)} 个关系")
            
            # 导入Neo4j
            self.importer.import_entities_and_relations(unique_entities, unique_relations)
            
            logger.info("知识图谱构建完成")
            
        except Exception as e:
            logger.error(f"处理文件失败: {e}")
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """分割文本"""
        # 清理文本，移除开头和结尾的空白
        text = text.strip()
        
        # 按段落分割，过滤掉空段落
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 过滤掉空的文本块
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        return chunks
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """去重实体"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.name.lower(), entity.type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """去重关系"""
        seen = set()
        unique_relations = []
        
        for relation in relations:
            key = (relation.source.lower(), relation.target.lower(), relation.relation_type)
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
        
        return unique_relations

def main():
    """主函数"""
    import sys
    import os
    
    try:
        # 创建知识图谱构建器
        builder = KnowledgeGraphBuilder()
        
        # 从命令行参数获取文件路径，如果没有提供则使用默认路径
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
        else:
            # 使用相对路径作为默认值
            file_path = os.path.join("data", "pajinsen.txt")
            
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            logger.info("使用方法: python intent_recognition_neo4j.py [文件路径]")
            return
            
        builder.process_text_file(file_path)
        
        logger.info("意图识别和数据导入完成！")
        logger.info("请在Neo4j Browser中查看构建的知识图谱")
        logger.info("Neo4j Browser: http://localhost:7474")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")

if __name__ == "__main__":
    main()