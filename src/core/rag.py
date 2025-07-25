import ollama
from ollama import Message
import numpy as np

# 创建带超时的 ollama 客户端
ollama_client = ollama.Client(host='http://localhost:11434', timeout=30)

class Kb:
    def __init__(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # 文件读取
        self.docs = self.spilit_content(content)
        self.embeds = self.encode((self.docs))
        self.collection_name = "ParkinsonKnowledge"

    @staticmethod
    def spilit_content(content, max_length=52):
        chunks = []
        # 按换行符分割成行，处理所有类型的换行符
        lines = content.splitlines()
        for line in lines:
            # 对每行按max_length进行分割
            if line.strip():
                chunks.append(line)
        return chunks

    @staticmethod
    def encode(texts):
        # 使用ollama的embeddings模型获取向量并储存
        embeds = []
        for text in texts:
            response = ollama_client.embeddings(model='nomic-embed-text', prompt=text)
            embeds.append(response['embedding'])
        return np.array(embeds)


    @staticmethod
    def similarity(e1, e2):
        # 计算余弦相似度
        dot_product = np.dot(e1, e2)
        # 点乘
        norm_e1 = np.linalg.norm(e1)
        norm_e2 = np.linalg.norm(e2)
        # 范数
        cosine_sim = dot_product / (norm_e1 * norm_e2)
        # 余弦相似度 = 点乘 / (范数1 * 范数2)
        return cosine_sim

    def search(self, text, top_k=5):
        # 文本解码
        e = self.encode([text])[0]

        # 相似度比较
        sims = [(idx, self.similarity(e, ke)) for idx, ke in enumerate(self.embeds)]
        sims.sort(key=lambda x: x[1], reverse=True)

        # 匹配前5
        best_matches = [self.docs[idx] for idx, _ in sims[:top_k]]
        return best_matches

class RAG:
    def __init__(self, model, kb:Kb):
        self.model = model
        self.kb = kb
        self.prompt_template = """
        基于以下知识:1:%s,2:%s,3:%s,4:%s,5:%s
        回答用户的问题:%s
        """

    def chat(self, text):
        # 先检索知识库再构建prompt传给ollama
        context = self.kb.search(text)
        prompt = self.prompt_template % (context[0], context[1], context[2], context[3], context[4], text)
        print(prompt)
        response = ollama_client.chat(self.model, [Message(role='system', content=prompt)])
        return response['message']

if __name__ == '__main__':
    kb = Kb('knowledgeBase/帕金森氏症en.txt')
    rag = RAG('deepseek-r1:7b', kb)

    while True:
        print(rag.prompt_template)
        q = input('Human:')
        r = rag.chat(q)
        print('Assistant: ', r['content'])