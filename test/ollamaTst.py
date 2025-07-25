from ollama import chat, Message

msgs = [
    Message(role='system', content='你是一个心理辅助机器人，你的任务是帮助用户解决心理问题。'),
    Message(role='user', content='我感到很焦虑。'),
]

response = chat(model='qwen2.5:0.5b', messages=msgs)
print(response)
