from ollama import embeddings
import numpy as np

response = embeddings(model='nomic-embed-text', prompt='北京建筑大学')
print(response)
print(response['embedding'])
print(len(response['embedding']))