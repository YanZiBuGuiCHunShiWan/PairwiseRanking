import faiss
import numpy as np
import os

query_embedding_path="embedding/query_embedding"
doc_embedding_path="embedding/doc_embedding"
train_query_embedding_path="embedding/train_query_embedding"

def load_embedding(embedding_path):
    with open(embedding_path,"r") as f:
        numpy_list=[]
        for line in f:
            line = line.split("\t")[1]
            vector_values = line.split(',')
            # 第三步：将分割得到的值转换为浮点数，并创建 NumPy 数组
            vector = np.array(vector_values, dtype=float).astype('float32')
            numpy_list.append(vector)
        return numpy_list
    
if not os.path.exists("embedding/faiss.index"):
    #query_embedding = load_embedding(query_embedding_path)
    doc_embedding = load_embedding(doc_embedding_path)
    #train_query_embedding = load_embedding(train_query_embedding_path)
    index = faiss.IndexFlatL2(128)  # d是向量维度
    index.add(np.array(doc_embedding))
    faiss.write_index(index,"embedding/faiss.index")
else:
    faiss.read_index("embedding/faiss.index")