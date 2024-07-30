import csv
import sys
import os
import torch
from tqdm import tqdm
import numpy as np

sys.path.append("..")
from model.models import BertForCL
from transformers import AutoTokenizer
import os
device = "cuda:0"
paths = os.listdir('/sharedata/models/Ecommerce-Retrieval/')
batch_size = 100
use_pinyin = False
last_one = 0
idxs = []


def encode_fun(texts, model):
    inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors="pt", max_length=110)
    inputs.to(device)
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
        embeddings = embeddings.squeeze(0).cpu().numpy()
    return embeddings



if __name__ == '__main__': 
    final_path="/sharedata/models/Ecommerce-Retrieval/"
    tokenizer = AutoTokenizer.from_pretrained('/sharedata/models/Ecommerce-Retrieval/')
    print("final path: {}".format(final_path))
    model = BertForCL.from_pretrained(final_path)
    model.to(device)
    corpus = [line[1] for line in csv.reader(open("./data/ecom/corpus.tsv"), delimiter='\t')]
    query = [line[1] for line in csv.reader(open("./data/ecom/dev.query.txt"), delimiter='\t')]
    train_query = [line[1] for line in csv.reader(open("./data/ecom/train.query.txt"), delimiter='\t')]

    # query_embedding_file = csv.writer(open('embedding/query_embedding', 'w'), delimiter='\t')
    # #query_embedding_file = csv.writer(open('query_embedding'+str(index), 'w'), delimiter='\t')
    # for i in tqdm(range(0, len(query), batch_size)):
    #     batch_text = query[i:i + batch_size]
    #     temp_embedding = encode_fun(batch_text, model)
    #     for j in range(len(temp_embedding)):
    #         writer_str = temp_embedding[j].tolist()
    #         writer_str = [format(s, '.8f') for s in writer_str]
    #         writer_str = ','.join(writer_str)
    #         query_embedding_file.writerow([i + j + 200001, writer_str])
    # #print(train_query[:10])
    # train_query_embedding_file = csv.writer(open('embedding/train_query_embedding', 'w'), delimiter='\t')
    # #query_embedding_file = csv.writer(open('query_embedding'+str(index), 'w'), delimiter='\t')
    # for i in tqdm(range(0, len(train_query), batch_size)):
    #     batch_text = train_query[i:i + batch_size]
    #     temp_embedding = encode_fun(batch_text, model)
    #     for j in range(len(temp_embedding)):
    #         writer_str = temp_embedding[j].tolist()
    #         writer_str = [format(s, '.8f') for s in writer_str]
    #         writer_str = ','.join(writer_str)
    #         train_query_embedding_file.writerow([i + j + 200001, writer_str])
            
            

    doc_embedding_file = csv.writer(open('embedding/doc_embedding', 'w'), delimiter='\t')
    #doc_embedding_file = csv.writer(open('doc_embedding'+str(index), 'w'), delimiter='\t')
    for i in tqdm(range(0, len(corpus), batch_size)):
        batch_text = corpus[i:i + batch_size]
        temp_embedding = encode_fun(batch_text, model)
        for j in range(len(temp_embedding)):
            writer_str = temp_embedding[j].tolist()
            writer_str = [format(s, '.8f') for s in writer_str]
            writer_str = ','.join(writer_str)
            doc_embedding_file.writerow([i + j + 1, writer_str])
