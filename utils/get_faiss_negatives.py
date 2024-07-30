import os
import numpy as np
import json
import faiss

def read_queries(queries):
    qmap = {}
    with open(queries) as f:
        for l in f:
            qid, qry = l.strip().split('\t')
            qmap[qid] = qry
    return qmap


def read_collections(collections):
    cmap = {}
    with open(collections) as f:
        for l in f:
            seg = l.strip().split('\t')
            if len(seg) == 2:
                did, content = seg
            if len(seg) == 3:
                did, title, text = seg
                content = title + ' ' + text
            cmap[did] = content
    return cmap


class CustomTrainerDataset():
    def __init__(self,ann_index:str,qrel_file,query_file,corpus_file,query_array_file):
        self.qrel_dict = self.read_qrel(qrel_file)
        self.query_dict = self.read_query(query_file)
        self.corpus_dict = self.read_corpus(corpus_file)
        self.ann_index = faiss.read_index(ann_index)
        self.query_array_file = query_array_file
        self.query_array_list = []
        with open(self.query_array_file,"r") as f:
            for line in f.readlines():
                vector = line.strip().split("\t")[1]
                vector = vector.split(',')
                float_array = np.array(vector).astype(np.float32)
                self.query_array_list.append(float_array)
                
    def read_qrel(self, qrel_file):
        qrel = {}
        with open(qrel_file, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                seg = line.split('\t')
                qid, pid = seg[0], seg[2]
                qrel[qid] = pid
        print ("Finish reading qrel dict")
        return qrel

    def read_query(self, query_file):
        query_dict = {}
        with open(query_file, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                seg = line.split('\t')
                qid, query = seg
                query_dict[qid] = query
        print ("Finish reading query dict")
        return query_dict

    def read_corpus(self, corpus_file):
        corpus_dict = {}
        with open(corpus_file, 'r') as f:
            for line in f:
                line = line.strip()
                seg = line.split('\t')
                doc_id = seg[0]
                content = ' '.join(seg[1:]).strip()
                corpus_dict[doc_id] = content
        print ("Finish reading corpus dict")
        return corpus_dict
    
    def make_json(self,k):
        distances,indexes = self.ann_index.search(np.array(self.query_array_list),k)
        data_list = []
        for i,index_list in enumerate(indexes):
            current_dict = {'query':self.query_dict[str(i+1)],"positive":self.corpus_dict[self.qrel_dict[str(i+1)]],"negative":[],"neg_dis":[]}
            for j,corpus_index in enumerate(index_list):
                relavant_docs = self.corpus_dict[str(corpus_index+1)]
                if relavant_docs!=current_dict['positive']:
                    current_dict['negative'].append(relavant_docs)
                    current_dict['neg_dis'].append(float(distances[i][j]))
            data_list.append(current_dict)
        with open("data/rerank/rerank-qd.json","w") as f:
            json.dump(data_list,f)
    
    
if __name__=="__main__":
    collection_path = "data/ecom/corpus.tsv"
    cmap = read_collections(collection_path)

    query_path = "data/ecom/train.query.txt"
    qrel_path = "data/ecom/qrels.train.tsv"
    dataset = CustomTrainerDataset(ann_index="embedding/faiss.index",
                                   qrel_file=qrel_path,
                                   query_file=query_path,
                                   corpus_file=collection_path,
                                   query_array_file="embedding/train_query_embedding")
    dataset.make_json(50)