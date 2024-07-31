import torch
import torch.nn as nn
import numpy as np
import faiss
import json,ujson
from transformers import BertTokenizer
from torch.utils.data import Dataset,DataLoader

class ReRankerDataset(Dataset):
    def __init__(self,rerank_datapath:str,model_path:str,max_length:int=128,negative_nums:int=30):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.max_length = max_length
        with open(rerank_datapath,"r") as f:
            self.data = ujson.load(f)
            
        self.expanded_negative_data = [] 
        self.expanded_positive_data = []
        self.adaptive_margin = []
        
        self.negative_nums = negative_nums
        for dict_info in self.data:
            current_expanded_qd = []
            current_expanded_qd_neagtive = []
            current_margin = []
            for i in range(self.negative_nums):
                current_expanded_qd.append([dict_info['query'],dict_info['positive']])
            self.expanded_positive_data.append(current_expanded_qd)
            
            for i,neg_docs in enumerate(dict_info['negative'][:negative_nums]):
                current_expanded_qd_neagtive.append([dict_info['query'],dict_info['negative'][i]])
                current_margin.append(dict_info['neg_dis'][i])
            self.expanded_negative_data.append(current_expanded_qd_neagtive)
            self.adaptive_margin.append(torch.Tensor(current_margin))
        #print(self.expanded_positive_data[0])
        assert len(self.adaptive_margin)==len(self.expanded_negative_data)==len(self.expanded_positive_data)
    
    def __getitem__(self, index):
        positive_dict = self.tokenizer(self.expanded_positive_data[index],
                                                   add_special_tokens=True,
                                                   return_tensors="pt",
                                                   max_length=self.max_length,
                                                   truncation=True,
                                                   padding="max_length")
        negative_dict = self.tokenizer(self.expanded_negative_data[index],
                                                   add_special_tokens=True,
                                                   return_tensors="pt",
                                                   max_length=self.max_length,
                                                   truncation=True,
                                                   padding="max_length")
        margin = self.adaptive_margin[index]
        return positive_dict,negative_dict,margin

    def __len__(self):
        return len(self.adaptive_margin)
    
    
if __name__=="__main__":
    # corpus_path="data/raw_data/multi-cpr/video/corpus.tsv"
    # query_path = "data/raw_data/multi-cpr/ecom/train.query.txt"
    # qrel_path = "data/raw_data/multi-cpr/ecom/qrels.train.tsv"
    bert_model_path="/sharedata/models/Bert-Base-Chinese"
    # ct = RerankJsonGenerator(ann_index="dump/AnnIndex/multi-cpr/ecom/myindex.faiss",
    #                       qrel_file=qrel_path,query_file=query_path,corpus_file="data/raw_data/multi-cpr/ecom/corpus.tsv",
    #                       query_array_file="dump/AnnIndex/multi-cpr/ecom/train_query_embedding")
    # data_list = ct.make_json(50)
    # json_str = ujson.dumps(data_list)
    # # 保存 JSON 字符串到文件
    # with open('data/format_data/multi-cpr/ecom/rerank_data.json', 'w') as f:
    #     f.write(json_str)
    rd_dataset = ReRankerDataset(rerank_datapath="data/rerank/rerank-qd-dev.json",model_path=bert_model_path)
    dataloader = DataLoader(rd_dataset,batch_size=3,shuffle=False)
    print("len dataset is :{}.".format(len(rd_dataset)))
    for (positive,negative,margin) in dataloader:
        print(positive.input_ids.shape)
        print(margin)
        break
