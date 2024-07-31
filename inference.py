import torch
import os
import torch.nn as nn
import torch.distributed as dist
from transformers import BertConfig,BertTokenizer,BertModel
from model.Pairwise_Reranker import PairWiseBertRanker
from dataset.Rerank_Dataset import ReRankerDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import faiss


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

        
def main(model_path,dev_path,query_embedding_path):
        
    index = faiss.read_index("embedding/faiss.index")
    query_doc_map_csv="data/ecom/qrels.dev.tsv"
    gr_list =[]
    with open(query_doc_map_csv,"r") as f:
        for line in f:
            gr=line.strip().split('\t')[2]
            gr_list.append(int(gr))
            
    query_dev_embedding = load_embedding(query_embedding_path)
    print("开始检索")
    distances,indices = index.search(np.array(query_dev_embedding),10)
    mrr_list = []
    for i,index_list in enumerate(indices):
        gt_label = gr_list[i]-1
        if gt_label not in index_list:
            mrr_list.append(0)
        else:
            ind= list(index_list).index(gt_label)+1
            mrr_list.append(1/ind)
    print("当前检索结果：{:.4f}".format(np.mean(mrr_list)))
        
    pretrained_model =  PairWiseBertRanker(model_path="/sharedata/models/nlp_rom_passage-ranking_chinese-base")
    checkpoint = torch.load(model_path)
    print("load state dict.........................")
    pretrained_model.load_state_dict(checkpoint['classifier'],strict=False)
    rerank_model = pretrained_model.cuda()
    rerank_model.eval()
    dataset = ReRankerDataset(rerank_datapath=dev_path,
                              model_path="/sharedata/models/nlp_rom_passage-ranking_chinese-base",
                              max_length=128,
                              negative_nums=10)

    rerank_mrr_list = []
    for i,index_list in enumerate(indices):
        gt_label = gr_list[i]-1
        if gt_label not in index_list:
            rerank_mrr_list.append(0)
        else:
            ind= list(index_list).index(gt_label)+1
            ##开始进行重排序
            _,current_docs,_ = dataset[i]
            negative_attention_mask= current_docs['attention_mask'].cuda()
            negative_input_ids =  current_docs['input_ids'].cuda()
            negative_inputs= {
                "attention_mask":negative_attention_mask,
                "input_ids":negative_input_ids
                    }
            with torch.no_grad():
                batch_scores = rerank_model(positive_inputs=None,negative_inputs=negative_inputs).cpu().numpy()
            assert batch_scores.shape[0]==10
            previous_index_score_dict = dict(zip(index_list,batch_scores))
            sorted_dict = sorted(previous_index_score_dict.items(), key=lambda item: item[1], reverse=True)
            sorted_index_list =list(dict(sorted_dict).keys())
            sorted_index = sorted_index_list.index(gt_label)+1
            rerank_mrr_list.append(1/sorted_index)
    print("重排序后结果：{:.4f}".format(np.mean(rerank_mrr_list)))
    print("增长:{:.4f}".format(np.mean(rerank_mrr_list)-np.mean(mrr_list)))
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

# 添加参数
    parser.add_argument('--model_path', type=str,default="dump/margin_ranking/model_2_37500.pt")
    parser.add_argument('--dev_path', type=str,default="data/rerank/rerank-qd-dev.json")
    parser.add_argument('--query_embedding',type=str,default='embedding/query_embedding')

# 解析命令行参数
    args = parser.parse_args()

    main(args.model_path,args.dev_path,args.query_embedding)