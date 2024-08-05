import torch
import os
import torch.nn as nn
import torch.distributed as dist
from transformers import BertConfig,BertTokenizer
from model.Pairwise_Reranker import PairWiseBertRanker
from dataset.Rerank_Dataset import ReRankerDataset
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW,Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,LambdaLR,ChainedScheduler, ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from model.loss.adaptive_margin_ranking_loss import AdaptiveMarginRankingLoss,sample_adpative_margin,RankNetLoss,rank_loss
from tqdm import tqdm
import argparse
from loguru import logger


# 定义一个 lambda 函数来实现学习率预热
def lr_lambda(step):
    warmup_steps = 100 # 设置预热步数
    if step < warmup_steps:
        return float(step) / float(warmup_steps)
    else:
        return 1.0
    
def save_checkpoint(saved_path:str,model,
                    epoch,steps,optimizer,scheduler):
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save({
            'epoch':epoch,
            'classifier':model_to_save.state_dict(), #DDP
            'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict()
        },os.path.join(saved_path,"model_{epoch}_{steps}.pt".format(epoch=epoch,steps=steps)))
    
    logger.info("*********************Saving Model Paramters Succeed!********************")


def main(args):
    #model_path = "/sharedata/models/nlp_rom_passage-ranking_chinese-base"
    model_path = "/sharedata/models/Bert-Base-Chinese"
    dist.init_process_group(backend="nccl")
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank=int(os.environ.get('LOCAL_RANK'))
    rank = int(os.environ['RANK'])

    model = PairWiseBertRanker(model_path=model_path)
    torch.cuda.set_device(local_rank)  # master gpu takes up extra memory
    torch.cuda.empty_cache()
    model.cuda(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)
    logger.info("模型加载完毕.............................................")
    MAX_LENGTH=128
    NEGATIVE_NUM = args.neg_num
    K = torch.tensor(args.adaptivek,requires_grad=False).cuda(local_rank)
    SAMPLE_MARGIN = args.sample_margin
    mode = args.mode
    if mode=="ranknet":
        creterion = RankNetLoss(sigma=torch.tensor(1,requires_grad=False).cuda(local_rank))
        SAVED_PATH = "dump/rank_net_{neg_num}".format(neg_num=NEGATIVE_NUM)
    else:
        if SAMPLE_MARGIN:
            creterion = AdaptiveMarginRankingLoss(k=torch.tensor(1,requires_grad=False).cuda(local_rank))
            SAVED_PATH = "dump/adaptive_margin_ranking_trick_sampling_{margin}_{neg_num}".format(margin=SAMPLE_MARGIN,neg_num=NEGATIVE_NUM)
        else:
            creterion = AdaptiveMarginRankingLoss(k=K)
            SAVED_PATH = "dump/adaptive_margin_ranking_trick_{margin}_{neg_num}".format(margin=args.adaptivek,neg_num=NEGATIVE_NUM)
        
    if not os.path.exists(SAVED_PATH):
        os.makedirs(SAVED_PATH)
    
    rd_dataset_third = ReRankerDataset(rerank_datapath="data/rerank/rerank-qd.json",
                                 model_path=model_path,
                                 max_length=MAX_LENGTH,
                                 negative_nums=NEGATIVE_NUM,
                                 start_index=5)
    sampler_third = DistributedSampler(rd_dataset_third, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    dataloader_third = DataLoader(rd_dataset_third,batch_size=8,sampler=sampler_third)
    
    rd_dataset_second = ReRankerDataset(rerank_datapath="data/rerank/rerank-qd.json",
                                 model_path=model_path,
                                 max_length=MAX_LENGTH,
                                 negative_nums=NEGATIVE_NUM,
                                 start_index=20)
    sampler_second = DistributedSampler(rd_dataset_second, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    dataloader_second = DataLoader(rd_dataset_second,batch_size=8,sampler=sampler_second)
    
    rd_dataset_first = ReRankerDataset(rerank_datapath="data/rerank/rerank-qd.json",
                                 model_path=model_path,
                                 max_length=MAX_LENGTH,
                                 negative_nums=NEGATIVE_NUM,
                                 start_index=40)
    sampler_first = DistributedSampler(rd_dataset_first, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    dataloader_first = DataLoader(rd_dataset_first,batch_size=8,sampler=sampler_first)
    
    dataloader_list = [dataloader_first,dataloader_second,dataloader_third]
    datasampler_list = [sampler_first,sampler_second,sampler_third]
    
    logger.info("数据集加载完毕.............................................")
    #SAVED_PATH = "dump/margin_ranking_{margin}_{neg_num}".format(margin=MARGIN,neg_num=NEGATIVE_NUM)

    optimizer = AdamW(model.parameters(), lr=3e-5,weight_decay=2e-6)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    steps = 0
    model.train()
    for epoch in range(2):
        sampler = datasampler_list[epoch]
        sampler.set_epoch(epoch)
        for positive_dict,negative_dict,adaptive_margin in tqdm(dataloader_list[epoch],desc="Training"):
            #positive_dict [batch_size,negative_nums,max_length],negative_dict [batch_size,negative_nums,max_length]
            positive_attention_mask,positive_input_ids = positive_dict["attention_mask"].reshape(-1,MAX_LENGTH),positive_dict['input_ids'].reshape(-1,MAX_LENGTH)
            negtive_attention_mask,negative_input_ids = negative_dict["attention_mask"].reshape(-1,MAX_LENGTH),negative_dict['input_ids'].reshape(-1,MAX_LENGTH)
            adaptive_margin = adaptive_margin.cuda(local_rank).reshape(-1,1)
            batch_size = adaptive_margin.shape[0]
            positive_inputs = {"attention_mask":positive_attention_mask.cuda(local_rank),
                               "input_ids":positive_input_ids.cuda(local_rank)}
            negative_inputs = {"attention_mask":negtive_attention_mask.cuda(local_rank),
                               "input_ids":negative_input_ids.cuda(local_rank)}
            
            if mode=="ranknet":
                _,_,(positive_logits,negative_logits) = model(positive_inputs,negative_inputs) #[Batch,1],[Batch,1]
                loss = creterion(positive_logits,negative_logits)
            else:
                positive_scores,negative_scores,(_,_) = model(positive_inputs,negative_inputs) #[Batch,1],[Batch,1]
                if SAMPLE_MARGIN:
                    adaptive_margin = sample_adpative_margin(mean=SAMPLE_MARGIN,std=0.01,batch_size=batch_size,sort=True).cuda(local_rank)       
                loss = creterion(positive_scores,negative_scores,adaptive_margin)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if local_rank==0:
                steps+=1
                l =loss.cpu().item()
                current_lr = lr_scheduler.get_last_lr()[0]
                logger.info("current epoch:{}. current step:{}. lr :{:.5}. loss:{:.4f}".format(epoch,steps,current_lr,loss.cpu().item()))
                if steps%400==0:
                    save_checkpoint(saved_path=SAVED_PATH,model=model,epoch=epoch,steps=steps,optimizer=optimizer,scheduler=lr_scheduler)
    
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

# 添加参数
    parser.add_argument('--mode', type=str,default="margin",choices=["margin","ranknet"])
    parser.add_argument('--adaptivek', type=int,default=0.3)
    parser.add_argument('--neg_num',type=int,default=5)
    # 添加布尔参数
    # 使用 store_true 添加一个默认为 False 的参数，如果命令行中指定了该参数，则其值为 True
    parser.add_argument('--sample_margin', type =float ,default=None,required=False)

    # 解析命令行参数
    args = parser.parse_args()

    main()



