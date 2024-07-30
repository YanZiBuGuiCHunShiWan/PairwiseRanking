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
from model.loss.adaptive_margin_ranking_loss import AdaptiveMarginRankingLoss,sample_adpative_margin
from tqdm import tqdm
from loguru import logger
# 定义一个 lambda 函数来实现学习率预热
def lr_lambda(step):
    warmup_steps = 500 # 设置预热步数
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


def main():
    model_path = "/sharedata/models/nlp_rom_passage-ranking_chinese-base"
    dist.init_process_group(backend="nccl")
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank=int(os.environ.get('LOCAL_RANK'))
    rank = int(os.environ['RANK'])
    creterion = nn.MarginRankingLoss(0.3)
    model = PairWiseBertRanker(model_path=model_path)
    torch.cuda.set_device(local_rank)  # master gpu takes up extra memory
    torch.cuda.empty_cache()
    model.cuda(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)
    logger.info("模型加载完毕.............................................")
    MAX_LENGTH=128
    rd_dataset = ReRankerDataset(rerank_datapath="data/rerank/rerank-qd.json",model_path=model_path,max_length=MAX_LENGTH)
    sampler = DistributedSampler(rd_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    dataloader = DataLoader(rd_dataset,batch_size=2,sampler=sampler)
    logger.info("数据集加载完毕.............................................")
    
    optimizer = AdamW(model.parameters(), lr=3e-5,weight_decay=2e-6)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    steps = 0
    model.train()
    for epoch in range(3):
        sampler.set_epoch(epoch)
        for positive_dict,negative_dict,faiss_margin in tqdm(dataloader,desc="Training"):
            #positive_dict [batch_size,negative_nums,max_length],negative_dict [batch_size,negative_nums,max_length]
            #reshape first
            positive_attention_mask,positive_input_ids = positive_dict["attention_mask"].reshape(-1,MAX_LENGTH),positive_dict['input_ids'].reshape(-1,MAX_LENGTH)
            negtive_attention_mask,negative_input_ids = negative_dict["attention_mask"].reshape(-1,MAX_LENGTH),negative_dict['input_ids'].reshape(-1,MAX_LENGTH)
            faiss_margin = faiss_margin.reshape(-1,1)
            batch_size = faiss_margin.shape[0]
            positive_inputs = {"attention_mask":positive_attention_mask.cuda(local_rank),
                               "input_ids":positive_input_ids.cuda(local_rank)}
            negative_inputs = {"attention_mask":negtive_attention_mask.cuda(local_rank),
                               "input_ids":negative_input_ids.cuda(local_rank)}
            positive_scores,negative_scores = model(positive_inputs,negative_inputs) #[Batch,1],[Batch,1]
            y = torch.ones(batch_size,dtype=torch.float32).cuda(local_rank)
            loss = creterion(positive_scores,negative_scores,y.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if local_rank==0:
                steps+=1
                l =loss.cpu().item()
                current_lr = lr_scheduler.get_last_lr()[0]
                logger.info("current epoch:{}. current step:{}. lr :{:.5}. loss:{:.4f}".format(epoch,steps,current_lr,loss.cpu().item()))
                if steps%500==0:
                    save_checkpoint(saved_path="dump/margin_ranking",model=model,epoch=epoch,steps=steps,optimizer=optimizer,scheduler=lr_scheduler)
    
    
    
if __name__=="__main__":
    main()



