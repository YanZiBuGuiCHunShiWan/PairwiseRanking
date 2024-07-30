import torch.nn as nn
import torch
import numpy as np

# 设置随机种子以确保结果的可复现性
torch.manual_seed(0)
SUPERIOR = 0.5
INFERIOR = 0.001
MEAN=0.3
STD=0.01

def sample_adpative_margin(mean,std,batch_size:int,sort:bool=True):
    samples = torch.normal(mean, std, size=(batch_size,))
    samples = torch.min(torch.max(samples, torch.tensor(0.0001)), torch.tensor(0.5))
    if sort:
        sorted_samples = torch.sort(samples)[0]
    return sorted_samples

class AdaptiveMarginRankingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
    
    def forward(self,x_positive,x_negativce,margin):
        #x_positive : [Batch,1] x_neagtive:[Batch,1] margin:[Batch,1]
        loss = self.activation(x_negativce-x_positive+margin)
        return loss
                
