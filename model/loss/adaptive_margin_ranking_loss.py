import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
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
        
                
def RankNetLoss(s_i,s_j):
    """_summary_

    Args:
        s_i (torch.Tensor): similarity(query_i,doc_i^+)
        s_j (torch.Tensor): similarity(query_i,doc_j^-)
        
    s_ij = 1. because doc_i^+ is much more relavant to query_i
    so the cost is $\log (sigmoid(s_i-s_j))$
    we can implement it by BCE Loss.
    """
    diff = s_i-s_j #[Batch]
    y_loss = -1*F.logsigmoid(s_i-s_j)
    return y_loss.mean()