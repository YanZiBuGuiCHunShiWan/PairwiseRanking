# PairwiseRanking

# 项目概述

​	本项目主要研究排序学习中常见的文本排序策略，比如margin ranking loss(又称margin triplet loss)，Adaptive margin ranking loss, RankNet, Adaptive RankNet.本项目主要实现了主流的Pairwise形式的Ranking方法。作者实现了三种最常见的Pairwise损失函数。

1. Margin Ranking Loss 

​				          		$L=\max(0,-l*(x_1-x_2)+m) $

​		用于文本排序场景，则有：

​				       	$x_1=sim(query,doc^{+}),x_2=sim(query,doc^-)$

​					$L=\max(0,sim(query,doc^-)-sim(query,doc^+)+m)$

2. Adaptive Margin Ranking Loss(注:称作Adaptive Margin Triplet Loss也行)

​				$L=\sum_{j \in neg(i)} \max(0,sim(query_i,doc_j^-)-sim(query_i,doc_i^+)+margin_j)$

​	其中$margin_j$是动态变化的，比如假设当前的$query_i$有$k$个对应的负样本$doc_j,j=1,...,k$，那么对于每一个pair,$query_i$和$doc_j^-$都有一个变化的$margin_j$。**显而易见**，在文档检索场景下，$margin_j$可以由$faiss$向量检索工具召回计算的向量距离$d_j$决定，最简单的，比如$margin_j=d_j$，进一步推广，可以有$margin_j=k*d_j;margin_j=k*\sqrt d_j;margin_j=d_j^{k}$，读者可以自行推展出各种各样的公式。adaptive的思想其实很自然，笔者在Google Scholar上搜索了一圈，发现有许多相关用法，比如[Adaptive margin ranking loss](https://arxiv.org/abs/1907.05336),[Adaptive margin loss](https://arxiv.org/abs/2005.13826),[Adaptive margin triplet loss](https://arxiv.org/abs/2107.06187)......总之有一堆。

​	此外，$margin_j$可以选择不由向量检索工具决定，比如可以从正太分布中采样$batchsize$个大小的$margin$，排序以后再作为$\text{adaptive margin}$，即$sample_j \sim \mathcal N(0.3,0.01)$，进一步，可以限制$margin_j$的范围，$margin_j = \min(\max(sample_j,0.0001),0.5)$，就是$margin_j$不能过大，也不能过小。

​	上述的$margin_j$都是不可学习的，我们也可以设计一个可学习的$margin_j$，假设文本编码器编码$query_i,doc_i^+$的语义向量记作$h_i^+$,编码$query_i,doc_j^-$的语义向量记作$h_j^-$，可以设计$margin_j=k*dis(h_i^+,h_j^-)$，其中$dis$是一个衡量向量相似度的函数，最普通的，可以用$L_2$距离来度量，$k$用来缩放。总之$\text{Adaptive}$是一种很常见，很直观的思想。

```python

class AdaptiveMarginRankingLoss(nn.Module):
    def __init__(self,k:torch.Tensor):
        '''
        K超参数，用于控制adpative margin
        '''
        super().__init__()
        self.activation = nn.ReLU()
        self.k = k
    def forward(self,x_positive,x_negativce,margin):
        #x_positive : [Batch,1] x_neagtive:[Batch,1] margin:[Batch,1]
        loss = self.activation(x_negativce-x_positive+self.k*margin)
        return loss.mean()
        

class RankNetLoss(nn.Module):
    def __init__(self,sigma:torch.Tensor):
        super().__init__()
        self.sigma = sigma
        
    def forward(self,positive_logits,negative_logits):
        """_summary_

        Args:
            s_i (torch.Tensor): similarity(query_i,doc_i^+)
            s_j (torch.Tensor): similarity(query_i,doc_j^-)
        
        s_ij = 1. because doc_i^+ is much more relavant to query_i
        so the cost is $\log (sigmoid(s_i-s_j))$
        """
        loss = -1*F.logsigmoid(self.sigma*(positive_logits-negative_logits))
        return loss.mean()
        
def rank_loss(p,n,sigma):
    loss = -1*F.logsigmoid(sigma*(p-n))
    return loss.mean()
        
```

$\text{RankNet}$

​										$\begin{aligned}P(U_i \gt U_j)=\frac{1}{1+\exp (-\sigma(s_i-s_j))} \end{aligned}$

​	$\text{RankNet}$认为，对于文档$U_i$与$query$的相关性大于$U_j$相关性的概率为$\bar{P_{ij}}$，其中$\bar{P_{ij}}=\frac{1}{2}(1+S_{ij})$，其中：

​											$\begin{aligned}S_{ij}= \begin{cases}  1 &\text{if }U_i \gt U_j \\ 0 &\text{if } U_i=U_j \\ -1 &\text{if }U_i \lt U_j \end{cases}\end{aligned}$

​	对于任意文档对$(U_i,U_j)$关系的概率公式则可以用公式$P_{ij}^{\bar{P_{ij}}}(1-P_{ij})^{1-\bar{P_{ij}}}$进行描述。加上$-\log$符号，则有：

​					              			$\begin{aligned}L=-{\bar{P_{ij}}}\log(P_{ij})-{(1-\bar{P_{ij}})}\log(1-P_{ij}) \end{aligned}$

带入公式$(1.1)$​则有：

​									     $\begin{aligned}L&=-{\bar{P_{ij}}}\log(P_{ij})-{(1-\bar{P_{ij}})}\log(1-P_{ij}) \\ &=-\frac{1}{2}(1-S_{ij})\sigma(s_i-s_j)+\log(1+\exp(-\sigma(s_i-s_j))) \end{aligned}$

```latex
\begin{aligned}L&=-{\bar{P_{ij}}}\log(P_{ij})-{(1-\bar{P_{ij}})}\log(1-P_{ij}) \\ &=-\frac{1}{2}(1-S_{ij})\sigma(s_i-s_j)+\log(1+\exp(-\sigma(s_i-s_j))) \end{aligned}
```

​	 如果我们强制另$S_{ij}=1$，则有：

​										$\begin{aligned}L&=-\log(sigmoid(\sigma(s_i-s_j)))\end{aligned}$​

​	如下是一个简洁的实现，假设$query$和$doc_i$是相关的，和$doc_j$是没那么相关的。以文本检索的场景为例，$h_i=Encoder(query\oplus doc_i)\in \mathbb R^{1 \times 768},h_j=Encoder(query\oplus doc_j)\in \mathbb R^{1 \times 768}$：

```python
import torch.nn.functional as F
import torch.nn as nn

class PairWiseBertRanker(nn.Module):    
    def __init__(self,model_path,learnable_margin:bool=False) -> None:
        super().__init__()
        self.config = BertConfig.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)        
        self.linear = nn.Linear(768,1)
        
    def forward(self,positive_inputs:Optional[Dict]=None,negative_inputs:Optional[Dict]=None):
        if positive_inputs is None:
            negative_outputs = self.model(**negative_inputs).last_hidden_state[:,0,:] #[Batch,768]
        	negative_logits = self.linear(negative_outputs)
            return neagtive_logits
        positive_outputs = self.model(**positive_inputs).last_hidden_state[:,0,:] #[Batch,768]
        positive_logits = self.linear(positive_outputs) #[Batch,1]
        negative_outputs = self.model(**negative_inputs).last_hidden_state[:,0,:] #[Batch,768]
        negative_logits = self.linear(negative_outputs) #[Batch,1]
        return positive_logits,negative_logits
    
class RankNetLoss(nn.Module):
    def __init__(self,sigma:torch.Tensor):
        super().__init__()
    	self.sigma = sigma
    def forward(self,positive_logits,negative_logits):
        loss = -1*F.logsigmoid(self.sigma*(positive_logits-negative_logits))
        return loss.mean()
```

$\text{RankNet}$损失函数如下（sigma=1）：

![image-20240801133543719](assets/1.png)

​	当$\sigma$等于其他值时，有：

![image-20240801133729331](assets/2.png)

​	即$\sigma$越大，惩罚越厉害。

$\text{Adaptive RankNet}$​​
>>>>>>> e532b2c360b9a9e787973ff39050c5b0cbbda9e8
