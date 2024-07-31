import torch
import torch.nn as nn
from transformers import BertModel,BertConfig,BertTokenizer
from typing import Optional,Dict


class PairWiseBertRanker(nn.Module):    
    def __init__(self,model_path,learnable_margin:bool=False) -> None:
        super().__init__()
        self.config = BertConfig.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)        
        self.linear = nn.Linear(768,1)
        self.activation = nn.Sigmoid()
        self.learnable_margin = learnable_margin
        
    def forward(self,positive_inputs:Optional[Dict]=None,negative_inputs:Optional[Dict]=None):
        if positive_inputs is None:
             negative_outputs = self.model(**negative_inputs).last_hidden_state[:,0,:] #[Batch,768]
             negative_scores= self.activation(self.linear(negative_outputs))
             return negative_scores
         
        positive_outputs = self.model(**positive_inputs).last_hidden_state[:,0,:] #[Batch,768]
        positive_logits = self.linear(positive_outputs)
        positive_scores = self.activation(positive_logits) #[Batch,1]
        negative_outputs = self.model(**negative_inputs).last_hidden_state[:,0,:] #[Batch,768]
        negative_logits = self.linear(negative_outputs)
        negative_scores = self.activation(negative_logits) #[Batch,1]
        if self.learnable_margin:
            extra_ouput = (positive_outputs,negative_outputs)
        else:
            extra_ouput = (positive_logits,negative_logits)
        return positive_scores,negative_scores,extra_ouput