#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:47:14 2022

@author: jeff
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from audtorch.metrics.functional import pearsonr

class Model(nn.Module):
    
    def __init__(self, input_dim, indicator_count):
        super(Model, self).__init__()
        
        self.linears = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for i in range(indicator_count)])
        self.indicator_count = indicator_count

    def forward(self, use_stock, emb_total):
        
        stock_coid = np.where(emb_total.iloc[:,0] == use_stock)[0][0]

        emb_total = emb_total.drop(columns=['s'])
        emb_total = emb_total.values
        emb_total = torch.tensor(emb_total)
     
        raw_scores = torch.zeros((emb_total.size(0), self.indicator_count), device = emb_total.device)
        
        for i, linear in enumerate(self.linears):
           raw_scores[:, i] = linear(emb_total).squeeze()
           
        max_raw_scores = raw_scores.max(dim=0, keepdim=True).values
        
        adjusted_scores = raw_scores - max_raw_scores
           
        normalized_scores = F.softmax(adjusted_scores, dim=0)
        
        use_normalized_scores = normalized_scores[stock_coid]
        
        
        # r = self.linears[W](embedding[0][0].to(torch.float64))
       
        # alpha_denominator = sum(self.linears[W](emb.to(torch.float64)) for emb in emb_total)
                
        # alpha = (r/alpha_denominator)
        
        # i_new = ind_tensor.T[0]*r

        return use_normalized_scores
    
    def optimize_indicator(self, original_indicator, normalized_scores):
        # 将原始指标值归一化到 [0, 1]
        original_indicator = original_indicator.squeeze()
        
        # 计算优化后的指标值
        optimized_indicator = original_indicator * normalized_scores
        
        return optimized_indicator.sum(dim=1)
    
    
    
    



def TTIO_loss(predictions,target):
        
    difference = pearsonr(predictions, target)

    return torch.abs(difference)*-1

# torch.abs(difference)[0]*-1


# r_1 = (self.nn1(emb).T @ torch.from_numpy(matrix_change.reshape(5,1)).to(torch.float32))[0]