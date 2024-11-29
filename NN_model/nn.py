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
    
# def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
#     """
#     計算斯皮爾曼相關係數。
#     :param x: 張量 (1D)
#     :param y: 張量 (1D)
#     :return: Spearman's Rank Correlation Coefficient
#     """
#     if x.shape != y.shape:
#         raise ValueError("x 和 y 必須具有相同的形狀")
    
#     # 計算排名
#     def rank_tensor(t):
#         sorted_indices = torch.argsort(t)
#         ranks = torch.zeros_like(sorted_indices, dtype=torch.float32)
#         ranks[sorted_indices] = torch.arange(1, len(t) + 1, dtype=torch.float32)
#         return ranks
    
#     rank_x = rank_tensor(x)
#     rank_y = rank_tensor(y)
    
#     # 計算皮爾森相關係數
#     mean_x = rank_x.mean()
#     mean_y = rank_y.mean()
#     cov_xy = ((rank_x - mean_x) * (rank_y - mean_y)).mean()
#     std_x = torch.sqrt(((rank_x - mean_x) ** 2).mean())
#     std_y = torch.sqrt(((rank_y - mean_y) ** 2).mean())
    
#     spearman_coefficient = cov_xy / (std_x * std_y)
#     return spearman_coefficient   

def differentiable_ic(predictions, targets):
    def soft_rank(t, regularization_strength=1e-6):
        pairwise_diff = t.unsqueeze(1) - t.unsqueeze(0)
        smoothing = torch.sigmoid(-pairwise_diff / regularization_strength)
        return smoothing.sum(dim=-1)

    pred_rank = soft_rank(predictions)
    target_rank = soft_rank(targets)

    mean_pred = pred_rank.mean()
    mean_target = target_rank.mean()
    cov = ((pred_rank - mean_pred) * (target_rank - mean_target)).mean()
    std_pred = torch.sqrt(((pred_rank - mean_pred) ** 2).mean())
    std_target = torch.sqrt(((target_rank - mean_target) ** 2).mean())
    rank_ic = cov / (std_pred * std_target)
    return rank_ic



def TTIO_loss(predictions,target):
        
    # difference = pearsonr(predictions, target)
    # difference = spearman_corr(predictions, target)
    difference = differentiable_ic(predictions, target)

    return torch.abs(difference)*-1

# torch.abs(difference)[0]*-1


# r_1 = (self.nn1(emb).T @ torch.from_numpy(matrix_change.reshape(5,1)).to(torch.float32))[0]