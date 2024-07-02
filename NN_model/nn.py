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
from audtorch.metrics.functional import pearsonr


class Model(nn.Module):
    
    
    def __init__(self,n):
        super(Model, self).__init__()

        # self.nn1 = nn.Linear(128, 1, bias = False) #第一層 Linear NN
        # self.nn2 = nn.Linear(128, 1, bias = False)
        # self.nn3 = nn.Linear(128, 1, bias = False)
        
        # self.linears = nn.ModuleList([nn.Linear(128, 1, bias = False)])
        self.linears = nn.ModuleList([nn.Linear(128, 1, bias = False) for i in range(n)])
        self.n = n
        
        
        

    def forward(self, ind_tensor, embedding):
        
        matrix_change = np.zeros(3)
        matrix_change[0] = 1
        
        self.ind_tensor = ind_tensor
        self.embedding = embedding
        
        r = []
        i_part = []
        
        for i in range(self.n):
        
            r.append(self.linears[i](embedding[0].to(torch.float32)))
        
        
        for t in range(self.n):
        
            i_part.append(ind_tensor.T[t]*torch.exp(r[t]))
        
        
        i_new = sum(i_part)/sum(torch.exp(r[i]) for i in range(self.n))
        

        # r_1 = (self.nn1(emb).T @ torch.from_numpy(matrix_change.reshape(3,1)).to(torch.float32))[0]
        # r_2 = (self.nn2(emb).T @ torch.from_numpy(matrix_change.reshape(3,1)).to(torch.float32))[0]
        # r_3 = (self.nn3(emb).T @ torch.from_numpy(matrix_change.reshape(3,1)).to(torch.float32))[0]
        

        # i_new = (i_1*torch.exp(r_1)+i_2*torch.exp(r_2)+i_3*torch.exp(r_3))/\
        #         (torch.exp(r_1)+torch.exp(r_2)+torch.exp(r_3))


        return i_new



def TTIO_loss(predictions,target):
    difference = pearsonr(target, predictions)

    return torch.abs(difference)[0]*-1

# torch.abs(difference)[0]*-1


# r_1 = (self.nn1(emb).T @ torch.from_numpy(matrix_change.reshape(5,1)).to(torch.float32))[0]