#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:04:11 2023

@author: jeff
"""

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch


class Indicator_Dataset(Dataset):
    def __init__(self, ind, targets, emb, transform=None, target_transform=None):
        self.indicator = ind
        self.labels = targets
        self.input_embedding = emb
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        ind = []
        
        for i in range(len(self.indicator.columns)):
            ind.append(self.indicator.iloc[idx,i])
            
        ind_tensor = torch.tensor(ind)
        label = torch.tensor(self.labels[idx])
        emb = torch.tensor(self.input_embedding.iloc[idx,:])
        # indicator_inp = self.indicator.iloc[:,idx]
        # indicator_inp_tensor = torch.tensor(indicator_inp)
 

        return ind_tensor, label, emb