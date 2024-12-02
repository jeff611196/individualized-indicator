#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:27:44 2022

@author: jeff
"""

import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from NN_model.nn import *
import statistics
import os
import shutil
import gcsfs
fs = gcsfs.GCSFileSystem(project="dst-dev2021")
from torch.utils.tensorboard import SummaryWriter

class Indicator_coefficient:
            
    def __init__(self, etl, indicator_top_list):
        
        self.etl = etl
        self.indicator_top_list = indicator_top_list
        self.number_list = list(range(0,len(self.etl.embeddings_concat.iloc[:,0])))
        self.result = pd.DataFrame(columns=['mean','max','min','std'],index = self.number_list)
        self.emb_path_list_train = self.etl.embeddings_concat.iloc[:,0].reset_index(drop = True)

    def calculate(self, indicator_count, file, window_size, stride):
        
        input_dim = 128  # 嵌入向量的维度
        file_ori = f'./emb/{file}'
        file_1 = f'./emb/{file}/top_{str(indicator_count)}'
        file_2 = f'./emb/{file}/minmax'
        file_3 = f'./emb/{file}/minmax/top_{str(indicator_count)}'
        self.etl.file_3 = f'./emb/{file}/minmax/top_{str(indicator_count)}'

        if not os.path.exists(file_ori):
            
            os.mkdir(file_ori)

        if not os.path.exists(file_1):
            
            os.mkdir(file_1)
        
        if not os.path.exists(file_2):
            
            os.mkdir(file_2)
            
        if not os.path.exists(file_3):
            
            os.mkdir(file_3)
        
        epochs = 30

        model = Model(input_dim, indicator_count)
        model.double()
        # print(model) #將模型print出來看看
        loss_function = TTIO_loss
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        # torch.optim.SGD
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        para_0 = []
        para_1 = []
        para_2 = []
        para_3 = []
        para_4 = []
        para_5 = []
        para_6 = []
        para_7 = []

        writer = SummaryWriter(log_dir=f'runs/experiment_1/{file}')

        for epoch in range(epochs):
            
            print(epoch)
            
            model.train()

            losses = []

            epoch_loss = 0
            
            # len(self.etl.embeddings_concat.iloc[:,0])
            
            for t in tqdm(range(0,len(self.etl.embeddings_concat.iloc[:,0]))): 
                
                use_stock = self.etl.embeddings_concat.iloc[t,0]
                
                try:  
                    new_ind_info = self.etl.New_indicator_IO(self.indicator_top_list, use_stock, self.emb_path_list_train)
                    
                except:
                    print("\n")
                    print("stock",use_stock)
                    
                else:
                    #105天(105個batch，batch_size=3，一個epoch train 35次)
                    indicator_dataset = self.etl.dataset_new_indicator(new_ind_info, use_stock, indicator_count, window_size, stride)
                    train_loader = DataLoader(indicator_dataset, batch_size=1, shuffle=False)
                        
                    for idx, (original_indicator, targets, embedding) in enumerate(train_loader):
                        
                        targets = targets.squeeze()
                        
                        embedding = embedding.squeeze()
    
                        normalized_scores = model(use_stock, new_ind_info[3])
    
                        optimized_indicator = model.optimize_indicator(original_indicator, normalized_scores)

                        loss = loss_function(optimized_indicator, targets)

                        # para.append(model.state_dict()['linears.0.weight'][0][0:5].numpy().tolist())
                        # para.append(list(model.parameters())[0].detach().numpy()[0][0])
                        
                        if not torch.isnan(loss):
                            optimizer.zero_grad()
                            loss.backward()
                            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            losses.append(loss.item())
                            para_0.append(model.state_dict()['linears.0.weight'][0][0:5].numpy().tolist())
                            para_1.append(model.state_dict()['linears.1.weight'][0][0:5].numpy().tolist())
                            para_2.append(model.state_dict()['linears.2.weight'][0][0:5].numpy().tolist())
                            para_3.append(model.state_dict()['linears.3.weight'][0][0:5].numpy().tolist())
                            para_4.append(model.state_dict()['linears.4.weight'][0][0:5].numpy().tolist())
                            para_5.append(model.state_dict()['linears.5.weight'][0][0:5].numpy().tolist())
                            para_6.append(model.state_dict()['linears.6.weight'][0][0:5].numpy().tolist())
                            # para_7.append(model.state_dict()['linears.7.weight'][0][0:5].numpy().tolist())

            
            if len(losses) > 0:
                epoch_loss = sum(losses) / len(losses)
            else:
                epoch_loss = 0
            
            scheduler.step()
            
            # if np.isnan(losses).any():
            #     break

            writer.add_scalar('Average Loss per Epoch', epoch_loss, epoch)
        
        writer.close()
            # self.result.iloc[s,0] = np.mean(loss_look_list)            
            # self.result.iloc[s,1] = max(loss_look_list)       
            # self.result.iloc[s,2] = min(loss_look_list)
            # self.result.iloc[s,3] = statistics.pstdev(loss_look_list)         

        emb_data = pd.DataFrame({self.indicator_top_list[i]:list(model.parameters())[i].detach().numpy()[0] for i in range(indicator_count)})
                                        
        title = self.indicator_top_list
                    
        emb_data.to_csv(file_1+'/'+'result_spearman.csv')

        result_folder_path_w = 'jeff-stock-wise/train/'
        result_path_w = result_folder_path_w + "result_"+file+".csv"  # 指定檔案名稱
        with fs.open(result_path_w, 'w') as f:
            emb_data.to_csv(f)


        print('end')