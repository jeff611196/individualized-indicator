#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:27:44 2022

@author: jeff
"""

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from NN_model.nn import *
import statistics
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='runs/experiment_1')

class Indicator_coefficient:
            
    def __init__(self, etl, indicator_top_list):
        
        self.etl = etl
        self.indicator_top_list = indicator_top_list
        self.number_list = list(range(0,len(self.etl.embeddings_concat.iloc[:,0])))
        self.result = pd.DataFrame(columns=['mean','max','min','std'],index = self.number_list)
        self.emb_path_list_train = self.etl.embeddings_concat.iloc[:,0].reset_index(drop = True)

    def calculate(self, top_k, file):
        
        file_ori = f'./emb/{file}'
        file_1 = f'./emb/{file}/top_{str(top_k)}'
        file_2 = f'./emb/{file}/minmax'
        file_3 = f'./emb/{file}/minmax/top_{str(top_k)}'
        self.etl.file_3 = f'./emb/{file}/minmax/top_{str(top_k)}'

        if not os.path.exists(file_ori):
            
            os.mkdir(file_ori)

        if not os.path.exists(file_1):
            
            os.mkdir(file_1)
        
        if not os.path.exists(file_2):
            
            os.mkdir(file_2)
            
        if not os.path.exists(file_3):
            
            os.mkdir(file_3)
    
        for s in tqdm(range(0,len(self.etl.embeddings_concat.iloc[:,0]))): 
                  
            try:  
        
                z = self.etl.New_indicator_IO(s,self.indicator_top_list,self.emb_path_list_train)
                
            except:
                print("\n")
                print("stock",self.etl.stock_chose)
                
                
            else:
                
                #105天(105個batch，batch_size=3，一個epoch train 35次)
                t = self.etl.dataset_new_indicator(z,self.etl.stock_chose,top_k)

                if t == "ind_can't_softmax":
                    continue
                                
                train_loader = DataLoader(t, batch_size=3, shuffle=False)
                
                # dataiter = iter(train_loader)
                # data = dataiter.next()
                
                # input_embedding = data[0]
                # target = data[4]
                
                # input_embedding = torch.Tensor(input_embedding.reshape(5,128))
                # target =torch.Tensor(target.reshape(5,1))
                
                # i_1 = data[1]
                # i_2 = data[2]
                # i_3 = data[3]

                model = Model(top_k)
                
                # print(model) #將模型print出來看看
                loss_function = TTIO_loss
                optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
                # optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
                
                epochs = 200
                
                loss_list = []
                para = []
                loss_look_list = []
                

                for epoch in range(epochs):
                    a = 0

                    model.train()
                    
                    for idx, (ind_tensor, targets, embedding) in enumerate(train_loader):

                        prediction = model(ind_tensor, embedding)

                        loss = loss_function(prediction, targets)

                        loss_look = loss.detach().numpy()
                        
                        if np.isnan(loss_look) == True:
                            break
                                                
                        loss_look_list.append(loss_look)
                
                        #loss_list.append(float(loss_look)) 
                        #loss_list.append(loss.detach().numpy())
                        a += loss_look 
                        para.append(model.state_dict()['linears.0.weight'][0][0:5].numpy().tolist())
                        # para.append(list(model.parameters())[0].detach().numpy()[0][0])

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    if np.isnan(loss_look) == True:
                        break    
                    
                    a=a/35
                    loss_list.append(a)    
                    writer.add_scalar('Average Loss per Epoch', a, epoch)

                if np.isnan(loss_look) == True:
                    continue    
                
                self.result.iloc[s,0] = np.mean(loss_list)            
                self.result.iloc[s,1] = max(loss_list)       
                self.result.iloc[s,2] = min(loss_list)
                self.result.iloc[s,3] = statistics.pstdev(loss_list)         
                    
                
                emb_data = pd.DataFrame({z[0].columns[i]:list(model.parameters())[i].detach().numpy()[0] for i in range(top_k)})
                                
                title = z[2]['s']
            
                emb_data.to_csv(file_1+'/'+title+'.csv')
                   
        print('end')


# zzz = pd.concat([pd.DataFrame(list(model.parameters())[i].detach().numpy()) for i in range(5)],axis=0)

# for s in range(5):
#     np.where((zzz.iloc[0,:] > 0.8)|(zzz.iloc[0,:] < -0.8))
