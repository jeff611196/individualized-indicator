#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:15:41 2024

@author: jeff
"""
import os

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录
current_directory = os.path.dirname(current_file_path)
# 切换工作目录到当前文件所在的目录
os.chdir(current_directory)

# os.chdir('GE')

# import os
# os.chdir('/Users/jeff/desktop/individualized-indicator/二原圖')
# fund = pd.read_csv('./fund/2023_12_01.csv',index_col = 0)
# os.chdir('/Users/jeff/desktop/project/factor/GE')
import numpy as np
import pandas as pd
import networkx as nx
# from packages.classify import read_node_label, Classifier
import gcsfs
fs = gcsfs.GCSFileSystem(project="dst-dev2021")
from .GE import DeepWalk
from .GE.packages.cosine_similar import cosine_similar

def Emb_gengrate(fund):
    """
    基金個股生成emb
    
    Args:
        fund

	Returns:
        emb
    """

    na_counts = fund.isna().sum(axis=1).tolist()
    
    na_total = sum(na_counts)
    
    rawdata_len = fund.shape[0]*fund.shape[1] - na_total
    
    picture_rawdata = pd.DataFrame(columns=['fund','stock'], index=range(0,rawdata_len))
    
    list_len = 0
    row_zero_len = fund.shape[1]-na_counts[0]
    start = row_zero_len
    
    for i in range(0,fund.shape[0]):
        start += list_len
        list_len = fund.shape[1]-na_counts[i]
        for t in range(0,list_len):
            picture_rawdata.iloc[start+t-row_zero_len,0] = fund.index[i]
            picture_rawdata.iloc[start+t-row_zero_len,1] = fund.iloc[i,t]
            
    picture_rawdata_contrary = pd.concat([picture_rawdata.iloc[:,1],picture_rawdata.iloc[:,0]],axis = 1)
    
    picture_rawdata_contrary.columns = ['fund','stock']
    
    picture_rawdata.iloc[:,1] = ["%d" % i for i in picture_rawdata.iloc[:,1]]
    
    picture_rawdata.iloc[:,1] = 's'+picture_rawdata.iloc[:,1]
    
    picture_rawdata_contrary.iloc[:,0] = ["%d" % i for i in picture_rawdata_contrary.iloc[:,0]]
    
    picture_rawdata_contrary.iloc[:,0] = 's'+picture_rawdata_contrary.iloc[:,0]
    
    picture_rawdata_final = pd.concat([picture_rawdata,picture_rawdata_contrary],axis = 0)
    
    picture_rawdata_final = picture_rawdata_final.reset_index(drop = bool)
    
    picture_rawdata_final_list = picture_rawdata_final.to_numpy()
    
    G = nx.DiGraph()
    G.add_edges_from(picture_rawdata_final_list)
    model = DeepWalk(G, walk_length=200, num_walks=100, workers=1)
    model.train(window_size=3, iter=10)
    embeddings = model.get_embeddings()
    # np.save('../emb/preprocessing/embeddings_length200_2023_09_01.npy', embeddings)
    
    # emb_folder_path_w = 'jeff-stock-wise/emb/'
    # emb_path_w = emb_folder_path_w + 'embeddings_length200_2023_09_01.npy'

    # with fs.open(emb_path_w, 'wb') as f:
    #     np.save(f, embeddings)