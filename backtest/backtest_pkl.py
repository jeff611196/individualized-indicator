#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:40:42 2023

@author: jeff
"""
import pandas as pd
import numpy as np
import glob
import yaml
from tqdm import tqdm, trange
from NN_model.dataset_DST import *

class Input_backtest_table:
    def __init__(self, recommend_stock, top_k, test_start, test_end, train_season):
        self.recommend_stock = recommend_stock
        self.train_season = train_season
        self.indicator_top_list = pd.read_csv('./emb/'+self.train_season+'/indicator_chose.csv')['0'].tolist()
        self.embeddings_concat = self.recommend_stock.embeddings_concat
        # self.start = self.recommend_stock.day_start
        # self.end = self.recommend_stock.day_end
        self.test_start = test_start
        self.test_end = test_end
        self.start_row = np.where(self.recommend_stock.close.index == test_start)[0][0]
        self.end_row = np.where(self.recommend_stock.close.index == test_end)[0][0]
        # self.start_row = np.where(self.recommend_stock.close.index == self.recommend_stock.day_start)[0][0]
        # self.end_row = np.where(self.recommend_stock.close.index == self.recommend_stock.day_end)[0][0]  
        self.top = top_k
        
    def softmax(self, row):
        exp_row = np.exp(row - np.max(row))
        return exp_row / exp_row.sum()
    
    
    def calculate(self):

        coe = pd.read_csv('./emb/'+self.train_season+'/'+self.top+'/result.csv',index_col = 0)
        raw_scores = pd.DataFrame(columns=range(coe.shape[1]), index=range(len(self.embeddings_concat)))

        for s in tqdm(range(0,len(self.embeddings_concat))):

            stock = self.embeddings_concat.iloc[s,0]
            stock_coid = np.where(self.embeddings_concat.iloc[:,0] == stock)[0][0]

            stock_minmax = pd.read_csv('./emb/'+self.train_season+'/minmax/'+self.top+'/'+stock+'.csv')
            stock_minmax.set_index(keys = ['stock','minmax'],inplace=True)

            emb = self.embeddings_concat.iloc[np.where(self.embeddings_concat.iloc[:,0] == stock)[0][0],:]

            emb_1 = emb.drop(labels = ['s'], inplace = False)

            for q in range(0,coe.shape[1]):        

                raw_scores.iloc[s,q] = emb_1 @ coe.iloc[:,q]

        raw_scores = raw_scores.astype(np.float64)
        softmax_df = raw_scores.apply(self.softmax, axis=0)   
        df = self.recommend_stock.New_indicator_IO(self.indicator_top_list, stock,'back_test')[0]
        # indicator_merge = pd.DataFrame(columns = df.columns)
        open_p = self.recommend_stock.open
        high_p = self.recommend_stock.high
        low_p = self.recommend_stock.low
        close_p = self.recommend_stock.close
        vol_p = self.recommend_stock.vol
        final_df = pd.DataFrame()
        
        for h in tqdm(range(0,len(self.embeddings_concat))):

            stock = self.embeddings_concat.iloc[h,0]
            indicator_total = self.recommend_stock.New_indicator_IO(self.indicator_top_list, stock,'back_test')[0]
            indicator = indicator_total.iloc[:,0].values
            indicator = indicator - stock_minmax.iloc[0,0]
            indicator = indicator/stock_minmax.iloc[1,0]
            col = np.where(open_p.columns == stock)[0][0]
            open_choose = open_p.iloc[self.start_row:self.end_row+1,col]
            high_choose = high_p.iloc[self.start_row:self.end_row+1,col]
            low_choose = low_p.iloc[self.start_row:self.end_row+1,col]
            close_choose = close_p.iloc[self.start_row:self.end_row+1,col]
            vol_choose = vol_p.iloc[self.start_row:self.end_row+1,col]
            
            for i in range(1,len(indicator_total.columns)):
            
                indicator_1 = indicator_total.iloc[:,i].values
                indicator_1 = indicator_1 - stock_minmax.iloc[0,i]
                indicator_1 = indicator_1/(stock_minmax.iloc[1,i]-stock_minmax.iloc[0,i])
                indicator = np.vstack((indicator,indicator_1))
            
            indicator = pd.DataFrame(indicator.T)
            indicator.index = indicator_total.index
            indicator.columns = indicator_total.columns
            
            for j in range(0,indicator.shape[1]):
                indicator.iloc[:,j] = indicator.iloc[:,j] * softmax_df.iloc[h,j]

            open_choose.index = indicator.index
            high_choose.index = indicator.index
            low_choose.index = indicator.index
            close_choose.index = indicator.index
            vol_choose.index = indicator.index            
            t_df = pd.concat([open_choose,high_choose,low_choose,close_choose,vol_choose,indicator],axis = 1)
            t_df.columns = ['open', 'high', 'low', 'close', 'vol', *t_df.columns[5:]]
            final_df = pd.concat([final_df, t_df], axis=0)

        new = pd.DataFrame(columns=['new'],index = final_df.index)
        row_sums = final_df.iloc[:,5:12].sum(axis=1)
        final_df = pd.concat([final_df.iloc[:,0:5], row_sums], axis=1)
        final_df.columns = ['open', 'high', 'low', 'close', 'vol', 'new']


        final_df.to_pickle('./emb/'+self.train_season+'/'+str(self.top)+'_'+self.test_start+'_'+self.test_end+'.pkl', compression='infer', protocol=5, storage_options=None)