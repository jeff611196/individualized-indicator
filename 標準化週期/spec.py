#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:26:26 2023

@author: jeff
"""
import numpy as np
import pandas as pd

class Top_k:
    
    def __init__(self, name):
            
        self.read_indicator = pd.read_csv('./NN_model/dataset/top_indicator/'+name+'.csv',index_col = 0)
        self.indicator_corr = pd.DataFrame(columns=['corr'],index = self.read_indicator.index)
    
    def clean_na_corr(self,indicator_top,indicator_table,use_indicator_ori, unique = 6):
        
        for i in range(0,use_indicator_ori.shape[0]):
              
            indicator_large_columns = np.where(indicator_table.columns == indicator_top)[0][0]
                        
            indicator_columns = np.where(indicator_table.columns == use_indicator_ori.index[i])[0][0]
            
            #與股價相關最大值的indicator與其他指標的相關係數
            self.indicator_corr.iloc[i,0] = (indicator_table.iloc[:,indicator_large_columns]).corr(indicator_table.iloc[:,indicator_columns])
        
        self.drop_nan_index = self.indicator_corr.index[list(np.where(np.isnan(list(self.indicator_corr.iloc[:,0])))[0])]
        
        self.indicator_corr = self.indicator_corr.drop(self.drop_nan_index, axis=0)
        
        self.indicator_table = indicator_table.drop(columns=list(self.drop_nan_index))
    
        scatter_indicator = []
    
        for s in range(len(self.indicator_table.columns)):
            
            if len(np.unique(self.indicator_table.iloc[:,s])) < unique:
                
                scatter_indicator.append(s)
        
        self.drop_scatter_index = self.indicator_table.columns[scatter_indicator]
        
        self.indicator_corr_drop = self.indicator_corr[~self.indicator_corr.index.isin(self.drop_scatter_index)]

        self.indicator_table = self.indicator_table.drop(columns=list(self.drop_scatter_index))
        
        table_na = []
        
        for t in range(len(self.indicator_table.columns)):
            if self.indicator_table.iloc[:,t].isna().sum() > int(0.5*self.indicator_table.shape[0]):
                table_na.append(t)
        
        self.drop_table_na = self.indicator_table.columns[table_na]
        
        self.indicator_table = self.indicator_table.drop(columns=list(self.drop_table_na))
             
        return self.drop_nan_index,self.indicator_corr,self.indicator_table
       
    def indicator_info(self,corr):
        
        #以絕對值排序
        
        indicator_table_col = self.indicator_table.columns
        
        indicator_corr_index = self.indicator_corr.index
        
        intersection =  indicator_table_col.intersection(indicator_corr_index)

        self.indicator_corr = self.indicator_corr.reindex(intersection)
        
        self.indicator_corr['temp_sort'] = abs(self.indicator_corr['corr'])
    
        self.indicator_corr = self.indicator_corr.sort_values(by=['temp_sort'],ascending=True).drop(columns=['temp_sort'])
    
        indicator_top_one = self.indicator_corr.index[0]
        
        indicator_drop = self.indicator_corr.index[np.where(abs(self.indicator_corr.iloc[:,0]) > corr)[0]]

        existing_columns = set(self.indicator_table.columns) & set(indicator_drop)

        indicator_table_reduce = self.indicator_table.drop(columns=list(existing_columns))
        
        self.indicator_corr = self.indicator_corr.drop(indicator_drop, axis=0)
        
        self.indicator_corr = pd.DataFrame(columns=['corr'],index = self.indicator_corr.index)
        
        return indicator_top_one, indicator_table_reduce, self.indicator_corr