#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:40:42 2023

@author: jeff
"""
import pandas as pd
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
        self.t_df_2 = pd.DataFrame()    
        self.top = top_k
    
    def calculate(self):

        for s in tqdm(range(0,len(self.recommend_stock.emb_path_list))):
        #for s in tqdm(range(61,62)):    
        #for s in tqdm(range(0,20)):
                
            title_2 = self.recommend_stock.emb_path_list[s]
            
            stock = title_2.split('/')[3][:-4]
            
            stock_minmax = pd.read_csv('./emb/'+self.train_season+'/minmax/'+self.top+'/'+stock+'.csv')
               
            stock_minmax.set_index(keys = ['stock','minmax'],inplace=True)
            
            coe = pd.read_csv('./emb/'+self.train_season+'/'+self.top+'/'+stock+'.csv',index_col = 0)    
            
            #需load前面model
            emb = self.embeddings_concat.iloc[np.where(self.embeddings_concat.iloc[:,0] == stock)[0][0],:]

            emb_1 = emb.drop(labels = ['s'], inplace = False)

            coe_list = list(coe.columns)
            coe_df = pd.DataFrame(columns=['value'], index=coe_list)

            for i in range(len(coe.columns)):
            
                r = np.array(coe.iloc[:,i]).reshape(1,128) @ np.array(emb_1, dtype = float).reshape(128,1)
                coe_df.iloc[i,0] = r[0][0]

            
            indicator_total = self.recommend_stock.New_indicator_IO(s,self.indicator_top_list,'back_test')[0]        
            
            indicator = indicator_total.iloc[:,0].values
            indicator = indicator - stock_minmax.iloc[0,0]
            indicator = indicator/stock_minmax.iloc[1,0]
            
            for i in range(1,len(indicator_total.columns)):
            
                indicator_1 = indicator_total.iloc[:,i].values
                indicator_1 = indicator_1 - stock_minmax.iloc[0,i]
                indicator_1 = indicator_1/stock_minmax.iloc[1,i]
                # if pd.Series(indicator_1).isnull()[0] == True:
                #     print(i)
                #     break
                
                indicator = np.vstack((indicator,indicator_1))
            
            indicator = pd.DataFrame(indicator.T)
            
            new = pd.DataFrame(0,columns=['new'],index = indicator.index)
            
            molecular = 0
            denominator = 0
            
            for t in range(0,len(indicator.index)):            
                for i in range(0,len(indicator.columns)):
                    molecular += indicator.iloc[t,i]*np.exp(coe_df.iloc[i,0])
                    denominator += np.exp(coe_df.iloc[i,0])
            
                new.iloc[t,0] = molecular/denominator
            
            open_p = self.recommend_stock.open
            col = np.where(open_p.columns == stock)[0][0]
            open_choose = open_p.iloc[self.start_row:self.end_row+1,col]
            
            high_p = self.recommend_stock.high
            high_choose = high_p.iloc[self.start_row:self.end_row+1,col]
            
            low_p = self.recommend_stock.low
            low_choose = low_p.iloc[self.start_row:self.end_row+1,col]
            
            close_p = self.recommend_stock.close
            close_choose = close_p.iloc[self.start_row:self.end_row+1,col]
            
            vol_p = self.recommend_stock.vol
            vol_choose = vol_p.iloc[self.start_row:self.end_row+1,col]
            
            date_chose = pd.Series(open_choose.index, index=open_choose.index, dtype='datetime64[ns]')
            
            new.index = date_chose.index
            
            t_df = pd.concat([pd.Series([],dtype = 'float64'),date_chose,open_choose,high_choose,low_choose,close_choose,vol_choose,new],axis = 1)
            
            t_df.columns = ['instrument','datetime','open','high','low','close','volume','new']
            
            t_df.iloc[:,0] = stock
            
            t_df.set_index(keys = ['instrument','datetime'],inplace=True)
            
            
            #2
            # chose = np.where(u_d.columns == stock)[0][0]
            # ups_downs_period = list(u_d.iloc[time_start_number+1:time_end_number+1,chose])
            # indicator_period = list(t_df.iloc[:,5])
    
            # test = np.corrcoef(ups_downs_period, indicator_period)[1][0]
            
            # all_r.append(test)
            # count.append(s)
            self.t_df_2 = pd.concat([self.t_df_2,t_df],axis = 0)
        
        self.t_df_2.to_pickle('./emb/'+self.train_season+'/'+str(self.top)+'_'+self.test_start+'_'+self.test_end+'.pkl', compression='infer', protocol=5, storage_options=None)