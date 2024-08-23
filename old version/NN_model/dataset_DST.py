#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:22:19 2022

@author: jeff
"""
import numpy as np
import pandas as pd
import glob
import talib
import copy
import torch
import warnings
import shutil
import talib
import json
import tidal as td
import ssl
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from pandas import Series
from dataset import *
from talib import abstract
from tqdm import tqdm, trange
from  copy import deepcopy
from pandas import Timestamp
from pathlib import Path
from datetime import datetime


class recommend_stock:
    
    def __init__(self, mode, ind_start, ind_end, start, end, t_start, t_end, train_season):

        self.ind_start = ind_start
        self.ind_end = ind_end
        self.day_start = start
        self.day_end = end
        self.test_start = t_start
        self.test_end = t_end
        self.fund = pd.read_csv('二原圖/fund/'+train_season+'.csv',index_col = 0)
        stock_use = np.unique(self.fund)        
        self.stock = pd.DataFrame(stock_use[~np.isnan(stock_use)])
        self.DEFAULT_STOCKS = list(self.stock[0].astype(int).astype(str))
        self.stock['0'] = np.nan
        standard = pd.read_csv('TEJ資料/基金/stock.csv',index_col = 0)
        
        for i in range(0,self.stock.shape[0]):
            self.stock.iloc[i,1] = standard.iloc[np.where(self.stock.iloc[i,0] == standard.iloc[:,0])[0][0],1]
        
        ssl._create_default_https_context = ssl._create_unverified_context

        PLUMBER_HOST = "https://dev-api.ddt-dst.cc/api/plumber/"
        with open(f'{str(Path.home())}/.config/gcloud/application_default_credentials.json') as plumber_token:
            token = json.load(plumber_token)

        quote_data = pd.read_parquet(
            f"{PLUMBER_HOST}stocks/tw/ohlcv",
            storage_options={
                "gcp-token": json.dumps(token),
                "start-date": self.ind_start,
                "end-date": self.ind_end,
                "tickers": ",".join([stock for stock in self.DEFAULT_STOCKS]),
            },
        )
        quote_data.index.set_levels(
            pd.to_datetime(quote_data.index.levels[1]),
            level=1,
            inplace=True,
        )
        quote_data.rename_axis(index={
            'ticker': 'instrument'
        }, inplace=True)
        self.quote_data = quote_data
        stock_df = quote_data.reset_index()
        datatime_unique = np.unique(stock_df.iloc[:,1])
        
        
        TSE = pd.read_parquet(
            f"{PLUMBER_HOST}stocks/tw/ohlcv",
            storage_options={
                "gcp-token": json.dumps(token),
                "start-date": self.ind_start,
                "end-date": self.ind_end,
                "tickers": ",".join([stock for stock in ['Y9999']]),
            },
        )
        TSE.index.set_levels(
            pd.to_datetime(TSE.index.levels[1]),
            level=1,
            inplace=True,
        )
        TSE.rename_axis(index={
            'ticker': 'instrument'
        }, inplace=True)
        
        TSE = TSE.reset_index()
        
        

        self.TSE = TSE
        
        if not os.path.exists('./price/'+train_season):
            
            os.mkdir('./price/'+train_season)
        
        TSE.to_csv('./price/'+train_season+'/TSE.csv')

        open_ = pd.DataFrame(index=datatime_unique, columns=self.DEFAULT_STOCKS)
        high_ = pd.DataFrame(index=datatime_unique, columns=self.DEFAULT_STOCKS)
        low_ = pd.DataFrame(index=datatime_unique, columns=self.DEFAULT_STOCKS)
        close_ = pd.DataFrame(index=datatime_unique, columns=self.DEFAULT_STOCKS)
        vol_ = pd.DataFrame(index=datatime_unique, columns=self.DEFAULT_STOCKS)
        
        for t in range(0,len(datatime_unique)):
            datatime_row = np.where(stock_df.iloc[:,1] == datatime_unique[t])[0]
            date_stock = stock_df.iloc[datatime_row,:]

            for i_0 in range(date_stock.shape[0]):
                    col_open = np.where(date_stock['instrument'].iloc[i_0] == open_.columns)[0][0]
                    open_.iloc[t,col_open] = date_stock['open'].iloc[i_0]
            for i_1 in range(date_stock.shape[0]):
                    col_high = np.where(date_stock['instrument'].iloc[i_1] == high_.columns)[0][0]
                    high_.iloc[t,col_high] = date_stock['high'].iloc[i_1]
            for i_2 in range(date_stock.shape[0]):
                    col_low = np.where(date_stock['instrument'].iloc[i_2] == open_.columns)[0][0]
                    low_.iloc[t,col_low] = date_stock['low'].iloc[i_2]
            for i_3 in range(date_stock.shape[0]):
                    col_close = np.where(date_stock['instrument'].iloc[i_3] == open_.columns)[0][0]
                    close_.iloc[t,col_close] = date_stock['close'].iloc[i_3]
            for i_4 in range(date_stock.shape[0]):
                    col_vol = np.where(date_stock['instrument'].iloc[i_4] == open_.columns)[0][0]
                    vol_.iloc[t,col_vol] = date_stock['volume'].iloc[i_4]

        open_filled = open_.fillna('NA')
        high_filled = high_.fillna('NA')
        low_filled = low_.fillna('NA')
        colse_filled = close_.fillna('NA')
        vol_filled = vol_.fillna('NA')
        
        open_.index = open_.index.strftime('%Y-%m-%d')
        high_.index = high_.index.strftime('%Y-%m-%d')
        low_.index = low_.index.strftime('%Y-%m-%d')
        close_.index = close_.index.strftime('%Y-%m-%d')
        vol_.index = vol_.index.strftime('%Y-%m-%d')

        self.open = open_
        self.high = high_
        self.low = low_
        self.close = close_
        self.high = high_
        
        open_.to_csv('./price/'+train_season+'/open.csv')
        high_.to_csv('./price/'+train_season+'/high.csv')
        low_.to_csv('./price/'+train_season+'/low.csv')
        close_.to_csv('./price/'+train_season+'/close.csv')
        vol_.to_csv('./price/'+train_season+'/vol.csv')



        self.TSE = pd.read_csv('./price/'+train_season+'/TSE.csv',index_col = 0)
        self.TSE.set_index(keys = ['instrument','datetime'],inplace=True)
        self.open = pd.read_csv('./price/'+train_season+'/open.csv',index_col = 0)
        self.high = pd.read_csv('./price/'+train_season+'/high.csv',index_col = 0)
        self.low = pd.read_csv('./price/'+train_season+'/low.csv',index_col = 0)
        self.close = pd.read_csv('./price/'+train_season+'/close.csv',index_col = 0)
        self.vol = pd.read_csv('./price/'+train_season+'/vol.csv',index_col = 0)
        
        ups_downs = pd.DataFrame(index=datatime_unique, columns=self.DEFAULT_STOCKS)
        
        for t in range(0,ups_downs.shape[1]):
            for i in range(0,ups_downs.shape[0]-1):
                ups_downs.iloc[i+1,t] = self.close.iloc[i+1,t]/self.close.iloc[i,t]
        
        self.ups_downs = ups_downs        
        self.all_ta_label = talib.get_functions()

        #本機
        #embedding(128維)
        embeddings = np.load('./emb/preprocessing/'+mode, allow_pickle='TRUE')[()]
        self.embeddings = np.array([embeddings.get(i,np.zeros(128,dtype =np.float32)) for i in embeddings.keys()],dtype='float32')
        #Dict照順序的個股編號
        self.keys = list(embeddings.keys())
        #430隻個股embedding(128維)
        embeddings_df = pd.DataFrame(self.embeddings.tolist())
        #個股排序        
        keys_reshape = np.array(self.keys).reshape(embeddings_df.shape[0],1)
        keys_df = pd.DataFrame(keys_reshape.tolist())
        keys_df.columns = list('s')
        #df加上個股編號
        embeddings_concat = pd.concat([keys_df,embeddings_df],axis=1)
        
        #對個股編號排序
        for i in range(0,len(embeddings_concat)):
            embeddings_concat.iloc[i,0] = embeddings_concat.iloc[i,0][1:]

        for t in range(0,len(embeddings_concat)):
            embeddings_concat.iloc[t,0] = int(embeddings_concat.iloc[t,0])
        
        self.embeddings_concat = embeddings_concat.sort_values(by=['s'])
        
        #將個股編號填入
        for s in range(0,len(embeddings_concat)):
        
            self.embeddings_concat.iloc[s,0] = self.stock.iloc[s,0]
         
        for h in range(0,len(self.embeddings_concat)):
        
            self.embeddings_concat.iloc[h,0] = str(int(self.embeddings_concat.iloc[h,0]))

        #lab
        #self.embeddings = np.load('/home/jeffhsu@cathayholdings.com.tw/stock_fund/'+mode, allow_pickle='TRUE')[()]
        # 尋找餘弦相似

    # def IO(self,indicator,s):
        
    #     #隔日收益
    #     ups_downs = self.ups_downs

    #     # r_list = []
        
    #     #時間設定
    #     t = pd.DataFrame(indicator.index)
    #     self.time_start_number = np.where( t == self.day_start)[0][0]
    #     self.time_end_number = np.where( t == self.day_end)[0][0]

    #     #指定個股在技術指標、隔日收益表中的col
    #     self.chose = np.where(indicator.columns == str(self.embeddings_concat.iloc[s,0]))[0][0]
    #     #將指定個股技術指標、隔日收益的指定日期值取出
    #     indicator_period = list(indicator.iloc[self.time_start_number:self.time_end_number,self.chose])
    #     # indicator_period = (np.array(indicator_period)-min(indicator_period)).tolist()
    #     # indicator_period = (np.array(indicator_period)/max(indicator_period)).tolist()
                
    #     ups_downs_period = list(ups_downs.iloc[self.time_start_number+1:self.time_end_number+1,self.chose])
                
    #     #轉array
    #     self.indicator_period_array = np.array(indicator_period)
    #     self.ups_downs_period_array = np.array(ups_downs_period)
        
    #     #indicator_period_tensor = indicator_period_tensor.reshape(105,1)

    #     #選取指定個股的embedding
    #     select_embedding = self.embeddings_concat.iloc[s,1:len(self.embeddings_concat.columns)]


    #     return self.indicator_period_array, self.ups_downs_period_array, select_embedding

    def New_indicator(self, name):
    
        stock_corr = pd.DataFrame(columns = self.all_ta_label,index = self.DEFAULT_STOCKS) 
        chose_stock = []
        
        for s in tqdm(range(0,len(self.DEFAULT_STOCKS))):
                
            stock  = self.DEFAULT_STOCKS[s]
    
            try:
                col = np.where(self.close.columns == stock)[0][0]
                chose_stock.append(s)
            except:
                pass
            
            else:
                open_p = self.open
                open_choose = open_p.iloc[:,col]
                
                high_p = self.high
                high_choose = high_p.iloc[:,col]
                
                low_p = self.low
                low_choose = low_p.iloc[:,col]
                
                close_p = self.close
                close_choose = close_p.iloc[:,col]
                
                vol_p = self.vol
                vol_choose = vol_p.iloc[:,col]
                
                date_chose = pd.Series(open_choose.index, index=open_choose.index, dtype='datetime64[ns]')
                
                t_df = pd.concat([pd.Series([],dtype = 'float64'),date_chose,open_choose,high_choose,low_choose,close_choose,vol_choose],axis = 1)
                
                t_df.columns = ['instrument','datetime','open','high','low','close','volume']
                
                t_df.iloc[:,0] = stock
                
                t_df_deepcopy = copy.deepcopy(t_df)
                
                t_df.set_index(keys = ['instrument','datetime'],inplace=True)
    
                day_start = (t_df.index[0][0], Timestamp(self.day_start))
                day_end = (t_df.index[0][0], Timestamp(self.day_end))
    
                day_start_deepcopy = copy.deepcopy(day_start)
                day_end_deepcopy = copy.deepcopy(day_end)
    
                #abstract => talib計算技術指標API
                example = getattr(abstract, self.all_ta_label[0])(t_df)
                d = pd.DataFrame(columns = self.all_ta_label,index = example.index) 
    
                day_start_row = np.where(d.index == day_start_deepcopy)[0][0]
                day_end_row = np.where(d.index == day_end_deepcopy)[0][0]+1
    
                d_deepcopy = copy.deepcopy(d)
    
                d = d.iloc[day_start_row:day_end_row,:]
    
    
                for z in range(0,len(self.all_ta_label)):
    
                    try:        
                        d_1 = pd.DataFrame({self.all_ta_label[z]:getattr(abstract, self.all_ta_label[z])(t_df)})
                        day_start_chose = np.where(d_1.index == day_start)[0][0]
                        day_end_chose = np.where(d_1.index == day_end)[0][0]+1        
                        d_1 = d_1.iloc[day_start_chose:day_end_chose,:]
                        d_1_colname = d_1.columns[0]
                        d.iloc[:,np.where(d.columns == d_1_colname)[0][0]] = d_1
                        
                    except:
                        pass
    
    
                u_p = self.ups_downs
    
                stock_chose = u_p.iloc[:,np.where(u_p.columns == d.index[0][0])[0][0]]
    
                e = []

                warnings.filterwarnings('ignore', category=RuntimeWarning)
    
                for zz in range(0,len(d.columns)):
    
                    try:
                        e_1 = d.iloc[:,zz]
                        e_1 = e_1 - np.nanmin(e_1)
                        e_1 = e_1/np.nanmax(e_1)
                        e_1 = pd.DataFrame(e_1)
                        e_2 = pd.DataFrame(stock_chose)
                        e_2 = e_2.iloc[np.where(e_2.index == self.day_start)[0][0]:np.where(e_2.index == self.day_end)[0][0]+1,:]  
                        e_2.index = e_1.index
                        e_3 = pd.concat([e_1,e_2],axis=1)
                        e_3 = e_3.apply(pd.to_numeric, errors='coerce')
                        stock_corr.iloc[s,zz] = e_3.corr().iloc[0,1]
                    except:
                        pass
                    
                warnings.filterwarnings('default', category=RuntimeWarning)
                    
        stock_corr_len = []            
    
        for i in range(0,len(stock_corr.columns)):
                        
            test = np.isnan(list(stock_corr.iloc[:,i]))   
            T_F_number_row = np.sum(test!=0)
            stock_corr_len.append(T_F_number_row)
            
        #na值超過5成drop    
        stock_drop = list(np.where(Series(stock_corr_len) > 0.5*stock_corr.shape[0])[0])
    
        stock_drop_name = stock_corr.columns[stock_drop]
    
        #去除NA值的所有個股與技術指標與隔日收益的相關係數表
    
        test_1 = stock_corr.drop(stock_drop_name, axis=1)

        test_1.to_csv('./NN_model/dataset/stock_ind_del_na/'+name+'.csv')

        corr_ave = pd.DataFrame(columns = test_1.columns,index = ['ave']) 
    
        # 平均
    
        for iii in range(0,len(test_1.columns)):
            mean = np.nanmean(test_1.iloc[:,iii])
            corr_ave.iloc[0,iii] = mean
    
        # corr_ave.to_csv('/Users/jeff/Desktop/corr_ave.csv')
    
        corr_ave_2 = corr_ave.T
    
        self.test_2 = abs(corr_ave_2).sort_values(by=['ave'],na_position ='first')
    
        self.test_2.iloc[:,0].fillna(value=0, inplace=True)
    
        self.test_2 = self.test_2.sort_values(by = ['ave'],ascending = False)
    
        self.test_2.to_csv('./NN_model/dataset/top_indicator/'+name+'.csv')
    
        # d:個股技術指標表，test_1:所有個股指標與隔日收益相關係數表
    
        return d, test_1


    def tech_indicator(self):
        
        taiex = self.TSE

        all_ta_label = talib.get_functions()

        example = getattr(abstract, all_ta_label[0])(taiex)

        indicator_df = pd.DataFrame(columns = all_ta_label,index = example.index) 

#some indicator col is not only one

        for z in range(0,158):
            
            try:
                d_1 = pd.DataFrame({all_ta_label[z]:getattr(abstract, all_ta_label[z])(taiex)})
                d_1_colname = d_1.columns[0]
                indicator_df.iloc[:,np.where(indicator_df.columns == d_1_colname)[0][0]] = d_1
            except:
                pass
            
        df = indicator_df.reset_index()
        
        # df.iloc[:,1] = df['datetime'].dt.date.astype(str)
        
        df = df.iloc[np.where(df.iloc[:,1] == self.day_start)[0][0]:np.where(df.iloc[:,1] == self.day_end)[0][0]+1,:]
        
        df.set_index(['instrument', 'datetime'], inplace=True)
        
        return df


    def New_indicator_IO(self,s,indicator_top_list,emb_path_list_train):# time need revise

        # 挑選使用指標
        use_indicator = pd.DataFrame(indicator_top_list)
        # 挑選的標的
        chose_stock = []

        if type(emb_path_list_train) == str:
            title_2 = self.emb_path_list[s]
            self.stock_chose = title_2.split('/')[3][:-4]
            
        else:
            self.stock_chose = emb_path_list_train[s]

        try:
            col = np.where(self.close.columns == self.stock_chose)[0][0]
            chose_stock.append(s)
            #標的是否存在、chose_stock存起來
        except:
            d = 'error'
        
        else:
            #產出開高低收表格
            open_p = self.open
            open_choose = open_p.iloc[:,col]
            
            high_p = self.high
            high_choose = high_p.iloc[:,col]
            
            low_p = self.low
            low_choose = low_p.iloc[:,col]
            
            close_p = self.close
            close_choose = close_p.iloc[:,col]
            
            vol_p = self.vol
            vol_choose = vol_p.iloc[:,col]
            
            date_chose = pd.Series(open_choose.index, index=open_choose.index, dtype='datetime64[ns]').dt.date
            
            t_df = pd.concat([pd.Series([],dtype = 'float64'),date_chose,open_choose,high_choose,low_choose,close_choose,vol_choose],axis = 1)
            
            t_df.columns = ['instrument','datetime','open','high','low','close','volume']
            
            t_df.iloc[:,0] = self.stock_chose
            
            t_df_deepcopy = copy.deepcopy(t_df)

            t_df.set_index(keys = ['instrument','datetime'],inplace=True)
            
            if type(emb_path_list_train) == str:
                day_start = (t_df.index[0][0], Timestamp(self.test_start))
                day_end = (t_df.index[0][0], Timestamp(self.test_end))
            
            else:
                day_start = (t_df.index[0][0], Timestamp(self.day_start))
                day_end = (t_df.index[0][0], Timestamp(self.day_end))
            
            day_start_deepcopy = copy.deepcopy(day_start)
            day_end_deepcopy = copy.deepcopy(day_end)


            #產出技術指標表格
            example = getattr(abstract, self.all_ta_label[0])(t_df)
            d = pd.DataFrame(columns = list(use_indicator.iloc[:,0]),index = example.index) 

            day_start_deepcopy = np.where(d.index == day_start_deepcopy)[0][0]
            day_end_deepcopy = np.where(d.index == day_end_deepcopy)[0][0]

            d_deepcopy = copy.deepcopy(d)

            d = d.iloc[day_start_deepcopy:day_end_deepcopy+1,:]
            
            for z in range(0,len(use_indicator.iloc[:,0])):

                try:        
                    d_1 = pd.DataFrame({use_indicator.iloc[:,0][z]:getattr(abstract, use_indicator.iloc[:,0][z])(t_df)})
                    day_start_chose = np.where(d_1.index == day_start)[0][0]
                    day_end_chose = np.where(d_1.index == day_end)[0][0]+1       
                    d_1 = d_1.iloc[day_start_chose:day_end_chose,:]
                    d_1_colname = d_1.columns[0]
                    d.iloc[:,np.where(d.columns == d_1_colname)[0][0]] = d_1
                    
                except:
                    pass

        #標的隔日漲幅
        ups_downs = self.ups_downs
        
        ups_downs_copy = copy.deepcopy(ups_downs)
        
        stock_col = np.where(ups_downs.columns == self.stock_chose)[0][0]

        time_start = np.where(ups_downs.index == self.day_start)[0][0]
        time_end = np.where(ups_downs.index == self.day_end)[0][0]
        
        ups_downs = ups_downs.iloc[time_start:time_end+1,stock_col]
        
        #標的embedding
        select_embedding = self.embeddings_concat

        select_embedding_row = np.where(pd.DataFrame(select_embedding.iloc[:,0]) == self.stock_chose)[0][0]

        select_stock_emb = select_embedding.iloc[select_embedding_row,:]
        
        ups_downs = np.array(ups_downs)

        return d, ups_downs, select_stock_emb


    def dataset_new_indicator(self,new_indicator,stock,top_k):

        new_indicator_copy = copy.deepcopy(new_indicator)

        emb = new_indicator_copy[2]

        del emb['s']

        emb = list(emb)

        test = new_indicator_copy[0]

        time_start_number = np.where(test.index == (test.index[0][0], Timestamp(self.day_start)))[0][0]

        time_end_number = np.where(test.index == (test.index[0][0], Timestamp(self.day_end)))[0][0]

        input_embedding = np.array(emb*(time_end_number - time_start_number+1)).reshape(time_end_number - time_start_number+1,128)        

        # self.input_embedding = torch.from_numpy(input_embedding).to(torch.float32)
       
        self.input_embedding = pd.DataFrame(input_embedding.reshape(test.shape[0],128))
       
        targets = new_indicator_copy[1]

        #歸一化
        minmax_table = pd.DataFrame(columns = ['stock','minmax']+list(new_indicator[0].columns),index=[0,1])
        
        minmax_table.iloc[:,0] = stock
        minmax_table.iloc[0,1] = 'min'
        minmax_table.iloc[1,1] = 'max'
        
        minmax_table.set_index(keys = ['stock','minmax'],inplace=True)
        
        indicator = new_indicator_copy[0].iloc[:,0].values 
        minmax_table.iloc[0,0] = np.min(indicator)
        indicator = indicator - np.min(indicator)
        minmax_table.iloc[1,0] = np.max(indicator)
        indicator = indicator/np.max(indicator)
      
        for i in range(1,len(new_indicator_copy[0].columns)):
        
            indicator_1 = new_indicator_copy[0].iloc[:,i].values
            minmax_table.iloc[0,i] = np.min(indicator_1)          
            indicator_1 = indicator_1 - np.min(indicator_1)
            minmax_table.iloc[1,i] = np.max(indicator_1)
            indicator_1 = indicator_1/np.max(indicator_1)
            warnings.filterwarnings('ignore')
            
            if np.isnan(indicator_1).any(axis = 0):
                print("\n")
                print("stock",stock,"num",i,"nan",sum(np.isnan(indicator_1)),sep=",")

                index = [i for i, x in enumerate(indicator_1) if np.isnan(x)]
                indicator_1[index] = 0

                return "ind_can't_softmax"

            indicator = np.vstack((indicator,indicator_1))
        
        indicator = pd.DataFrame(indicator.T)
        
        indicator_dataset = Indicator_Dataset(indicator, targets, self.input_embedding)
  
        minmax_table.to_csv(self.file_3+'/'+stock+'.csv')
        
        return indicator_dataset