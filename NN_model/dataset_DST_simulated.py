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
import gcsfs
fs = gcsfs.GCSFileSystem(project="dst-dev2021")
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
        # self.fund = pd.read_csv('二原圖/fund/'+train_season+'.csv',index_col = 0)
        fund_path_r = 'jeff-stock-wise/fund/'+train_season+'.csv'
        
        with fs.open(fund_path_r, 'r') as f:
            self.fund = pd.read_csv(f, index_col=0)
        
        stock_use = np.unique(self.fund)        
        self.stock = pd.DataFrame(stock_use[~np.isnan(stock_use)])
        self.DEFAULT_STOCKS = list(self.stock[0].astype(int).astype(str))
        self.stock['0'] = np.nan
        # standard = pd.read_csv('TEJ資料/基金/stock.csv',index_col = 0)
        standard_path_r = 'jeff-stock-wise/stock.csv'
        
        with fs.open(standard_path_r, 'r') as f:
            standard = pd.read_csv(f, index_col=0)
        
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
        
        quote_data.index = quote_data.index.set_levels(
            pd.to_datetime(quote_data.index.levels[1]),level=1)

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
        
        TSE.index = TSE.index.set_levels(
            pd.to_datetime(TSE.index.levels[1]),level=1)
        
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
        #embeddings = np.load('./emb/preprocessing/'+mode, allow_pickle='TRUE')[()]
        emb_folder_path_r = 'jeff-stock-wise/emb/'
        emb_path_r = emb_folder_path_r + mode
        
        with fs.open(emb_path_r, 'rb') as f:
            embeddings = np.load(f, allow_pickle=True)[()]

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
        
        print('ok')



    def New_indicator_IO(self, use_ind, use_stock, emb_path_list_train):# time need revise

        date_chose = pd.Series(self.open.index, index=self.open.index, dtype='datetime64[ns]').dt.date
        
        stock_col = np.where(self.close.columns == use_stock)[0][0]

        open_choose = self.open.iloc[:,stock_col]
        
        high_choose = self.high.iloc[:,stock_col]
        
        low_choose = self.low.iloc[:,stock_col]
        
        close_choose = self.close.iloc[:,stock_col]
        
        vol_choose = self.vol.iloc[:,stock_col]
        
        t_df = pd.concat([pd.Series([],dtype = 'float64'),date_chose,open_choose,high_choose,low_choose,close_choose,vol_choose],axis = 1)
        
        t_df.columns = ['instrument','datetime','open','high','low','close','volume']
        
        t_df.iloc[:,0] = use_stock

        t_df.set_index(keys = ['instrument','datetime'],inplace=True)
        
        t_df = t_df.fillna(method = 'ffill', axis = 0)
        
        if type(emb_path_list_train) == str:
            day_start = (t_df.index[0][0], Timestamp(self.test_start))
            day_start_time = day_start[1]
            day_start_value = day_start_time.date()
            day_start = (day_start[0], day_start_value)
            
            day_end = (t_df.index[0][0], Timestamp(self.test_end))
            day_end_time = day_end[1]
            day_end_value = day_end_time.date()
            day_end = (day_end[0], day_end_value)

        else:
            day_start = (t_df.index[0][0], Timestamp(self.day_start))
            day_start_time = day_start[1]
            day_start_value = day_start_time.date()
            day_start = (day_start[0], day_start_value)
            
            day_end = (t_df.index[0][0], Timestamp(self.day_end))
            day_end_time = day_end[1]
            day_end_value = day_end_time.date()
            day_end = (day_end[0], day_end_value)
        
        # day_start_deepcopy = copy.deepcopy(day_start)
        # day_end_deepcopy = copy.deepcopy(day_end)


        #產出技術指標表格
        # example = getattr(abstract, self.all_ta_label[0])(t_df)
        # d = pd.DataFrame(columns = use_ind, index = example.index)

        # day_start_deepcopy = np.where(d.index == day_start_deepcopy)[0][0]
        # day_end_deepcopy = np.where(d.index == day_end_deepcopy)[0][0]
        # d_deepcopy = copy.deepcopy(d)

        # d = d.iloc[day_start_deepcopy:day_end_deepcopy+1,:]
        use_ind_df = pd.DataFrame({use_ind_num:getattr(abstract, use_ind_num)(t_df) for use_ind_num in use_ind})
        day_start_chose = np.where(use_ind_df.index == day_start)[0][0]
        day_end_chose = np.where(use_ind_df.index == day_end)[0][0]+1       
        use_ind_df = use_ind_df.iloc[day_start_chose:day_end_chose,:]
        
        #標的隔日漲幅
        ups_downs = self.ups_downs
        
        ups_downs = ups_downs.fillna(method = 'ffill', axis = 0)
        
        stock_col = np.where(ups_downs.columns == use_stock)[0][0]

        time_start = np.where(ups_downs.index == self.day_start)[0][0]
        time_end = np.where(ups_downs.index == self.day_end)[0][0]
        
        ups_downs = ups_downs.iloc[time_start:time_end+1,stock_col]
        
        #標的embedding
        select_embedding = self.embeddings_concat

        select_embedding_row = np.where(pd.DataFrame(select_embedding.iloc[:,0]) == use_stock)[0][0]

        select_stock_emb = select_embedding.iloc[select_embedding_row,:]
        
        ups_downs = np.array(ups_downs)

        return use_ind_df, ups_downs, select_stock_emb, self.embeddings_concat