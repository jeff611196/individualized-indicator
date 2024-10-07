# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Feb 21 11:44:36 2023

# @author: jeff
# """
import os
# # 获取当前文件的绝对路径
# current_file_path_ori = os.path.abspath(__file__)
# # 获取当前文件所在的目录
# current_directory_ori = os.path.dirname(current_file_path_ori)
# # 切换工作目录到当前文件所在的目录
# os.chdir(current_directory_ori)
os.chdir('/Users/jeff/Desktop/individualized-indicator')

import yaml
import glob
import pandas as pd
from  copy import deepcopy
from NN_model.dataset_DST import *
from 標準化週期.spec import *
from indicator_coefficient import *
from backtest.backtest_pkl import *
from 二原圖.fund_module import *

ind_start = '2021-11-01'
ind_end = '2024-09-30'
day_start = '2024-01-02'
day_end = '2024-05-31'
test_start = '2024-06-03'
test_end = '2024-09-30'
train_season = '2024_03_01'
emb_length = 'embeddings_length200_2024_03_01.npy'
parameter = "emb/2024_03_01/top_8/*.csv"

new_date_str = train_season.replace('_', '-')
pre_processing = Graph_main(fund_monthly, new_date_str, fund_basic)

fund = pd.read_csv('./二原圖/fund/2024_03_01.csv',index_col = 0)

from 二原圖.emb_gengrate import *
emb_birth = Emb_gengrate(fund)

# os.chdir('/Users/jeff/Desktop/individualized-indicator')

# os.chdir(current_directory_ori)

# stock embedding
etl = recommend_stock(emb_length, ind_start, ind_end, day_start, day_end, test_start, test_end, train_season)

# 計算完的indicator係數
TOP_k = 8
corr = 0.3
window_size = 21
stride = 5

etl.emb_path_list = glob.glob(parameter)

# # 回測
# input_backtest_table = Input_backtest_table(etl,'top_'+str(TOP_k), test_start, test_end, train_season)
# input_backtest_table_calculate = input_backtest_table.calculate()

# save top_indicator(技術指標的相關係數依高低排序)
stock_indicator_corr = etl.New_indicator(train_season)

# 大盤技術指標
indicator_table_ori = etl.tech_indicator()
indicator_table = copy.deepcopy(indicator_table_ori)

# Load top_indicator
strategy = Top_k(train_season)

# max corr with stock price fluctuations
indicator_top = strategy.read_indicator.index[0]

# list related to stock price fluctuations
use_indicator_ori = strategy.read_indicator

# clear NA
drop_nan_index, indicator_corr, indicator_table = strategy.clean_na_corr(indicator_top,indicator_table,use_indicator_ori)

indicator_top_list = [indicator_top]

# indicator_top_list
for i in range(0,TOP_k-1):
   
    indicator_top_one, indicator_table_reduce, chose_indicator = strategy.indicator_info(corr)

    drop_nan_index, indicator_corr, indicator_table = strategy.clean_na_corr(indicator_top_one, indicator_table_reduce, chose_indicator)

    indicator_top_list.append(indicator_top_one)
    

if not os.path.exists('./emb/'+train_season):
    
    os.mkdir('./emb/'+train_season)
    
pd.DataFrame(indicator_top_list).to_csv('./emb/'+train_season+'/indicator_chose.csv')

unique_list = list(set(indicator_top_list))

TOP_k = len(unique_list)

chose_col = []    
    
for t in range(0,len(indicator_top_list)):

    col = np.where(indicator_table_ori.columns == indicator_top_list[t])[0][0]

    chose_col.append(col)


# Model train
indicator_coefficient = Indicator_coefficient(etl, indicator_top_list)

indicator_coefficient_calculate = indicator_coefficient.calculate(TOP_k, train_season, window_size, stride)

# etl.emb_path_list = glob.glob(parameter)

print('回測')
# 回測
input_backtest_table = Input_backtest_table(etl,'top_'+str(TOP_k), test_start, test_end, train_season)
input_backtest_table_calculate = input_backtest_table.calculate()