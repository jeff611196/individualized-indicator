#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:39:31 2024

@author: jeff
"""
from  copy import deepcopy
import numpy as np
import tejapi
from pandas import read_parquet
import pandas as pd
import gcsfs
fs = gcsfs.GCSFileSystem(project="dst-dev2021")

stock = pd.read_csv('TEJ資料/基金/stock.csv',index_col = 0)
taiwan = pd.read_csv('TEJ資料/基金/本國信託基金.csv')
gcs_read = 'gs://dst-tej/fund/twn/amm/raw-data/20230901-20230930.parquet'
fund_monthly = read_parquet(gcs_read)
fund_read = 'gs://dst-tej/fund/twn/aatt/raw-data/20230101-20231231.parquet'
fund_basic = read_parquet(fund_read)

try:
    fund_basic = fund_basic.drop(columns=['fld023'])
except KeyError as e:
    pass
    
def convert_index_to_datetime(fund_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    將 MultiIndex DataFrame 的日期欄位轉換為 datetime

    Args:
        fund_monthly(pd.DataFrame): MultiIndex DataFrame

    Returns:
        pd.DataFrame: 日期轉換後的 DataFrame
	"""
    fund_monthly_copy = deepcopy(fund_monthly)
    fund_monthly_copy['mdate'] = fund_monthly_copy['mdate'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return fund_monthly_copy

def chose_index_to_datetime(fund_monthly_datetime: pd.DataFrame) -> pd.DataFrame:
    """
    使用coid(基金代號)、mdata(時間)、key3(個股代號)

	Args:
        fund_monthly(pd.DataFrame): MultiIndex DataFrame

	Returns:
        pd.DataFrame: 挑選過 col 的 DataFrame
    """
    fund_monthly_use_col = fund_monthly_datetime.iloc[:,0:3]
    return fund_monthly_use_col

def chose_data_to_df(fund_monthly_use_col: pd.DataFrame, year: str) -> pd.DataFrame:
    """
    指定時間(ex:202203第一季詳細持股)

	Args:
        fund_monthly_use_col(pd.DataFrame): MultiIndex DataFrame

	Returns:
        pd.DataFrame: 指定時間的 DataFrame
    """
    row = np.where(fund_monthly_use_col.iloc[:,1].str.contains(year))[0]
    fund_monthly_specified_time = fund_monthly_use_col.iloc[row,:]
    return fund_monthly_specified_time

def select_taiwan_stock(fund_monthly_specified_time: pd.DataFrame) -> pd.DataFrame:
    """
    挑選4位數字的台灣股票，ex: M1100為產業指數
    需執行過convert_index_to_datetime、chose_index_to_datetime、chose_data_to_df
    
    Args:
        fund_monthly_specified_time(pd.DataFrame): MultiIndex DataFrame，做過
    convert_index_to_datetime、chose_index_to_datetime、chose_data_to_df處理

	Returns:
        pd.DataFrame: 挑選含有台灣股票的基金 DataFrame(不一定所有基金成份均為台灣股票)
    """
    use_fund_True = fund_monthly_specified_time['key3'].str.contains('^\d{4}$')
    taiwan_stock = fund_monthly_specified_time[use_fund_True]
    taiwan_stock = taiwan_stock.reset_index(drop = True)
    return taiwan_stock
    
def fund_taiwan_stock(taiwan_stock: pd.DataFrame) -> pd.DataFrame:
    """
    唯一化基金(已除去非台灣股票)，需執行過select_taiwan_stock-----非產出必要
    
    Args:
        taiwan_stock(pd.DataFrame): MultiIndex DataFrame，含有台灣股票的基金

	Returns:
        pd.DataFrame: 唯一化基金
    """
    fund_name = pd.DataFrame(np.unique(taiwan_stock.iloc[:,0]))
    return fund_name
    
def _fund_use_col(fund_basic: pd.DataFrame) -> pd.DataFrame:
    """
    挑選coid(基金代號)、mdate(時間)、fld004(中文名稱)、fld017_c(基金分類)
    
    Args:
        fund_basic(pd.DataFrame): MultiIndex DataFrame

	Returns:
        pd.DataFrame: 挑選過 col 的 DataFrame
    """
    selected_cols = ['coid', 'mdate', 'fld004', 'fld017_c']
    fund_basic = fund_basic[selected_cols]
    return fund_basic

def fund_classify(fund_basic: pd.DataFrame) -> pd.DataFrame:
    """
    所有基金類型-----非產出必要
    
    Args:
        fund_basic(pd.DataFrame): MultiIndex DataFrame

	Returns:
        array: 所有基金類型
    """
    classify = np.unique(_fund_use_col(fund_basic).iloc[:,3])
    return classify

def _select_classify(fund_basic: pd.DataFrame) -> pd.DataFrame:
    """
    挑選基金商品只有股票的類型
    
    Args:
        fund_basic(pd.DataFrame): MultiIndex DataFrame

	Returns:
        pd.DataFrame: 指定類型後的基金
    """
    indices = np.where(
    (_fund_use_col(fund_basic).iloc[:, 3] == '上櫃股票型') | 
    (_fund_use_col(fund_basic).iloc[:, 3] == '開放式一般型') | 
    (_fund_use_col(fund_basic).iloc[:, 3] == '開放式中小型') | 
    (_fund_use_col(fund_basic).iloc[:, 3] == '開放式科技類') | 
    (_fund_use_col(fund_basic).iloc[:, 3] == '開放式價值型')
    )[0]
    
    fund = _fund_use_col(fund_basic).iloc[indices,:]
    return fund

def fund_take_intersection(fund_basic: pd.DataFrame, taiwan_stock: pd.DataFrame) -> pd.DataFrame:
    """
    含有台灣股票標的基金、分好類型的基金取交集得到成分含有台灣股票標的且只有股票類型的基金，
    需執行過select_taiwan_stock
    
    Args:
        fund_basic(pd.DataFrame)、taiwan_stock(pd.DataFrame): MultiIndex DataFrame，
        基金基本資料、基金月持股篩選含台灣股票
        
	Returns:
        list: 交集的基金
    """
    fund_use_coid = pd.DataFrame(np.unique(_select_classify(fund_basic).iloc[:,0]))
    fund_use_classify = pd.DataFrame(np.unique(taiwan_stock.iloc[:,0]))
    
    #修改column name
    new_column_names = {old_col: str(old_col) for old_col in fund_use_coid.columns}    
    
    fund_use_coid.columns = fund_use_coid.columns.map(new_column_names)
    fund_use_classify.columns = fund_use_classify.columns.map(new_column_names)
    
    new_column_names = {'0': 'fund'}
    
    fund_use_coid = fund_use_coid.rename(columns=new_column_names)
    fund_use_classify = fund_use_classify.rename(columns=new_column_names)
    
    intersection = pd.merge(fund_use_coid, fund_use_classify, left_on='fund', right_on='fund', how='inner')
    intersection_list = intersection.values.flatten().tolist()
    return intersection_list

def intersection_in_select_taiwan_stock(taiwan_stock: pd.DataFrame, intersection_list: list) -> pd.DataFrame:
    """
    從含有台灣股票標的中挑出交集的基金
    
    Args:
        use_fund(pd.DataFrame): MultiIndex DataFrame，只有台灣股票的基金(需執行過select_taiwan_stock)、
        intersection_list(list): 交集的基金

	Returns:
        pd.DataFrame: 含有台灣股票且指定類型的基金持有標的
    """

    use_fund_intersection = taiwan_stock.iloc[np.where(taiwan_stock.iloc[:,0].isin(intersection_list))[0],:]

    use_fund_intersection = use_fund_intersection.reset_index(drop = True)
    return use_fund_intersection

def find_fund_hold_max(use_fund_intersection: pd.DataFrame) -> pd.Series:
    """
    列出所有基金的持有檔數
    
    Args:
        use_fund_intersection(pd.DataFrame): MultiIndex DataFrame，
        含有台灣股票且指定類型的基金持有標的

	Returns:
        pd.Series: 每隻基金最大持有檔數
    """
    counts = use_fund_intersection['coid'].value_counts()
    return counts

def make_fund_df(intersection_list: list, counts: pd.Series, taiwan_stock: pd.DataFrame) -> pd.DataFrame:
    """
    製作基金大表(基金及其所包含的所有持股)，需先執行fund_take_intersection、
    find_fund_hold_max、select_taiwan_stock
    
    Args:
        intersection_list(list): 從含有台灣股票標的且指定類型的條件中挑出交集的基金
        counts(pd.Series): 所有基金持股數
        taiwan_stock: 含有台灣股票的基金
	
    Returns:
        pd.DataFrame: 基金大表
    """
    fund_df = pd.DataFrame(np.nan, index=range(len(intersection_list)),\
              columns=[f'col_{col+1}' for col in range(np.max(counts))])
    fund_df.index = intersection_list
    
    for i in range(0,len(intersection_list)):#單隻基金的所有個股
        single_fund = taiwan_stock.iloc[np.where(taiwan_stock.iloc[:,0] ==\
                                           intersection_list[i])[0],2]
        fund_df.loc[intersection_list[i]] = single_fund.values.tolist() +\
            [None] * (len(fund_df.columns) - len(single_fund))
    
    return fund_df

def unique_fund(fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    只留下主基金(子基金持股相同)，須先執行make_fund_df
    
    Args:
        fund_df: 基金大表

	Returns:
        pd.DataFrame: 不重複的基金大表
    """
    new_index = [name[:-1] for name in fund_df.index]
    
    fund_df.index = new_index
    
    fund_df_unique_index = fund_df.drop_duplicates(keep='first')
    
    fund_df_unique_index = fund_df_unique_index.fillna('NA')
    return  fund_df_unique_index

def del_etf(fund_df_unique_index: pd.DataFrame, code: str) -> pd.DataFrame:
    """
    去除ETF，須先執行unique_fund-----非產出必要
    
    Args:
        fund_df_unique_index: 不重複的基金大表

	Returns:
        pd.DataFrame: 基金大表去除ETF(0050)
    """
    row = np.where(fund_df_unique_index == code)[0]
    row_ori = deepcopy(row)
    length = len(row)

    for i in range(0,length):
        row_specify = np.where(fund_df_unique_index == code)[0][0]
        specify_row = fund_df_unique_index.iloc[row_specify,:]
        specify_row[specify_row == code] = 'NA'
        specify_row = specify_row.replace('NA', np.nan)
        specify_row_drop_na = specify_row.dropna()
        na_count = len(specify_row) - len(specify_row_drop_na)
        specify_row_drop_na = pd.concat([specify_row_drop_na, pd.Series(['NA'] * na_count)])
        specify_row_drop_na.index = specify_row.index
        fund_df_unique_index.iloc[row_specify,:] = specify_row_drop_na

    return fund_df_unique_index

def fund_stock(fund_df_unique_index: pd.DataFrame) -> pd.DataFrame:
    """
    產生基金、股票兩個col的df，須先執行unique_fund or del_etf-----非產出必要
    
    Args:
        fund_df_unique_index: 不重複的基金大表

	Returns:
        pd.DataFrame: 基金對股票兩個col
    """
    rawdata_len = 0

    for i in range(0,fund_df_unique_index.shape[0]):
        rawdata_len += fund_df_unique_index.shape[1]-sum(fund_df_unique_index.iloc[i,:] == 'NA')

    fund_stock_node = pd.DataFrame(columns=['fund','stock'], index=range(0,rawdata_len))
    list_len = 0
    row_zero_len = fund_df_unique_index.shape[1]-sum(fund_df_unique_index.iloc[0,:] == 'NA')
    start = row_zero_len

    for i in range(0,fund_df_unique_index.shape[0]):
        start += list_len
        list_len = fund_df_unique_index.shape[1]-sum(fund_df_unique_index.iloc[i,:] == 'NA')
        for t in range(0,list_len):
            fund_stock_node.iloc[start+t-row_zero_len,0] = fund_df_unique_index.index[i]
            fund_stock_node.iloc[start+t-row_zero_len,1] = fund_df_unique_index.iloc[i,t]
    return fund_stock_node

def stock_stock(fund_df_unique_index: pd.DataFrame) -> pd.DataFrame:
    """
    產生股票、股票兩個col的df，須先執行unique_fund or del_etf
    
    Args:
        fund_df_unique_index: 不重複的基金大表

	Returns:
        pd.DataFrame: 股票對股票兩個col
    """
    stock_stock_node = pd.DataFrame()
    fund_include_stock = []        
            
    for i in range(0,fund_df_unique_index.shape[0]):
        not_NA = fund_df_unique_index.shape[1]-sum(fund_df_unique_index.iloc[i,:] == 'NA')
        fund_include_stock.append(not_NA)      

    for i in range(0,len(fund_include_stock)): 
        # sum_of_squares = sum(x**2 for x in fund_include_stock)
        sum_of_squares = fund_include_stock[i]**2
        
        data = {
            'col1': [np.nan] * sum_of_squares,
            'col2': [np.nan] * sum_of_squares
        }
        
        # 創建 DataFrame
        graph_data = pd.DataFrame(data)
        num = fund_include_stock[i]
        df = fund_df_unique_index.iloc[i,0:num]
        graph_data['col1'] = df.apply(lambda x: [x] * num).explode(ignore_index=True)
        repeated_values = np.tile(df, num)
        graph_data['col2'] = repeated_values
        stock_stock_node = pd.concat([stock_stock_node, graph_data])
        stock_stock_node = stock_stock_node.reset_index(drop = True)
    return stock_stock_node

def col_contrary(stock_stock_node: pd.DataFrame) -> pd.DataFrame:
    """
    相對的節點左右重複，須先執行stock_stock
    
    Args:
        stock_stock_node: 產生股票對股票兩個col的df

	Returns:
        pd.DataFrame: 相對的節點左右重複一次的df
    """
    stock_node_contrary = pd.concat([stock_stock_node.iloc[:,1],stock_stock_node.iloc[:,0]],axis = 1)
    stock_node_contrary.columns = ['col1','col2']
    stock_node_contrary = pd.concat([stock_stock_node,stock_node_contrary],axis = 0)
    stock_node_contrary = stock_node_contrary.reset_index(drop = True)
    return stock_node_contrary

def delete_col_equal(stock_node_contrary: pd.DataFrame) -> pd.DataFrame:
    """
    刪除col1、col2相同的row，須先執行col_contrary
    
    Args:
        stock_node_contrary: 相對的節點左右重複一次的df

	Returns:
        pd.DataFrame: 刪除col1、col2相同row的df
    """
    repeat = np.where(stock_node_contrary.iloc[:,0] == stock_node_contrary.iloc[:,1])[0]
    node_del_col_equal = deepcopy(stock_node_contrary)
    node_del_col_equal.drop(repeat, inplace=True)
    return node_del_col_equal

def input_made(node_del_col_equal: pd.DataFrame) -> pd.DataFrame:
    """
    製作成GNN input graph
    
    Args:
        node_del_col_equal: 刪除col1、col2相同row的df

	Returns:
        pd.DataFrame: GNN input graph
    """
    node_del_col_equal_unique = node_del_col_equal.drop_duplicates().copy()
    node_del_col_equal_unique = node_del_col_equal_unique.reset_index(drop = True)
    node_del_col_equal_unique['name'] = np.nan
    node_del_col_equal_unique.insert(0, 'main_name', [np.nan] * len(node_del_col_equal_unique))
    node_del_col_equal_unique['col1'] = node_del_col_equal_unique['col1'].astype(int)
    node_del_col_equal_unique['col2'] = node_del_col_equal_unique['col2'].astype(int)
    node_del_col_equal_unique = node_del_col_equal_unique.rename(columns={'col1': 'main_code', 'col2': 'code'})
    
    for i in range(0,len(node_del_col_equal_unique)):
        
        try:
            row = np.where(node_del_col_equal_unique.iloc[i,1] == stock.iloc[:,0])[0][0]
            node_del_col_equal_unique.iloc[i,0] = stock.iloc[row,1]

        except Exception as e:
            print("Error occurred:", e, i)

    for t in range(0,len(node_del_col_equal_unique)):
        row = np.where(node_del_col_equal_unique.iloc[t,2] == stock.iloc[:,0])[0][0]
        node_del_col_equal_unique.iloc[t,3] = stock.iloc[row,1]
    return node_del_col_equal_unique

def Graph_main(fund_monthly, fund_season, fund_basic):
    """
    基金大表生成主程式
    
    Args:
        fund_monthly
        fund_season
        fund_basic

	Returns:
        pd.DataFrame: 基金大表
    """
    fund_monthly_datetime = convert_index_to_datetime(fund_monthly)
    fund_monthly_use_col = chose_index_to_datetime(fund_monthly_datetime)
    fund_monthly_specified_time = chose_data_to_df(fund_monthly_use_col, fund_season)
    taiwan_stock = select_taiwan_stock(fund_monthly_specified_time)
    # fund_name = fund_taiwan_stock(taiwan_stock)
    # classify = fund_classify(fund_basic)
    intersection_list = fund_take_intersection(fund_basic,taiwan_stock)
    use_fund_intersection = intersection_in_select_taiwan_stock(taiwan_stock, intersection_list)
    counts = find_fund_hold_max(use_fund_intersection)
    fund_df = make_fund_df(intersection_list, counts, taiwan_stock)
    fund_df_unique_index = unique_fund(fund_df)
    fund_df_unique_index = del_etf(fund_df_unique_index,'0050')
    fund_df_unique_index = del_etf(fund_df_unique_index,'0052')
    fund_df_unique_index = del_etf(fund_df_unique_index,'0056')
    
    fund_season = fund_season.replace('-', '_')
    # 基金大表
    fund_df_unique_index.to_csv('./二原圖/fund/'+fund_season +'.csv')

    fund_folder_path_w = 'jeff-stock-wise/fund/'
    fund_path_w = fund_folder_path_w + fund_season +'.csv'  # 指定檔案名稱
    with fs.open(fund_path_w, 'w') as f:
        fund_df_unique_index.to_csv(f)

    # fund_stock_node = fund_stock(fund_df_unique_index)
    # stock_stock_node = stock_stock(fund_df_unique_index)
    # stock_node_contrary = col_contrary(stock_stock_node)
    # node_del_col_equal = delete_col_equal(stock_node_contrary)
    # input_graph = input_made(node_del_col_equal)


    # input_graph.to_csv('input graph_230601.csv', encoding='utf-8-sig')














if __name__ == "__main__":
    fund_monthly_datetime = convert_index_to_datetime(fund_monthly)
    fund_monthly_use_col = chose_index_to_datetime(fund_monthly_datetime)
    fund_monthly_specified_time = chose_data_to_df(fund_monthly_use_col, '2022-03-01')
    taiwan_stock = select_taiwan_stock(fund_monthly_specified_time)
    # fund_name = fund_taiwan_stock(taiwan_stock)
    # classify = fund_classify(fund_basic)
    intersection_list = fund_take_intersection(fund_basic,taiwan_stock)
    use_fund_intersection = intersection_in_select_taiwan_stock(taiwan_stock, intersection_list)
    counts = find_fund_hold_max(use_fund_intersection)
    fund_df = make_fund_df(intersection_list, counts, taiwan_stock)
    fund_df_unique_index = unique_fund(fund_df)
    fund_df_unique_index = del_etf(fund_df_unique_index,'0050')
    # fund_stock_node = fund_stock(fund_df_unique_index)
    stock_stock_node = stock_stock(fund_df_unique_index)
    stock_node_contrary = col_contrary(stock_stock_node)
    node_del_col_equal = delete_col_equal(stock_node_contrary)
    input_graph = input_made(node_del_col_equal)
    