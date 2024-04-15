#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:37:09 2024

@author: jeff
"""
import os
from fund_module import gcs_read
from fund_module import fund_monthly
from fund_module import fund_read
from fund_module import fund_basic
from fund_module import convert_index_to_datetime
from fund_module import chose_index_to_datetime
from fund_module import chose_data_to_df
from fund_module import select_taiwan_stock
from fund_module import fund_taiwan_stock
from fund_module import _fund_use_col
from fund_module import fund_classify
from fund_module import _select_classify
from fund_module import fund_take_intersection
from fund_module import intersection_to_list
from fund_module import intersection_in_select_taiwan_stock
from fund_module import find_fund_hold_max
from fund_module import make_fund_df
from fund_module import unique_fund
from fund_module import del_etf
from fund_module import fund_stock
from fund_module import stock_stock
from fund_module import col_contrary
from fund_module import delete_col_equal
from fund_module import input_made



fund_monthly_datetime = convert_index_to_datetime(fund_monthly)
fund_monthly_use_col = chose_index_to_datetime(fund_monthly_datetime)
fund_monthly_specified_time = chose_data_to_df(fund_monthly_use_col, '2022-03-01')
taiwan_stock = select_taiwan_stock(fund_monthly_specified_time)
fund_name = fund_taiwan_stock(taiwan_stock)
classify = fund_classify(fund_basic)
intersection = fund_take_intersection(fund_basic,fund_monthly_specified_time)
intersection_list = intersection_to_list(intersection)
use_fund_intersection = intersection_in_select_taiwan_stock(taiwan_stock, intersection_list)
max_col = find_fund_hold_max(use_fund_intersection)
fund_df = make_fund_df(intersection_list, max_col, taiwan_stock)
fund_df_unique_index = unique_fund(fund_df)
fund_df_unique_index = del_etf(fund_df_unique_index,'0050')
fund_stock_node = fund_stock(fund_df_unique_index)
stock_stock_node = stock_stock(fund_df_unique_index)
stock_node_contrary = col_contrary(stock_stock_node)
node_del_col_equal = delete_col_equal(stock_node_contrary)
input_graph = input_made(node_del_col_equal)