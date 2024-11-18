from pathlib import Path
import pandas as pd
import numpy as np
import json
import ssl
import glob
import os

# current_directory = os.path.dirname(os.path.abspath(__file__))
# os.chdir(current_directory)
os.chdir('/Users/jeff/Desktop/individualized-indicator')

ssl._create_default_https_context = ssl._create_unverified_context

PLUMBER_HOST = "https://dev-api.ddt-dst.cc/api/plumber/"
with open(f'{str(Path.home())}/.config/gcloud/application_default_credentials.json') as plumber_token:
    token = json.load(plumber_token)


total_data = pd.read_parquet(
    f"{PLUMBER_HOST}tej/fund/twn/anav",
    storage_options={
        "gcp-token": json.dumps(token),
        "start-date": "2022-09-01",
        "end-date": "2022-11-30"
    },
)


## fund = pd.read_csv('./二原圖/fund/2024_06_01.csv',index_col = 0)
folder_path = './二原圖/fund'

## 使用 glob 找到所有 .csv 檔案
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

files_sorted = sorted(csv_files, key=lambda x: (
    int(os.path.basename(x).split('_')[0]),  # 年份
    int(os.path.basename(x).split('_')[1].split('.')[0])  # 月份
))

df_list = [pd.read_csv(file,index_col = 0) for file in files_sorted[1:]]
## df_list = [pd.read_csv(file,index_col = 0) for file in files_sorted[1:-1]]


#變數月份
total_fund = list(df_list[1].index)


time = sorted(total_data.iloc[:,1].unique().tolist())

ori_df = pd.DataFrame(index=time, columns=total_fund)
ori_reward_df = pd.DataFrame(index=time, columns=total_fund)
ori_max_min_df = pd.DataFrame(index=time, columns=["max","min"])
date = pd.read_csv('/Users/jeff/Desktop/project/factor/fund_net_value/date_2022_09_01.csv',index_col = 0)
df = ori_df[~ori_df.index.isin(date.iloc[:,0])]
reward_df = ori_reward_df[~ori_reward_df.index.isin(date.iloc[:,0])]
max_min_df = ori_max_min_df[~ori_max_min_df.index.isin(date.iloc[:,0])]

# df = ori_df
# reward_df = ori_reward_df
# max_min_df = ori_max_min_df

for i in range(0,len(total_fund)):

    use_fund = total_fund[i]
    # pattern = '|'.join(use_fund)
    # indices = np.where(total_data['coid'].isin(use_fund))[0]
    # indices = np.where(total_data['coid'].str.contains(pattern))[0]
    row = np.where(total_data['coid'].str.contains(use_fund))[0]
    use_df = total_data.iloc[row,:]
    if len(np.unique(use_df['coid'])) > 1:
        chose_fund = np.unique(use_df['coid'])[0]
        row = np.where(total_data['coid'] == chose_fund)[0]
        use_df_2 = total_data.iloc[row,:]
    else:
        use_df_2 = use_df
    dataframe_sorted = use_df_2.sort_values(by='mdate') 
    if i == 5:
        dataframe_sorted_use = use_df_2.sort_values(by='mdate')
    source_values = dataframe_sorted['fld004'].values
    df[df.columns[i]] = list(source_values) + [np.nan] * (len(df) - len(source_values))

for k in range(0,len(total_fund)):
    for j in range(1,len(reward_df)):
        reward_df.iloc[j,k] = (df.iloc[j,k] - df.iloc[0,k])/df.iloc[0,k]

max_min_df['max'] = reward_df.max(axis=1)
max_min_df['min'] = reward_df.min(axis=1)
## max_min_df.iloc[0,:] = 0
# max_min_df.to_csv('/Users/jeff/Desktop/project/factor/fund_net_value/2024_06_01.csv')


#首輪
look_df = df.index
# look_dataframe_sorted = dataframe_sorted['mdate']
look_dataframe_sorted = dataframe_sorted_use['mdate']
unique_dates_in_df = pd.DataFrame(set(look_df) - set(look_dataframe_sorted))

# unique_dates_in_df.to_csv('/Users/jeff/Desktop/project/factor/fund_net_value/date_2024_06_01.csv')



#concat time
import os
import glob
import copy
import pandas as pd
import matplotlib.pyplot as plt

# 指定目錄路徑
folder_path = '/Users/jeff/Desktop/project/factor/fund_net_value'

## 使用 glob 找到所有 .csv 檔案
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

files_sorted = sorted(csv_files, key=lambda x: (
    int(os.path.basename(x).split('_')[0]),  # 年份
    int(os.path.basename(x).split('_')[1].split('.')[0])  # 月份
))

df_list = [pd.read_csv(file,index_col = 0) for file in files_sorted]

ori = copy.deepcopy(df_list[0])

for i in range(1,len(df_list)):

    use = copy.deepcopy(df_list[i])
    ori_num = ori.iloc[-1, :]
    use['max'] = use['max'].fillna(0) + ori_num[0]
    use['min'] = use['min'].fillna(0) + ori_num[1]
    ori = ori.drop(ori.index[-1])
    ori = pd.concat([ori, use], axis=0)



ori.iloc[102:520, 1] = ori.iloc[102:520, 1] + 0.5743353035794835

# plt.plot(ori.index, ori['max'], marker='o', color='blue', label='Max')
# plt.plot(ori.index, ori['min'], marker='o', color='red', label='Min')
plt.plot(range(len(ori)), ori['max'], marker='o', color='blue', label='Max')
plt.plot(range(len(ori)), ori['min'], marker='o', color='red', label='Min')

plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Max and Min Line Chart')
plt.legend()
plt.xticks(rotation=45)  # 旋轉 x 軸標籤
plt.grid(True)
plt.tight_layout()
plt.show()