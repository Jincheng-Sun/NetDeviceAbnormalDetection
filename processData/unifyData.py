import pandas as pd
import numpy as np
import sklearn.preprocessing as pre

'''load data'''
pd.set_option('display.max_columns', 50)
# data = pd.read_parquet('/Users/sunjincheng/Desktop/Ciena_EU_project/EU_ciena_pro/data/Tokyo_Network_Data_1Day.parquet')
data_eu = pd.read_parquet('../data/Europe_Network_data.parquet')
# data = data.drop(['ID', 'TIME'], axis=1)
# data_eu = data_eu.drop(['ID', 'TIME'], axis=1)
# data_eu['ALARM'] = data_eu['ALARM'].fillna('normal')
# data_eu.fillna(0)
'''data analyse, it might takes a long time'''

data = pd.read_parquet('../data/Europe_Network_Data_May13.parquet')
data = data.drop(['LASTOCCURRENCE'], axis=1)
new_idx = np.arange(data.index.shape[0],data.index.shape[0] + data_eu.index.shape[0] )
data_eu.index = new_idx

new_data = pd.concat([data, data_eu],axis=0)

# all = pd.concat([data_eu,data.drop(['LASTOCCURRENCE'],axis = 1)],axis = 0)
# '''replace labels'''
# # label dictionary
# rep_list = {
#     'CV-E':'E-CV',
#     'CV-PCS':_'PCS-CV',
#     'INFRAMESERR-E_INFRAMES-E_/': 'E-INFRAMESERR_E-INFRAMES_/',
#     'ES-PCS': 'PCS-ES',
#     'UAS-PCS': 'PCS-UAS',
#     'SPANLOSSMAX-OCH_SPANLOSSMIN-OCH_-': 'OCH-SPANLOSSMAX_OCH-SPANLOSSMIN_-',
#     'UAS-E': 'E-UAS',
#     'ES-E': 'S-ES',
#     'OUTFRAMESERR-E_OUTFRAMES-E_/': 'E-OUTFRAMESERR_E-OUTFRAMES_/',
#     'SPANLOSSAVG-OCH': 'OCH-SPANLOSSAVG'
# }
# # rename the columns
# data = data.rename(columns = rep_list)
#
# '''Unify the columns'''
# listA = data.columns.tolist()
# listB = data_eu.columns.tolist()
# common = list(set(listA).intersection(set(listB)))
# diff = list(set(listB).difference(set(listA)))
# emp_columns = pd.DataFrame(columns=diff)
# data = pd.concat([data,emp_columns],axis = 1)
# data = data[data_eu.columns]

'''Save file'''
# data.to_parquet('../data/Tokyo_Network_data.parquet')


'''statistic'''
#
# pd.set_option('display.max_rows', 500) #最大行数
# part = data_eu[['GROUPBYKEY','ALARM']]
# part = part.fillna('Normal')
# part['Num'] = 1
# group = part.groupby(['ALARM','GROUPBYKEY'])
# print(group.count())
#
# EU_device_list = data_eu['GROUPBYKEY'].value_counts().index.tolist()
# TK_device_list = data['GROUPBYKEY'].value_counts().index.tolist()
# dev_common = list(set(EU_device_list).intersection(set(TK_device_list)))
# dev_diff1 = list(set(EU_device_list).difference(set(dev_common)))
# dev_diff2 = list(set(TK_device_list).difference(set(dev_common)))
# ----------------------------------------------------------------------------------------
# analyse = pd.concat([pd.get_dummies(data_eu['GROUPBYKEY']),pd.get_dummies(data_eu['LABEL']),pd.get_dummies(data_eu['ALARM']), data_eu],axis=1).corr()
# # analyse = pd.concat([pd.get_dummies(data['GROUPBYKEY']), pd.get_dummies(data['LABEL']), data], axis=1).corr()
# # list = ['ETH10G', 'ETHN', 'ETTP']
# # list = ['OTM', 'OTM2', 'OTUTTP']
# # list = ['OC12', 'OC192', 'OC3', 'OC48']
# # list = ['CHMON', 'NMCMON']
# # others = ['AMP', 'OPTMON', 'OSC', 'PTP', 'STM16', 'STM64']
#
# list = ['ETH10G', 'ETH', 'ETTP', 'ETHN']
# list = ['OTM', 'OTM2', 'OTUTTP','PTP']
# list = ['STM4', 'STM64', 'STTP']
# others = ['AMP','CHMON']
#
# # a = data.loc[data['GROUPBYKEY'].isin(list)]
# # a = a.pivot_table(index=['GROUPBYKEY', 'LABEL'])
# a = data_eu.loc[data_eu['GROUPBYKEY'].isin(list)]
# a = a.pivot_table(index=['GROUPBYKEY', 'LABEL','ALARM'])