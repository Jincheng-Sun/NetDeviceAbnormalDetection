import pandas as pd
import numpy as np
import sklearn.preprocessing as pre

pd.set_option('display.max_columns', 50)
data = pd.read_parquet('../data/Europe_Network_data.parquet')
data = data.drop(['ID', 'TIME'], axis=1)
scaler = pre.MinMaxScaler()

# case one: Drop the fewer case, left only 11 device type (aka GROUPBYKEY) and 14 alarms
# Concatenate device type and PM values before encoding

'''drop data'''

dev_list = ['CHMON', 'STM64', 'OC192', 'STTP', 'STM4']
#
label_list = ['OOS-AU', 'OOS', 'IS-ANR', 'OOS-MA', 'OOS-AUMA', 'OOS-MAANR', 'n/a']
alarm = data['ALARM'].value_counts()

alm_list = alarm[(alarm > 50) & (alarm < 10000)].index.tolist()

'''drop the devices in dev_list'''
data = data.drop(data[data['GROUPBYKEY'].isin(dev_list)].index)

'''data normalization'''
data.iloc[:, 2:47] = scaler.fit_transform(data.iloc[:, 2:47])
'''fill na with 0'''
data = data.fillna(0)  # shape = (11533,47)

# -------------------------------------------------------------

'''Consider all the OOSs are abnormal cases'''
def in_out_service(data):
    if (data['ALARM'] == 0):
        if (data['LABEL'] == 'IS' or data['LABEL'] == 'n/a'):
            return 'Normal'
        else:
            return 'Malfunction'
    else:
        return data['ALARM']


data['ALARM'] = data.apply(in_out_service, axis=1)
data = data.drop(['LABEL'], axis=1)

# -------------------------------------------------------------

# '''one-hot the device type and concatenate with the original data'''
# data = pd.concat([pd.get_dummies(data['GROUPBYKEY']), data], axis=1)
# data = data.drop(['GROUPBYKEY'], axis=1)
#
# '''extract data with alarms in alm_list'''
# # data with alarms that occurs 50 times more in the dataset (OOS not included)
# real_alarms = data[data['ALARM'].isin(alm_list)]
# data = data.drop(real_alarms.index)
# real_alarm_labels = real_alarms['ALARM']
# real_alarms = real_alarms.drop(['ALARM'], axis=1)
#
# # OOS/IS-ANR data
# fake_normal = data[data['ALARM'] == 'Malfunction']
# data = data.drop(fake_normal.index)
# fake_normal_labels = fake_normal['ALARM']
# fake_normal = fake_normal.drop(['ALARM'], axis=1)
#
# # Normal data
# real_normal = data[data['ALARM'] == 'Normal']
# del data
# real_normal_labels = real_normal['ALARM']
# real_normal = real_normal.drop(['ALARM'], axis=1)
#
# file_path = 'preconcat'

# -------------------------------------------------------------
#
'''extract data with alarms in alm_list'''
# data with alarms that occurs 50 times more in the dataset (OOS not included)
real_alarms = data[data['ALARM'].isin(alm_list)]
data = data.drop(real_alarms.index)
real_alarm_labels = real_alarms[['ALARM', 'GROUPBYKEY']]
real_alarms = real_alarms.drop(['ALARM', 'GROUPBYKEY'], axis=1)

# OOS/IS-ANR data
fake_normal = data[data['ALARM'] == 'Malfunction']
data = data.drop(fake_normal.index)
fake_normal_labels = fake_normal[['ALARM', 'GROUPBYKEY']]
fake_normal = fake_normal.drop(['ALARM', 'GROUPBYKEY'], axis=1)

# Normal data
real_normal = data[data['ALARM'] == 'Normal']
del data
real_normal_labels = real_normal[['ALARM', 'GROUPBYKEY']]
real_normal = real_normal.drop(['ALARM', 'GROUPBYKEY'], axis=1)

file_path = 'aftconcat'

# -------------------------------------------------------------

'''save the scaler'''
np.save('../data/scaler', scaler)

'''save file'''
np.save('../data/%s/real_alarm_x.npy' % file_path, real_alarms)
np.save('../data/%s/real_alarm_y.npy' % file_path, real_alarm_labels)
np.save('../data/%s/fake_normal_x.npy' % file_path, fake_normal)
np.save('../data/%s/fake_normal_y.npy' % file_path, fake_normal_labels)
np.save('../data/%s/real_normal_x.npy' % file_path, real_normal)
np.save('../data/%s/real_normal_y.npy' % file_path, real_normal_labels)
