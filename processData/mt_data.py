import pandas as pd

pd.set_option('display.max_columns', 50)

import numpy as np
import sklearn.preprocessing as pre
from sklearn.externals import joblib

scaler_path = 'Minmax_scaler'

# 'Europe' or 'Tokyo'
data_apostrophe = 'Europe'
data_path = '../data/%s_Network_data.parquet' % data_apostrophe

data_path2 = '../data/Tokyo_Network_data.parquet'
data_eu = pd.read_parquet(data_path)
try:
    data_eu = data_eu.drop(['ID', 'TIME'], axis=1)
except:
    pass

data_tk = pd.read_parquet(data_path2)

'''drop data'''
alarm = data_eu['ALARM'].value_counts()
# alm_list = alarm[(alarm > 50) & (alarm < 10000)].index.tolist()
alm_list = alarm[alarm <= 50].index.tolist()

data_eu = data_eu.drop(data_eu[data_eu['ALARM'].isin(alm_list)].index)


def drop_data(data):
    dev_list = ['CHMON', 'STM64', 'OC192', 'STTP', 'STM4', 'STM16', 'NMCMON', 'OC48', 'OC12', 'OC3']

    '''drop the devices in dev_list'''
    data = data.drop(data[data['GROUPBYKEY'].isin(dev_list)].index)

    '''fill na with 0'''
    data = data.fillna(0)

    '''data normalization'''
    try:
        scaler = joblib.load('../models/%s' % scaler_path)
        data.iloc[:, 2:47] = scaler.transform(data.iloc[:, 2:47])

    except:
        # should choose the right scaler
        scaler = pre.MinMaxScaler()
        data.iloc[:, 2:47] = scaler.fit_transform(data.iloc[:, 2:47])
        joblib.dump(scaler, '../models/%s' % scaler_path)

    return data


data_eu = drop_data(data_eu)
data_tk = drop_data(data_tk)

'''Consider all the OOSs are abnormal cases'''


def in_out_service(data):
    if (data['ALARM'] == 0):
        '''OOS does not have to be malfunction, it could be simply not in use'''
        # although the IS-ANR has bigger correlation with not normal
        if (data['LABEL'] == 'IS' or data['LABEL'] == 'n/a'):
            return -1
        else:
            return 'Notsure'


    else:
        return data['ALARM']


data_eu['ALARM'] = data_eu.apply(in_out_service, axis=1)
data_eu = data_eu.drop(['LABEL'], axis=1)
data_eu = data_eu.drop(data_eu[data_eu['ALARM'] == 'Notsure'].index)
data_tk = data_tk.drop(['LABEL'], axis=1)
data_tk['ALARM'] = -1

data = pd.concat([data_eu, data_tk], axis=0)

labeled = data[data['ALARM'] != -1]
unlabeled = data[data['ALARM'] == -1]

'''encode the data'''
from keras.models import load_model

encoder_name = 'enlarge_s'
encoder = load_model('../models/encoder_%s' % encoder_name)

lbEncoder = joblib.load('../models/LE_GBK')
ohEncoder = joblib.load('../models/OH_GBK')
lbEncoder2 = joblib.load('../models/LE_ALM')
ohEncoder2 = joblib.load('../models/OH_ALM')

alarm = lbEncoder2.transform(np.reshape(labeled['ALARM'].tolist(), [-1, 1]))


def device_pms(data):
    pms = encoder.predict(data.iloc[:,1:46])
    device = lbEncoder.transform(np.reshape(data['GROUPBYKEY'].tolist(), [-1, 1]))
    device = np.reshape(device, [-1, 1])
    device = ohEncoder.transform(device)
    device = pd.DataFrame(device.toarray())

    x = np.concatenate([device,pms],axis=1)
    y = data['ALARM']
    return x, y
label_x,_ = device_pms(labeled)
unlabel_x, unlabel_y = device_pms(unlabeled)
x = np.concatenate([label_x,unlabel_x],axis=0)
y = np.concatenate([alarm,unlabel_y],axis=0)
'''save data'''
np.save('../data/mean_teacher/train_x',x)
np.save('../data/mean_teacher/train_y',y)
