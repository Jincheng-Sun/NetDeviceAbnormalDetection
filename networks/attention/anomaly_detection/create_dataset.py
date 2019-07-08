import pandas as pd
pd.set_option('display.max_columns', 50)
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np

data = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/Europe_Network_data.parquet')
dev_list = ['AMP', 'ETH', 'ETH10G', 'ETHN', 'ETTP', 'OPTMON', 'OSC', 'OTM', 'OTM2', 'OTUTTP', 'PTP']
alarm_list = ['Excessive Error Ratio',  # 1
              'Frequency Out Of Range',  # 2
              'GCC0 Link Failure', 'Gauge Threshold Crossing Alert Summary',  # 4
              'Link Down', 'Local Fault', 'Loss Of Clock', 'Loss Of Frame', 'Loss Of Signal',  # 9
              'OSC OSPF Adjacency Loss', 'OTU Signal Degrade',  # 11
              'Rx Power Out Of Range']  # 12
state_list = ['IS', 'n/a', 'IS-ANR']

scaler = StandardScaler(with_mean=False)
scaler.fit(data.iloc[:, 4:49])
le_1 = LabelEncoder()
le_1.fit(dev_list)
ohe_1 = OneHotEncoder()
ohe_1.fit(np.arange(0,len(dev_list)).reshape([-1,1]))

def keep_valid_and_split(raw_data):
    # keep data of certain devices
    raw_data = raw_data[raw_data['GROUPBYKEY'].isin(dev_list)]

    # keep data of certain alarms and normal data
    raw_data = raw_data[raw_data['ALARM'].isin(alarm_list+[None])]
    raw_data = raw_data.fillna(0)

    # divide dataset into normal and anomaly group
    normal = raw_data[raw_data['ALARM'] == 0]
    anomaly = raw_data[raw_data['ALARM'] != 0]

    print(normal.shape)
    print(anomaly.shape)

    # normal data should be 'in service'
    normal = normal[normal['LABEL'].isin(state_list)]
    # random sample same amount with anomaly data of normal data
    # as labeled data, and the rest as unlabeled data.
    normal_labeled, normal_unlabeled = train_test_split(
        normal, train_size=anomaly.shape[0], random_state=42)

    return anomaly, normal_labeled, normal_unlabeled

def label_data(data, value):
    data['ALARM'] = value
    return data

def process_attention(data):

    data = data.drop(['ID', 'TIME', 'LABEL'], axis=1)
    PMs = data.iloc[:, 1:46]
    PMs = scaler.transform(PMs)
    PMs = PMs.reshape([-1,45,1])

    # device onehot-encoding
    device = data['GROUPBYKEY']
    device = le_1.transform(device)
    device = device.reshape([-1,1])

    device = ohe_1.transform(device).toarray()

    # alarm label-to-number
    alarm = data['ALARM'].values
    return PMs, device, alarm


def process_concat(data):
    data = data.drop(['ID', 'TIME', 'LABEL'], axis=1)
    PMs = data.iloc[:, 1:46]
    PMs = scaler.transform(PMs)

    # device onehot-encoding
    device = data['GROUPBYKEY']
    device = le_1.transform(device)
    device = device.reshape([-1,1])

    device = ohe_1.transform(device).toarray()
    features = np.concatenate([PMs, device], axis=1).reshape([-1,56,1])
    # alarm label-to-number
    alarm = data['ALARM'].values

    return features, alarm



anomaly, normal, _ = keep_valid_and_split(data)
anomaly = label_data(anomaly, 1)
normal = label_data(normal, 0)
dataset = pd.concat([anomaly, normal], axis=0)
train, test = train_test_split(dataset, test_size=0.2, random_state=42)

# X1, dev1, y1 = process_attention(train)
# np.save('data/a_PMs_train',X1)
# np.save('data/a_dev_train',dev1)
# np.save('data/a_alm_train',y1)
# #
# X2, dev2, y2 = process_attention(test)
# np.save('data/a_PMs_test',X2)
# np.save('data/a_dev_test',dev2)
# np.save('data/a_alm_test',y2)

X1, y1 = process_concat(train)
np.save('data/c_PMs_train',X1)
np.save('data/c_alm_train',y1)

X2, y2 = process_concat(test)
np.save('data/c_PMs_test',X2)
np.save('data/c_alm_test',y2)