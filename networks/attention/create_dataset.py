import pandas as pd
pd.set_option('display.max_columns', 50)
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np

data = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/Europe_Network_data.parquet')
dev_drop_list = ['CHMON', 'STM64', 'STTP', 'STM4', 'STM16', 'OC48', 'OC12', 'OC3', 'RAMAN']
alarm_list = ['Excessive Error Ratio', 'Frequency Out Of Range', 'GCC0 Link Failure',
              'Gauge Threshold Crossing Alert Summary', 'Link Down', 'Local Fault',
              'Loss Of Clock', 'Loss Of Frame', 'Loss Of Signal', 'OSC OSPF Adjacency Loss',
              'OTU Signal Degrade', 'Rx Power Out Of Range']

print(data['ALARM'].value_counts())
data = data[data['ALARM'].isin(alarm_list)]
print(data['ALARM'].value_counts())

print(data['GROUPBYKEY'].value_counts())
data = data.drop(data[data['GROUPBYKEY'].isin(dev_drop_list)].index)
print(data['GROUPBYKEY'].value_counts())

data = data.fillna(0)

data = data.drop(['ID','TIME','LABEL'], axis=1)
scaler = StandardScaler(with_mean=False)
scaler.fit(data.iloc[:, 1:46])
le_1 = LabelEncoder()
dev_keep_list = ['AMP', 'ETH10G', 'ETHN', 'ETTP', 'OC192', 'OPTMON', 'OSC', 'OTM', 'OTM2', 'OTUTTP', 'PTP']
# le_1.fit(['AMP', 'ETH10G', 'ETHN', 'ETTP', 'FLEX', 'OC192', 'OPTMON', 'OSC', 'OTM', 'OTM2', 'OTUTTP', 'PTP'])
le_1.fit(dev_keep_list)

ohe_1 = OneHotEncoder()
ohe_1.fit(np.arange(0,len(dev_keep_list)).reshape([-1,1]))

le_2 = LabelEncoder()
le_2.fit(alarm_list)
ohe_2 = OneHotEncoder()
ohe_2.fit(np.arange(0,len(alarm_list)).reshape([-1,1]))

# split
train_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=42)
train = data.loc[train_idx]
test = data.loc[test_idx]

def process(data):
    PMs = data.iloc[:, 1:46]
    PMs = scaler.transform(PMs)
    PMs = PMs.reshape([-1,45,1])

    # device onehot-encoding
    device = data['GROUPBYKEY']
    device = le_1.transform(device)
    device = device.reshape([-1,1])

    device = ohe_1.transform(device).toarray()

    # alarm label-to-number
    alarm = data['ALARM']
    alarm = le_2.transform(alarm)
    alarm = alarm.reshape([-1,1])
    alarm = ohe_2.transform(alarm).toarray()
    return PMs, device, alarm

X1, dev1, y1 = process(train)
np.save('/home/oem/Projects/NetDeviceAbnormalDetection/data/attention/c_PMs_train',X1)
np.save('/home/oem/Projects/NetDeviceAbnormalDetection/data/attention/c_dev_train',dev1)
np.save('/home/oem/Projects/NetDeviceAbnormalDetection/data/attention/c_alm_train',y1)
#
X2, dev2, y2 = process(test)
np.save('/home/oem/Projects/NetDeviceAbnormalDetection/data/attention/c_PMs_test',X2)
np.save('/home/oem/Projects/NetDeviceAbnormalDetection/data/attention/c_dev_test',dev2)
np.save('/home/oem/Projects/NetDeviceAbnormalDetection/data/attention/c_alm_test',y2)