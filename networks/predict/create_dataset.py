import pandas as pd
import numpy as np



m = 3
n = 2
dev_type = 'ETH'
file_path = '/home/oem/Projects/NetDeviceAbnormalDetection/data/Europe_Network_data.parquet'
#--------------------------------------------------------------------------------
# dataset 1
data = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/Europe_network_data.parquet')
alarm = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/Europe_network_labels.parquet')
alarm = alarm.loc[~alarm['description'].str.contains('Demo for Josh')]
alarm['TIME'] = pd.to_datetime(alarm['time'], infer_datetime_format=True).dt.floor('1D')
alarm = alarm.drop(['description','time','timestamp','extraAttributes'], axis=1).rename({'category':'ALARM'}, axis=1)
alarm = alarm.drop_duplicates()
data = pd.merge(data,alarm,'left', on=['ID', 'TIME'])
# --------------------------------------------------------------------------------
#dataset 2

# data = pd.read_parquet(file_path)
print(data.shape)
devices = data[data['GROUPBYKEY'] == dev_type]  # extract certain type of device

dev_count = devices['ID'].value_counts()
dev_count = dev_count.drop(dev_count[dev_count < m + n].index).index.tolist()
#
devices = devices[devices['ID'].isin(dev_count)]  # filter out devices that has less data than the time window
devices = devices.drop_duplicates()
devices = devices.set_index(['ID', 'TIME']).sort_index().reset_index().fillna(0)
devices['ALARM'] = devices['ALARM'].map(lambda x: 0 if x == 0 else 1)  # mask all alarms 1

print(devices['ID'].value_counts())
print(devices['ALARM'].value_counts())

label_list = ['IS', 'n/a', 'IS-ANR']
x = []
y = []
for idx in devices.index:
    device_type = devices.loc[idx:idx + m + n, 'GROUPBYKEY']
    if device_type.nunique() != 1:  # make sure all devices in this window are the same
        continue

    m_labels = devices.loc[idx:idx + m, 'LABEL']
    if ~m_labels.isin(label_list).all():  # make sure m data are all in service
        continue

    m_alarms = devices.loc[idx:idx + m, 'ALARM']
    if m_alarms.any():  # make sure m data are all normal
        continue

    x.append(devices.iloc[idx:idx + m, 4:49].values)

    if (devices.loc[idx + m:idx + m + n, 'ALARM'].values.any()):
        y.append([1])
    else:
        y.append([0])

    if idx % 10000 == 0:
        print(idx)

    if idx + m + n == devices.index[-1]:
        break

x = np.expand_dims(x, 3)
y = np.array(y)

import collections
print(collections.Counter(y.flatten()))
np.save('/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_pms_3_45.npy'%dev_type,x)
np.save('/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_alarms_2days.npy'%dev_type,y)