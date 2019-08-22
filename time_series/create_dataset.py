from os import listdir
from os.path import isfile, join
import fastparquet as fp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.externals import joblib
import numpy as np
from numpy.lib.stride_tricks import as_strided as strided
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

status_list = ['IS', 'IS-ANR', 'n/a']

mypath = '/home/oem/Projects/NetDeviceAbnormalDetection/data/15m_binned_network_data_v1.parquet/'

onlyfiles = [mypath + f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles = [f for f in onlyfiles if '.snappy' in f]

df = fp.ParquetFile(onlyfiles).to_pandas()

df = df.drop(['pk_id', 'pk_timestamp', 'meta_nhpId', 'meta_nodeName', 'meta_aid', 'meta_shelf',
              'meta_slot', 'meta_port', 'meta_channel', 'meta_macAddress', 'meta_pec'], axis=1)

df = df.drop_duplicates(subset=['ID', 'TIME'])


dev_count = df['ID'].value_counts()
# drop the devices that has data less than a day
dev_count = dev_count.drop(dev_count[dev_count < 96].index).index.tolist()
df = df[df['ID'].isin(dev_count)]
#
df['oos'] = df['meta_status'].apply(lambda x: 0 if x in status_list else 1)
# count oos data
oos_count = df.groupby(['ID'])['oos'].sum()
all_count = df.groupby(['ID'])['oos'].count()
# calculate oos ratio
oos_ratio = oos_count.div(all_count)
# drop devices that has more than 80% data that are oos
oos_ratio = oos_ratio.drop(oos_ratio[oos_ratio == 1].index).index.tolist()
df = df[df['ID'].isin(oos_ratio)]
df.drop(['oos'], axis = 1, inplace = True)
# load standard scaler
scaler = StandardScaler(copy=False)

# take only otm4 devices
otm4 = df[df['meta_facility']=='OTM4'].fillna(0)
# PM values of otm4
keep_pms = ['OTU-QAVG', 'OTU-ES', 'OPRMAX-OCH_OPRMIN-OCH_-', 'OPRAVG-OCH', 'OTU-QSTDEV', 'CV-OTU']
# Scale pms to have mean of 0 and variance of 1
# otm4[keep_pms] = scaler.fit_transform(otm4[keep_pms])
# split device ID
# train_dev, test_dev = train_test_split(otm4['ID'].unique(), test_size=0.2, random_state=10)
# # list of data
# devices_train = [otm4[otm4['ID']==i][keep_pms] for i in train_dev]
# devices_test = [otm4[otm4['ID']==i][keep_pms] for i in test_dev]

def features_and_targets(devices, past_point, future_point):

    def slid_generate(data, past_days, future_days):

        def get_sliding_windows(dataFrame, windowSize, returnAs2D=False):
            stride0, stride1 = dataFrame.values.strides
            rows, columns = dataFrame.shape
            output = strided(dataFrame, shape=(rows - windowSize + 1, windowSize, columns),
                             strides=(stride0, stride0, stride1))
            if returnAs2D == True:
                return output.reshape(dataFrame.shape[0] - windowSize + 1, -1)
            else:
                return output

        data = get_sliding_windows(data[:-1], past_days + future_days)

        return data[:, :past_days, :], data[:, past_days:, :]

    slides = [slid_generate(device, past_point, future_point) for device in devices]

    slide = slides.pop(0)
    features = slide[0]
    targets = slide[1]

    while len(slides) != 0:
        slide = slides.pop(0)
        features = np.concatenate([features, slide[0]], axis=0)
        targets = np.concatenate([targets, slide[1]], axis=0)

    return features, targets

# f_train, t_train = features_and_targets(devices_train, 24*4, 24*4)
# f_test, t_test = features_and_targets(devices_test, 24*4, 24*4)
#
# np.save('data/OTM4_features_96x6_train', f_train)
# np.save('data/OTM4_targets_96x6_train', t_train)
#
# np.save('data/OTM4_features_96x6_test', f_test)
# np.save('data/OTM4_targets_96x6_test', t_test)
#
# np.save('data/OTM4_test_list_id', test_dev)
# joblib.dump(scaler, 'data/scaler_OTM4')

data = devices_train = [otm4[otm4['ID']==i][keep_pms] for i in otm4['ID'].unique()]
features, targets = features_and_targets(data, 96, 96)
f_train, f_test, t_train, t_test = train_test_split(features, targets, test_size=0.2, random_state=22)