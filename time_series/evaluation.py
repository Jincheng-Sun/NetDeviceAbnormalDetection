from keras.models import load_model
import pandas as pd
from os import listdir
from os.path import isfile, join
import fastparquet as fp
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt

status_list = ['IS', 'IS-ANR', 'n/a'
               ]
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

# --------------------------------------------------------------------------
# OTM4 devices
otm4 = df[df['meta_facility'] == 'OTM4'].fillna(0)

keep_pms = ['OTU-QAVG', 'OTU-ES', 'OPRMAX-OCH_OPRMIN-OCH_-', 'OPRAVG-OCH', 'OTU-QSTDEV', 'CV-OTU']
# keep columns
otm4 = otm4[['ID', 'TIME']+keep_pms]
# load scaler
scaler = joblib.load('data/scaler_OTM4')
# load test device ID
test_id = np.load('data/OTM4_test_list_id.npy')
# load model
model = load_model('models/OTM4_model')
# per device
per_device =otm4[otm4['ID'] == 'Node385:OTM4-1-3-1']
#
index = np.random.randint(0, per_device.shape[0]-1-192)
# two days of data
two_days = per_device[index:index+192]

two_days_2 = two_days.copy()

first_day = two_days[:96]

second_day = two_days[96:]

# first_day = scaler.transform(first_day[keep_pms].values)
first_day = first_day[keep_pms].values
second_day_pred = model.predict(first_day.reshape([1,96,6]))

second_day_pred = second_day_pred.reshape([96, 6])

# second_day_pred = scaler.inverse_transform(second_day_pred)

two_days_2[96:][keep_pms] = second_day_pred

# plot
def plot(data):
    plt.rcParams['figure.figsize'] = [8, 6]
    plt.rcParams['figure.dpi'] = 200

    data.set_index('TIME', inplace=True)
    data.index = pd.to_datetime(data.index)
    axs = data.plot(subplots=True, grid=True, marker='o', markersize=2)
    for ax in axs:
        ax.legend(loc=2, prop={'size': 5})
    plt.show()

plot(two_days)
plot(two_days_2)