from os import listdir
from os.path import isfile, join
import fastparquet as fp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.externals import joblib
import numpy as np
from numpy.lib.stride_tricks import as_strided as strided
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
# anomaly data
anomaly = df[df['category'].notna()]
anomaly_list = anomaly['ID'].value_counts().index.tolist()

# statistics
## Count devices
# devices = df.groupby(['meta_facility'])['ID'].value_counts().index.to_frame()
# devices['meta_facility'].value_counts()
## Count anomaly devices
# devices = anomaly.groupby(['category', 'meta_facility']).nunique()['ID']

def plot_per_device(data, device_id, time_range=pd.date_range('20190523', end='201906232100', freq='15T')):

    plt.rcParams['figure.figsize']=[8,6]
    plt.rcParams['figure.dpi'] = 200
    # get data by device id
    per_device = data[data.ID == device_id]
    per_device.set_index('TIME', inplace=True)
    per_device.index = pd.to_datetime(per_device.index)
    # data from 2018-11-18 to 2019-03-19, frequency: day
    # reindex the data and fill the gaps with NaN
    per_device = per_device.reindex(time_range)
    # drop NaN columns
    per_device.dropna('columns', inplace=True, how='all')

    try:
        anomaly = per_device.loc[per_device['category'].notna()]
        print(anomaly['category'])
    except:
        print('No alarm happened'
              '')
    oos = per_device[~per_device['meta_status'].isin(status_list)]
    # get device type
    device_type = per_device['meta_facility'].values[0]
    # get status
    status = per_device['meta_status'].values
    print(status)



    plt.figure()
    # plot a subplot for each PM value
    axs = per_device.plot(subplots=True, grid=True, marker='o', markersize=2)

    try:
        axs = oos.plot(ax = axs.flatten()[:], legend = False, subplots=True, grid=True, style='yx', markersize=5)
    except:
        print('All data in-service')
    # plot a vline for each anomaly in each ax
    for ax in axs:
        ax.legend(loc=2, prop={'size': 5})
        try:
            for vline in anomaly.index.to_list():
                ax.axvline(vline, color='r', lw=1, ls='dashed', label='anomaly')
                # print(anomaly.loc[vline, ['TIME', 'ALARM']])
        except:
            print('No anomaly exsists')

    plt.xlabel('ID: ' + str(device_id) + '   Device Type: '+str(device_type))

    plt.show()


# # one_day = pd.date_range('20190610', end='20190611', freq='15T')
# dev_groupby_type = anomaly[anomaly['meta_facility']=='OPTMON']
#
# plot_per_device(df, 'Node339:ETH10G-1-2-8')

# local_fault = anomaly[anomaly['category'] == 'local fault - service affecting unplanned']['ID'].value_counts()
# loss_of_clock = anomaly[anomaly['category'] == 'loss of clock - service affecting unplanned']['ID'].value_counts()
# loss_of_data = anomaly[anomaly['category'] == 'loss of data synch - service affecting unplanned']['ID'].value_counts()
# loss_of_frame = anomaly[anomaly['category'] == 'loss of frame - service affecting unplanned']['ID'].value_counts()
# # loss_of_multiframe = anomaly[anomaly['category'] == 'loss of multiframe - service affecting unplanned']['ID'].value_counts()
# loss_of_signal = anomaly[anomaly['category'] == 'loss of signal - service affecting unplanned']['ID'].value_counts()
# odu_ais = anomaly[anomaly['category'] == 'odu ais - service affecting unplanned']['ID'].value_counts()
# odu_lck = anomaly[anomaly['category'] == 'odu lck - service affecting unplanned']['ID'].value_counts()
# pre_fec = anomaly[anomaly['category'] == 'pre-fec signal fail - service affecting unplanned']['ID'].value_counts()




