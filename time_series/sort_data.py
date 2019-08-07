import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.externals import joblib
import numpy as np
dev_list = ['AMP', 'ETH', 'ETH10G', 'ETHN', 'ETTP', 'OPTMON', 'OSC', 'OTM', 'OTM2', 'OTUTTP', 'PTP']

alarm_list = [None, 'Excessive Error Ratio',  # 1
              'Frequency Out Of Range',  # 2
              'GCC0 Link Failure', 'Gauge Threshold Crossing Alert Summary',  # 4
              'Link Down', 'Local Fault', 'Loss Of Clock', 'Loss Of Frame', 'Loss Of Signal',  # 9
              'OSC OSPF Adjacency Loss', 'OTU Signal Degrade',  # 11
              'Rx Power Out Of Range']  # 12

state_list = ['IS', 'n/a', 'IS-ANR']


# # May dataset
# data = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/Europe_Network_Data_May13.parquet')
# data.drop('LASTOCCURRENCE', axis=1, inplace=True)


# # May 23 dataset
# data = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/v2_Europe_network_data_pivoted.parquet')
# labels = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/v2_Europe_network_labels.parquet')
#
# # remove PEC for join
# tmp, _ = data['meta_NHP_ID'].str.rsplit(':',1).str
# data['ID'] = tmp
#
# labels['ID'] = labels['fk']
#
# # floor to nearest day
# data['TIME'] = pd.to_datetime(data['pk_timestamp'], infer_datetime_format=True).dt.floor('1D')
# labels['TIME'] = pd.to_datetime(labels['time'], infer_datetime_format=True).dt.floor('1D')
#
# data = pd.merge(data, labels, on=['ID', 'TIME'])
# data = data.drop(['pk_id', 'pk_timestamp', 'meta_NHP_ID', 'meta_PEC', 'meta_TID', 'meta_AID', 'meta_SHELF', 'meta_SLOT', 'meta_PORT', 'meta_CHANNEL',
#             'createdAt', 'description', 'extraAttributes', 'fk', 'time', 'timestamp'], axis=1)
# rep_list = {'meta_FACILITY': 'GROUPBYKEY',
#             'meta_STATUS': 'LABEL',
#             'category': 'ALARM'
#             }
# # rename the columns
# data = data.rename(columns=rep_list)
# sup_list = ['OPTAVG-OCH', 'OPTMAX-OCH_OPTMIN-OCH_-']
# data[sup_list] = np.nan

# # March data
data= pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/Europe_Network_data.parquet')
index = data.columns
# scaler = joblib.load('standard_scaler.pkl')

# filter data
devices = data[data['GROUPBYKEY'].isin(dev_list)]
devices = devices[devices['ALARM'].isin(alarm_list)]  # filter out devices with alarm out of the list
devices = devices.set_index(['ID', 'TIME']).sort_index().reset_index()
devices = devices.drop_duplicates(subset=['ID', 'TIME'])

# drop data that has less than 100 days
dev_count = devices['ID'].value_counts()
dev_count = dev_count.drop(dev_count[dev_count < 30].index).index.tolist()
devices = devices[devices['ID'].isin(dev_count)]  # filter out devices that has less data than the time window

# devices.iloc[:, 4:49] = scaler.transform(devices.iloc[:, 4:49].values)


anomalies = devices.loc[devices['ALARM'].notna()]
# 'Device10111'

def plot_per_device(data, device_id):

    plt.rcParams['figure.figsize']=[8,6]
    plt.rcParams['figure.dpi'] = 200
    # get data by device id
    per_device = data[data.ID == device_id]
    anomaly = per_device.loc[per_device['ALARM'].notna()]
    oos = per_device[~per_device['LABEL'].isin(state_list)]
    # get device type
    device_type = per_device['GROUPBYKEY'].values[0]
    # drop NaN columns
    per_device.dropna('columns', inplace=True)
    # set `TIME` column as index and convert the format to '%y-%mm-%dd'
    per_device.set_index('TIME', inplace=True)
    per_device.index = pd.to_datetime(per_device.index, format='%y-%mm-%dd', unit='s')
    # data from 2018-11-18 to 2019-03-19, frequency: day
    days = pd.date_range('20181118', end='20190319', freq='D')
    # reindex the data and fill the gaps with NaN
    per_device = per_device.reindex(days)
    # set `TIME` column as index and convert the format to '%y-%mm-%dd'
    anomaly.set_index('TIME', inplace=True)
    anomaly.index = pd.to_datetime(anomaly.index, format='%y-%mm-%dd', unit='s')
    # alarm_type = anomaly['ALARM']
    print(anomaly['ALARM'])
    oos.set_index('TIME', inplace=True)
    oos.index = pd.to_datetime(oos.index, format='%y-%mm-%dd', unit='s')
    oos = oos[per_device.columns]

    plt.figure()
    # plot a subplot for each PM value
    axs = per_device.plot(subplots=True, grid=True, marker='o', markersize=2)
    # plot scatter for not-in-service data
    try:
        axs = oos.plot(ax = axs.flatten()[:], legend = False, subplots=True, grid=True, style='yx', markersize=5)
    except:
        print('All data in-service')
    # # plot a vline for each anomaly in each ax
    for ax in axs:
        ax.legend(loc=2, prop={'size': 5})
        try:
            for vline in anomaly.index.to_list():
                ax.axvline(vline, color='r', lw=1, ls='dashed', label='anomaly')
                # print(anomaly.loc[vline, ['TIME', 'ALARM']])
        except:
            print('No anomaly exsists')

    plt.xlabel('ID: ' + device_id + '   Device Type: '+device_type)

    plt.show()

devs = anomalies[anomalies['GROUPBYKEY']==dev_list[0]]['ID'].value_counts().index.to_list()
for dev in devs:
    plot_per_device(devices, dev)

