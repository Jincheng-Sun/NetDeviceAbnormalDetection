import pandas as pd
from fbprophet import Prophet
from os import listdir
from os.path import isfile, join
import fastparquet as fp
import matplotlib.pyplot as plt
from time_series.sort_data_15m import plot_per_device
import numpy as np

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

otm4 = df[df['meta_facility'] == 'OTM4'].fillna(0)
otm4_copy = otm4.copy()

# array(['Node19:OTM4-2-1-1', 'Node363:OTM4-1-1-1', 'Node207:OTM4-1-1-1',
#        'Node339:OTM4-1-1-1', 'Node279:OTM4-1-17-1', 'Node77:OTM4-1-5-1',
#        'Node406:OTM4-6-7-1', 'Node13:OTM4-1-7-1', 'Node173:OTM4-2-3-1',
#        'Node240:OTM4-2-5-1', 'Node150:OTM4-1-1-1', 'Node141:OTM4-2-5-1',
#        'Node368:OTM4-1-10-1', 'Node64:OTM4-1-7-1', 'Node240:OTM4-3-1-1',
#        'Node71:OTM4-2-2-1', 'Node266:OTM4-1-15-1', 'Node173:OTM4-2-4-1',
#        'Node148:OTM4-1-1-1', 'Node77:OTM4-1-1-1', 'Node150:OTM4-2-1-1',
#        'Node222:OTM4-1-5-1', 'Node406:OTM4-6-5-1', 'Node306:OTM4-1-1-1',
#        'Node115:OTM4-3-5-1', 'Node141:OTM4-2-3-1', 'Node173:OTM4-1-1-1',
#        'Node13:OTM4-1-28-1', 'Node19:OTM4-1-1-1', 'Node17:OTM4-1-4-1',
#        'Node275:OTM4-1-7-1', 'Node419:OTM4-3-1-1', 'Node303:OTM4-1-1-1',
#        'Node352:OTM4-1-8-1', 'Node249:OTM4-1-4-1', 'Node222:OTM4-1-1-1',
#        'Node240:OTM4-2-1-1', 'Node115:OTM4-3-6-1', 'Node196:OTM4-1-1-1',
#        'Node363:OTM4-1-3-1', 'Node13:OTM4-1-7-2', 'Node77:OTM4-1-9-1',
#        'Node363:OTM4-1-2-1', 'Node266:OTM4-1-5-1', 'Node196:OTM4-2-5-1',
#        'Node141:OTM4-1-3-1', 'Node406:OTM4-1-7-1', 'Node85:OTM4-1-4-1',
#        'Node115:OTM4-2-1-1', 'Node324:OTM4-2-4-1', 'Node214:OTM4-1-3-1',
#        'Node419:OTM4-3-4-1', 'Node173:OTM4-1-4-1', 'Node148:OTM4-1-4-1',
#        'Node173:OTM4-1-3-1', 'Node85:OTM4-1-4-2', 'Node249:OTM4-2-4-1',
#        'Node249:OTM4-1-1-1', 'Node363:OTM4-1-4-1', 'Node275:OTM4-1-5-1',
#        'Node17:OTM4-2-2-1', 'Node266:OTM4-3-3-1', 'Node420:OTM4-2-5-1',
#        'Node115:OTM4-2-4-1', 'Node380:OTM4-1-1-1', 'Node249:OTM4-2-1-1',
#        'Node279:OTM4-1-17-2', 'Node240:OTM4-1-1-1', 'Node214:OTM4-1-11-1',
#        'Node385:OTM4-1-3-1', 'Node412:OTM4-1-1-1', 'Node347:OTM4-2-1-1'],
#       dtype=object)

# anomaly = otm4[otm4['category'] != 0]
# array(['Node207:OTM4-1-1-1', 'Node279:OTM4-1-17-1', 'Node13:OTM4-1-7-1',
#        'Node240:OTM4-2-5-1', 'Node240:OTM4-3-1-1', 'Node115:OTM4-3-6-1',
#        'Node13:OTM4-1-7-2', 'Node115:OTM4-2-1-1', 'Node85:OTM4-1-4-2',
#        'Node115:OTM4-2-4-1', 'Node279:OTM4-1-17-2', 'Node240:OTM4-1-1-1'],
#       dtype=object)
keep_pms = ['OTU-QAVG', 'OTU-ES', 'OPRMAX-OCH_OPRMIN-OCH_-', 'OPRAVG-OCH', 'OTU-QSTDEV', 'CV-OTU']

otm4 = otm4[['ID', 'TIME'] + keep_pms]

def plot_original(data, Device):
    plt.rcParams['figure.figsize'] = [8, 6]
    plt.rcParams['figure.dpi'] = 200

    per_device = data[data['ID'] == Device]
    per_device.set_index('TIME', inplace=True)
    per_device.index = pd.to_datetime(per_device.index)

    axs = per_device.plot(subplots=True, grid=True, marker='o', markersize=2)
    for ax in axs:
        ax.legend(loc=2, prop={'size': 5})
    plt.show()

def plot(data, Device='Node385:OTM4-1-3-1', PM='OTU-QAVG', train_portion=0.9):

    additional_pms = ['OTU-QAVG', 'OTU-ES', 'OPRMAX-OCH_OPRMIN-OCH_-', 'OPRAVG-OCH', 'OTU-QSTDEV', 'CV-OTU']
    additional_pms.remove(PM)
    # # keep columns
    # data = data[['ID', 'TIME'] + [PM]]

    per_device = data[data['ID'] == Device]

    per_device = per_device.rename(columns={'TIME': 'ds', PM: 'y'})

    m = Prophet()

    for i in additional_pms:
        m.add_regressor(i)

    dt = per_device[:int(per_device.shape[0] * train_portion)].copy()

    m.fit(dt)

    future = m.make_future_dataframe(periods=per_device.shape[0] - int(per_device.shape[0] * train_portion),
                                     freq='15min', include_history=True)
    for i in additional_pms:
        future[i] = pd.Series(dt[i].to_list())

    forecast = m.predict(future)

    plt.rcParams['figure.figsize'] = [8, 6]
    plt.rcParams['figure.dpi'] = 200


    fig1 = m.plot(forecast)
    fig2 = m.plot_components(forecast)
    fig1.show()
    fig2.show()

    plt.cla()

    per_device.set_index('ds', inplace=True)
    per_device.index = pd.to_datetime(per_device.index)
    per_device = per_device.reindex(pd.date_range('20190523', end='201906232100', freq='15T'))

    pd.plotting.register_matplotlib_converters()

    per_device['y'].plot()

    plt.show()
    plt.cla()
    return per_device['y']



