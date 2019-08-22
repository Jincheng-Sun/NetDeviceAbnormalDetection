import pandas as pd
from fbprophet import Prophet
from os import listdir
from os.path import isfile, join
import fastparquet as fp
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

otm4 = df[df['meta_facility'] == 'OTM4'].fillna(0)

keep_pms = ['OTU-QAVG', 'OTU-ES', 'OPRMAX-OCH_OPRMIN-OCH_-', 'OPRAVG-OCH', 'OTU-QSTDEV', 'CV-OTU']

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

    # keep columns
    data = data[['ID', 'TIME'] + [PM]]

    per_device = data[data['ID'] == Device]

    per_device = per_device.rename(columns={'TIME': 'ds', PM: 'y'})

    m = Prophet()

    dt = per_device[:int(per_device.shape[0] * train_portion)]
    m.fit(dt)

    future = m.make_future_dataframe(periods=per_device.shape[0] - int(per_device.shape[0] * train_portion),
                                     freq='15min', include_history=True)

    forecast = m.predict(future)

    plt.rcParams['figure.figsize'] = [8, 6]
    plt.rcParams['figure.dpi'] = 200

    per_device.set_index('ds', inplace=True)
    per_device.index = pd.to_datetime(per_device.index)
    per_device['y'].plot()
    plt.show()

    fig1 = m.plot(forecast)
    fig2 = m.plot_components(forecast)
    fig1.show()
    fig2.show()


