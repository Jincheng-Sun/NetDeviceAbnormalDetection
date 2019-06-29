import pandas as pd
import numpy as np
import gc
import sys
import collections

sys.path.insert(0,'/home/oem/Projects')
from Kylearn.utils.log import log_down

logfile = 'create_data_log'
logger = log_down(logfile)
gc.enable()


# --------------------------------------------------------------------------------
# dataset 3
# data = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/v2_Europe_network_data_pivoted.parquet')
# alarm = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/v2_Europe_network_labels.parquet')
# alarm = alarm.loc[~alarm['description'].str.contains('Demo for Josh')]
#
# alarm['TIME'] = pd.to_datetime(alarm['time'], infer_datetime_format=True).dt.floor('1D')
#
# alarm = alarm.drop(['description', 'time', 'timestamp', 'extraAttributes'], axis=1).rename({'category': 'ALARM'}, axis=1)
#
# alarm = alarm.drop_duplicates()
#
# alarm = alarm.rename(columns={ 'fk' : 'ID'})
#
#
# data['TIME'] = pd.to_datetime(data['pk_timestamp'], infer_datetime_format=True).dt.floor('1D')
# tmp, _ = data['meta_NHP_ID'].str.rsplit(':',1).str
# data['ID'] = tmp
#
# data = data.rename(columns={ 'meta_FACILITY' : 'GROUPBYKEY', 'meta_STATUS' : 'LABEL'})
#
#
# data = data.drop(columns=['meta_TID', 'pk_timestamp', 'meta_PORT', 'meta_SHELF', 'meta_SLOT', 'meta_PEC', 'meta_CHANNEL', 'meta_AID', 'meta_PEC', 'meta_NHP_ID', 'pk_id'], axis=0)
#
# data['OPTAVG-OCH'] = 0
#
# data['OPTMAX-OCH_OPTMIN-OCH_-'] = 0
#
# data = pd.merge(data, alarm, 'left', on=['ID', 'TIME'])
#
# data = data.drop(columns=['createdAt'], axis=1)
# data = data[['ID', 'TIME', 'LABEL', 'GROUPBYKEY', 'BBE-RS', 'CV-OTU', 'CV-S',
#
#        'DROPGAINAVG-OTS', 'DROPGAINMAX-OTS_DROPGAINMIN-OTS_-', 'E-CV', 'E-ES',
#
#        'E-INFRAMESERR_E-INFRAMES_/', 'E-OUTFRAMESERR_E-OUTFRAMES_/', 'E-UAS',
#
#        'ES-OTU', 'ES-RS', 'ES-S', 'OCH-OPRAVG', 'OCH-OPRMAX_OCH-OPRMIN_-',
#
#        'OCH-SPANLOSSAVG', 'OCH-SPANLOSSMAX_OCH-SPANLOSSMIN_-', 'OPINAVG-OTS',
#
#        'OPINMAX-OTS_OPINMIN-OTS_-', 'OPOUTAVG-OTS',
#
#        'OPOUTAVG-OTS_OPINAVG-OTS_-', 'OPOUTMAX-OTS_OPOUTMIN-OTS_-',
#
#        'OPRAVG-OCH', 'OPRAVG-OTS', 'OPRMAX-OCH_OPRMIN-OCH_-',
#
#        'OPRMAX-OTS_OPRMIN-OTS_-', 'OPTAVG-OCH', 'OPTAVG-OTS',
#
#        'OPTMAX-OCH_OPTMIN-OCH_-', 'OPTMAX-OTS_OPTMIN-OTS_-', 'ORLAVG-OTS',
#
#        'ORLMIN-OTS', 'OTU-CV', 'OTU-ES', 'OTU-QAVG', 'OTU-QSTDEV', 'PCS-CV',
#
#        'PCS-ES', 'PCS-UAS', 'QAVG-OTU', 'QSTDEV-OTU', 'RS-BBE', 'RS-ES',
#
#        'S-CV', 'S-ES', 'ALARM']]




# --------------------------------------------------------------------------------
# dataset 2
# data = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/Europe_network_data.parquet')
# alarm = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/Europe_network_labels.parquet')
# alarm = alarm.loc[~alarm['description'].str.contains('Demo for Josh')]
# alarm['TIME'] = pd.to_datetime(alarm['time'], infer_datetime_format=True).dt.floor('1D')
# alarm = alarm.drop(['description','time','timestamp','extraAttributes'], axis=1).rename({'category':'ALARM'}, axis=1)
# alarm = alarm.drop_duplicates()
# data = pd.merge(data,alarm,'left', on=['ID', 'TIME'])
# --------------------------------------------------------------------------------
# dataset 1



# # data = data.rename(columns={
# #                             'meta_NHP_ID':'ID',
# #                             'far_end_client_signal_fail_unplanned':'ALARM',
# #                             'meta_FACILITY':'GROUPBYKEY',
# #                             'meta_STATUS':'LABEL'
# #                             })
#
# try:
#     data = data.drop(['LASTOCCURRENCE'], axis=1)
# except:
#     pass
#
# # Case all
# # devices = data[~data['GROUPBYKEY'].isin(drop_list)]
# # PM_list['ALL'] = data.columns[4:49]
#
# # apply the auto encoder here
#
# # Case one device
# devices = data[data['GROUPBYKEY'].isin(dev_list[dev_type])]  # extract certain type of device
# # devices = data[data['GROUPBYKEY'] == dev_type]
# # alarm = data['ALARM'].value_counts()
# # # focus on alarms that appear more than 50 times
# # alm_list = alarm[(alarm > 50) & (alarm < 10000)].index.tolist()
#
#

# devices we are focusing on
drop_list = ['CHMON', 'STM64', 'OC192', 'STTP', 'STM4', 'STM16', 'NMCMON', 'OC48', 'OC12', 'OC3', 'FLEX', 'RAMAN']
# alarms we are focusing on

alarm_list = ['Loss Of Signal', None]
label_list = ['IS', 'n/a', 'IS-ANR']


dev_dict = {
            'OTM': ['OTM', 'OTM0', 'OTM1', 'OTM2', 'OTM3', 'OTM4', 'OTMC2'],
            # 'OTM': ['OTM0', 'OTM1', 'OTM2', 'OTM3', 'OTM4'],
            'ETH': ['ETH', 'ETHN', 'ETH10G', 'ETH40G', 'ETH100', 'ETH100G', 'ETHFlex'],
            'OPTMON': ['OPTMON']
}
PM_dict = {'OTM': ['OPRMAX-OCH_OPRMIN-OCH_-', 'OPRAVG-OCH', 'OTU-CV','OTU-ES', 'OTU-QAVG', 'OTU-QSTDEV'],
           'ETH': [
               'E-CV',
               'E-ES',
               'E-INFRAMESERR_E-INFRAMES_/',
               'E-OUTFRAMESERR_E-OUTFRAMES_/',
               'E-UAS',
               'PCS-CV',
               'PCS-ES',
               'PCS-UAS'
           ], 'OPTMON': [
        'OPRAVG-OTS',
        'OPRMAX-OTS_OPRMIN-OTS_-',
        'OPTAVG-OTS',
        'OPTMAX-OTS_OPTMIN-OTS_-',
        "OPTAVG-OTS_OPRAVG-OTS_-"
    ],
           'ETH10G':[
               "E-UAS", "E-ES", "E-CV", "E-INFRAMESERR_E-INFRAMES_/", "E-OUTFRAMESERR_E-OUTFRAMES_/","PCS-UAS", "PCS-ES", "PCS-CV"
           ]}


file_path = '/home/oem/Projects/NetDeviceAbnormalDetection/data/Europe_Network_Data_May13.parquet'
# file_path = '/home/oem/Projects/NetDeviceAbnormalDetection/data/Europe_all_PMs_anon_no_CHMON_22_02_19_to_03_06_19_null_dropped.parquet'
data = pd.read_parquet(file_path)

m = 3
n = 2
dev_type = 'OPTMON'
device_list = dev_dict[dev_type]
pm_list = PM_dict[dev_type]

try:
    data = data.drop(['LASTOCCURRENCE'], axis=1)
except:
    pass

def slid_generate(m, n, data,label_list = None, pm_list = None, device_list=None,drop_list=None, alarm_list=None, all_devices = False, all_alarms = False):
    logger.info('Past day(s): %s Future day(s): %s'%(m, n))
    if all_devices:
        pm_list = data.columns[4:49]
        devices = data[~data['GROUPBYKEY'].isin(drop_list)]
        logger.info('Device type: ALL devices except %s'%drop_list)
    else:
        devices = data[data['GROUPBYKEY'].isin(device_list)]
        logger.info('Device type: %s'%device_list)

    if all_alarms:
        logger.info('ALARM: ALL')
        logger.info('\n'+ str(devices['ALARM'].value_counts()))
    else:
        devices = devices[devices['ALARM'].isin(alarm_list)]  # filter out devices with alarm out of the list
        logger.info('ALARM: %s'%alarm_list)
    logger.info('DEVICE COUNT')
    logger.info('\n'+ str(devices['GROUPBYKEY'].value_counts()))  # print device count
    dev_count = devices['ID'].value_counts()
    dev_count = dev_count.drop(dev_count[dev_count < m + n].index).index.tolist()
    devices = devices[devices['ID'].isin(dev_count)]  # filter out devices that has less data than the time window
    devices = devices.drop_duplicates()
    devices = devices.set_index(['ID', 'TIME']).sort_index().reset_index().fillna(0)
    devices['ALARM'] = devices['ALARM'].map(lambda x: 0 if x == 0 else 1)  # mask all alarms 1
    logger.info('SAMPLE COUNT')
    logger.info('\n'+ str(devices['ALARM'].value_counts()))

    x = []
    y = []
    for idx in devices.index:
        device_type = devices.loc[idx:idx + m + n - 1, 'GROUPBYKEY']
        if device_type.nunique() != 1:  # make sure all devices in this window are the same
            continue

        m_labels = devices.loc[idx:idx + m - 1, 'LABEL']
        n_labels = devices.loc[idx + m:idx + m + n - 1, 'LABEL']
        if ~m_labels.isin(label_list).all():  # make sure m data are all in service
            continue

        m_alarms = devices.loc[idx:idx + m - 1, 'ALARM']
        if m_alarms.any():  # make sure m data are all normal
            continue

        if (devices.loc[idx + m:idx + m + n - 1, 'ALARM'].values.any()):
            y.append([1])
        else:
            if (n_labels.isin(
                    label_list).all()):  # if no alarm happens, then it's a normal sample and the labels should all be in service.
                y.append([0])
            else:
                continue

        x.append(devices.loc[idx:idx + m - 1, pm_list].values)

        if idx % 10000 == 0:
            print(idx)

        if idx + m + n == devices.index[-1]:
            break

    X = np.expand_dims(x, 3)
    Y = np.array(y)
    logger.info('DATA COUNT')
    logger.info('\n'+ str(collections.Counter(Y.flatten())))
    return X, Y
X,Y = slid_generate(3,2,data, label_list, pm_list, device_list,drop_list,alarm_list, False, all_alarms=True)
# X,Y = slid_generate(3,2,data, label_list, None, None,drop_list,alarm_list, True, True)


np.save('/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_pms_partial_3days_may_los.npy' % dev_type, X)
np.save('/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_alarms_2days_may_los.npy' % dev_type, Y)


