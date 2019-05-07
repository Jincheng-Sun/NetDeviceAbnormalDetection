from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.utils import shuffle

#
# x = np.load('data/autoencoder/train_x.npy')
# y = np.load('data/train_y_all.npy')
# pca = PCA(2)
# dataset = pca.fit_transform(x)
# clust = KMeans(n_clusters=15)
# opt = clust.fit(dataset)
# center = opt.cluster_centers_
# opt = KMeans(n_clusters=15, algorithm='full', init=center).fit(dataset)
#
# pred = opt.predict(dataset)

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d, Axes3D
#
#
# # fig = plt.figure()
# # ax = Axes3D(fig)
# # ax.scatter(dataset[:,0],dataset[:,1],dataset[:,2],c = pred)
# # # fig.colorbar(ax)
# # plt.show()
#
# plt.scatter(dataset[:,0],dataset[:,1],c = pred)
# plt.colorbar()
# plt.show()

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(x[:,0],x[:,1],x[:,2])
# # fig.colorbar(ax)
# plt.show()

# ---------------------------------------------
# pca visualization

'''load data'''
print('[INFO] Start loading data')

EU_data = pd.read_parquet('../data/normalized_data_Europe.parquet', engine='pyarrow').drop(
    ['ALARM', 'GROUPBYKEY', 'LABEL'], axis=1)
TK_data = pd.read_parquet('../data/normalized_data_Tokyo.parquet', engine='pyarrow').drop(
    ['ALARM', 'GROUPBYKEY', 'LABEL'], axis=1)

EU_data['AREA'] = 'Europe'
TK_data['AREA'] = 'Tokyo'

print('[INFO] Finish loading data')

'''load encoder'''
encoder = load_model('../models/aftconcat/encoder_3layers')

'''encode data'''
print('[INFO] Start encoding data')
EU_data_new = encoder.predict(EU_data.loc[:, EU_data.columns != 'AREA'])
EU_data_new = np.concatenate((EU_data_new, EU_data['AREA'].as_matrix().reshape(-1, 1)), axis=1)
TK_data_new = encoder.predict(TK_data.loc[:, TK_data.columns != 'AREA'])
TK_data_new = np.concatenate((TK_data_new, TK_data['AREA'].as_matrix().reshape(-1, 1)), axis=1)

data_new = np.concatenate((EU_data_new, TK_data_new), axis=0)
data_new = shuffle(data_new)
print('[INFO] Finish encoding data')

'''PCA'''

print('[INFO] Start PCA')
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data_new[:, 0:20])
principalComponents = pd.DataFrame(data=principalComponents, columns=['x', 'y'])
principalComponent = pd.concat([principalComponents, pd.DataFrame(data_new[:, 20])], axis=1, ignore_index=True)
print('[INFO] Finish PCA')


'''visualization'''
import matplotlib.pyplot as plt
print('[INFO] Start Plotting')
targets = ['Europe', 'Tokyo']
colors = ['r', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = principalComponent.iloc[:,2] == target
    plt.scatter(x=principalComponent.loc[indicesToKeep, 0], y=principalComponent.loc[indicesToKeep, 1], c=color, s = 50, marker='.')

plt.xlabel('X1 after reduction', fontsize = 15)
plt.ylabel('X2 after reduction', fontsize = 15)
plt.title('2-Dimensional Reduction using PCA - increase Dimension')
plt.legend(targets)
plt.show()
print('[INFO] Finish Plotting')
