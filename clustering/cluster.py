from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np


# use t-sne for visualization
x = np.load('data/train_x.npy')
y = np.load('data/train_y_all.npy')
pca = PCA(2)
dataset = pca.fit_transform(x)
clust = KMeans(n_clusters=15)
opt = clust.fit(dataset)
center = opt.cluster_centers_
opt = KMeans(n_clusters=15, algorithm='full', init=center).fit(dataset)

pred = opt.predict(dataset)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(dataset[:,0],dataset[:,1],dataset[:,2],c = pred)
# # fig.colorbar(ax)
# plt.show()

plt.scatter(dataset[:,0],dataset[:,1],c = pred)
plt.colorbar()
plt.show()

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(x[:,0],x[:,1],x[:,2])
# # fig.colorbar(ax)
# plt.show()