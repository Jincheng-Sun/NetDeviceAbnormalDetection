import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import math
from sklearn.metrics import confusion_matrix

def cm_metrix(y_test, y_pred):
    return confusion_matrix(y_true=y_test, y_pred=y_pred)

def cm_analysis(cm, labels, x_rotation=90, y_rotation=0, font_size=0.33, precision=False):
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率

    if (precision):
        '''flip and rotate the confusion metrix'''
        labels = labels[::-1]
        cm = np.rot90(np.flip(cm, axis=0))

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif (c == 0):
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)

    if (precision):
        cm.columns.name = 'True Label'
        cm.index.name = 'Predict Label'
    else:
        cm.index.name = 'True Label'
        cm.columns.name = 'Predict Label'

    sb.set(font_scale=font_size)

    sb.heatmap(cm, annot=annot, fmt='', cmap='Blues')
    plt.xticks(rotation=x_rotation)
    plt.yticks(rotation=y_rotation)

    plt.show()
