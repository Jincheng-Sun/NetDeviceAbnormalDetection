from keras import Model
from keras.layers import Input, Dense
from keras import Model
import numpy as np
from sklearn.externals import joblib

# '''build model'''
#
# input_classification = Input(shape=(20,))
# input_binary = Input(shape=(20,))
#
#
# def share(input):
#     layer1 = Dense(64, activation='relu')(input)
#     layer2 = Dense(128, activation='relu')(layer1)
#     return layer2
#
#
# mid_classification = share(input_classification)
# mid_binary = share(input_binary)
#
# output_classification = Dense(15, activation='softmax')(mid_classification)
# output_binary = Dense(2, activation='softmax')(mid_binary)
#
# model = Model(input=[input_classification, input_binary], output=[output_classification, output_binary])
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
#
# '''load data'''
#
# x = np.load('../data/train_x.npy')
# y = np.load('../data/train_y_all.npy')
#
# labelencoder = joblib.load('../models/labelencoder')
# ohencoder = joblib.load('../models/OneHotEncoder')
# y[y == 0] = 'normal'
# y = labelencoder.transform(y).reshape([-1, 1])
# y = ohencoder.fit_transform(y)
# del labelencoder, ohencoder
#
# from sklearn.preprocessing import OneHotEncoder
#
# y2 = np.load('../data/train_y_bi.npy')
# ohencoder = OneHotEncoder()
# y2 = ohencoder.fit_transform(y2.reshape([-1, 1]))
#
# '''split dataset'''
# from sklearn.model_selection import train_test_split
#
# x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
#
# x_train2, x_val2, y_train2, y_val2 = train_test_split(x, y2, test_size=0.2, random_state=43)
# del x, y, y2
#
# '''train'''
# from keras.callbacks import EarlyStopping
#
# monitor = EarlyStopping(monitor='dense_5_loss', patience=10, restore_best_weights=True)
# model.fit(x=[x_train, x_train2], y=[y_train, y_train2], batch_size=50, epochs=1000, callbacks=[monitor])
# model.save('../models/oneclassmodel')

import pandas as pd
from keras.models import load_model

'''load testset and encoders'''
x_test = np.load('../data/test_x.npy')
y_test = np.load('../data/test_y_all.npy')
y_test2 = np.load('../data/test_y_bi.npy')
y_test2 = y_test2.reshape([-1,1])
labelencoder = joblib.load('../models/labelencoder')
model = load_model('../models/oneclassmodel')
y_test[y_test == 0] = 'normal'
y_test = labelencoder.transform(y_test).reshape([-1, 1])
label = labelencoder.classes_.tolist()

'''predict'''
y_pred,y_pred2 = model.predict([x_test,x_test])
y_pred = np.argmax(y_pred, axis=1).reshape([-1, 1])
y_pred2 = np.argmax(y_pred2, axis=1).reshape([-1, 1])
'''Assessment'''
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

print(classification_report(y_true=y_test, y_pred=y_pred))
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
acc = accuracy_score(y_true=y_test, y_pred=y_pred)
'''Visualization'''
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def cm_analysis(cm, labels, x_rotation=90, y_rotation=0):
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
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'True Label'
    cm.columns.name = 'Predict Label'
    sns.set(font_scale=0.5)

    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues')
    plt.xticks(rotation=x_rotation)
    plt.yticks(rotation=y_rotation)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.show()


cm_analysis(cm, label)

'''for binary'''
print(classification_report(y_true=y_test2, y_pred=y_pred2))
cm2 = confusion_matrix(y_true=y_test2, y_pred=y_pred2)
acc2 = accuracy_score(y_true=y_test2, y_pred=y_pred2)

cm_analysis(cm2, ['normal','malfunction'])
