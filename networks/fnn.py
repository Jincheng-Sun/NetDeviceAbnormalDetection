import keras
from keras.layers import Dense, Input
from keras.models import load_model
from keras import Model
import numpy as np
from sklearn.externals import joblib
'''build model'''
input = Input(shape=(20,))
layer1 = Dense(64,activation='relu')(input)
layer2 = Dense(128,activation='relu')(layer1)
output = Dense(15,activation='softmax')(layer2)

model = Model(input = input, output = output)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

'''load data'''

x = np.load('../data/train_x.npy')
y = np.load('../data/train_y_all.npy')

'''fill 0 with normal'''

y[y == 0 ] = 'normal'

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

'''turn label into one-hot form'''
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
y = y.reshape([-1,1])
ohencoder = OneHotEncoder()
y = ohencoder.fit_transform(y)

'''save encoders'''

joblib.dump(labelencoder,'../models/labelencoder')
joblib.dump(ohencoder,'../models/OneHotEncoder')
del labelencoder,ohencoder

'''split train-valid set'''
from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=42)
del x,y


'''train'''
from keras.callbacks import EarlyStopping


monitor = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
model.fit(x_train,y_train,batch_size=50,epochs=1000,validation_data=(x_val,y_val),callbacks=[monitor])
model.save('models/fnnmodel')
del model
del x_train,x_val,y_train,y_val




import pandas as pd
'''load testset and encoders'''
x_test=np.load('../data/test_x.npy')
y_test=np.load('../data/test_y_all.npy')
labelencoder = joblib.load('../models/labelencoder')
ohencoder = joblib.load('../models/OneHotEncoder')
model = load_model('../models/fnnmodel')
y_test[y_test == 0 ] = 'normal'
y_test = labelencoder.transform(y_test).reshape([-1,1])
label = labelencoder.classes_.tolist()

'''predict'''
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1).reshape([-1,1])
'''Assessment'''
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

print(classification_report(y_true=y_test,y_pred=y_pred))
cm = confusion_matrix(y_true=y_test,y_pred=y_pred)
acc = accuracy_score(y_true=y_test,y_pred=y_pred)
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