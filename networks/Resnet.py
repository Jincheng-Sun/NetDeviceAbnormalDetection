from keras.layers import Conv1D, BatchNormalization, Activation, Dropout, add, Input, Dense, Flatten
from keras import Model
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import RandomOverSampler
from keras.optimizers import Adam
def bn_relu(layer, dropout=0, **params):
    layer = BatchNormalization()(layer)
    layer = Activation(params['conv_activation'])(layer)

    if dropout > 0:
        layer = Dropout(dropout)(layer)
    return layer


def resnet_block(layer, filters, kernels, dropout, activation,
                 cross_block=False, is_first=False, is_last=False, shrink=False):
    # -BN-Act-Conv-BN-Act-Conv--
    # ↳-----------------------↑
    strides = 1
    if shrink:
        strides = 2

    if cross_block:

        shortcut = Conv1D(filters=filters,
                          kernel_size=strides,
                          kernel_initializer='random_uniform',
                          # kernel_regularizer=regularizers.l2(0.01),
                          strides=strides,
                          padding='same')(layer)
    else:
        shortcut = layer

    if not is_first:
        layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    layer = Conv1D(filters=filters,
                   kernel_size=kernels,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=strides,
                   padding='same')(layer)
    layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    layer = Conv1D(filters=filters,
                   kernel_size=kernels,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=1,
                   padding='same')(layer)
    layer = add([shortcut, layer])

    if is_last:
        layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    return layer


'''global variant'''

ecd = 'None'
'''load dataset'''
train_x = np.load('../data/new/%s_train_x_bi.npy' % ecd)
train_y = pd.DataFrame(np.load('../data/new/%s_train_y_bi.npy' % ecd))
# train_y = np.load('../data/new/%s_train_y_all.npy' % ecd)
le_encoder = joblib.load('../models/LE_ALM')
label = le_encoder.classes_

'''split dataset'''
shape = 56
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

train_y = pd.get_dummies(train_y.iloc[:, 0])
# train_y2 = np.reshape(train_y,[-1,1,14])
# train_y = np.reshape(train_y.values,[-1,1,2])

'''upsampling '''
# sampler = RandomOverSampler(random_state=42)
# train_x, train_y = sampler.fit_resample(train_x, train_y)


train_x = np.reshape(train_x, [-1, 1, shape])
test_x = np.reshape(test_x, [-1, 1, shape])
'''create model'''
input = Input(shape=(1, shape))
layer = Conv1D(filters=32,
               kernel_size=3,
               kernel_initializer='random_uniform',
               # kernel_regularizer=regularizers.l2(0.01),
               strides=1,
               padding='same')(input)
layer = resnet_block(layer=layer, filters=32, kernels=3, dropout=0, activation='relu')

layer = resnet_block(layer, 64, 3, 0, 'relu', cross_block=True, shrink=True)
layer = resnet_block(layer, 64, 3, 0, 'relu')
#
layer = resnet_block(layer, 128, 3, 0, 'relu', cross_block=True, shrink=True)
layer = resnet_block(layer, 128, 3, 0, 'relu')

layer = resnet_block(layer, 256, 3, 0, 'relu', cross_block=True, shrink=True)
layer = resnet_block(layer, 256, 3, 0, 'relu')
layer = Flatten()(layer)
output = Dense(units=2, activation='softmax')(layer)

model = Model(inputs=[input], outputs=[output])

# input = Input(shape=(56,))
# layer1 = Dense(64,activation='relu')(input)
# layer2 = Dense(128,activation='relu')(layer1)
# output = Dense(2,activation='softmax')(layer2)
#
# model = Model(input = input, output = output)
optimizer = Adam(lr=0.005)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
monitor = EarlyStopping(monitor='val_acc', min_delta=1e-3, patience=3, mode='auto', restore_best_weights=True)

model.fit(train_x, train_y,
          batch_size=50,
          epochs=1000,
          validation_split=0.1,
          callbacks=[monitor])

model.save('../models/newmodels/model_%s_bi' % ecd)

'''validation'''
# test
# abnormal    4653
# normal    4574
y_pred = model.predict(test_x)
y_pred = np.argmax(y_pred, axis=1).reshape([-1, 1])
y_test = test_y.values

'''Assessment'''

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

print(classification_report(y_true=y_test, y_pred=y_pred))
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
acc = accuracy_score(y_true=y_test, y_pred=y_pred)

'''Visualization'''

from toolPackage.draw_cm import cm_analysis

cm_analysis(cm, ['Normal', 'Malfunction'], precision=True)
# cm_analysis(cm, label, precision=False,x_rotation=90)
