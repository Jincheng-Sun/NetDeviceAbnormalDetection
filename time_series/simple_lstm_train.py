from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, TimeDistributed, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping
import numpy as np

features_train = np.load('data/OTM4_features_96x6_train.npy')
targets_train = np.load('data/OTM4_targets_96x6_train.npy')

features_test = np.load('data/OTM4_features_96x6_test.npy')
targets_test = np.load('data/OTM4_targets_96x6_test.npy')


# design network
model = Sequential()
model.add(LSTM(units=64, input_shape=(features_train.shape[1], features_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
# model.add(LSTM(units=128, return_sequences=True))
# model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(features_train.shape[2])))
# model.add(Activation("tanh"))

print(model.summary())
model.compile(loss='mae', optimizer='rmsprop')
# fit network

early_stop = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(features_train, targets_train, epochs=100, batch_size=50,
                    validation_split=0.2, shuffle=True, callbacks=[early_stop]
                    )
# 2 epoches

from matplotlib import pyplot
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

model.save('models/OTM4_model')
