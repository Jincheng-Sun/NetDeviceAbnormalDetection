from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, TimeDistributed, Activation, SimpleRNN
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping
import numpy as np

# features_train = np.load('data/OTM4_features_96x6_train.npy')
# targets_train = np.load('data/OTM4_targets_96x6_train.npy')
#
# features_test = np.load('data/OTM4_features_96x6_test.npy')
# targets_test = np.load('data/OTM4_targets_96x6_test.npy')
#
# # train one target
# index = 0
# targets_train = targets_train[:, :, index]
# targets_train = np.expand_dims(targets_train, -1)
# targets_test = targets_test[:, :, index]
# targets_test = np.expand_dims(targets_test, -1)
#
#
# # design network
# model = Sequential()
# model.add(LSTM(units=64, input_shape=(features_train.shape[1], features_train.shape[2]), return_sequences=True))
# model.add(Dropout(0.2))
# # model.add(LSTM(units=128, return_sequences=True))
# # model.add(Dropout(0.2))
# # model.add(TimeDistributed(Dense(features_train.shape[2])))
# # model.add(TimeDistributed(Dense(1)))
# model.add(Dense(1))
#
# # model.add(Activation("tanh"))
#
# print(model.summary())
#
# model.compile(loss='mae', optimizer='rmsprop')
# # fit network
#
# early_stop = EarlyStopping(patience=5, restore_best_weights=True)
# history = model.fit(features_train, targets_train, epochs=100, batch_size=50,
#                     validation_split=0.2, shuffle=True, callbacks=[early_stop]
#                     )
# # 2 epoches
#
# from matplotlib import pyplot
# # plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()
#
# model.save('models/OTM4_model')

# design network
# timesteps = 10
# features = 5
# model = Sequential()
# model.add(SimpleRNN(20, input_shape=(timesteps, features), return_sequences=True))
# model.add(Dropout(0.2))
# # model.add(LSTM(units=128, return_sequences=True))
# # model.add(Dropout(0.2))
# # model.add(TimeDistributed(Dense(6)))
# model.add(Dense(6))
# # model.add(Activation("tanh"))
#
# print(model.summary())
#
# model = Sequential()
# model.add(Dense(4, input_shape=(10, 5)))
# print(model.summary())
