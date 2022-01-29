import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
import tensorflow as tf
from keras.models import save_model, load_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import keras

df_train = pdr.DataReader('005930.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))
df_train2 = pdr.DataReader('035420.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))
df_train3 = pdr.DataReader('034730.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21)) 
df_train4 = pdr.DataReader('051910.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))
df_train10 = pdr.DataReader('005380.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))

df_train11 = pdr.DataReader('012330.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))
df_train12 = pdr.DataReader('010140.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))
df_train13 = pdr.DataReader('006405.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))
df_train14 = pdr.DataReader('204320.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))
df_train15 = pdr.DataReader('000270.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))
df_train16 = pdr.DataReader('010620.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))
df_train17 = pdr.DataReader('034020.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))
df_train18 = pdr.DataReader('068760.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))
df_train19 = pdr.DataReader('033780.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))

df_train20 = pdr.DataReader('001360.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))
df_train24 = pdr.DataReader('010950.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))
df_train25 = pdr.DataReader('251270.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))
df_train26 = pdr.DataReader('051600.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))
df_train27 = pdr.DataReader('047040.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))
df_train28 = pdr.DataReader('010620.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))
df_train29 = pdr.DataReader('097950.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))
df_train30 = pdr.DataReader('009540.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))


df_test = pdr.DataReader('005930.KS','yahoo',datetime(2000,1,1), datetime(2020,5,21))

training_set1 = df_train.iloc[:, 3:4].values
training_set2 = df_train2.iloc[:, 3:4].values  # df_train의 3번 째 Columne 선택
training_set3 = df_train3.iloc[:, 3:4].values
training_set4 = df_train4.iloc[:, 3:4].values
training_set10 = df_train10.iloc[:, 3:4].values
training_set11 = df_train11.iloc[:, 3:4].values
training_set12 = df_train12.iloc[:, 3:4].values
training_set13 = df_train13.iloc[:, 3:4].values
training_set14 = df_train14.iloc[:, 3:4].values
training_set15 = df_train15.iloc[:, 3:4].values
training_set16 = df_train16.iloc[:, 3:4].values
training_set17 = df_train17.iloc[:, 3:4].values
training_set18 = df_train18.iloc[:, 3:4].values
training_set19 = df_train19.iloc[:, 3:4].values
training_set20 = df_train20.iloc[:, 3:4].values
training_set24 = df_train24.iloc[:, 3:4].values
training_set25 = df_train25.iloc[:, 3:4].values
training_set26 = df_train26.iloc[:, 3:4].values
training_set27 = df_train27.iloc[:, 3:4].values
training_set28 = df_train28.iloc[:, 3:4].values
training_set29 = df_train29.iloc[:, 3:4].values
training_set30 = df_train30.iloc[:, 3:4].values

test_set = df_test.iloc[:, 3:4].values

training_set = np.append(training_set1, training_set2, axis = 0)
training_set = np.append(training_set, training_set3, axis = 0)
training_set = np.append(training_set, training_set4, axis = 0)
training_set = np.append(training_set, training_set10, axis = 0)
training_set = np.append(training_set, training_set11, axis = 0)
training_set = np.append(training_set, training_set12, axis = 0)
training_set = np.append(training_set, training_set13, axis = 0)
training_set = np.append(training_set, training_set14, axis = 0)
training_set = np.append(training_set, training_set15, axis = 0)
training_set = np.append(training_set, training_set16, axis = 0)
training_set = np.append(training_set, training_set17, axis = 0)
training_set = np.append(training_set, training_set18, axis = 0)
training_set = np.append(training_set, training_set19, axis = 0)
training_set = np.append(training_set, training_set20, axis = 0)
training_set = np.append(training_set, training_set24, axis = 0)
training_set = np.append(training_set, training_set25, axis = 0)
training_set = np.append(training_set, training_set26, axis = 0)
training_set = np.append(training_set, training_set27, axis = 0)
training_set = np.append(training_set, training_set28, axis = 0)
training_set = np.append(training_set, training_set29, axis = 0)
training_set = np.append(training_set, training_set30, axis = 0)


print(training_set.shape)
print(test_set.shape)


sc1 = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc1.fit_transform(training_set)
sc2 = MinMaxScaler(feature_range=(0,1))
test_set_scaled = sc2.fit_transform(test_set)

X_train = []
Y_train = []
for i in range(10, training_set_scaled.shape[0]):  # 30일 선으로 
    X_train.append(training_set_scaled[i-10 : i, 0]) 
    Y_train.append(training_set_scaled[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_test = []
Y_test = []
for i in range(10, test_set_scaled.shape[0]):
    X_test.append(test_set_scaled[i-10:i, 0])
    Y_test.append(test_set_scaled[i, 0])
X_test, Y_test = np.array(X_test), np.array(Y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

Custom = Sequential()
Custom.add(LSTM(10,  input_shape = (10,1), return_sequences=False))
Custom.add(Dense(1))
Custom.summary()

Custom.compile(optimizer = 'adam', loss = 'mean_squared_error')
Custom.fit(X_train, Y_train, epochs = 70, batch_size = 128)

predicted_X_train = Custom.predict(X_train)
predicted_X_train = sc1.inverse_transform(predicted_X_train)
predicted_X_test = Custom.predict(X_test)
predicted_X_test = sc2.inverse_transform(predicted_X_test)

mean_train_error = np.average(np.abs(training_set[9:-1,:] - predicted_X_train))
mean_test_error = np.average(np.abs(test_set[9:-1,:] - predicted_X_test))

print("평균 training error:", mean_train_error)
print("평균 test error:",mean_test_error)

plt.figure(figsize = (16,16))

plt.subplot(2, 1, 1)
plt.plot(training_set[9:-1,:], color = 'black', label = 'GroundTruth')
plt.plot(predicted_X_train, color = 'red', label = 'Prediction')
plt.title('Naver Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.figure(figsize = (16,16))

plt.subplot(2, 1, 2)
plt.plot(test_set[9:-1,:], color = 'black', label = 'GroundTruth')
plt.plot(predicted_X_test, color = 'red', label = 'Prediction')
plt.title('Naver Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

Custom.save('drive/MyDrive/Corab notebook/이종수_2014741002.h5')
