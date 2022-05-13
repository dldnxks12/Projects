import keras
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.models import save_model, load_model

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler



### START CODE HERE ###
df_train1  = fdr.DataReader("TSLA", start = "2010-07-10", end = "2021-12-31") 
df_train2  = fdr.DataReader("AAPL", start = "2000-01-1", end = "2021-12-31")  
df_train3  = fdr.DataReader("KO", start = "2000-01-1", end = "2021-12-31")    
df_train4  = fdr.DataReader("035420", start = "2003-01-1", end = "2021-12-31") 
df_train5  = fdr.DataReader("005380", start = "2000-01-1", end = "2021-12-31") 
df_train6  = fdr.DataReader("012330", start = "2000-01-1", end = "2021-12-31") 
df_train7  = fdr.DataReader("GOOG", start = "2005-01-1", end = "2021-12-31") 
df_train8  = fdr.DataReader("005930", start = "2001-01-1", end = "2021-12-31") 
df_train9  = fdr.DataReader("035720", start = "2001-01-1", end = "2021-12-31") 
df_train10 = fdr.DataReader("MSFT", start = "2000-01-1", end = "2021-12-31") 
df_train11 = fdr.DataReader("ASML", start = "2000-01-1", end = "2021-12-31") 
df_train12 = fdr.DataReader("010140", start = "2000-01-1", end = "2021-12-31")
df_train13 = fdr.DataReader("006405", start = "2000-01-1", end = "2021-12-31")
df_train14 = fdr.DataReader("204320", start = "2000-01-1", end = "2021-12-31")
df_train15 = fdr.DataReader("000270", start = "2000-01-1", end = "2021-12-31")
df_train16 = fdr.DataReader("010620", start = "2000-01-1", end = "2021-12-31")
df_train17 = fdr.DataReader("034020", start = "2000-01-1", end = "2021-12-31")
df_train18 = fdr.DataReader("068760", start = "2000-01-1", end = "2021-12-31")
df_train19 = fdr.DataReader("033780", start = "2000-01-1", end = "2021-12-31")
df_train20 = fdr.DataReader("001360", start = "2000-01-1", end = "2021-12-31")
df_train21 = fdr.DataReader("010950", start = "2000-01-1", end = "2021-12-31")
df_train22 = fdr.DataReader("251270", start = "2000-01-1", end = "2021-12-31") 
df_train23 = fdr.DataReader("051600", start = "2000-01-1", end = "2021-12-31") 
df_train24 = fdr.DataReader("047040", start = "2000-01-1", end = "2021-12-31") 
df_train25 = fdr.DataReader("NVDA", start = "2000-01-1", end = "2021-12-31") 
df_train26 = fdr.DataReader("097950", start = "2000-01-1", end = "2021-12-31")
df_train27 = fdr.DataReader("009540", start = "2000-01-1", end = "2021-12-31")
df_train28 = fdr.DataReader("034730", start = "2000-01-1", end = "2021-12-31")
df_train29 = fdr.DataReader("V", start = "2009-01-1", end = "2021-12-31") 
df_train30 = fdr.DataReader("NKE", start = "2000-01-1", end = "2021-12-31")
df_train31 = fdr.DataReader("SBUX", start = "2000-01-1", end = "2021-12-31") 
df_train32 = fdr.DataReader("MA", start = "2007-01-1", end = "2021-12-31") 
df_train33 = fdr.DataReader("BABA", start = "2015-01-1", end = "2021-12-31") 
df_train34 = fdr.DataReader("ADBE", start = "2000-01-1", end = "2021-12-31")
df_train35 = fdr.DataReader("DIS", start = "2000-01-1", end = "2021-12-31") 
df_train36 = fdr.DataReader("BA", start = "2000-01-1", end = "2021-12-31") 
df_train37 = fdr.DataReader("BIDU", start = "2006-01-1", end = "2021-12-31") 
df_train38 = fdr.DataReader("PLUG", start = "2000-01-1", end = "2021-12-31") 
df_train39 = fdr.DataReader("CCL", start = "2000-01-1", end = "2021-12-31") 
df_train40 = fdr.DataReader("GM", start = "2011-01-1", end = "2021-12-31") 

training_set1  = df_train1.iloc[:, 3:4].values
training_set2  = df_train2.iloc[:, 3:4].values
training_set3  = df_train3.iloc[:, 3:4].values
training_set4  = df_train4.iloc[:, 3:4].values
training_set5  = df_train5.iloc[:, 3:4].values
training_set6  = df_train6.iloc[:, 3:4].values
training_set7  = df_train7.iloc[:, 3:4].values
training_set8  = df_train8.iloc[:, 3:4].values
training_set9  = df_train9.iloc[:, 3:4].values
training_set10 = df_train10.iloc[:,3:4].values
training_set11 = df_train11.iloc[:,3:4].values
training_set12 = df_train12.iloc[:,3:4].values
training_set13 = df_train13.iloc[:,3:4].values
training_set14 = df_train14.iloc[:,3:4].values
training_set15 = df_train15.iloc[:,3:4].values
training_set16 = df_train16.iloc[:,3:4].values
training_set17 = df_train17.iloc[:,3:4].values
training_set18 = df_train18.iloc[:,3:4].values
training_set19 = df_train19.iloc[:,3:4].values
training_set20 = df_train20.iloc[:,3:4].values
training_set21 = df_train21.iloc[:,3:4].values
training_set22 = df_train22.iloc[:,3:4].values
training_set23 = df_train23.iloc[:,3:4].values
training_set24 = df_train24.iloc[:,3:4].values
training_set25 = df_train25.iloc[:,3:4].values
training_set26 = df_train26.iloc[:,3:4].values
training_set27 = df_train27.iloc[:,3:4].values
training_set28 = df_train28.iloc[:,3:4].values
training_set29 = df_train29.iloc[:,3:4].values
training_set30 = df_train30.iloc[:,3:4].values
training_set31 = df_train31.iloc[:,3:4].values
training_set32 = df_train32.iloc[:,3:4].values
training_set33 = df_train33.iloc[:,3:4].values
training_set34 = df_train34.iloc[:,3:4].values
training_set35 = df_train35.iloc[:,3:4].values
training_set36 = df_train36.iloc[:,3:4].values
training_set37 = df_train37.iloc[:,3:4].values
training_set38 = df_train38.iloc[:,3:4].values
training_set39 = df_train39.iloc[:,3:4].values
training_set40 = df_train40.iloc[:,3:4].values

print(training_set1.shape)
print(training_set2.shape)
print(training_set3.shape)
print(training_set4.shape)
print(training_set5.shape)
print(training_set6.shape)
print(training_set7.shape)
print(training_set8.shape)
print(training_set9.shape)
print(training_set10.shape)
print(training_set11.shape)
print(training_set12.shape)
print(training_set13.shape)
print(training_set14.shape)
print(training_set15.shape)
print(training_set16.shape)
print(training_set17.shape)
print(training_set18.shape)
print(training_set19.shape)
print(training_set20.shape)
print(training_set21.shape)
print(training_set22.shape)
print(training_set23.shape)
print(training_set24.shape)
print(training_set25.shape)
print(training_set26.shape)
print(training_set27.shape)
print(training_set28.shape)
print(training_set29.shape)
print(training_set30.shape)
print(training_set31.shape)
print(training_set32.shape)
print(training_set33.shape)
print(training_set34.shape)
print(training_set35.shape)
print(training_set36.shape)
print(training_set37.shape)
print(training_set38.shape)
print(training_set39.shape)
print(training_set40.shape)

training_set = np.append(training_set1, training_set2,  axis = 0)
training_set = np.append(training_set,  training_set3,  axis = 0)
training_set = np.append(training_set,  training_set4,  axis = 0)
training_set = np.append(training_set,  training_set5,  axis = 0)
training_set = np.append(training_set,  training_set6,  axis = 0)
training_set = np.append(training_set,  training_set7,  axis = 0)
training_set = np.append(training_set,  training_set8,  axis = 0)
training_set = np.append(training_set,  training_set9,  axis = 0)
training_set = np.append(training_set,  training_set10, axis = 0)
training_set = np.append(training_set,  training_set11, axis = 0)
training_set = np.append(training_set,  training_set12, axis = 0)
training_set = np.append(training_set,  training_set13, axis = 0)
training_set = np.append(training_set,  training_set14, axis = 0)
training_set = np.append(training_set,  training_set15, axis = 0)
training_set = np.append(training_set,  training_set16, axis = 0)
training_set = np.append(training_set,  training_set17, axis = 0)
training_set = np.append(training_set,  training_set18, axis = 0)
training_set = np.append(training_set,  training_set19, axis = 0)
training_set = np.append(training_set,  training_set20, axis = 0)
training_set = np.append(training_set,  training_set21, axis = 0)
training_set = np.append(training_set,  training_set22, axis = 0)
training_set = np.append(training_set,  training_set23, axis = 0)
training_set = np.append(training_set,  training_set24, axis = 0)
training_set = np.append(training_set,  training_set25, axis = 0)
training_set = np.append(training_set,  training_set26, axis = 0)
training_set = np.append(training_set,  training_set27, axis = 0)
training_set = np.append(training_set,  training_set28, axis = 0)
training_set = np.append(training_set,  training_set29, axis = 0)
training_set = np.append(training_set,  training_set30, axis = 0)
training_set = np.append(training_set,  training_set31, axis = 0)
training_set = np.append(training_set,  training_set32, axis = 0)
training_set = np.append(training_set,  training_set33, axis = 0)
training_set = np.append(training_set,  training_set34, axis = 0)
training_set = np.append(training_set,  training_set35, axis = 0)
training_set = np.append(training_set,  training_set36, axis = 0)
training_set = np.append(training_set,  training_set37, axis = 0)
training_set = np.append(training_set,  training_set38, axis = 0)
training_set = np.append(training_set,  training_set39, axis = 0)
training_set = np.append(training_set,  training_set40, axis = 0)

df_test = fdr.DataReader("AMD", start = "2001-01-1", end = "2021-12-31") 
test_set = df_test.iloc[:, 3:4].values

print(training_set.shape)
print(test_set.shape)

sc1 = MinMaxScaler(feature_range=(0,1))
sc2 = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc1.fit_transform(training_set)
test_set_scaled     = sc2.fit_transform(test_set)

X_train = []
Y_train = []

for i in range(10, training_set_scaled.shape[0]):  # 10일 선으로 
    X_train.append(training_set_scaled[i-10 : i, 0]) 
    Y_train.append(training_set_scaled[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_test = []
Y_test = []
for i in range(10, test_set_scaled.shape[0]):     # 10일 선으로 
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

plt.figure(figsize = (8,8))
plt.subplot(2, 1, 1)
plt.plot(training_set[9:-1,:], color = 'black', label = 'GroundTruth')
plt.plot(predicted_X_train, color = 'red', label = 'Prediction')
plt.title('Naver Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.figure(figsize = (8,8))
plt.subplot(2, 1, 2)
plt.plot(test_set[9:-1,:], color = 'black', label = 'GroundTruth')
plt.plot(predicted_X_test, color = 'red', label = 'Prediction')
plt.title('Naver Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

Custom.save('/content/drive/MyDrive/Colab Notebooks/2022-1 딥러닝/이종수_2014741002.h5')
