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

# 모델 불러오기 

recent_model = load_model('drive/MyDrive/Corab notebook/이종수_2014741002.h5')

predicted_X_train = recent_model.predict(X_train)
predicted_X_train = sc1.inverse_transform(predicted_X_train)

predicted_X_test = recent_model.predict(X_test)
predicted_X_test = sc2.inverse_transform(predicted_X_test)

mean_train_error = np.average(np.abs(training_set[9:-1,:] - predicted_X_train))
mean_test_error = np.average(np.abs(test_set[9:-1,:] - predicted_X_test))


print("평균 training error:", mean_train_error)
print("평균 test error:",mean_test_error)

plt.subplot(2, 1, 1)
plt.plot(training_set[9:-1,:], color = 'blue', label = 'GroundTruth')
plt.plot(predicted_X_train, color = 'red', label = 'Prediction')
plt.title('Naver Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.subplot(2, 1, 2)
plt.plot(test_set[9:-1,:], color = 'blue', label = 'GroundTruth')
plt.plot(predicted_X_test, color = 'red', label = 'Prediction')
plt.title('Naver Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
