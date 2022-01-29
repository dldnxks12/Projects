
'''

1. Weight load -> Pickle Setting (Security 문제로 Pickle False로 되어있으면 np.load로 불러들여지지 않음)
2. Data 전처리 후 Face Classification 모델에 넣고 Prediction Check

'''


import numpy as np
import cv2
import keras
from keras import models
from keras import layers
from PIL import Image
import os 

sample = Image.open("/content/drive/MyDrive/Colab Notebooks/Project/Sample/6.jpg") 
sample = np.array(sample)
reconstruct = keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/Project/cat_recognize") # 가중치 불러오기 -> 경로 재설정 필수 

# data 전처리

ni = sample

im = np.zeros((128,128,3) , dtype = np.uint8)

if ni.shape[0] >= ni.shape[1]:
    scale = ni.shape[0] / 128
    new_width = int(ni.shape[1] / scale)
    diff = (128-new_width) // 2
    ni = cv2.resize(ni, (new_width, 128))
    im[:, diff:diff+new_width,:] = ni

else:
    scale = ni.shape[1] / 128
    new_height = int(ni.shape[0] / scale)
    diff = (128 - new_height) // 2
    ni = cv2.resize(ni, (128, new_height))  
    im[diff : diff + new_height, :, :] = ni

sample = im
sample = sample /255.0

train_sample = np.expand_dims(sample, axis = 0) # 1개씩 predict 할 때 차원 맞춰주기

pred = reconstruct.predict(train_sample)

print(pred)
