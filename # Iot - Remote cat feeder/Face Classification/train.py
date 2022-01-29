
'''


Dataset이 적어 keras의 datagenerator 사용 
총 1500 개의 train set , 500 개의 test 셋으로 basic Cnn 구성 


'''


import tensorflow as tf
import numpy as np
import cv2 
import os
from PIL import Image

os.chdir("/content/drive/MyDrive/Colab Notebooks/Project/Data/kiki") 
filename = os.listdir() # 현재 가리키는 directory 내에 모든 파일 이름 
image_list = [] 

# numpy로 변환 
for i in filename:
  train_image = Image.open(i) # JPG 파일 열기 
  cat_image = np.array(train_image) # Numpy로 형변환
  image_list.append(cat_image)  # list에 추가 
 
np.save("/content/drive/MyDrive/Colab Notebooks/Project/kiki_image", image_list)

os.chdir("/content/drive/MyDrive/Colab Notebooks/Project/Data/kong") 
filename = os.listdir() # 현재 가리키는 directory 내에 모든 파일 이름 

image_list = [] 

# numpy로 변환 
for i in filename:
  train_image = Image.open(i) # JPG 파일 열기 
  cat_image = np.array(train_image) # Numpy로 형변환
  image_list.append(cat_image)  # list에 추가 
  
np.save("/content/drive/MyDrive/Colab Notebooks/Project/kong_image", image_list)
  
  
# Label Tagging 
kiki_list = np.load("/content/drive/MyDrive/Colab Notebooks/Project/kiki_image.npy", allow_pickle= True)
kong_list = np.load("/content/drive/MyDrive/Colab Notebooks/Project/kong_image.npy", allow_pickle= True)

train_label = []

for _ in kiki_list:
  train_label.append('kiki')  
 
for _ in kong_list:
  train_label.append('kong')    

train_label = np.array(train_label)

# kiki data 전처리

kiki_320 = []

for i in kiki_list:
  ni = i

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

  kiki_320.append(im)

  
# kong data 전처리

kong_320 = []

for i in kong_list:
  ni = i

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
      
  kong_320.append(im)
  
kiki_320 = np.array(kiki_320)
kong_320 = np.array(kong_320)

kiki_320 = kiki_320 / 255.0
kong_320 = kong_320 / 255.0

# Kong + KiKi Train set & dimension setting
train_images  = np.concatenate((kiki_320, kong_320), axis = 0)
train_label = np.expand_dims(train_label, axis=1)

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

## Train set , Test set divide

train_images, test_images, train_labels, test_labels = train_test_split( train_images, train_label, test_size = 0.3, shuffle = True,  random_state = 1004)

## data generator 사용해서 늘리기

import keras
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Flatten 
from keras.models import Sequential
from sklearn.model_selection import train_test_split


train_div = train_images
test_div = test_images
train_label_div = train_labels
test_label_div = test_labels

data_argumentation = Sequential([
                                 tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.2) 
])

data_argumentation1 = Sequential([
                                 tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.7) 
                                                                
])

data_argumentation2 = Sequential([
                                 tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.2,0.3)  
                                  
])

data_argumentation3 = Sequential([
                                 tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.5), 
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.1,0.5)   
])

data_argumentation4 = Sequential([
                                 tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.5), 
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.2,0.3)   
])

train_div1 = data_argumentation(train_div)
test_div1 = data_argumentation(test_div)

train_div2 = data_argumentation1(train_div)
test_div2 = data_argumentation1(test_div)

train_div3 = data_argumentation2(train_div)
test_div3 = data_argumentation2(test_div)

train_div4 = data_argumentation3(train_div)
test_div4 = data_argumentation3(test_div)



train_images = np.append(train_images, train_div1, axis = 0)
train_labels = np.append(train_labels, train_label_div, axis = 0)
test_images = np.append(test_images, test_div1, axis = 0)
test_labels = np.append(test_labels, test_label_div, axis = 0 )

train_images = np.append(train_images, train_div2, axis = 0)
train_labels = np.append(train_labels, train_label_div, axis = 0)
test_images = np.append(test_images, test_div2, axis = 0)
test_labels = np.append(test_labels, test_label_div, axis = 0 )

train_images = np.append(train_images, train_div3, axis = 0)
train_labels = np.append(train_labels, train_label_div, axis = 0)
test_images = np.append(test_images, test_div3, axis = 0)
test_labels = np.append(test_labels, test_label_div, axis = 0 )

train_images = np.append(train_images, train_div4, axis = 0)
train_labels = np.append(train_labels, train_label_div, axis = 0)
test_images = np.append(test_images, test_div4, axis = 0)
test_labels = np.append(test_labels, test_label_div, axis = 0 )


print(train_images.shape)
print(train_labels.shape)

print(test_images.shape)
print(test_labels.shape)

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
train_labels_new = train_labels.reshape(-1,1)
enc.fit(train_labels_new) # label one-hot encoding

train_labels_onehot = np.array(enc.transform(train_labels_new).toarray())


Custom_Model = keras.Sequential()
character = 2

Custom_Model.add(layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu', input_shape = (128,128,3)))
Custom_Model.add(layers.MaxPool2D(pool_size=(2,2)))

Custom_Model.add(layers.Conv2D(64,(3,3), padding = 'same', activation = 'relu'))
Custom_Model.add(layers.MaxPool2D(pool_size=(2,2)))

Custom_Model.add(layers.Conv2D(128,(3,3), padding = 'same', activation = 'relu'))
Custom_Model.add(layers.MaxPool2D(pool_size=(2,2)))

Custom_Model.add(layers.Conv2D(512,(3,3), padding = 'same', activation = 'relu'))
Custom_Model.add(layers.MaxPool2D(pool_size=(2,2)))

Custom_Model.add(layers.Flatten())
Custom_Model.add(layers.Dense(4096, activation = 'relu'))

Custom_Model.add(layers.Dense(1024, activation = 'relu'))

Custom_Model.add(layers.Dense(64, activation = 'relu'))

Custom_Model.add(layers.Dense(character, activation = 'softmax'))

Custom_Model.compile(optimizer = 'SGD',
                     loss = 'categorical_crossentropy',
                     metrics=['accuracy'],
                     )

batch_size = 64

history = Custom_Model.fit(train_images, train_labels_onehot, batch_size= batch_size , epochs = 30, verbose = 1 )

test_sample = np.expand_dims(test_images[400], axis = 0)
pred = Custom_Model.predict(test_sample)
print(pred)

decoded = enc.inverse_transform(pred)
print(decoded)

## 가중치 Save
Custom_Model.save("/content/drive/MyDrive/Colab Notebooks/Project/cat_recognize")
