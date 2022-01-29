'''

1. Data가 적을 경우 utils augumentation
2. Weight를 저장하고 로드하는 방법

'''


from google.colab import drive
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Flatten 
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


drive.mount('/content/drive/')
if os.getcwd() != '/content/drive/My Drive':
    os.chdir('./drive/My Drive')


train_images = np.load("/content/drive/MyDrive/Colab Notebooks/MiniProj#1/train.npy")
train_labels = np.load("/content/drive/MyDrive/Colab Notebooks/MiniProj#1/train_label.npy")

train_size = train_images.shape[0] 
train_images =train_images.reshape(train_size, 64, 64, 3)
train_images = train_images / 255.0


train_images, valid_images, train_labels, valid_labels = train_test_split( train_images, train_labels, test_size = 0.3, shuffle = True,  random_state = 1004)
valid_images, test_images, valid_labels, test_labels = train_test_split( train_images, train_labels, test_size = 0.2, shuffle = True,  random_state = 1004)

import random

train_div = train_images
valid_div = valid_images
train_label_div = train_labels
valid_label_div = valid_labels

data_argumentation = Sequential([
                                 layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 layers.experimental.preprocessing.RandomRotation(0.2) 
])

data_argumentation1 = Sequential([
                                 layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 layers.experimental.preprocessing.RandomRotation(0.7) 
                                                                
])

data_argumentation2 = Sequential([
                                 layers.experimental.preprocessing.RandomFlip('horizontal'),
                                 layers.experimental.preprocessing.RandomRotation(0.1),
                                 layers.experimental.preprocessing.RandomZoom(0.2,0.3)  
                                  
])

data_argumentation3 = Sequential([
                                 layers.experimental.preprocessing.RandomFlip('horizontal'),
                                 layers.experimental.preprocessing.RandomRotation(0.5), 
                                 layers.experimental.preprocessing.RandomZoom(0.1,0.5)   
])

data_argumentation4 = Sequential([
                                 layers.experimental.preprocessing.RandomFlip('vertical'),
                                 layers.experimental.preprocessing.RandomRotation(0.5), 
                                 layers.experimental.preprocessing.RandomZoom(0.2,0.3)   
])

data_argumentation5 = Sequential([
                                 layers.experimental.preprocessing.RandomFlip('vertical'),
                                 layers.experimental.preprocessing.RandomRotation(0.5), 
                                 layers.experimental.preprocessing.RandomZoom(0.1,0.5)   
])


train_div1 = data_argumentation(train_div)
valid_div1 = data_argumentation(valid_div)

train_div2 = data_argumentation1(train_div)
valid_div2 = data_argumentation1(valid_div)

train_div3 = data_argumentation2(train_div)
valid_div3 = data_argumentation2(valid_div)

train_div4 = data_argumentation3(train_div)
valid_div4 = data_argumentation3(valid_div)

train_div5 = data_argumentation4(train_div)
train_div6 = data_argumentation5(train_div)


train_images = np.append(train_images, train_div1, axis = 0)
train_labels = np.append(train_labels, train_label_div, axis = 0)
valid_images = np.append(valid_images, valid_div1, axis = 0)
valid_labels = np.append(valid_labels, valid_label_div, axis = 0 )

train_images = np.append(train_images, train_div2, axis = 0)
train_labels = np.append(train_labels, train_label_div, axis = 0)
valid_images = np.append(valid_images, valid_div2, axis = 0)
valid_labels = np.append(valid_labels, valid_label_div, axis = 0 )

train_images = np.append(train_images, train_div3, axis = 0)
train_labels = np.append(train_labels, train_label_div, axis = 0)
valid_images = np.append(valid_images, valid_div3, axis = 0)
valid_labels = np.append(valid_labels, valid_label_div, axis = 0 )

train_images = np.append(train_images, train_div4, axis = 0)
train_labels = np.append(train_labels, train_label_div, axis = 0)
valid_images = np.append(valid_images, valid_div4, axis = 0)
valid_labels = np.append(valid_labels, valid_label_div, axis = 0 )

train_images = np.append(train_images, train_div5, axis = 0)
train_labels = np.append(train_labels, train_label_div, axis = 0)

train_images = np.append(train_images, train_div6, axis = 0)
train_labels = np.append(train_labels, train_label_div, axis = 0)

print(train_images.shape)
print(train_labels.shape)

print(valid_images.shape)
print(valid_labels.shape)

enc= OneHotEncoder()
train_labels_new = train_labels.reshape(-1,1) 
enc.fit(train_labels_new) 

train_labels = np.array(enc.transform(train_labels_new).toarray())

valid_labels_new = valid_labels.reshape(-1,1) 
enc.fit(valid_labels_new) 

valid_labels = np.array(enc.transform(valid_labels_new).toarray())
with tf.device('/gpu:0'):  

  characters = 20 
  Custom_Model = Sequential()  # 새 모델 객체 생성 

  
  Custom_Model.add(layers.Conv2D(32, (3, 3), padding = 'same', activation='relu', input_shape=(64,64,3)))
  Custom_Model.add(layers.MaxPooling2D(pool_size=(2, 2)))

  Custom_Model.add(layers.Conv2D(64, (3, 3), padding = 'same', activation='relu'))
  Custom_Model.add(layers.MaxPooling2D(pool_size=(2, 2)))

  Custom_Model.add(layers.Conv2D(128, (3, 3), padding = 'same', activation='relu'))
  Custom_Model.add(layers.MaxPooling2D(pool_size=(2, 2)))

  Custom_Model.add(layers.Conv2D(512, (3, 3), padding = 'same', activation='relu'))
  Custom_Model.add(layers.MaxPooling2D(pool_size=(2, 2)))

  Custom_Model.add(layers.Flatten()) 
  
  Custom_Model.add(layers.Dense(4096, activation='relu'))

  Custom_Model.add(layers.Dropout(0.5))

  Custom_Model.add(layers.Dense(1024, activation='relu'))

  Custom_Model.add(layers.Dense(64, activation='relu'))

  Custom_Model.add(layers.Dense(characters, activation='softmax'))

# ----------------------- train Model ---------------------

with tf.device('/gpu:0'):
  Custom_Model.compile(optimizer='SGD',
                loss='categorical_crossentropy',
                metrics=['accuracy'],
                )
  batch_size = 64

  history=Custom_Model.fit(train_images, train_labels, batch_size=batch_size, epochs=30, verbose=1, validation_data = (valid_images, valid_labels)) 
  
test_labels_new = test_labels.reshape(-1,1) 
enc.fit(test_labels_new) 

test_labels = np.array(enc.transform(test_labels_new).toarray())

with tf.device('/gpu:0'):
  score = Custom_Model.evaluate(test_images, test_labels, verbose = 1)

print(score)  # Score == [ loss , Acc ]

Custom_Model.save("/content/drive/MyDrive/Colab Notebooks/Simpson/saved_model")
