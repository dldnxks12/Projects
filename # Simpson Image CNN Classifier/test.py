from sklearn.preprocessing import OneHotEncoder
import numpy as np
from tensorflow import keras
from keras import models
import tensorflow as tf
from google.colab import drive
import os

def test_accuracy(dataset, model_name):
  #구글드라이브 연결 
  drive.mount('/content/drive/')
  
  #이미 현재 경로가 설정되있다면
  if os.getcwd() != '/content/drive/My Drive':
    os.chdir('./drive/My Drive')
  print(os.getcwd())
  #원핫인코딩 설정
  origin_labels = np.array([1,2,3])
  
  if os.path.isfile("./train_label.npy"):#Simpson폴더에 없는경우
    origin_labels = np.load("./train_label.npy")
  else : #Simpson폴더에 있는경우
    origin_labels = np.load("./Simpson/train_label.npy")
  enc= OneHotEncoder()
  origin_labels = origin_labels.reshape(-1,1)
  enc.fit(origin_labels)

  images = np.load("./Simpson/"+dataset+".npy")
  labels = np.load("./Simpson/"+dataset+"_label.npy")

  #이미지 전처리
  images = images / 255.0

  #원핫 인코딩
  labels_new = labels.reshape(-1,1)
  labels_onehot = np.array(enc.transform(labels_new).toarray())

  #저장된 모델을 불러옴
  reconstructed_model = keras.models.load_model("./Simpson/"+model_name)
  with tf.device('/gpu:0'):
      score = reconstructed_model.evaluate(images, labels_onehot, verbose=0)

  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

#첫 번째 인자를 수정하세요.
test_accuracy(dataset="test",model_name="saved_model")
