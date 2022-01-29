'''

U Net을 이용한 7. Segmentation-Semantic

'''


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras.layers import Input, concatenate, Dropout, Reshape, Permute, Activation, ZeroPadding2D, Cropping2D, Add
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, DepthwiseConv2D
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_HEIGHT = 256 
IMG_WIDTH = 256
BATCH_SIZE = 16


## Corab의 CPU 제한으로 상당히 줄여서 학습을 진행 
x_train = np.load('/content/drive/MyDrive/Colab Notebooks/human_segmentation/preprocess/x_train.npz')['data'][:10].astype(np.float32)
y_train = np.load('/content/drive/MyDrive/Colab Notebooks/human_segmentation/preprocess/y_train.npz')['data'][:10].astype(np.float32)
#x_val = np.load('/content/drive/MyDrive/Colab Notebooks/human_segmentation/preprocess/x_val.npz')['data'][:1000].astype(np.float32)
#y_val = np.load('/content/drive/MyDrive/Colab Notebooks/human_segmentation/preprocess/y_val.npz')['data'][:1000].astype(np.float32)

plt.figure(figsize=(16,16))

plt.subplot(1, 3, 1)
plt.imshow(x_train[0]/255)
plt.subplot(1, 3, 2)
plt.imshow(y_train[0,:,:,0])
plt.subplot(1, 3, 3)
plt.imshow(y_train[0,:,:,1])

x_train[5] = cv2.cvtColor(x_train[5], cv2.COLOR_BGR2RGB)

plt.figure(figsize=(16,16))
plt.subplot(1, 3, 1)
plt.imshow(x_train[5]/255)
plt.subplot(1, 3, 2)
plt.imshow(y_train[5,:,:,0])
plt.subplot(1, 3, 3)
plt.imshow(y_train[5,:,:,1])

|train_datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.7, 1.3]
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_gen = train_datagen.flow(
    x_train,    
    y_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_gen = val_datagen.flow(
    x_val,
    y_val,
    batch_size=BATCH_SIZE,
    shuffle=False
)

inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# orthogonal convolution 사용 -> Up Sampling에서 좋은 효과를 보인다고 하여 사용해본다. 

# 인코딩 (feature 축소)
conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='orthogonal')(inputs)
conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='orthogonal')(conv1)
pool1 = MaxPooling2D(pool_size=2)(conv1)
# (128, 128, 64)

# For U-Net Upsampling -> 대응되는 layer에서 Concatenate를 통해 더해줄 것 
shortcut_1 = pool1

conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='orthogonal')(pool1)
conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='orthogonal')(conv2)
pool2 = MaxPooling2D(pool_size=2)(conv2)
# (64, 64, 128)
# For U-Net Upsampling -> 대응되는 layer에서 Concatenate를 통해 더해줄 것 
shortcut_2 = pool2

conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='orthogonal')(pool2)
conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='orthogonal')(conv3)
pool3 = MaxPooling2D(pool_size=2)(conv3)
# (32, 32, 256)
# For U-Net Upsampling -> 대응되는 layer에서 Concatenate를 통해 더해줄 것 
shortcut_3 = pool3

# Depth wise Conv -> 보고서에서 따로 설명 
mid = DepthwiseConv2D(3, activation='relu', padding='same', kernel_initializer='orthogonal')(pool3)
mid = Conv2D(256, 1, activation='relu', padding='same', kernel_initializer='orthogonal')(mid)

mid = DepthwiseConv2D(3, activation='relu', padding='same', kernel_initializer='orthogonal')(mid)
mid = Conv2D(256, 1, activation='relu', padding='same', kernel_initializer='orthogonal')(mid)

# layers.add 함수는 같은 크기의 두 개의 tensor를 더 해준다. -> FCN과 비슷 
mid = Add()([shortcut_3, mid])

# 디코딩 

up8 = UpSampling2D(size=2)(mid)
up8 = concatenate([up8, conv3], axis=-1) # concatenate로 pooling 하기 전의 layer (conv)를 이어붙인다.
conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='orthogonal')(up8)
conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='orthogonal')(conv8)
# (64, 64, 128)

conv8 = Add()([shortcut_2, conv8]) 

up9 = UpSampling2D(size=2)(conv8)
up9 = concatenate([up9, conv2], axis=-1) 
conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='orthogonal')(up9)
conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='orthogonal')(conv9)
# (128, 128, 64)

conv9 = Add()([shortcut_1, conv9])

up10 = UpSampling2D(size=2)(conv9)
up10 = concatenate([up10, conv1], axis=-1)
conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='orthogonal')(up10)
conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='orthogonal')(conv10)
# (256, 256, 32)


# pixel을 모두 feature map 크기로 Class 분류하기 위해 2 채널로 만든 후 flatten 
conv11 = Conv2D(2, 1, padding='same', activation='relu',kernel_initializer='he_normal', kernel_regularizer=l2(0.005))(conv10)
conv11 = Reshape((IMG_HEIGHT * IMG_WIDTH, 2))(conv11) 
# (256, 256, 2)

# 최종 출력
conv11 = Activation('softmax')(conv11) 
outputs = Reshape((IMG_HEIGHT, IMG_WIDTH, 2))(conv11)

# Model 객체 생성
model = Model(inputs=inputs, outputs=outputs)
# Segmentation을 위해 Categorical loss 사용
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  

history = model.fit_generator(train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[
        ModelCheckpoint('models/unet_no_drop.h5', monitor='val_acc', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=10, verbose=1, min_lr=1e-05)
    ]
)
