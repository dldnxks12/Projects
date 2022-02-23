
'''
U-Net을 이용한 Segmentaion을 진행 
utils Resize and Save as npy file
'''


import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm  # for 문 상태바 
import glob, os # glob -> 특정 file만 추출하기 

img = cv2.imread('/content/drive/MyDrive/Colab Notebooks/human_segmentation/imgs/03.jpg', cv2.IMREAD_COLOR)
label_img = cv2.imread('/content/drive/MyDrive/Colab Notebooks/human_segmentation/imgs/03.jpg', cv2.IMREAD_GRAYSCALE)

BASE_PATH = '/'
IMG_WIDTH, IMG_HEIGHT = 256, 256
N_CLASSES = 2 

with open('/content/drive/MyDrive/Colab Notebooks/human_segmentation/data/seg_train.txt', 'r') as f:
    train_list = f.readlines()
    
x_train = np.zeros((len(train_list), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
y_train = np.zeros((len(train_list), IMG_HEIGHT, IMG_WIDTH, 2), dtype=np.uint8) # channel 이 2개 -> 2개 색으로 구별할 것

for i, train_path in enumerate(tqdm(train_list)):
    # Image path setting 
    
    img_path = os.path.join(BASE_PATH, train_path.split(' ')[0])
    label_path = os.path.join(BASE_PATH, train_path.split(' ')[-1].strip()) # 60000만 개 
    
    print("img_path",img_path)
    print("label_path",label_path)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    print(img.shape)
    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    
    # result
    im = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3), dtype=np.uint8) # RGC color scale
    lim = np.zeros((IMG_WIDTH, IMG_HEIGHT), dtype=np.uint8)   # grayscale 

    # image size setting -> 256 x 256으로 : 크기가 맞지 않는 부분은 0 값으로 모두 만들어서 붙여주자
    if img.shape[0] >= img.shape[1]:
        scale = img.shape[0] / IMG_HEIGHT
        new_width = int(img.shape[1] / scale)
        diff = (IMG_WIDTH - new_width) // 2
        img = cv2.resize(img, (new_width, IMG_HEIGHT))
        label_img = cv2.resize(label_img, (new_width, IMG_HEIGHT))

        im[:, diff:diff + new_width, :] = img
        lim[:, diff:diff + new_width] = label_img
    else:
        scale = img.shape[1] / IMG_WIDTH
        new_height = int(img.shape[0] / scale)
        diff = (IMG_HEIGHT - new_height) // 2
        img = cv2.resize(img, (IMG_WIDTH, new_height))
        label_img = cv2.resize(label_img, (IMG_WIDTH, new_height))
        im[diff:diff + new_height, :, :] = img
        lim[diff:diff + new_height, :] = label_img

    #label 이미지에 대한 7. Segmentation-Semantic map 생성
    seg_labels = np.zeros((IMG_HEIGHT, IMG_WIDTH, N_CLASSES), dtype=np.uint8)

    for c in range(N_CLASSES):
        seg_labels[:, :, c] = (lim == c).astype(np.uint8)
    
    x_train[i] = im
    y_train[i] = seg_labels
    
np.savez_compressed('data/x_train.npz', data=x_train)
np.savez_compressed('data/y_train.npz', data=y_train)

with open('/content/drive/MyDrive/Colab Notebooks/human_segmentation/data/seg_test.txt', 'r') as f:
    test_list = f.readlines()
    
x_val = np.zeros((len(test_list), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
y_val = np.zeros((len(test_list), IMG_HEIGHT, IMG_WIDTH, 2), dtype=np.uint8)
    
for i, test_path in enumerate(tqdm(test_list)):
    img_path = os.path.join(BASE_PATH, test_path.split(' ')[0])
    label_path = os.path.join(BASE_PATH, test_path.split(' ')[-1].strip())
    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    
    # result
    im = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3), dtype=np.uint8)
    lim = np.zeros((IMG_WIDTH, IMG_HEIGHT), dtype=np.uint8)

    if img.shape[0] >= img.shape[1]:
        scale = img.shape[0] / IMG_HEIGHT
        new_width = int(img.shape[1] / scale)
        diff = (IMG_WIDTH - new_width) // 2
        img = cv2.resize(img, (new_width, IMG_HEIGHT))
        label_img = cv2.resize(label_img, (new_width, IMG_HEIGHT))

        im[:, diff:diff + new_width, :] = img
        lim[:, diff:diff + new_width] = label_img
    else:
        scale = img.shape[1] / IMG_WIDTH
        new_height = int(img.shape[0] / scale)
        diff = (IMG_HEIGHT - new_height) // 2
        img = cv2.resize(img, (IMG_WIDTH, new_height))
        label_img = cv2.resize(label_img, (IMG_WIDTH, new_height))
        im[diff:diff + new_height, :, :] = img
        lim[diff:diff + new_height, :] = label_img
        
    seg_labels = np.zeros((IMG_HEIGHT, IMG_WIDTH, N_CLASSES), dtype=np.uint8)
    for c in range(N_CLASSES):
        seg_labels[:, :, c] = (lim == c).astype(np.uint8)

    x_val[i] = im
    y_val[i] = seg_labels
    
plt.subplot(1, 3, 1)
plt.imshow(x_val[-1])
plt.subplot(1, 3, 2)
plt.imshow(y_val[-1,:,:,0])
plt.subplot(1, 3, 3)
plt.imshow(y_val[-1,:,:,1])


np.savez_compressed('data/x_val.npz', data=x_val)
np.savez_compressed('data/y_val.npz', data=y_val)
