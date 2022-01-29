'''

학습된 모델과 Adain을 통해 얻은 이미지를 이용한 Image Style Transfer 

'''


from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# 학습된 가중치 불러오기 
model = load_model('/content/drive/MyDrive/Colab Notebooks/human_segmentation/models/unet_no_drop.h5')
IMG_PATH = '/content/drive/MyDrive/Colab Notebooks/human_segmentation/imgs/01.jpg'

# cv2에서 이미지는 BGR 채널 순서로 받아오므로 RGB 순서로 바꿔주기 

img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
img_ori = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

plt.figure(figsize=(16, 16))
plt.imshow(img_ori)

img_ori.shape

IMG_WIDTH, IMG_HEIGHT = 256, 256

def preprocess(img): # 256x256 크기로 만들기 

    # Reshape한 이미지를 넣어주기 위해 빈 array 생성 
    im = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3) , dtype=np.uint8)  

    if img.shape[0] >= img.shape[1]: # heignt가 사이즈랑 다르다면 height 조정
        scale = img.shape[0] / IMG_HEIGHT
        new_width = int(img.shape[1] / scale)
        diff = (IMG_WIDTH - new_width) // 2
        img = cv2.resize(img, (new_width, IMG_HEIGHT))

        # 사진을 256x256 크기로 바꾸고, 남는 공간은 모두 0으로 채우기 
        im[:, diff:diff + new_width, :] = img 
    else:
        scale = img.shape[1] / IMG_WIDTH # width가 사이즈랑 다르다면 width 조정
        new_height = int(img.shape[0] / scale)
        diff = (IMG_HEIGHT - new_height) // 2
        img = cv2.resize(img, (IMG_WIDTH, new_height))

        # 사진을 256x256 크기로 바꾸고, 남는 공간은 모두 0으로 채우기 
        im[diff:diff + new_height, :, :] = img
        
    return im

img = preprocess(img)

# 크기 256 x 256 x 3 이미지 get

plt.figure(figsize=(8, 8))
plt.imshow(img)

img.shape

# 0~1 로 정규화 
# Model 에 넣어주기 위해 차원 추가 -> 1 x 256 x 256 x 3 
input_img = img.reshape((1, IMG_WIDTH, IMG_HEIGHT, 3)).astype(np.float32) / 255.
pred = model.predict(input_img)
plt.figure(figsize = (8,8))
plt.subplot(1,2,1)

pred1 = pred.squeeze()[:, :, 1]
plt.imshow(pred1)

plt.subplot(1,2,2)
pred2 = pred.squeeze()[:, :, 0]

THRESHOLD = 0.4
EROSION = 1

# 이미지 후처리 
def DoSegment(img_ori, pred):
    
    # Original 이미지 사이즈로 바꾸기 위해 값 받아오기 : img_ori.shape = (935, 640, 3)
    # h, w = (935, 640)
    h, w = img_ori.shape[:2]  

    # threshold 를 이용해서 0과 1로 값 매칭하기 
    mask_ori = ( pred.squeeze()[:, :, 1] > THRESHOLD ).astype(np.uint8)  

    # 사이즈를 줄일 떄 가장 큰 폭을 기준으로 줄였기 때문에 마찬가지로 큰 폭을 기준으로 늘려주기 
    # 우선 정방형으로 Resize한 다음 Cutting 
    max_size = max(h, w) 
    result_mask = cv2.resize(mask_ori, dsize=(max_size, max_size)) 

    # Size 맞추기 1. Height 기준 2. width 기준 
    if h >= w:
        diff = (max_size - w) // 2
        if diff > 0:
            result_mask = result_mask[:, diff:-diff]
    else:
        diff = (max_size - h) // 2
        if diff > 0:
            result_mask = result_mask[diff:-diff, :]
     
    result_mask = cv2.resize(result_mask, dsize=(w, h))

    # fill holes
    cv2.floodFill(result_mask, mask=np.zeros((h+2, w+2), np.uint8), seedPoint=(0, 0), newVal=255)
    result_mask_fill = cv2.bitwise_not(result_mask)
    result_mask = result_mask_fill

    # eroding : Open -> Close (객체 내의 구멍 막기 but 모서리가 두터워지는 문제가 있으므로 일단은 사용 x)
    # kernal 생성 
    #element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*EROSION + 1, 2*EROSION+1), (EROSION, EROSION)) 
    #result_mask2 = cv2.erode(result_mask, element)
    reuslt_mask = result_mask * 255
    
    # smoothen edges
    result_mask = cv2.GaussianBlur(result_mask, ksize=(9, 9), sigmaX=5, sigmaY=5)
    
    return result_mask_fill, result_mask

fill_mask, mask = DoSegment(img_ori, pred)

plt.figure(figsize=(32, 32))
plt.subplot(1, 3, 1)
plt.imshow(pred[0, :, :, 1])
plt.subplot(1, 3, 2)
plt.imshow(fill_mask)
plt.subplot(1, 3, 3)
plt.imshow(mask)

import imageio

# bit 연산을 위해 mask -> 3 channel BGR로 

converted_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
converted_mask2 = ~converted_mask

plt.figure(figsize=(8, 8))
plt.imshow(converted_mask)

plt.figure(figsize=(8, 8))
plt.imshow(converted_mask2)

result_img = cv2.subtract(converted_mask, img_ori)

plt.figure(figsize=(16, 16))
plt.subplot(1, 2, 1)
plt.imshow(img_ori)
plt.subplot(1, 2, 2)
plt.imshow(result_img)

result_img

result_img2 = - result_img #-> 이런식으로하면 경계값들이 살아나서 이상함

result_img3 = cv2.subtract(converted_mask, result_img) # 배경은 건들지 않으면서 반전된 이미지 다시 반전 

plt.figure(figsize=(16, 16))
plt.subplot(1, 2, 1)
plt.imshow(result_img2)
plt.subplot(1, 2, 2)
plt.imshow(result_img3)

res1 = img_ori - result_img3

plt.figure(figsize = (16,16))
plt.imshow(res1)

imageio.imwrite('/content/drive/MyDrive/Colab Notebooks/human_segmentation/content/img2.png', res1)

returned = '/content/drive/MyDrive/Colab Notebooks/human_segmentation/backstyle/1/output3.png'
img2 = cv2.imread(returned, cv2.IMREAD_COLOR)

plt.figure(figsize = (8,8))
plt.imshow(img2)

img_ori2 = cv2.cvtColor(img2.copy(), cv2.COLOR_BGR2RGB)
img_ori2 = cv2.resize( img_ori2, dsize=(640, 444))

plt.figure(figsize = (8,8))
plt.imshow(img_ori2)

failed_img = img_ori2 + result_img3

plt.figure(figsize = (8,8))
plt.imshow(failed_img)

test_img = cv2.subtract(img_ori2,converted_mask)

plt.figure(figsize = (8,8))
plt.imshow(test_img)

test_img2 = '/content/drive/MyDrive/Colab Notebooks/human_segmentation/c3/output3.png'
test_img2 = cv2.imread(test_img2, cv2.IMREAD_COLOR)

plt.figure(figsize = (8,8))
plt.imshow(test_img2)

test_img2 = cv2.resize( test_img2, dsize=(640, 444))

plt.figure(figsize = (8,8))
plt.imshow(test_img2)

test_img2 = cv2.subtract(test_img2,  converted_mask2)

plt.figure(figsize=(8, 8))
plt.imshow(test_img2)

RESIMG1 = test_img2 + test_img

plt.figure(figsize=(8, 8))
plt.imshow(RESIMG1)

RESIMG2 = test_img + result_img3

plt.figure(figsize=(8, 8))
plt.imshow(RESIMG2)

# image 저장
imageio.imwrite('/content/drive/MyDrive/Colab Notebooks/human_segmentation/Result_img/Result_img2.png', RESIMG) # saves file to png
plt.imshow(pred2)
