# BUSI Dataset Loading 후 224x224 크기로 resize
# Size --- X_train : 447 x 224 x 224 x 3
# Size --- Y_train : 447 x 224 x 224 x 3 
# Y_train은 rgb channel 모두 같은 값을 가지고 있기 때문에 1개 채널만 따로 사용할 것 --- cv2.split() 사용 

import os
import shutil

#os.mkdir("./sample_data")
#os.mkdir("./sample_mask")

src = './sample/'
data_dir = './sample_data/'
mask_dir = './sample_mask/'
files = os.listdir("./sample")

for idx, file in enumerate(files):
    if 'mask' in file:
        shutil.move(src + file , mask_dir + file)
    else:
        shutil.move(src + file , data_dir + file)

sample_data = os.listdir('./sample_data/')
sample_mask = os.listdir('./sample_mask/')

# 이미지 다 읽어와서 Numpy로 저장하기 

import cv2
import numpy as np

width  = 224
height = 224

x_train = []
y_train = []

for img, msk in zip(sample_data,sample_mask):
        
    image = cv2.imread('./sample_data/'+ img, cv2.IMREAD_COLOR)
    mask  = cv2.imread('./sample_mask/'+ msk, cv2.IMREAD_COLOR)
    
    x = np.zeros((width, height, 3), dtype = np.uint8) # 3 channle input 
    y = np.zeros((width, height, 3), dtype = np.uint8) # 7. Segmentation-Semantic map 1 또는 0으로 mapping할 것
    
    if image.shape[0] >= image.shape[1]: # height > width
        scale = image.shape[0] / height
        new_width = int(image.shape[1] / scale)
        diff = (width - new_width) // 2
        image = cv2.resize(image, (new_width, height))
        mask  = cv2.resize(mask, (new_width, height))
        
        x[:, diff:diff+new_width, :] = image 
        y[:, diff:diff+new_width, :] = mask
            
    elif image.shape[0] < image.shape[1]: # height < width 
        
        scale = image.shape[1] / width
        new_width = int(image.shape[0] / scale)
        diff = (height - new_height) // 2
        image = cv2.resize(image, (width, new_height))
        mask  = cv2.resize(mask, (width, new_height))
        
        x[diff:diff+new_height, :, :] = image 
        y[diff:diff+new_height, :, :] = mask        
        
    x_train.append(x)
    y_train.append(y)
        
    
