'''

64 x 64 x 3 크기로 학습시켜줄 것이기 떄문에 모든 사진을 해당 크기로 바꿔주는 과정이 필요 
남는 여백은 모두 0의 값으로 채운다.

'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

train_images = Image.open("3.jpg")

ni = np.array(train_images)

print(ni.shape)


im = np.zeros( (64,64,3) , dtype = np.uint8)

if ni.shape[0] >= ni.shape[1]:
    scale = ni.shape[0] / 64
    new_width = int(ni.shape[1] / scale)
    diff = (64-new_width) // 2
    ni = cv2.resize(ni, (new_width, 64))
    print(ni.shape)
    im[:, diff:diff+new_width,:] = ni

else:
    scale = ni.shape[1] / 64
    new_height = int(ni.shape[0] / scale)
    diff = (64 - new_height) // 2
    ni = cv2.resize(ni, (64, new_height))
    
    print(ni.shape)
    im[diff : diff + new_height, :, :] = ni
    
    
print(im.shape)    
plt.imshow(im)
    
