'''

Opencv의 cv2.imread() 사용 시 Numpy type으로 변환되어 Loading된다.

If PIL 의 Image.open() 사용 시에는 추가적으로 numpy type으로 변환하는 과정이 필요하다

'''

import numpy as np
import os # 경로 파악 
from PIL import Image # JPG 이미지 불러올 무듈  


# 현재 directory로 바꾸기
os.chdir("/content/drive/MyDrive/Colab Notebooks/Project/kiki") 
filename = os.listdir() # 현재 가리키는 directory 내에 모든 파일 이름 

image_list = [] 

for i in filename:
  train_image = Image.open(i) # JPG 파일 열기 
  numpy_image = np.array(train_image) # Numpy로 형변환
  image_list.append(numpy_image)  # list에 추가 
  
  
# Check
import matplotlib.pyplot as plt

plt.imshow(image_list[1])  


# npy file로 저장 -> 경로 재설정할 것 
np.save("/content/drive/MyDrive/Colab Notebooks/Project/kiki/numpy_image", image_list)

# Making label 
kiki_list = np.load(kiki_folder)
kong_list = np.load(kong_folder)

test_label = []

for _ in kiki_list:
  test_label.append('kiki')  
 
for _ in kong_list:
  test_label.append('kong')    
  
   
  
  
