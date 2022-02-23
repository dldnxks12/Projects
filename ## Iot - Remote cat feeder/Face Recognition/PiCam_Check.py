'''
Clasffication Model을 학습시킨 후 아래 코드와 병합하여 콩이랑 구일이 구별하는 모델 만들 예정
'''

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while(True):
  ret, frame = cap.read()
  frame = cv2.flip(frame, -1)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  cv2.imshow('frame',frame)
  cv2.imshow('gray',gray)
  
  k = cv2.waitKey(30)
  if k == 27:
    break
    
cap.release()
cv2.destroyAllWindows()
