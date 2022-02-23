'''

Daemon - Cron 설정 
crontae -e 

* * * * * sudo python3 /home/pi/ML/Project/7079_Servo_up.py >> /home/pi/ML/Project/log.txt 2>1& : log check 
* * * * * sudo pkill -f /home/pi/ML/Project/7079_Servo_up.py

sudo service cron restart 

crontab -l


*주의 

Raspberry Pi를 VNC를 통해 연결했기 때문에 Terminal에서 해당 Cron을 실행 했을 때 , gtk display cannot open error 발생 

  1. VNC가 아닌 UV4L or 직접 모니터 연결해서 Image display 연결하기
  2. cv2.imshow() 함수 제거 -> led로 loop 문 진입 확인하기   

Crontab -e Setting 시 Python file에 한글이 포함되어 있다면 문자 인코딩 필요 -- utf-8

'''


# Servo Down Code

import cv2 as cv
import numpy as np
import RPi.GPIO as GPIO
import time
import tensorflow as tf
import keras
from keras import models
from keras import layers

reconstruct = tf.keras.models.load_model("cat_recognize_small") # 가중치 불러오기

GPIO.cleanup()

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

Servo_PIN = 12
TRIG = 23
ECHO = 24

print("Distance is in Measurment")

GPIO.setup(Servo_PIN, GPIO.OUT)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

servo = GPIO.PWM(Servo_PIN, 50)
GPIO.output(TRIG, False)
print("Waiting for setting")
time.sleep(2)

def ImgProcessing(sample):
    ni = sample
    im = np.zeros((64,64,3) , dtype = np.uint8)

    if ni.shape[0] >= ni.shape[1]:
        scale = ni.shape[0] / 64
        new_width = int(ni.shape[1] / scale)
        diff = (64-new_width) // 2
        ni = cv.resize(ni, (new_width, 64))
        im[:, diff:diff+new_width,:] = ni

    else:
        scale = ni.shape[1] / 64
        new_height = int(ni.shape[0] / scale)
        diff = (64 - new_height) // 2
        ni = cv.resize(ni, (64, new_height))  
        im[diff : diff + new_height, :, :] = ni

    sample = im
    sample = sample /255.0

    return sample

def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    faceROI = np.zeros_like(frame) # Detected Face 
    
    for (x,y,w,h) in faces:             
        if( x > 30 and y > 30) :           
            faceROI = frame[y-30:y+h+30 , x-30:x+w+30, :]               
        frame = cv.rectangle(frame, (x-30,y-30),(x+w+30, y+h+30), (255, 0, 255), 3)        

    cv.imshow('Cat-detector1', faceROI)
    cv.imshow('Cat-detector2', frame)    

    return faceROI
        
face_cascade = cv.CascadeClassifier()

if not face_cascade.load(cv.samples.findFile('/home/pi/ML/Project/haarcascade_frontalcatface.xml')):    
    exit(0)

count = 0
Valid = 0
Check = 0 
End_Program = 0

while True:

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    while GPIO.input(ECHO) == 0:
        start = time.time()
    while GPIO.input(ECHO) == 1:
        stop  = time.time()

    check_time = stop - start
    distance = check_time * 34300 / 2
    print(distance)
    time.sleep(0.4)    
    
    if(distance < 10):
        count += 1         
        print("count", count)

        if(count == 10):            

            servo.start(0) # 모터 시작 

            cap = cv.VideoCapture(-1) # Video 시작 
            cap.set(3,640)
            cap.set(4,480)

            if not cap.isOpened:
                print('--(!)Error opening video capture')
                exit(0)            

            while True:
                ret, frame = cap.read()                
                frame = cv.flip(frame, -1)
                if frame is None:
                    print('--(!) No captured frame -- Break!')
                    break
                                
                face = detectAndDisplay(frame)                                
                if(len(face) != 480):         
                    print("valid",Valid)           
                    Valid += 1
                    if(Valid == 3):                                        
                        train_sample = ImgProcessing(face)
                        train_sample = np.expand_dims(train_sample, axis = 0)
                        pred = reconstruct.predict(train_sample)
                        print("pred",pred)                        
                        if(pred[0][0] > 0.9):                            
                            Check += 1                             
                            if( Check == 2):
                                # Food Open                         
                                print("Check 2")
                                print("Hi, KiKi")
                                servo.ChangeDutyCycle(12.5)                        
                                time.sleep(1)                                                                       
                                servo.stop()                            
                                End_Program = 1
                                break  
                            print("Check 1")
                            Valid = 0                                                   
                        else:
                            print("Not Clean")
                            Check = 0 
                            Valid = 0
                        
                if cv.waitKey(10) == 27:
                    break
                                                                
            cap.release()    
            cv.destroyAllWindows()
    else:
        count = 0 
    if( End_Program == 1):
        break
        
