'''

Simple Classification + Servo + LED Check

'''

import cv2 as cv
import numpy as np
import RPi.GPIO as GPIO
import time

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

while True:

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    while GPIO.input(ECHO) == 0:
        start = time.time()
    while GPIO.input(ECHO) == 1:
        stop = time.time()

    check_time = stop - start
    distance = check_time * 34300 / 2
    print(distance)
    time.sleep(0.4)
    

    if(distance < 10):
        count += 1 
        # 초음파 센서 작동하ᅧ면 While loop 진입할 것 
        print(count)
        if(count == 5):

            servo.start(0)

            cap = cv.VideoCapture(-1)

            cap.set(3,640)
            cap.set(4,480)

            if not cap.isOpened:
                print('--(!)Error opening video capture')
                exit(0)

            while True:
                ret, frame = cap.read()
                # print(type(frame)) : Numpy
                frame = cv.flip(frame, -1)
                if frame is None:
                    print('--(!) No captured frame -- Break!')
                    break

                face = detectAndDisplay(frame)

                if(len(face) != 480):
                    servo.ChangeDutyCycle(7.5)
                    time.sleep(2)  
                    servo.ChangeDutyCycle(12.5)
                    time.sleep(2)                                        
                    servo.ChangeDutyCycle(2.5)
                    time.sleep(2)  
                    print(len(face))        
                    # Face Classification On the Board
                else:
                    servo.ChangeDutyCycle(0)                        
                # Servo 모터 작동하면 Stop 할 것             
                if cv.waitKey(10) == 27:
                    break
                
            servo.stop()            
            count = 0            
            cap.release()    
            GPIO.cleanup()
            cv.destroyAllWindows()

