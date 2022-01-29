'''

Daemon - Cron 설정 

crontae -e 

* * * * * sudo python3 /home/pi/ML/Project/7079_Servo_down.py >> /home/pi/ML/Project/log.txt 2>1& : log check 
* * * * * sudo pkill -f /home/pi/ML/Project/7079_Servo_down.py

sudo service cron restart 

crontab -l

'''

# Servo Down Code

import numpy as np
import RPi.GPIO as GPIO
import time

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

count = 0

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

    if(distance > 10):
        count += 1
        print("count: ", count)
    if(count > 10):        
        
        servo.start(0)
        servo.ChangeDutyCycle(2.5)                        
        time.sleep(1)
        servo.stop()
        break
    
