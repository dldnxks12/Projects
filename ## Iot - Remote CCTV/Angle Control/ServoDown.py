import RPi.GPIO as GPIO
import time

GPIO.cleanup()
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

def Down():
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    Servo_PIN = 12

    GPIO.setup(Servo_PIN, GPIO.OUT)
    servo1 = GPIO.PWM(Servo_PIN, 50)

    servo1.start(0)
    time.sleep(1)
    servo1.ChangeDutyCycle(2.5)
    time.sleep(1)
    servo1.stop()


