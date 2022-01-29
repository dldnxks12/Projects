#!/usr/bin/env python
from flask import Flask, render_template, Response, redirect, url_for, request
from camera import Camera
import RPi.GPIO as GPIO
import time

app = Flask(__name__)
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

Servo_PIN = 12

GPIO.setup(Servo_PIN, GPIO.OUT)
servo = GPIO.PWM(Servo_PIN, 50)

servo.start(0)
time.sleep(2)

@app.route('/<state>')
def gate2(state):
    if state == 'GY':        
        return redirect(url_for('home'))
    else:
        return redirect(url_for('NoGY'))

@app.route('/NotHome')
def NoGY():
    return "You are not Gayeong"

@app.route('/home')
def home():    
    return render_template('index.html')

@app.route('/servo', methods = ['POST']) # Button clink -> button? value -> python? ?? 
def servo():
    servo = request.form['servo']    
    if servo == 'Up':
        servo.ChangeDutyCycle(12.5)
        time.sleep(1)
        return home()    
    elif servo == 'Down':
        servo.ChangeDutyCycle(2.5)
        time.sleep(1)
        return home()
    
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
