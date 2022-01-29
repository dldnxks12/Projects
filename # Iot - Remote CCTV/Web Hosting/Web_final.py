'''

Port Forwarding or SuperDMZ 를 이용하여 
Remote CCTV와 Servo Control 기능 구현 

'''

#!/usr/bin/env python
from flask import Flask, render_template, Response, redirect, url_for, request
from camera import Camera
import ServoUp
import ServoDown

app = Flask(__name__)


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
        ServoUp.Up()
        return home()    
    elif servo == 'Down':
        ServoDown.Down()
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

