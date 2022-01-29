
from flask import Flask, request
from flask import render_template
import RPi.GPIO as GPIO

app = Flask(__name__)
GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.OUT, initial=GPIO.LOW)
 
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/led/on")
def led_on():
    try:
        GPIO.output(26, GPIO.HIGH)
        return "ok"
    except expression as identifier:
        return "fail"


@app.route("/led/off")
def led_off():
    try:
        GPIO.output(26, GPIO.LOW)
        return "ok"
    except expression as identifier:
        return "fail"



if __name__ == "__main__":
        app.run(host="0.0.0.0")
