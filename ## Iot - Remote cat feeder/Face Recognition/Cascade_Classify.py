'''

라즈베리 파이 4에서 간단하게 단순 Pi Camera를 이용하여 Classification을 진행

'''


import cv2 as cv

def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:        
        frame = cv.rectangle(frame, (x-30,y-30),(x+w+30, y+h+30), (255, 0, 255), 3)

    cv.imshow('Cat-detector', frame)

face_cascade = cv.CascadeClassifier()

if not face_cascade.load(cv.samples.findFile('/home/pi/ML/Project/haarcascade_frontalcatface.xml')):    
    exit(0)

cap = cv.VideoCapture(-1)

cap.set(3,320)
cap.set(4,240)

if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    frame = cv.flip(frame, -1)
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    detectAndDisplay(frame)

    if cv.waitKey(10) == 27:
        break

