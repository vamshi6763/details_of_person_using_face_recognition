import cv2
from deepface import DeepFace
import numpy as np
import os

face_cascade_name = os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"  #getting a haarcascade xml file
face_cascade = cv2.CascadeClassifier(face_cascade_name)  #processing it for our project
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):  #adding a fallback event
    print("Error loading xml file")

video=cv2.VideoCapture(0,cv2.CAP_DSHOW)  #requisting the input from the webcam or camera

while video.isOpened():  #checking if are getting video feed and using it
    _,frame = video.read()

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  #changing the video to grayscale to make the face analisis work properly
    face=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

    for x,y,w,h in face:
      img=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),1)
      try:
          analysis = DeepFace.analyze(frame, actions = ["age", "gender", "emotion", "race"])
          print("age = ",analysis["age"]," gender = ",analysis["gender"]," Emotion = ",analysis["dominant_emotion"]," race = ",analysis["dominant_race"])
      except:
          print("no face")
          video.release()
          cv2.destroyAllWindows()
          
      cv2.imshow('video', frame)
      key=cv2.waitKey(1) 
      if key==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
