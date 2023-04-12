import os
import cv2
import time
import json
import argparse
import requests
import numpy as np

if __name__ == "__main__":
    print("<Facial Recognition Recognizer>")
    
    parser = argparse.ArgumentParser(description="Facial Recognition Recognizer")
    parser.add_argument("-w", "--weight", default="face.yml", type=str, help="Model Weight Path")
    parser.add_argument("-v", "--video", default="http://admin:admin@192.168.50.2/video.cgi?identify_key=984852984&pipe=0", type=str, help="Video Streaming Path")
    parser.add_argument("-t", "--time", default=5, type=int, help="Waiting time between open door")
    args = parser.parse_args()
    
    print(" - Prepare Face Detection and Recognition Models...")
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(args.weight)
    
    with open('name.json') as json_file:
        name_dict = json.load(json_file)
    
    while True:
        cap = cv2.VideoCapture('http://admin:admin@192.168.50.2/video.cgi?identify_key=984852984&pipe=0')
        if not cap.isOpened():
            print(" - Waiting Camera...")
            continue
        while True:
            ret, img = cap.read()
            if not ret:
                print(" - Cannot Receive Frame")
                break

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray)
            
            print("No face")
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                if confidence < 40:
                    text = name_dict[str(id)]
                    print("Hello ", text)
                    response = requests.get(r"http://admin:admin@192.168.50.2/DP/doorunlock.ncgi?id=2635107228")
                    time.sleep(5)
                else:
                    print("Who are you?")