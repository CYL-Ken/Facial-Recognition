import os
import cv2
import json
import argparse
import requests
from datetime import datetime
from threading import Timer


class Door():
    def __init__(self) -> None:
        self.status = True
        self.person = ""

    def enable_door(self):
        self.status = True

if __name__ == "__main__":
    print("<Facial Recognition Recognizer>")
    
    parser = argparse.ArgumentParser(description="Facial Recognition Recognizer")
    parser.add_argument("-w", "--weight", default="face.yml", type=str, help="Model Weight Path")
    parser.add_argument("-v", "--video", default="http://admin:admin@192.168.50.2/video.cgi?identify_key=984852984&pipe=0", type=str, help="Video Streaming Path")
    parser.add_argument("-t", "--time", default=3, type=int, help="Waiting time between open door")
    args = parser.parse_args()
    
    print(" - Prepare Face Detection and Recognition Models...")
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(args.weight)
    
    with open('name.json') as json_file:
        name_dict = json.load(json_file)
    
    door = Door()
    door_timer = Timer(args.time, door.enable_door)
    
    
    print(" - Start Running System...")
    while True:
        cap = cv2.VideoCapture('http://admin:admin@192.168.50.2/video.cgi?identify_key=984852984&pipe=0')
        if not cap.isOpened():
            print(" - Waiting Camera...")
            continue
        while True:
            ret, img = cap.read()
            if not ret:
                print(" - Cannot Receive Frame!")
                break

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray)
            
            # print("No face", door.status)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                try:
                    if confidence < 40 and door.status == True:
                        text = name_dict[str(id)]
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Hello ", text)
                        
                        if text == door.person:
                            break
                        door.person = text
                        
                        response = requests.get(r"http://admin:admin@192.168.50.2/DP/doorunlock.ncgi?id=2635107228")
                        door_timer.start()
                        door.status = False
                    else:
                        print(f"[{datetime.now()}] Who are you?")
                        door.person = None
                except:
                    print("==")