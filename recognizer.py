import cv2
import json
import time
import argparse
import requests
from datetime import datetime
from threading import Timer

from predict import recognize


class Door():
    def __init__(self, escape_time) -> None:
        self.status = True
        self.person = ""
        
        self.escape = escape_time
        self.open_timer = 0
        
        self.counter = {
            "Yes": 0,
            "No": 0
        }

    def set_door_status(self, enable=True):
        self.status = enable
        
    def open(self):
        response = requests.get(r"http://admin:admin@192.168.50.2/DP/doorunlock.ncgi?id=2635107228")
        print("OPEN!")
        self.open_timer = time.time()


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
    
    door = Door(escape_time=args.time)
    
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
            
            # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # faces = face_detector.detectMultiScale(gray)
            
            text = "Not in dataset"
            text = recognize(img)
            
            # for (x, y, w, h) in faces:
            #     cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
            #     id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                
            #     text = name_dict[str(id)] if confidence < 40 else "Not in dataset"
                
            #     print("DEBUG:", text, door.person, door.status)
            
            
            if text != "Not in dataset":
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Hello ", text)
                if (time.time()-door.open_timer) > door.escape:
                    if door.person == text:
                        door.counter['Yes'] += 1
                    
                    door.person = text
                    if door.counter['Yes'] > 6 and door.status == True:
                        door.open()
                        door.set_door_status(False)
                        # print("Disable Door")
                        door.counter['Yes'] = 0
                        door.counter['No'] = 0
                else:
                    pass
                    print("Too Fast")
            else:
                if door.person == text:
                    door.counter['No'] += 1
                
                door.person = text
                if door.counter['No'] > 8:
                    # print("Enable Door")
                    door.set_door_status(True)
                    door.counter['Yes'] = 0
                    door.counter['No'] = 0
