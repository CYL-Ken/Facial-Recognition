"""
    Facial-Recognition
"""
import cv2
import time
import argparse
import requests

from logger import Log
from facial_recognizer import FacialRecognizer

        
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
        
    def process(self, text):
        if text != "Guest":
            log.info(f"Hello {text}")
            if (time.time()-self.open_timer) > self.escape:
                if self.person == text:
                    self.counter['Yes'] += 1
                
                self.person = text
                if self.counter['Yes'] > 3 and self.status == True:
                    # self.open()
                    self.set_door_status(False)
                    # print("Disable Door")
                    self.counter['Yes'] = 0
                    self.counter['No'] = 0
            else:
                print("Too Fast")
        else:
            if self.person == text:
                self.counter['No'] += 1
            
            self.person = text
            if self.counter['No'] > 5:
                # print("Enable Door")
                self.set_door_status(True)
                self.counter['Yes'] = 0
                self.counter['No'] = 0

if __name__ == "__main__":
    
    log = Log().get_log()
    
    parser = argparse.ArgumentParser(description="Facial Recognition")
    parser.add_argument("-x", "--dataset", default="face_dataset", type=str, help="Dataset Path")
    parser.add_argument("-v", "--video", default="http://admin:admin@192.168.50.2/video.cgi?identify_key=984852984&pipe=0", type=str, help="Video Streaming Path")
    parser.add_argument("-t", "--time", default=5, type=int, help="Waiting time between open door")
    parser.add_argument("-i", "--image", default=None)
    parser.add_argument("-d", "--door", default=True, help="For open door")
    parser.add_argument("-e", "--embedding", default=None, help="Custom dataset")

    args = parser.parse_args()
    
    # Door Setting
    if args.door == True:
        door = Door(escape_time=args.time)
    
    # Prepare Embedding Data and Models
    recognizer = FacialRecognizer(dataset_path=args.dataset)
    recognizer.prepare_model()
    recognizer.create_embeddings(embedding_data=args.embedding)
    
    if args.image != None:
        log.info("<Image Mode>")
        image = cv2.imread(args.image)
        if len(image) > 0:
            log.info("Start recognize image")
            recognizer.inference(image)
    else:
        while True:
            video = args.video
            cap = cv2.VideoCapture(video)
            
            if not cap.isOpened():
                log.warning("Waiting camera...")
            log.info("Start recognize image from video")
            while True:
                ret, image = cap.read()
                if not ret:
                    log.warning("Cannot receive frame!")
                    break
            
                text, (x1, y1, x2, y2) = recognizer.inference(image)
                w, h, c = image.shape[0], image.shape[1], image.shape[2]
                if w > 0 and h > 0:
                    
                    if args.door == True and text != "No face" and len(text)>0:
                        door.process(text)
                    
                    cv2.rectangle(image, (x1, y1), (x2, y2), (214, 217, 8), 2, cv2.LINE_AA)
                    cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (214, 217, 8), 2, cv2.LINE_AA)
                    image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
                    cv2.imshow("Result", image)
                    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:
                        break
                else:
                    continue
            cap.release()
            cv2.destroyAllWindows()
            
            
            
            # cap = cv2.VideoCapture('http://admin:admin@192.168.50.2/video.cgi?identify_key=984852984&pipe=0')
            # if not cap.isOpened():
            #     log.warning(" - Waiting Camera...")
            #     continue
            # while True:
            #     ret, img = cap.read()
            #     if not ret:
            #         log.warning(" - Cannot Receive Frame!")
            #         break
                
            #     text = recognizer.inference(img)
                
            #     if text == "No face":
            #         continue
                
            #     if text != "Not in dataset":
            #         log.info(f"Hello ", text)
            #         if (time.time()-door.open_timer) > door.escape:
            #             if door.person == text:
            #                 door.counter['Yes'] += 1
                        
            #             door.person = text
            #             if door.counter['Yes'] > 3 and door.status == True:
            #                 door.open()
            #                 door.set_door_status(False)
            #                 # print("Disable Door")
            #                 door.counter['Yes'] = 0
            #                 door.counter['No'] = 0
            #         else:
            #             print("Too Fast")
            #     else:
            #         if door.person == text:
            #             door.counter['No'] += 1
                    
            #         door.person = text
            #         if door.counter['No'] > 5:
            #             # print("Enable Door")
            #             door.set_door_status(True)
            #             door.counter['Yes'] = 0
            #             door.counter['No'] = 0