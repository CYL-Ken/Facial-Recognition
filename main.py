import os
import cv2
import json
import time
import argparse

from logger import Logger

from door_controller import DoorController
from face_recognizer import FaceRecognizer


def recognize_frame(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ret, face, boxes = faceRecognizer.detectFace(original_image=image)
    if len(boxes) == 0:
        return None, (0,0,0,0)
    else:
        result = faceRecognizer.recognizeFace(face=face, debug=True, threshold=12)
        return result, boxes

def start_streaming(video_path, show_result=True):
    while True:
        video_path = config["video"] if video_path == "cyl" else video_path
        video = video_path
        cap = cv2.VideoCapture(video)

        if not cap.isOpened():
            log.warning("Waiting camera...")
        
        while True:
            ret, image = cap.read()
            #cap.release()
            if not ret:
                log.warning("Cannot receive frame!")
                break
            try:
                start_time = time.time()
                result, boxes = recognize_frame(image=image)
                log.debug(f"Inference time: {time.time() - start_time}")
            except Exception as e:
                log.warning(f"Got Exception: {e}")
                result = None
                
            if args.door:
                ret, name = door.visit(result)
                if ret == True:
                    log.info(f"Hello, {name}.")
                elif name != "No Person":
                    log.info(f"Found guest!")
            
            if show_result == False:
                continue
            
            if result != None:
                x1 = boxes[0][0]
                y1 = boxes[0][1]
                x2 = boxes[0][2]
                y2 = boxes[0][3]

                cv2.rectangle(image, (x1, y1), (x2, y2), (214, 217, 8), 2, cv2.LINE_AA)
                cv2.putText(image, result, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (214, 217, 8), 2, cv2.LINE_AA)

            image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
            cv2.imshow("Result", image)
            if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:
                break
        
        cap.release()       
        cv2.destroyAllWindows()
        
        
        
        
if __name__ == "__main__":
    log = Logger().get_log()
    
    parser = argparse.ArgumentParser(description="Facial Recognition")
    parser.add_argument("-d", "--dataset", default="face_dataset", type=str, help="Dataset Path")
    parser.add_argument("-v", "--video", default=0, help="Video Stream")
    parser.add_argument("-i", "--image", default=None, help="Video Stream")
    parser.add_argument("--show", default=False, help="Show Result")
    parser.add_argument("--door", default=False, type=bool, help="Control Door")
    args = parser.parse_args()
    
    faceRecognizer = FaceRecognizer()
    
    log.info(f"Prepare Dataset: {args.dataset}")
    faceRecognizer.initFeatureDataset(dataset_path=args.dataset, 
                                      label_list=os.listdir(args.dataset))
    
    config = json.load(open("config.json"))
    
    if args.door:
        door = DoorController(open_link = config["open"])
    
    if args.image != None:
        log.info("Image mode")
        # Not yet!
        pass
    else:
        log.info("Video mode")
        start_streaming(args.video, args.show)
