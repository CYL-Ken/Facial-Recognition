import cv2
import logging

from face_recognizer import FaceRecognizer

path = "rtsp://admin:Aa83446416!@192.168.50.3:554/rtspstream/channel=0/stream=1"

faceRecognizer = FaceRecognizer()

count = 0

while True:
    cap = cv2.VideoCapture(path)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame!")
            break
        try:
            image = frame[100:900, 640:1111]
            image = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2))
            ret, face, boxes = faceRecognizer.detectFace(image)
            if ret:
                face_width = boxes[0][2] - boxes[0][0]
                face_height = boxes[0][3] - boxes[0][1]
                if face_height > 30 and face_width > 30:
                    count += 1
                    cv2.imwrite(f"data/output_{count}.jpg", face)
            
                x1 = boxes[0][0]
                y1 = boxes[0][1]
                x2 = boxes[0][2]
                y2 = boxes[0][3]
                cv2.rectangle(image, (x1, y1), (x2, y2), (214, 217, 8), 2, cv2.LINE_AA)
            
        except Exception as e:
            print(f"Got Exception: {e}")
            result = None
        
        cv2.imshow("Result", image)
        if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:
            break

    cap.release()       
    cv2.destroyAllWindows()