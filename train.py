import os
import cv2
import json
import argparse
import numpy as np


def prepare_face_data(dataset_path, face_detector):
    face_list, id_list = [], []
    name_dict = {}
    count = 0
    for person in os.listdir(dataset_path):
        count += 1
        name_dict[count] = person
        person_path = os.path.join(dataset_path, person)
        for file in os.listdir(person_path):
            file_path = os.path.join(person_path, file)
            
            img = cv2.imread(file_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            img_np = np.array(gray,'uint8')
            face = face_detector.detectMultiScale(gray) 
            for(x,y,w,h) in face:
                face_list.append(img_np[y:y+h,x:x+w])
                id_list.append(count)
    return face_list, id_list, name_dict

if __name__ == "__main__":
    print("<Facial Recognition Trainer>")
    
    parser = argparse.ArgumentParser(description="Facial Recognition Trainer")
    parser.add_argument("-d", "--dataset", default="face_dataset", type=str, help="Dataset Path")
    parser.add_argument("-w", "--weight", default="face.yml", type=str, help="Model Weight Path")
    args = parser.parse_args()
    
    
    dataset_path = args.dataset
    weight_path = args.weight
    
    print(" - Prepare Face Detection and Recognition Models...")
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    
    print(f" - Prepare Face Dataset from {dataset_path}...")
    face_list, id_list, name_dict = prepare_face_data(dataset_path, face_detector)
    
    
    print(" - Start Training Recognition Model...")
    recognizer.train(face_list, np.array(id_list))
    recognizer.save(weight_path)
    
    with open("name.json", "w") as outfile:
        json.dump(name_dict, outfile)
    
    print(" - Finish Training!")
    print(f" - Model Weight Path: {weight_path}")