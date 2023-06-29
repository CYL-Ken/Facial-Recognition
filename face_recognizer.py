import os
import cv2
import numpy as np
import onnxruntime as ort

import utils

class FaceRecognizer():
    def __init__(self, 
                 detection_model_path="models/face_detector.onnx", 
                 recognition_model="models/FaceNet_vggface2_optimized.onnx") -> None:
        
        self.detection_model = ort.InferenceSession(detection_model_path)
        self.recognition_model = ort.InferenceSession(recognition_model)
        
        self.feature_dataset = []
    
    def detectFace(self, original_image, threshold=0.7) -> list:
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        input_name = self.detection_model.get_inputs()[0].name
        confidences, raw_boxes = self.detection_model.run(None, {input_name: image})
        
        boxes, _, _ = utils.choose_boxes(original_image.shape[1], 
                                         original_image.shape[0], 
                                         confidences, 
                                         raw_boxes, 
                                         threshold)
        ret, face = utils.crop_face(boxes=boxes, image=original_image)
        return ret, face, boxes
    
    def getFaceFeature(self, face_image):
        image = utils.facenet_preprocess(face_image)
        input_name = self.recognition_model.get_inputs()[0].name
        result = self.recognition_model.run(None, {input_name: image})
        return np.array(result)[0]
    
    def initFeatureDataset(self, dataset_path, label_list):
        dataset = []
        for person in label_list:
            folder = os.path.join(dataset_path, person)
            for path in os.listdir(folder):
                img_path = os.path.join(folder, path)
                
                img = cv2.imread(img_path)
                ret, face, _ = self.detectFace(original_image=img)
                if ret is False:
                    continue

                print(f" - Loading {img_path}, {face.shape}, {person}")
                face_feature = self.getFaceFeature(face)
                
                dataset.append((person, face_feature))
        self.feature_dataset = dataset
    
    def recognizeFace(self, face, threshold=6, debug=False):
        feature = self.getFaceFeature(face)
        
        min_dist = -1
        name = ""
        for data in self.feature_dataset:
            dist = np.linalg.norm(data[1] - feature)
            if debug:
                print(data[0], dist)
            if min_dist>dist or min_dist==-1:
                min_dist = dist
                name = data[0]
                
        if min_dist<threshold:
            return name
        else:
            return "Guest"

    
if __name__ == "__main__":
    
    """
        Function Test 1
         - Input: 2 images
         - Output: The distance of 2 images
         
        Function Test 2
         - Input: 1 image and 1 dataset
         - Output: Who is it
    """
    
    # Function Test 1
    image_A = cv2.imread("face_dataset/Ken/S__24674311.jpg")
    image_B = cv2.imread("face_dataset/Mark/S__25403464.jpg")
    
    faceRecognizer = FaceRecognizer()
    ret, crop_A, _ = faceRecognizer.detectFace(original_image=image_A)
    if ret is False:
        print("No Face in A.")
        exit()
        
    ret, crop_B, _ = faceRecognizer.detectFace(original_image=image_B)
    if ret is False:
        print("No Face in B.")
        exit()
    
    vector_A = faceRecognizer.getFaceFeature(face_image=crop_A)
    vector_B = faceRecognizer.getFaceFeature(face_image=crop_B)
    
    dist = np.linalg.norm(vector_A - vector_B)
    print("Function Test 1")
    print(f"Difference between A and B is {dist}")

    # Function Test 2
    image_C = cv2.imread("face_dataset/Mark/S__25403461.jpg")
    
    dataset_path = "cyl_dataset/"
    label_list = os.listdir(dataset_path)
    
    faceRecognizer.initFeatureDataset(dataset_path=dataset_path, label_list=label_list)
    
    ret, face, _ = faceRecognizer.detectFace(original_image=image_C)
    print("Function Test 2")
    
    if ret is False:
        print("No Face Detected")
    else:
        answer = faceRecognizer.recognizeFace(face=face, debug=True)
        print("Answer is", answer)