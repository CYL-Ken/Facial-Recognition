import os
import time
import argparse
import cv2
import numpy as np
import onnx

# onnx runtime
import onnxruntime as ort

import utils as box_utils

class FaceDetector():
    def __init__(self, 
                onnx_path="models/version-slim-320.onnx", 
                label_path="models/voc-model-labels.txt",
                img_size=(320, 240), 
                result_path="",
                threshold=0.7):
        self.onnx_path = onnx_path
        self.img_size = img_size
        self.result_path = result_path
        self.class_names = [name.strip() for name in open(label_path).readlines()]

        self.predictor = onnx.load(onnx_path)
        onnx.checker.check_model(self.predictor)
        onnx.helper.printable_graph(self.predictor.graph)

        self.ort_session = ort.InferenceSession(onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name

        self.threshold = threshold
    
    def inference(self, ori_image=None):
        
        # Preprocessing
        image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        confidences, boxes = self.ort_session.run(None, {self.input_name: image})
        boxes, labels, probs = self.model_predict(ori_image.shape[1], ori_image.shape[0], confidences, boxes, self.threshold)
        return boxes
        sum = 0
        
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            print(f" - Face {i+1}: {box}")
            label = f"{self.class_names[labels[i]]}: {probs[i]:.2f}"

            cv2.rectangle(ori_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 1)

            # cv2.putText(orig_image, label,
            #             (box[0] + 20, box[1] + 40),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             1,  # font scale
            #             (255, 0, 255),
            #             2)  # line type
        
        return boxes

    def model_predict(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = box_utils.hard_nms(box_probs,
                                        iou_threshold=iou_threshold,
                                        top_k=top_k,
                                        )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='my description')
    parser.add_argument('--img_path', type=str)
    args = parser.parse_args()

    FaceDetector = FaceDetector(result_path="results")
    img_path = args.img_path
    boxes = FaceDetector.inference(img_path=img_path)
    print(boxes)