import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets
from facenet_pytorch import MTCNN, InceptionResnetV1

filename = "face_embedding.npy"
dataset = np.load(filename, allow_pickle=True).tolist()

print("<Facial Recognition Recognizer>")

workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(' - Running on device: {}'.format(device))

print(" - Prepare MTCNN and FaceNet")
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def recognize(img):
    result, prob = mtcnn(img, return_prob=True)
    if result != None:
        embedding = resnet(result.unsqueeze(0).to(device))

        comparison = []
        names = []
        for data in dataset:
            dist = (embedding-data[0]).norm().item()
            comparison.append(dist)
            names.append(data[1])
            
        name = names[np.argmin(comparison)]
        return name
    else:
        return "Not in dataset"