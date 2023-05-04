import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1

from dataloader import faceDataset

class Recognizer():
    def __init__(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(device=self.device)
        
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
    
    def create_embeddings(self, loader):
        for x, y in loader:
            boxes, prob = self.mtcnn.detect(x)
            box = list(map(int, boxes[0]))
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            x.crop((x1, y1, x2, y2))

transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize((160,160)),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = faceDataset(path="cyl_dataset", transform=transform)

dataloader = DataLoader(dataset=dataset, collate_fn=lambda x: x[0])

recognizer = Recognizer()
recognizer.create_embeddings(dataloader)
# for image, label in dataloader:
#     np_image = image[0].permute(1, 2, 0).numpy()
#     print(label)
#     print(np_image.shape)
#     cv2.imshow(str(label), (np_image))
#     if cv2.waitKey(0) == ord('q'):
#         continue