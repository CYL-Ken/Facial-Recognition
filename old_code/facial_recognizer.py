import os
import cv2
import json
import numpy as np

import torch
from torchvision import datasets

from logger import Log
from facenet_pytorch import MTCNN, InceptionResnetV1

log = Log().get_log()

class FacialRecognizer():
    def __init__(self, dataset_path="", weight="", embedding="face_embedding.npy") -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataset_path = dataset_path
        self.weight = weight
        self.embedding_path = embedding
        self.face_embedding = []
    
    def prepare_model(self):
        log.info("Prepare MTCNN and FaceNet")
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )
        
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def collate_fn(self, x):
        return x[0]
    
    def load_dataset(self, dataset_path):
        if os.path.exists(dataset_path):
            self.dataset = datasets.ImageFolder(dataset_path)
            self.dataset.idx_to_class = {i:c for c, i in self.dataset.class_to_idx.items()}
            loader = torch.utils.data.DataLoader(self.dataset, collate_fn=lambda x: x[0], num_workers=0)
            return True, loader
        else:
            return False, None


    def create_embeddings(self, embedding_data=None):
        if embedding_data != None:
            filename = "face_embedding.npy"
            self.face_embedding = np.load(filename, allow_pickle=True).tolist()
        else:
            log.info("Creating embedding file.")
            ret, loader = self.load_dataset(self.dataset_path)
            if ret == False:
                return False
            
            self.face_embedding = []
            aligned = []
            names = []
            for x, y in loader:
                boxes, prob = self.mtcnn.detect(x)
                if boxes is not None:
                    log.info(' -> Face detected with probability: {:8f}'.format(prob[0]))
                    box = list(map(int, boxes[0]))
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    x_aligned = x.crop((x1, y1, x2, y2))
                    if (x2-x1)<50 or (y2-y1)<50:
                        continue
                    x_aligned = cv2.cvtColor(np.asarray(x_aligned), cv2.COLOR_RGB2BGR)
                    x_aligned = cv2.resize(x_aligned, (160, 160))
                    x_aligned = torch.from_numpy(x_aligned).permute(2,0,1)
                    print(x_aligned.shape)
                    aligned.append(x_aligned)
                    names.append(self.dataset.idx_to_class[y])

            aligned = torch.stack(aligned).to(self.device)
            embeddings = self.facenet(aligned.to(torch.float32)).detach().cpu()

            log.info(f"Finish Training on {len(names)} Data")
            
            for e, n in zip(embeddings, names):
                data = (e,n)
                self.face_embedding.append(data)
            save_data = np.array(self.face_embedding, dtype=object)
            log.info(f"Save Embedding Feature at {self.embedding_path}")
            np.save(self.embedding_path, save_data)
            with open("name.json", "w") as outfile:
                json.dump(self.dataset.idx_to_class, outfile)
        
    
    def inference(self, image):
        # result, prob = self.mtcnn(image, return_prob=True)
        boxes, _ = self.mtcnn.detect(image)
        x1, y1, x2, y2 = 0, 0, 0, 0
        if boxes is not None:
            box = list(map(int, boxes[0]))
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            
            image_crop = image[y1:y2, x1:x2]
            if image_crop.shape[0]<50 or image_crop.shape[1]<50:
                return "", (x1, y1, x2, y2)
            
            input = torch.from_numpy(image_crop).permute(2,0,1)
            embedding = self.facenet(input.unsqueeze(0).to(self.device).float())
            
            comparison = []
            names = []
            
            for data in self.face_embedding:
                dist = (embedding-data[0].to(self.device)).norm().item()
                comparison.append(dist)
                names.append(data[1])
            
            # print(comparison)
            if min(comparison) < 2:
                name = names[np.argmin(comparison)]
                return name, (x1, y1, x2, y2)
            
            return "Guest", (x1, y1, x2, y2)
        else:
            return "No face", (x1, y1, x2, y2)
            
        
        
        
        
        
        # if result != None:
        #     embedding = self.facenet(result.unsqueeze(0).to(self.device))

        #     else:
        #         return "Not in dataset"
        # else:
        #     return "No face"