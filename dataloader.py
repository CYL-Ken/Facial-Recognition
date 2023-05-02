import os
from torch.utils.data.dataset import Dataset

class customDataset(Dataset):
    def __init__(self, path, transform):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.transform = transform
        
        self.name = {}
        
        self.people_list = []
        self.face = []
        
        for person in path:
            if person not in self.people_list:
                self.people.append(person)
                
            person_dataset = os.path.join(path, person)
            for i in os.listdir(person_dataset):
                self.face.append(os.path.join(person_dataset, i))
                self.people_list.append(person)
            
            
        
    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        
    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------s