from torch.utils.data import Dataset
from torchvision.io import read_image
import torch

class Achilles(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.studies = data.drop_duplicates(subset=['Folder Name'])['Folder Name']
        self.labels = data.drop_duplicates(subset=['Folder Name'])['Healthy?']
        self.transform = transform

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        study_name = self.studies.iloc[idx]
        
        # read and group images with study
        img_paths = self.data['Given Name'].loc[self.data['Folder Name']==study_name]
        study_imgs = []
        for path in img_paths:
            img = read_image('../Achilles_Classification/roi_data/'+path)
            if self.transform:
                img = self.transform(img)
            study_imgs.append(img)
        study = torch.concat(study_imgs)
        label = self.labels.iloc[idx]
        record = {
            "name": study_name,
            "study": study,
            "label": label
        }
        # return study, label
        return record
       