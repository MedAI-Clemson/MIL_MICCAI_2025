import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

class ImageDataBags(Dataset):
    """Image CSV Dataset for Multiple Instance Learning"""
    type_map = {1: 0, 2: 1, 3: 2, 4: 2}
    inv_type_map = {0: '1-part', 1: '2-part', 2: '3- or 4-part'}

    def __init__(self, data: pd.DataFrame, transforms, seed=1, train=True, view_filter = None):
        self.transforms = transforms
        self.data = data
        self.train = train
        self.r = np.random.RandomState(seed=1)
        # Recode targets
        self.data['trg'] = self.data['num_parts'].map(ImageDataBags.type_map)

        # Fill nan views with "Unknown"
        self.data['view_group'] = self.data['view_group'].fillna('Unknown')
        
        # Filter to given views
        if view_filter is not None:
            self.data = self.data.loc[self.data['view_group'].isin(view_filter)]

        self.bags_list, self.labels_list, self.patient_ids_list = self._create_bags()

    def _create_bags(self):
        bags_list = []
        labels_list = []
        patient_ids_list = []  
        patient_ids = self.data['patient_id'].unique()  

        # Determine the maximum number of images in any bag based on the patient with the most images
        max_images = max(self.data.groupby('patient_id').size())
        #print(max_images)
        for patient_id in patient_ids:
            patient_data = self.data[self.data['patient_id'] == patient_id]
            # print(patient_data)
            bag_images = []
            bag_labels = []
            #print("$" * 40)
            # Fill the bag with images from the patient, applying transformations
            for idx, row in patient_data.iterrows():
                img = self.transforms(Image.open(row['path']))
                trg = row['trg']
                # print(img.shape)
                bag_images.append(img)
                bag_labels.append(trg)
                #bag_labels = [1 if label == 1 else 0 for label in bag_labels]
                C, H, W = bag_images[0].shape
            bags_list.append(torch.stack(bag_images))
            bag_label = bag_labels[0]
            labels_list.append((torch.tensor(bag_label, dtype=torch.long), torch.tensor(bag_labels, dtype=torch.long)))
            patient_ids_list.append(patient_id)
        return bags_list, labels_list, patient_ids_list
    def __len__(self):
        #print("len")
        return len(self.bags_list)
        #return len(self.labels_list)

    def __getitem__(self, index):
        bag, (bag_label, instance_labels) = self.bags_list[index], self.labels_list[index]
        patient_id = self.patient_ids_list[index]
        #print(self.labels_list[index])


        return bag, (bag_label, instance_labels, patient_id)
