from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import os
import pandas as pd
import numpy as np
import pydicom

class MILAchilles(Dataset):
    def __init__(self, data, data_path=None, transforms=None, save_dir=None):
        self.data = data
        self.data_path = data_path
        self.studies = data.drop_duplicates(subset=['Folder Name'])['Folder Name']
        self.labels = data.drop_duplicates(subset=['Folder Name'])['Healthy?']
        self.transform = transforms
        
        if save_dir is not None:
            self.data.to_csv(save_dir)

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        study_name = self.studies.iloc[idx]
        
        # read and group images with study
        img_paths = self.data['Given Name'].loc[self.data['Folder Name']==study_name]
        study_imgs = []
        for path in img_paths:
            img = read_image(self.data_path+path)
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
       
        
class Achilles(Dataset):
    def __init__(self, data, data_path=None, transforms=None, save_dir=None):
        self.data = data
        self.data_path = data_path
        self.transform = transforms
        
        if save_dir is not None:
            self.data.to_csv(save_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data['Given Name'].iloc[idx]
        image = read_image(self.data_path+img_path)
        label = self.data['Healthy?'].iloc[idx]
        patient = self.data['Folder Name'].iloc[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
       
        

class MILXray(Dataset):
    """Image CSV Dataset for Multiple Instance Learning"""
    type_map = {1: 0, 2: 1, 3: 2, 4: 2}
    inv_type_map = {0: '1-part', 1: '2-part', 2: '3- or 4-part'}

    def __init__(self, data: pd.DataFrame, data_path=None, transforms=None, seed=1, train=True, view_filter=None, save_dir=None):
        self.transforms = transforms
        self.data = data
        self.data_path = data_path
        self.train = train
        self.r = np.random.RandomState(seed=1)
        # Recode targets
        self.data['trg'] = self.data['num_parts'].map(Xray.type_map)

        # Fill nan views with "Unknown"
        self.data['view_group'] = self.data['view_group'].fillna('Unknown')
        
        # Filter to given views
        if view_filter is not None:
            self.data = self.data.loc[self.data['view_group'].isin(view_filter)]

        self.bags_list, self.labels_list, self.patient_ids_list = self._create_bags()
        
        if save_dir is not None:
            self.data.to_csv(save_dir)

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
                img = self.transforms(read_image(self.data_path+row['path']))
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
        record = {
            "name": patient_id,
            "study": bag,
            "label": bag_label
        }

        return record
    

class Xray(Dataset):
    """Image CSV Dataset for Multiple Instance Learning"""
    type_map = {1: 0, 2: 1, 3: 2, 4: 2}
    inv_type_map = {0: '1-part', 1: '2-part', 2: '3- or 4-part'}

    def __init__(self, data: pd.DataFrame, data_path=None, transforms=None, seed=1, train=True, view_filter = None, save_dir=None):
        self.transforms = transforms
        self.data = data
        self.data_path = data_path
        self.train = train
        self.r = np.random.RandomState(seed=1)
        # Recode targets
        self.data['trg'] = self.data['num_parts'].map(Xray.type_map)

        # Fill nan views with "Unknown"
        self.data['view_group'] = self.data['view_group'].fillna('Unknown')
        
        # Filter to given views
        if view_filter is not None:
            self.data = self.data.loc[self.data['view_group'].isin(view_filter)]
    
        if save_dir is not None:
            self.data.to_csv(save_dir)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        img = self.transforms(read_image(self.data_path+row['path']))
        trg = row['trg']

        return img, trg

    
    
class MILMRI(Dataset):
    def __init__(self, data, data_path=None, transforms=None, view_filter=None, save_dir=None):
        self.data = data
        self.data_path = data_path
        self.transforms = transforms
        
        # Filter to given views
        if view_filter is not None:
            self.data = self.data.loc[self.data['DCMSerDescr'].isin(view_filter)]
        
        if save_dir is not None:
            self.data.to_csv(save_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        series_name = self.data['SeriesUID'].iloc[idx]
        
        # read and group images with study
        # Read the DICOM object using pydicom.
        path = self.data_path+series_name
        files = os.listdir(path)
        study_imgs = []
        for i,file in enumerate(files):
            if '.dcm' in file:
                dicom = pydicom.dcmread(path+'/'+file)
                pixels = dicom.pixel_array
                img = torch.tensor(pixels,dtype=torch.uint8).unsqueeze(0)
                if self.transforms:
                    img = self.transforms(img)
                study_imgs.append(img)
        study = torch.concat(study_imgs)
        label = self.data['ClinSig'].iloc[idx]
        record = {
            "name": series_name,
            "study": study,
            "label": label
        }
        # return study, label
        return record
    
    
class MRI(Dataset):
    def __init__(self, data, data_path=None, transforms=None, view_filter=None, save_dir=None):
        self.data = data
        self.data_path = data_path
        self.transforms = transforms
        
        # Filter to given views
        if view_filter is not None:
            self.data = self.data.loc[self.data['DCMSerDescr'].isin(view_filter)]
        
        # Creating instance level dataframe
        temp_df = pd.DataFrame()
        num_ls = np.array([])
        count = 0
        for i,series in self.data.iterrows():
            num_slices = int(series['Dim'].split('x')[2])
            count+=num_slices
            df = pd.DataFrame([series]*num_slices, columns=self.data.columns)
            temp_df = pd.concat([temp_df, df])
            num_ls = np.append(num_ls,range(num_slices))
        slice_id = num_ls.flatten()
        self.data = temp_df
        self.data['sliceid'] = slice_id
        
        if save_dir is not None:
            self.data.to_csv(save_dir)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        series_name = self.data['SeriesUID'].iloc[idx]
        slice_id = int(self.data['sliceid'].iloc[idx])
        
        
        # read and group images with study
        # Read the DICOM object using pydicom.
        path = self.data_path+series_name
        files = os.listdir(path)
        files = [file for file in files if '.dcm' in file]
        file = files[slice_id]
        dicom = pydicom.dcmread(path+'/'+file)
        pixels = dicom.pixel_array
        img = torch.tensor(pixels,dtype=torch.uint8).unsqueeze(0)
        if self.transforms:
            img = self.transforms(img)
        label = self.data['ClinSig'].iloc[idx]
        
        return img, label
    
    
        
class MILCT(Dataset):
    def __init__(self, data, data_path=None, transforms=None, view_filter=None):
        self.data = data
        self.data_path = data_path
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        series_name = self.data['SeriesUID'].iloc[idx]
        
        # read and group images with study
        # Read the DICOM object using pydicom.
        path = self.data_path+series_name
        files = os.listdir(path)
        study_imgs = []
        for i,file in enumerate(files):
            if '.dcm' in file:
                dicom = pydicom.dcmread(path+'/'+file)
                pixels = dicom.pixel_array
                img = torch.tensor(pixels,dtype=torch.uint8).unsqueeze(0)
                if self.transforms:
                    img = self.transforms(img)
                study_imgs.append(img)
        study = torch.concat(study_imgs)
        label = self.data['Diagnosis'].iloc[idx]
        record = {
            "name": series_name,
            "study": study,
            "label": label
        }
        # return study, label
        return record
    
    
class CT(Dataset):
    def __init__(self, data, data_path=None, transforms=None, view_filter=None):
        self.data = data
        self.data_path = data_path
        self.transforms = transforms
        
        # Creating instance level dataframe
        temp_df = pd.DataFrame()
        num_ls = np.array([])
        count = 0
        for i,series in self.data.iterrows():
            path = self.data_path+series['SeriesUID']
            files = [file for file in os.listdir(path) if '.dcm' in file]
            num_slices = len(files)
            df = pd.DataFrame([series]*num_slices, columns=self.data.columns)
            temp_df = pd.concat([temp_df, df])
            num_ls = np.append(num_ls,range(num_slices))
        slice_id = num_ls.flatten()
        self.data = temp_df
        self.data['sliceid'] = slice_id
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        series_name = self.data['SeriesUID'].iloc[idx]
        slice_id = int(self.data['sliceid'].iloc[idx])
        
        
        # read and group images with study
        # Read the DICOM object using pydicom.
        path = self.data_path+series_name
        files = os.listdir(path)
        files = [file for file in files if '.dcm' in file]
        file = files[slice_id]
        dicom = pydicom.dcmread(path+'/'+file)
        pixels = dicom.pixel_array
        img = torch.tensor(pixels,dtype=torch.uint8).unsqueeze(0)
        if self.transforms:
            img = self.transforms(img)
        label = self.data['Diagnosis'].iloc[idx]
        
        return img, label