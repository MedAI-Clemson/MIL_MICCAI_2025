# scripts/create_dataset.py
import pandas as pd
import torch
import pickle
from pathlib import Path
from dataset import ImageDataBags
from transforms import get_transforms
from torch.utils.data import DataLoader

def collate_fn(batch):
    combined_bag_images = []
    combined_instance_labels = []
    combined_bag_labels = []
    combined_patient_ids = []
    markers = [0]
    
    total_images = 0
    for bag, (bag_label, instance_labels, patient_id) in batch:
        num_images = bag.shape[0]
        total_images += num_images
        combined_bag_images.append(bag)
        combined_instance_labels.append(instance_labels)
        combined_bag_labels.append(bag_label)
        combined_patient_ids.append(patient_id)
        markers.append(total_images)
        
    combined_bag_images = torch.cat(combined_bag_images, dim=0)
    combined_instance_labels = torch.cat(combined_instance_labels, dim=0)
    
    return combined_bag_images, (combined_bag_labels, combined_instance_labels, combined_patient_ids, markers)

def main(targets_path, images_path, output_path, resolution=224):
    # Get transforms
    transform_train = get_transforms(res=resolution, train=True)
    transform_test = get_transforms(res=resolution, train=False)
    
    # Read data
    df_trg = pd.read_csv(targets_path)
    df_img = pd.read_csv(images_path)
    
    # Load splits
    with open('results/cv_splits.pkl', 'rb') as f:
        saved_splits = pickle.load(f)
    
    # Parameters for dataloaders
    bs_train = 32
    bs_test = 32
    num_workers = 4
    keep_views = None
    
    # Process each split
    for ix, (train_ix, test_ix) in enumerate(saved_splits):
        print(f"Processing split {ix+1} of {len(saved_splits)}")
        
        # Create train and test dataframes for this split
        df_train = df_trg.iloc[train_ix].merge(df_img)
        df_test = df_trg.iloc[test_ix].merge(df_img)
        
        # Save to CSV with split number
        df_train.to_csv(f'results/train_dataset_split_{ix+1}.csv', index=False)
        df_test.to_csv(f'results/test_dataset_split_{ix+1}.csv', index=False)
        
        # Create datasets
        train_dataset = ImageDataBags(
            df_train, 
            transforms=transform_train, 
            view_filter=keep_views
        )
        
        test_dataset = ImageDataBags(
            df_test, 
            transforms=transform_test, 
            view_filter=keep_views
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=bs_train, 
            num_workers=num_workers, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=bs_test, 
            num_workers=num_workers, 
            collate_fn=collate_fn
        )
        
        # Save datasets and dataloaders
        torch.save(train_dataset, f'results/train_dataset_split_{ix+1}.pt')
        torch.save(test_dataset, f'results/test_dataset_split_{ix+1}.pt')
        torch.save(train_loader, f'results/train_loader_split_{ix+1}.pt')
        torch.save(test_loader, f'results/test_loader_split_{ix+1}.pt')

if __name__ == "__main__":
    snakemake_in_targets = snakemake.input.targets
    snakemake_in_images = snakemake.input.images
    snakemake_out = snakemake.output[0]
    resolution = snakemake.config.get('resolution', 224)
    main(snakemake_in_targets, snakemake_in_images, snakemake_out, resolution)