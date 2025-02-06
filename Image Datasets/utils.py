import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import default_collate

def pad(h, num_imgs):
    # reshape as batch of studies and pad
    start_ix = 0
    h_ls = []
    for n in num_imgs:
        h_ls.append(h[start_ix:(start_ix+n)])
        start_ix += n
    h = pad_sequence(h_ls)

    return h

def num_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

class EarlyStopper:
    def __init__(self, patience, init_best_value = 1e10):
        self.patience = patience
        self._count_no_improvement = 0
        self.best_val = init_best_value
        
    def stop(self, val):
        if val >= self.best_val:
            self._count_no_improvement += 1
        else:
            self.best_val = val
            self._count_no_improvement = 0
            
        if self._count_no_improvement > self.patience:
            return True
        else:
            return False
        
    def __repr__(self):
        return f"EarlyStopper(patience={self.patience}). Current best: {self.best_val:0.5f}. Steps without improvement: {self._count_no_improvement}"

    
'''
Collate function based on Dr. Smith's implementation
'''
def custom_collate(batch):
    batch_studies = torch.concat([record["study"] for record in batch])
    batch_lengths = [len(record["study"]) for record in batch]
    
    record = {
        "study": batch_studies,
        "num_imgs": batch_lengths
    }
    for b in batch:
        b.pop('study')
    record.update(default_collate(batch))
    
    return record

'''
Collate function by Xiaofeng for xray dataset
'''
def collate_fn(batch):
    # print("collate_fn called")
    # Initialize empty lists for combined bags and labels
    combined_bag_images = []
    combined_instance_labels = []
    combined_bag_labels = []  # Collect all bag labels
    combined_patient_ids = []
    markers = [0]  # Start with 0, the index for the start of the first bag
    
    total_images = 0
    for bag, (bag_label, instance_labels, patient_id) in batch:
        num_images = bag.shape[0]  # Number of images in the current bag
        total_images += num_images
        combined_bag_images.append(bag)
        combined_instance_labels.append(instance_labels)
        combined_bag_labels.append(bag_label)  # Collect bag labels for each study bag
        combined_patient_ids.append(patient_id)
        markers.append(total_images)  # Append cumulative image count

    # Combine all study bags into one along the 0th dimension (number of images)
    combined_bag_images = torch.cat(combined_bag_images, dim=0)  # Combined shape: [n1 + n2 + ..., channel, h, w]
    combined_instance_labels = torch.cat(combined_instance_labels, dim=0)  # Combined instance labels

    return combined_bag_images, (combined_bag_labels, combined_instance_labels, combined_patient_ids, markers)