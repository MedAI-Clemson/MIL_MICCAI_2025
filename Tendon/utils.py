import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import default_collate

def pad(h, num_imgs):
    # reshape as batch of vids and pad
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
