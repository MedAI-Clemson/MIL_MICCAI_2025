import torchvision.transforms as tfm
import torch

def get_transforms(res, train=True):
    if train:
        return tfm.Compose([
            tfm.Resize(res),
            tfm.CenterCrop(res),
            tfm.RandomEqualize(p=0.5),
            tfm.RandAugment(),
            tfm.ToTensor(),
            tfm.Normalize(
                mean=torch.Tensor([0.4850, 0.4560, 0.4060]), 
                std=torch.Tensor([0.2290, 0.2240, 0.2250])
            ),
            tfm.RandomHorizontalFlip(),
            tfm.RandomRotation((-45, 45)),
            tfm.RandomInvert(),
            tfm.Grayscale()
        ])
    else:
        return tfm.Compose([
            tfm.Resize(res),
            tfm.CenterCrop(res),
            tfm.ToTensor(),
            tfm.Normalize(
                mean=torch.Tensor([0.4850, 0.4560, 0.4060]), 
                std=torch.Tensor([0.2290, 0.2240, 0.2250])
            ),
            tfm.Grayscale()
        ])