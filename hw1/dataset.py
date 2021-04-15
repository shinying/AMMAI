import glob
import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import torch


train_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.001, .5)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
])
val_transforms = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
])


def get_loader(cfg):
    trainset = ImageFolder(cfg.rec, transform=train_transform)
    valset = ImageFolder(cfg.rec, transform=val_transforms)
    val_size = int(len(trainset) * cfg.val_split)
    
    print("Checking data ...", end=' ', flush=True)
    all_indices = set(range(len(trainset)))
    train_indices = []
    has_data = set()
    for i in range(len(trainset)):
        _, label = trainset[i]
        if label not in has_data:
            train_indices.append(i)
            has_data.add(label)
        if len(has_data) == len(trainset.classes):
            break
    
    all_indices = all_indices - set(train_indices)
    val_indices = np.random.choice(list(all_indices), val_size, replace=False)
    train_indices += list(all_indices - set(val_indices))
    
    print('Done')
    print(f"Classes: {len(trainset.classes)},",
          f"training size: {len(train_indices)},",
          f"validation size: {len(val_indices)}")
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(trainset, cfg.batch_size-cfg.mixup_batch_size*2, sampler=train_sampler,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(valset, cfg.test_batch_size, sampler=val_sampler, 
                            num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    
    return train_loader, val_loader, get_mixup_loader(cfg, trainset.class_to_idx)


class MixUp(Dataset):

    def __init__(self, root, class_to_idx, transform):
        self.files = glob.glob(os.path.join(root, '*.jpg'))
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        p1, p2 = os.path.basename(file).split('.')[0].split('_')
        p1, p2 = self.class_to_idx[p1], self.class_to_idx[p2]

        return self.transform(Image.open(file)), torch.tensor((p1, p2))


def get_mixup_loader(cfg, class_to_idx):
    trainset = MixUp(cfg.mixupdir, class_to_idx, train_transform)
    loader = DataLoader(trainset, cfg.mixup_batch_size, num_workers=cfg.num_workers, 
                        pin_memory=True, drop_last=True)
    return loader