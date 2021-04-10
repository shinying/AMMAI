import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_loader(cfg):
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
    train_loader = DataLoader(trainset, cfg.batch_size, sampler=train_sampler,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(valset, cfg.test_batch_size, sampler=val_sampler, 
                            num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    
    return train_loader, val_loader

