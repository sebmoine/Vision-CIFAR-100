import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import random_split, DataLoader, Dataset
from src.scripts.plot_data import show_image, show_image_per_label, show_labels_distribution, show_transformed_samples

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def get_dataloaders(config,use_cuda):

    batch_size  = config["batch_size"]
    num_workers = config["num_workers"]
    val_ratio   = config["val_ratio"]

    transform_train = transforms.Compose([
        # transforms.RandAugment(num_ops=2,               #Number of augmentation transformations to apply sequentially.
        #                        magnitude=7,             # Magnitude for all the transformations
        #                        num_magnitude_bins=30,   #The number of different magnitude values.
        #                        fill=None),

        # Geometry
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4), #Train the model to handle partial views.
        #Color & Appearance
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25), # Adjusts brightness, contrast, saturation, hue.

        transforms.ToTensor(),  #normalise par 255
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                            std=[0.2675, 0.2565, 0.2761]), #paramètres pour CIFAR100
        # Occlusion & Region-based
        transforms.RandomErasing(p=0.5,             # Removes a some shapes from the image.
                                scale=(0.02, 0.2),  # proportion de l'image à effacer
                                ratio=(0.3, 3.3),   # ratio hauteur/largeur de l'erasure
                                value=0,            # pixels noirs (0) ou 'mean'
                                inplace=False)      # True modifie directement le tensor
                                 
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                            std=[0.2675, 0.2565, 0.2761]) #paramètres pour CIFAR100

    ])

    train_full_dataset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=None)
    test_dataset  = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

    train_size = int((1-val_ratio) * len(train_full_dataset))
    val_size   = len(train_full_dataset) - train_size
    train_subset, val_subset = random_split(train_full_dataset, [train_size, val_size])

    train_dataset = SubsetWithTransform(train_subset, transform=transform_train)
    val_dataset   = SubsetWithTransform(val_subset, transform=transform_test)

    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size, pin_memory=use_cuda, drop_last=True, prefetch_factor = 2 if num_workers > 0 else None, persistent_workers=num_workers > 0)
    val_loader   = DataLoader(val_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size, pin_memory=use_cuda, drop_last=False, prefetch_factor = 2 if num_workers > 0 else None, persistent_workers=num_workers > 0)
    test_loader  = DataLoader(test_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size, pin_memory=use_cuda, drop_last=False, prefetch_factor = 2 if num_workers > 0 else None, persistent_workers=num_workers > 0)

    num_classes = len(train_dataset.subset.dataset.classes)
    images, _ = next(iter(train_loader))
    input_size = images.shape[1:]

    show_image(train_dataset)
    show_image_per_label(train_dataset)
    show_labels_distribution(train_dataset)
    show_transformed_samples(train_dataset)

    return train_loader, val_loader, test_loader, input_size, num_classes


class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def log_loader_info(loader):
    dataset = loader.dataset
    transform = getattr(dataset, "transform", None)
    
    if transform:
        transforms_str = "\n\t\t- ".join([str(t) for t in transform.transforms])
        transforms_str = "\n\t\t- " + transforms_str
    else:
        transforms_str = "\n\t\t- None"
    
    return f"Dataset CIFAR100\n\tNumber of datapoints : {len(dataset)}\n\tTransforms:{transforms_str}"
