import logging
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import random_split
from torch.utils.data import DataLoader
from src.scripts.plot_data import show_image, show_image_per_label, show_labels_distribution

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def get_dataloaders(config,use_cuda):

    batch_size  = config["batch_size"]
    num_workers = config["num_workers"]
    val_ratio   = config["val_ratio"]

    transform = transforms.Compose([
        transforms.ToTensor(),  #normalise par 255
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761]) #paramètres pour CIFAR100
    ])

    train_dataset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    test_dataset  = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

    train_size = int((1-val_ratio) * len(train_dataset))
    val_size   = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    #val_dataset.dataset.transform = transform_test

    logging.info(f"  - I loaded {len(train_dataset)} TRAIN samples")
    logging.info(f"  - I loaded {len(val_dataset)}  VAL samples")
    logging.info(f"  - I loaded {len(test_dataset)} TEST samples")


    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size, pin_memory=use_cuda, drop_last=True, prefetch_factor = 2 if num_workers > 0 else None, persistent_workers=num_workers > 0)
    val_loader   = DataLoader(val_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size, pin_memory=use_cuda, drop_last=False, prefetch_factor = 2 if num_workers > 0 else None, persistent_workers=num_workers > 0)
    test_loader  = DataLoader(test_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size, pin_memory=use_cuda, drop_last=False, prefetch_factor = 2 if num_workers > 0 else None, persistent_workers=num_workers > 0)

    num_classes = len(train_dataset.dataset.classes)
    images, _ = next(iter(train_loader))
    input_size = images.shape[1:]

    show_image(train_loader.dataset)
    show_image_per_label(train_loader.dataset)
    show_labels_distribution(train_loader.dataset)

    return train_loader, val_loader, test_loader, input_size, num_classes