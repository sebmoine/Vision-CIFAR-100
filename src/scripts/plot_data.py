import math
import matplotlib.pyplot as plt
import torch
import numpy as np
from collections import Counter
from pathlib import Path
from torchvision.utils import make_grid
FIGURES_ROOT = Path("outputs/figures")

def show_image(dataset):
    save_path = Path(f'{FIGURES_ROOT}/random_image.png')
    if save_path.exists():
        return
    idx = torch.randint(len(dataset), (1,)).item()
    img, label = dataset[idx]
    plt.figure()
    plt.imshow(img.permute(1,2,0))
    plt.title(f"Label: {dataset.dataset.classes[label] if hasattr(dataset, 'dataset') else dataset.classes[label]}")
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()

def show_image_per_label(dataset):
    save_path = Path(f'{FIGURES_ROOT}/every_labels.png')
    if save_path.exists():
        return
    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    classes = base_dataset.classes
    num_classes = len(classes)
    found = {}

    for i in range(len(dataset)):
        img, label = dataset[i]
        if label not in found:
            found[label] = img
        if len(found) == num_classes:
            break

    # grille automatique
    cols = math.ceil(math.sqrt(num_classes))
    rows = math.ceil(num_classes / cols)

    plt.figure(figsize=(cols*2, rows*2))
    for label, img in found.items():
        plt.subplot(rows, cols, label+1)
        plt.imshow(img.permute(1,2,0))
        plt.title(classes[label], fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def show_transformed_samples(dataset, num_samples=16):
    """
    Affiche des images après transformations appliquées par le dataset.
    
    Args:
        dataset: instance de torch Dataset avec des transforms
        num_samples: nombre d'images à afficher
        seed: pour reproductibilité
    """
    save_path = Path(f'{FIGURES_ROOT}/random_transformed_images.png')

    # Tirer quelques indices aléatoires
    indices = torch.randperm(len(dataset))[:num_samples]
    
    # Charger les images transformées
    images = []
    labels = []
    for idx in indices:
        img, label = dataset[idx]  # le transform est appliqué ici
        images.append(img)
        labels.append(label)
    
    # Faire une grille d'images
    grid = make_grid(images, nrow=int(num_samples**0.5), padding=2)
    
    # Convertir en format affichable (C, H, W) -> (H, W, C)
    np_grid = grid.permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(np.clip(np_grid, 0, 1))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def show_transformed_image_per_label(dataset, n=5):
    save_path = Path(f'{FIGURES_ROOT}/every_transformed_labels.png')
    if save_path.exists():
        return
    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    classes = base_dataset.classes

    found = {i: [] for i in range(len(classes))}

    for i in range(len(dataset)):
        img, label = dataset[i]
        if len(found[label]) < n:
            found[label].append(img)

        if all(len(v) == n for v in found.values()):
            break

    plt.figure(figsize=(n*2, len(classes)*2))

    for label in range(len(classes)):
        for j in range(n):
            plt.subplot(len(classes), n, label*n + j + 1)
            plt.imshow(denormalize(found[label][j]).permute(1,2,0))
            plt.axis("off")
            if j == 0:
                plt.ylabel(classes[label], fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def show_labels_distribution(dataset):
    save_path = Path(f'{FIGURES_ROOT}/labels_distibution.png')
    if save_path.exists():
        return
    labels = [dataset[i][1] for i in range(len(dataset))]
    counts = Counter(labels)

    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    classes = base_dataset.classes

    xs = list(counts.keys())
    ys = [counts[x] for x in xs]

    plt.figure(figsize=(20,5))
    plt.bar(xs, ys)
    plt.xticks(xs, [classes[x] for x in xs], rotation=90)
    plt.title("Label distribution")
    plt.savefig(save_path)
    plt.close()
