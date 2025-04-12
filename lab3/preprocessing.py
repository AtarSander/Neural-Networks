from PIL import Image
from torch.utils.data import Dataset
from torchvision import utils
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def visualize_images(dataloader, classes, figsize=(20, 10), batch_size=16):
    plt.figure(figsize=figsize)
    images, labels = next(iter(dataloader))
    images, labels = next(iter(dataloader))

    imshow(torchvision.utils.make_grid(images))

    print(" ".join("%5s" % classes[labels[j]] for j in range(batch_size)))


def verify_image_sizes(dataloader):
    sizes = {}
    for images, _ in dataloader:
        for image in images:
            size = image.size()
            sizes[size] = sizes.get(size, 0) + 1
    return sizes


def plot_bar(values, title, size=(10, 4)):
    plt.figure(figsize=size)
    plt.bar(*zip(*values.items()))
    plt.title(title)
    plt.show()


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.data = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(path)


def show_class_grid(dataset, class_name, num_images=16, nrow=4):
    class_idx = dataset.class_to_idx[class_name]
    found_images = []

    for img, label in dataset:
        if label == class_idx:
            found_images.append(img)
            if len(found_images) == num_images:
                break
    if not found_images:
        print(f"No images found for class '{class_name}'")
        return

    grid = utils.make_grid(torch.stack(found_images), nrow=nrow, padding=2)
    imshow(grid)
