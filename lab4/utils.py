import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import utils
from collections import Counter


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def visualize_images(dataloader, classes, figsize=(20, 10), batch_size=16):
    plt.figure(figsize=figsize)
    images, labels = next(iter(dataloader))
    images, labels = next(iter(dataloader))

    imshow(utils.make_grid(images))

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


def create_conditioning_distribution(dataset, sample_size=1000):
    frequency = Counter(dataset.targets)
    total_count = len(dataset)
    exact = {label: (count / total_count) * sample_size for label, count in frequency.items()}
    floored = {label: int(np.floor(count)) for label, count in exact.items()}
    remainder = sample_size -  sum(floored.values())
    fractional_parts = {label: count - floored[label] for label, count in exact.items()}
    sorted_labels = sorted(fractional_parts, key=fractional_parts.get, reverse=True)
    for i in range(remainder):
        floored[sorted_labels[i]] += 1
    return floored


def create_conditioning_vector(frequency):
    conditioning_vector = []
    for label, count in frequency.items():
        conditioning_vector.extend([label] * count)
    return torch.tensor(conditioning_vector)


def show_generated(generated_images, range_low=0, range_high=-1):
    imshow(utils.make_grid(generated_images[range_low:range_high]))