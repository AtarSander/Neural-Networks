import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import utils


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


# def to_pil(x_tensor):
#     x = (x_tensor.clamp(-1, 1) + 1) * 0.5      
#     return transforms.ToPILImage()(x.cpu())